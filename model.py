# -*- coding: utf-8 -*-
"""
TinyGPT —— 从零实现的 Decoder-only Transformer
现代架构要点:
  - RMSNorm 归一化 (前置)
  - RoPE 旋转位置编码
  - GQA 分组查询注意力 (省 KV cache)
  - SwiGLU 前馈网络
  - 权重共享 embedding / lm_head
纯手写, 不依赖 transformers 的模型代码。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 用 float32 计算方差, 数值更稳
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def precompute_rope(head_dim, max_seq_len, base=10000.0, device=None):
    """预计算 RoPE 的 cos/sin 表"""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)          # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)   # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: (B, H, T, D)
    T = q.shape[-2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # GQA: 把 kv 复制到 query 的头数
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)  # gate
        self.w3 = nn.Linear(d_model, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, d_model, bias=False)  # down
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, inter, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, inter, dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=8, n_heads=6,
                 n_kv_heads=2, max_seq_len=256, dropout=0.0, tie_weights=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # SwiGLU 中间维度, 按 8/3 比例对齐到 64 的倍数
        inter = int(8 * d_model / 3)
        inter = ((inter + 63) // 64) * 64

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, n_kv_heads, inter, dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE 表 (buffer, 随设备移动)
        cos, sin = precompute_rope(d_model // n_heads, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_emb=False):
        n = sum(p.numel() for p in self.parameters())
        if non_emb:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"seq len {T} > {self.max_seq_len}"
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=128, temperature=0.8, top_k=40,
                 top_p=0.9, repetition_penalty=1.1, eos_id=None, rep_window=48):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            # 重复惩罚: 只惩罚"近期出现过"的 token, 避免误伤常用字
            if repetition_penalty != 1.0:
                for b in range(idx.shape[0]):
                    recent = torch.unique(idx[b, -rep_window:])
                    tgt = logits[b, recent]
                    logits[b, recent] = torch.where(
                        tgt > 0, tgt / repetition_penalty, tgt * repetition_penalty
                    )

            # top-k
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")

            # top-p (nucleus)
            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                remove = cum - probs > top_p
                sorted_logits[remove] = -float("inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx


if __name__ == "__main__":
    m = TinyGPT(vocab_size=6359, d_model=384, n_layers=8, n_heads=6, n_kv_heads=2)
    print(f"总参数: {m.num_params()/1e6:.2f}M")
    print(f"非嵌入参数: {m.num_params(non_emb=True)/1e6:.2f}M")
    x = torch.randint(0, 6359, (2, 64))
    logits, _ = m(x)
    print("前向输出:", logits.shape)
