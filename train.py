# -*- coding: utf-8 -*-
"""
TinyGPT 训练脚本 (CPU, 内存预算 8G)

设计:
  - 小 batch + 梯度累积 -> 在有限内存下达到大有效批大小
  - 余弦退火 + warmup
  - 梯度裁剪、验证集早停保护
  - 保存 loss 曲线与检查点
"""
import json, math, time, os, argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import TinyGPT

HERE = Path(__file__).parent
DATA = HERE / "data"
CKPT = HERE / "ckpt"
CKPT.mkdir(exist_ok=True)

# ---------------- 特殊 token ----------------
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
USER_ID = 4
ASSIST_ID = 5
IGNORE = -100


def load_tokenizer(name="tokenizer.json"):
    return Tokenizer.from_file(str(HERE / name))


def encode_conversation(tok, messages, max_len):
    """
    把一轮会话编码成训练序列。
    结构: <bos> <|user|> ... <|assistant|> ... <eos>
    loss 只在 assistant 的回答部分计算。

    重要: 返回的 labels 已经"右移对齐" ——
      labels[i] 是模型在位置 i 处应当预测的下一个 token。
      即 logits[i] (由 idx[0..i] 得到) 对应 labels[i]。
    因此 labels[i] == ids[i+1]。
    """
    ids, mask = [BOS_ID], [IGNORE]
    for m in messages:
        role_id = USER_ID if m["role"] == "user" else ASSIST_ID
        content_ids = tok.encode(m["content"]).ids
        ids.append(role_id)
        mask.append(1 if m["role"] == "assistant" else 0)
        ids.extend(content_ids)
        mask.extend([1 if m["role"] == "assistant" else 0] * len(content_ids))
    ids.append(EOS_ID)
    mask.append(1)  # 模型要学会在回答结束后输出 EOS

    ids = ids[:max_len]
    mask = mask[:max_len]

    # 右移对齐: 位置 i 的目标 = 原序列的 i+1 位置
    labels = [ids[i + 1] if (i + 1 < len(ids) and mask[i + 1] == 1) else IGNORE
              for i in range(len(ids))]
    return ids, labels


def build_dataset(tok, path, max_len):
    seqs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids, labels = encode_conversation(tok, obj["messages"], max_len)
            if sum(1 for x in labels if x != IGNORE) >= 4:  # 至少有点可学的
                seqs.append((ids, labels))
    return seqs


def make_batch(batch, pad_id=PAD_ID):
    maxlen = max(len(x[0]) for x in batch)
    B = len(batch)
    ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    labels = torch.full((B, maxlen), IGNORE, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
    return ids, labels


def lr_at(step, total, warmup, base_lr, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, prog)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tokenizer", type=str, default="tokenizer.json")
    ap.add_argument("--data", type=str, default="train_v4.jsonl")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    tok = load_tokenizer(args.tokenizer)
    vocab_size = tok.get_vocab_size()
    print(f"词表: {vocab_size}")

    print("加载数据...")
    all_seqs = build_dataset(tok, DATA / args.data, args.seq)
    print(f"样本数: {len(all_seqs)}")
    total_tokens = sum(len(s[0]) for s in all_seqs)
    print(f"总 token: {total_tokens}")

    # 划分验证集
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(all_seqs), generator=rng).tolist()
    n_val = max(64, len(all_seqs) // 20)
    val_seqs = [all_seqs[i] for i in perm[:n_val]]
    train_seqs = [all_seqs[i] for i in perm[n_val:]]
    print(f"训练 {len(train_seqs)} / 验证 {len(val_seqs)}")

    model = TinyGPT(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.layers,
        n_heads=args.heads, n_kv_heads=args.kv_heads, max_seq_len=args.seq,
        dropout=0.1,
    )
    print(f"参数量: {model.num_params()/1e6:.2f}M (非嵌入 {model.num_params(True)/1e6:.2f}M)")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    history = {"step": [], "train_loss": [], "val_loss": [], "lr": [], "elapsed": []}
    t_start = time.time()
    best_val = float("inf")
    step = 0
    ptr = 0

    def get_batch(seqs, bs):
        nonlocal ptr
        out = []
        for _ in range(bs):
            if ptr >= len(seqs):
                perm2 = torch.randperm(len(seqs), generator=rng).tolist()
                seqs[:] = [seqs[i] for i in perm2]
                ptr = 0
            out.append(seqs[ptr]); ptr += 1
        return make_batch(out)

    @torch.no_grad()
    def evaluate(n=32):
        model.eval()
        losses = []
        idx = 0
        for _ in range(n):
            b = [val_seqs[(idx + j) % len(val_seqs)] for j in range(args.batch)]
            idx += args.batch
            x, y = make_batch(b)
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    print(f"\n开始训练: {args.steps} 步, 有效 batch = {args.batch*args.accum}\n")
    print(f"{'step':>6} {'loss':>8} {'val':>8} {'lr':>9} {'p/pl':>7} {'耗时':>8}")

    running = 0.0
    run_cnt = 0
    while step < args.steps:
        lr = lr_at(step, args.steps, args.warmup, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.accum):
            x, y = get_batch(train_seqs, args.batch)
            _, loss = model(x, y)
            (loss / args.accum).backward()
            step_loss += loss.item() / args.accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running += step_loss; run_cnt += 1
        step += 1

        if step % 25 == 0 or step == 1:
            avg = running / run_cnt
            running, run_cnt = 0.0, 0
            el = time.time() - t_start
            ppl = math.exp(min(20, avg))
            print(f"{step:>6} {avg:>8.4f} {'-':>8} {lr:>9.2e} {ppl:>7.2f} {el/60:>7.1f}m")

        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate()
            avg = step_loss
            el = time.time() - t_start
            print(f"{step:>6} {avg:>8.4f} {v:>8.4f} {lr:>9.2e} {math.exp(min(20,v)):>7.2f} {el/60:>7.1f}m  <-- eval")
            history["step"].append(step)
            history["train_loss"].append(avg)
            history["val_loss"].append(v)
            history["lr"].append(lr)
            history["elapsed"].append(el)

            if v < best_val:
                best_val = v
                torch.save({
                    "model": model.state_dict(),
                    "args": vars(args),
                    "vocab_size": vocab_size,
                    "step": step, "val_loss": v, "tokenizer": args.tokenizer,
                }, CKPT / "best.pt")
                print(f"        新的最佳验证 loss {v:.4f}, 已保存")

    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "vocab_size": vocab_size,
        "step": step, "val_loss": best_val, "tokenizer": args.tokenizer,
    }, CKPT / "final.pt")

    with (CKPT / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    print(f"\n训练完成, 用时 {(time.time()-t_start)/60:.1f} 分钟")
    print(f"最佳验证 loss: {best_val:.4f} (ppl {math.exp(min(20,best_val)):.2f})")
    print(f"检查点: {CKPT/'best.pt'}")


if __name__ == "__main__":
    main()
