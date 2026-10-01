# -*- coding: utf-8 -*-
"""模型评测: 用一组留出的问题检测回答质量"""
import json, torch
from pathlib import Path
from tokenizers import Tokenizer
from model import TinyGPT

HERE = Path(__file__).parent
EOS_ID, USER_ID, ASSIST_ID, BOS_ID = 2, 4, 5, 1

# 留出测试集: 用与训练不同的问法, 检验泛化
TEST = [
    ("你好", None),
    ("你是谁", None),
    ("在吗", None),
    ("早上好", None),
    ("晚上的时候，天空为什么是黑的", None),
    ("雨是怎么来的", None),
    ("月亮为什么有时候圆有时候缺", None),
    ("人为什么要喝水", None),
    ("为什么打哈欠", None),
    ("我压力好大", None),
    ("我觉得很孤单", None),
    ("我今天特别高兴", None),
    ("我搞砸了一件事", None),
    ("如果加班我就不去，今天加班了，我去吗", None),
    ("所有的鸟都会飞，企鹅是鸟，企鹅会飞吗", None),
    ("苹果比橘子贵，橘子比香蕉贵，哪个最便宜", None),
    ("三个人平分九十个苹果，每人多少", None),
    ("什么是机器学习", None),
    ("缓存是什么意思", None),
    ("怎么才能睡得好", None),
    ("怎么准备考试", None),
    ("怎么改掉拖延", None),
    ("谢谢你", None),
    ("再见", None),
    ("我有个问题", None),
]


@torch.no_grad()
def generate(model, tok, prompt, max_new=100, greedy=True, temperature=0.8, top_k=40):
    ids = [BOS_ID, USER_ID] + tok.encode(prompt).ids + [ASSIST_ID]
    x = torch.tensor([ids])
    out = []
    for _ in range(max_new):
        logits, _ = model(x[:, -model.max_seq_len:])
        logits = logits[0, -1]
        if greedy:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = -float("inf")
            nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
        if nxt == EOS_ID:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    return tok.decode(out).strip()


def main():
    ck = torch.load(HERE / "ckpt" / "best.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    model = TinyGPT(vocab_size=ck["vocab_size"], d_model=a["d_model"], n_layers=a["layers"],
                    n_heads=a["heads"], n_kv_heads=a["kv_heads"], max_seq_len=a["seq"])
    model.load_state_dict(ck["model"]); model.eval()
    tok = Tokenizer.from_file(str(HERE / ck.get("tokenizer", "tokenizer_v5.json")))

    print(f"模型: {model.num_params()/1e6:.1f}M 参数 | 训练 {ck['step']} 步 | val_loss {ck['val_loss']:.4f}")
    print("=" * 70)
    ok = 0
    for q, _ in TEST:
        a_ = generate(model, tok, q)
        # 简单判定: 是否非空且不是明显退化
        good = bool(a_) and len(a_) > 1 and not (len(set(a_)) <= 3)
        ok += good
        print(f"[{'OK' if good else '??'}] 问: {q}")
        print(f"     答: {a_ if a_ else '(空)'}")
        print()
    print("=" * 70)
    print(f"有效回答: {ok}/{len(TEST)} ({ok/len(TEST)*100:.0f}%)")


if __name__ == "__main__":
    main()
