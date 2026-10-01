# -*- coding: utf-8 -*-
"""
TinyGPT 演示 —— 快速查看模型对不同问题的回答
直接运行: python3 demo.py
"""
import torch
from pathlib import Path
from tokenizers import Tokenizer
from model import TinyGPT

HERE = Path(__file__).parent
EOS_ID, USER_ID, ASSIST_ID, BOS_ID = 2, 4, 5, 1

DEMO = [
    "你好",
    "你是谁",
    "再见",
    "为什么会下雨",
    "太阳有多远",
    "人为什么要喝水",
    "什么是机器学习",
    "人工智能是什么",
    "我心情不好",
    "我压力好大",
    "我很开心",
    "怎么才能睡得好",
    "怎么改掉拖延",
    "苹果比橘子贵，橘子比香蕉贵，哪个最便宜",
    "如果加班我就不去，今天加班了，我去吗",
    "谢谢你",
]


@torch.no_grad()
def generate(model, tok, prompt, max_new=100, temperature=None, top_k=None):
    ids = [BOS_ID, USER_ID] + tok.encode(prompt).ids + [ASSIST_ID]
    x = torch.tensor([ids])
    out = []
    for _ in range(max_new):
        logits, _ = model(x[:, -model.max_seq_len:])
        logits = logits[0, -1]
        if temperature:                       # 采样模式
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = -float("inf")
            nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
        else:                                  # 贪心模式 (默认, 最稳)
            nxt = int(logits.argmax())
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
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(str(HERE / "tokenizer_v5.json"))

    print("=" * 66)
    print(f" TinyGPT  ——  从零训练的中文对话模型")
    print(f" 参数 {model.num_params()/1e6:.2f}M  |  训练 {ck['step']} 步  |  验证 loss {ck['val_loss']:.4f}")
    print("=" * 66)

    for q in DEMO:
        ans = generate(model, tok, q)
        print(f"\n你  > {q}")
        print(f"AI  > {ans}")

    print("\n" + "=" * 66)
    print("提示: 运行 python3 chat.py 可以进入交互式对话")


if __name__ == "__main__":
    main()
