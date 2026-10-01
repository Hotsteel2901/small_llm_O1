# -*- coding: utf-8 -*-
"""与训练好的 TinyGPT 对话"""
import sys, argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer

from model import TinyGPT

HERE = Path(__file__).parent
PAD_ID, BOS_ID, EOS_ID, USER_ID, ASSIST_ID = 0, 1, 2, 4, 5


def load(model_path=None):
    ck = torch.load(model_path or (HERE / "ckpt" / "best.pt"), map_location="cpu", weights_only=False)
    a = ck["args"]
    model = TinyGPT(
        vocab_size=ck["vocab_size"], d_model=a["d_model"], n_layers=a["layers"],
        n_heads=a["heads"], n_kv_heads=a["kv_heads"], max_seq_len=a["seq"],
    )
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(str(HERE / (ck.get("tokenizer", "tokenizer_v4.json"))))
    return model, tok, ck


def build_prompt(tok, history):
    ids = [BOS_ID]
    for role, text in history:
        ids.append(USER_ID if role == "user" else ASSIST_ID)
        ids.extend(tok.encode(text).ids)
    ids.append(ASSIST_ID)
    return ids


@torch.no_grad()
def reply(model, tok, history, max_new=120, temperature=0.75, top_k=40,
          top_p=0.9, rep=1.15):
    ids = build_prompt(tok, history)
    x = torch.tensor([ids[-model.max_seq_len:]], dtype=torch.long)
    out = model.generate(x, max_new_tokens=max_new, temperature=temperature,
                         top_k=top_k, top_p=top_p, repetition_penalty=rep,
                         eos_id=EOS_ID)
    gen = out[0, len(ids[-model.max_seq_len:]):].tolist()
    # 截断到 eos, 并去掉特殊 token
    clean = []
    for t in gen:
        if t == EOS_ID:
            break
        if t in (PAD_ID, BOS_ID, USER_ID, ASSIST_ID):
            continue
        clean.append(t)
    return tok.decode(clean).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--temp", type=float, default=0.75)
    ap.add_argument("--max-new", type=int, default=120)
    args = ap.parse_args()

    model, tok, ck = load(args.model)
    print(f"[模型] {model.num_params()/1e6:.1f}M 参数, 训练 {ck['step']} 步, val_loss={ck['val_loss']:.4f}\n")

    if args.prompt:
        ans = reply(model, tok, [("user", args.prompt)], args.max_new, args.temp)
        print(f"你: {args.prompt}")
        print(f"AI: {ans}")
        return

    print("进入对话模式 (输入 exit 退出, clear 清空上下文)")
    history = []
    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("exit", "quit", "退出"):
            break
        if q == "clear":
            history = []
            print("(已清空上下文)")
            continue
        history.append(("user", q))
        a = reply(model, tok, history, args.max_new, args.temp)
        print(f"AI: {a}\n")
        history.append(("assistant", a))
        if len(history) > 12:
            history = history[-12:]


if __name__ == "__main__":
    main()
