# -*- coding: utf-8 -*-
"""绘制训练曲线 + 生成模型自测报告"""
import json, argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent

# 中文字体
for cand in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "training_curve.png"))
    args = ap.parse_args()

    h = json.loads((HERE / "ckpt" / "history.json").read_text())
    steps = h["step"]
    tr, va = h["train_loss"], h["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ax.plot(steps, tr, "o-", color="#c0392b", lw=2, ms=5, label="训练 loss")
    ax.plot(steps, va, "s-", color="#2980b9", lw=2, ms=5, label="验证 loss")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("交叉熵 loss")
    ax.set_title("TinyGPT 训练曲线 (15.04M 参数, CPU 从零训练)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    import math
    ax2.plot(steps, [math.exp(min(20, x)) for x in tr], "o-", color="#c0392b", lw=2, ms=5, label="训练 PPL")
    ax2.plot(steps, [math.exp(min(20, x)) for x in va], "s-", color="#2980b9", lw=2, ms=5, label="验证 PPL")
    ax2.set_xlabel("训练步数")
    ax2.set_ylabel("困惑度 Perplexity")
    ax2.set_title("困惑度下降曲线 (越低越好)")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, facecolor="white")
    print("已保存:", args.out)
    print(f"最终验证 loss: {va[-1]:.4f}  PPL: {math.exp(min(20,va[-1])):.3f}")


if __name__ == "__main__":
    main()
