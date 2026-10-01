---
language:
  - zh
license: mit
library_name: pytorch
tags:
  - text-generation
  - conversational
  - chinese
  - gpt
  - from-scratch
  - tiny-llm
pipeline_tag: text-generation
model-index:
  - name: TinyGPT-zh-15M
    results:
      - task:
          type: text-generation
        dataset:
          name: self-built Chinese dialogue corpus
          type: custom
        metrics:
          - name: Validation Loss
            type: loss
            value: 0.1128
          - name: Validation Perplexity
            type: perplexity
            value: 1.12
---

# TinyGPT — 从零训练的中文对话模型

一个用 PyTorch 手写实现、从随机初始化开始训练的中文对话语言模型。**没有使用任何预训练权重** —— 全部参数都是在 32 核 CPU 上训练出来的（无 GPU）。

## 模型详情

| 项目 | 数值 |
|---|---|
| 参数量 | 15.04M（非嵌入 12.59M） |
| 架构 | Decoder-only Transformer |
| 层数 | 8 |
| 隐藏维度 | 384 |
| 注意力头 | 6 query heads / 2 KV heads（GQA） |
| 序列长度 | 256 |
| 词表大小 | 6374（自训练 BPE） |
| 训练数据 | 84.5 万 token / 1.97 万条对话 |
| 训练步数 | 6000 |
| 最终验证 loss | **0.1128** |
| 验证困惑度 | 1.12 |
| 训练耗时 | 134 分钟（32 核 CPU） |

## 架构特点

全部手写实现，不依赖 `transformers` 的模型代码：

- **RMSNorm** 前置归一化
- **RoPE** 旋转位置编码
- **GQA** 分组查询注意力，8 个 query 头共享 2 组 KV
- **SwiGLU** 前馈网络
- **权重共享**：输入 embedding 与输出投影共用参数

## 使用方法

```python
import torch
from tokenizers import Tokenizer
from model import TinyGPT

EOS_ID, USER_ID, ASSIST_ID, BOS_ID = 2, 4, 5, 1

ck = torch.load("ckpt/best.pt", map_location="cpu", weights_only=False)
a = ck["args"]
model = TinyGPT(
    vocab_size=ck["vocab_size"], d_model=a["d_model"], n_layers=a["layers"],
    n_heads=a["heads"], n_kv_heads=a["kv_heads"], max_seq_len=a["seq"],
)
model.load_state_dict(ck["model"])
model.eval()
tok = Tokenizer.from_file("tokenizer_v5.json")

ids = [BOS_ID, USER_ID] + tok.encode("为什么会下雨").ids + [ASSIST_ID]
x = torch.tensor([ids])
out = []
with torch.no_grad():
    for _ in range(100):
        logits, _ = model(x[:, -model.max_seq_len:])
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS_ID:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
print(tok.decode(out))
# 地面的水蒸发成水蒸气升到空中，遇冷凝结成小水滴形成云，
# 小水滴不断聚集变大，空气托不住时就落下来，形成雨。
```

## 对话示例

```
你  > 为什么会下雨
AI  > 地面的水蒸发成水蒸气升到空中，遇冷凝结成小水滴形成云，
      小水滴不断聚集变大，空气托不住时就落下来，形成雨。

你  > 我心情不好
AI  > 难过是很自然的反应，说明这件事对你确实重要。慢慢来，不着急。

你  > 苹果比橘子贵，橘子比香蕉贵，哪个最便宜
AI  > 香蕉最便宜。价格排序是苹果最贵、橘子居中、香蕉最低。

你  > 如果加班我就不去，今天加班了，我去吗
AI  > 按你的规则，加班就不去，今天加班了，所以不去。

你  > 怎么才能睡得好
AI  > 可以这样一步步来：每天固定时间上床；睡前一小时别刷刺激内容；
      房间调暗调静；白天适当运动。
```

## 训练数据

语料由教师模型生成（知识蒸馏），按任务类型均衡构建：

| 任务类型 | 内容 |
|---|---|
| 常识问答 | 天气、天文、生物、生活等 41 个主题 |
| 概念定义 | 25 个技术/抽象概念 |
| 操作教程 | 16 个「怎么做」的分步骤说明 |
| 情绪陪伴 | 8 类情绪，每类多个恰当回应 |
| 逻辑推理 | 20 道推理题 |
| 闲聊对话 | 29 种日常问法与自我认知 |
| 算术换算 | 计算与单位换算 |

随后包装成多轮对话流（最多 8 轮），把语料从 27 万 token 扩到 84.5 万 token。

## 训练曲线

验证 loss 从 0.43 平滑下降到 0.113，全程与训练 loss 贴合，没有过拟合。

见仓库中的 `training_curve.png`。

## 能力边界

这是一个 15M 参数的小模型，请合理预期。

**擅长**：常见常识问题、情绪回应、分步骤说明、简单逻辑推理、日常闲聊

**不擅长**：
- 算术计算（会算错，如 `9×7` 答成 27）
- 训练语料之外的话题
- 长文本生成（超过两三句后容易跑题）
- 需要多步推理的复杂问题

**重要**：模型只是在模仿训练语料的语言模式，**回答不保证事实正确**。

## 复现训练

```bash
python3 train.py \
  --tokenizer tokenizer_v5.json \
  --data train_v5.jsonl \
  --steps 6000 \
  --d-model 384 --layers 8 --heads 6 --kv-heads 2 \
  --seq 256 --batch 8 --accum 3 \
  --lr 2.5e-3 --eval-every 500
```

## 仓库结构

```
tiny-gpt/
├── model.py             GPT 架构实现（RoPE / GQA / SwiGLU / RMSNorm）
├── train.py             训练脚本（梯度累积 / 余弦退火 / 验证集）
├── chat.py              交互式对话
├── demo.py              批量演示
├── evaluate.py          评测脚本
├── plot.py              训练曲线绘制
├── gen_v4.py            语料生成 v4（知识库构建）
├── gen_v5.py            语料生成 v5（扩充成对话流）
├── tokenizer_v5.json    训练好的 BPE 分词器
├── training_curve.png   训练曲线
├── ckpt/best.pt         模型权重
├── data/train_v5.jsonl  训练语料
└── logs/train_final.log 完整训练日志
```

## 引用

```
@misc{tinygpt-zh-15m,
  title  = {TinyGPT: A 15M Small Chinese Dialogue Model Trained From Scratch},
  author = {hotsteel09},
  year   = {2026},
  url    = {https://huggingface.co/hotsteel09/small_llm_O1}
}
```
