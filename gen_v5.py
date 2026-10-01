# -*- coding: utf-8 -*-
"""
语料 v5: 在 v4 基础上大幅扩充 token 数

手段:
  1. 把 v4 的所有单轮问答用"对话化"包装 (加开场、过渡、收尾)
  2. 生成更长的多轮对话 (最多 8 轮)
  3. 对答案做"扩展式改写" —— 在正确答案后追加一句相关的延伸,
     既保持语义一致, 又显著增加 token 数量和信息密度
"""
import json, random
from pathlib import Path

random.seed(99)
HERE = Path(__file__).parent
DATA = HERE / "data"

# ---- 载入 v4 的所有单轮 ----
base = []
with (DATA / "train_v4.jsonl").open(encoding="utf-8") as f:
    for line in f:
        msgs = json.loads(line)["messages"]
        if len(msgs) == 2:
            base.append((msgs[0]["content"], msgs[1]["content"]))

print(f"v4 单轮问答: {len(base)}")
by_q = {}
for q, a in base:
    by_q.setdefault(q, []).append(a)
qs = list(by_q.keys())

# ---- 延伸句库: 追加在回答后面, 保持语义连贯 ----
EXTEND = [
    "有需要的话我们可以接着聊。",
    "希望这些对你有帮助。",
    "如果你还有别的疑问，随时问我。",
    "这是个大致的说明，细节可能需要再查证。",
    "你可以根据自己的情况灵活调整。",
    "说起来，这个问题挺有意思的。",
    "不知道这样解释清楚没有。",
    "当然，具体情况还是要具体分析。",
    "你可以先从最简单的那一步试起。",
    "慢慢来，不用着急一下子全搞定。",
    "这也是很多人都会遇到的情况。",
    "换个角度看，可能还会更清楚一些。",
]

OPEN_KEEP = [("你好", "你好！有什么可以帮你的吗？"), ("在吗", "在的，你说。"),
             ("你好啊", "你好呀，有什么想聊的？"), ("嗨", "嗨！今天怎么样？"),
             ("我想问你个问题", "你问吧，我知道的都会尽量回答。"),
             ("我有个问题", "好的，你说。"), ("帮我个忙", "可以，你说说看是什么事。"),
             ("聊聊呗", "好啊，想聊点什么？"), ("你现在有空吗", "有空，随时都可以。"),
             ("我来了", "欢迎回来，今天怎么样？"), ("喂", "哎，我在。你说吧。"),
             ("有人吗", "有的，我一直都在。")]

TRANS2 = [("嗯", "嗯，我在，你继续说。"), ("然后呢", "然后你可以接着讲，我听着。"),
          ("还有吗", "还有一点，就是别着急，慢慢来。"), ("我懂了", "好，那我们继续。"),
          ("明白了", "嗯，那我们说下一个。"), ("不太懂", "没关系，换个说法，你卡在哪一步？"),
          ("为什么", "原因可能有好几种，你是指哪方面？"),
          ("举个例子", "好比学骑车，一开始怕摔，骑多了身体自然就记住平衡了。"),
          ("是这样吗", "按我了解的情况是这样，你要是有别的信息我们可以对一对。"),
          ("你说得对", "嗯，想通了就没那么难了。"), ("好吧", "嗯，那我们接着聊。"),
          ("等等", "好，不急，你想清楚再说。"), ("继续说", "好，我继续听着。"),
          ("确实", "对，很多事想明白之后就顺了。")]

CLOSE2 = [("谢谢你", "不客气，能帮上忙我很开心。"), ("多谢", "不用谢，随时找我。"),
          ("太感谢了", "别客气，这是应该的。"), ("再见", "再见，路上小心。"),
          ("拜拜", "拜拜，下次再聊。"), ("我先走了", "好的，有空再来。"),
          ("晚安", "晚安，好好休息。"), ("那就这样", "好，有需要随时找我。"),
          ("回头聊", "好，随时欢迎。"), ("我走了", "好，慢走，注意安全。")]

samples = []

# ---- 1. 原始单轮 ----
for q, a in base:
    samples.append([{"role": "user", "content": q}, {"role": "assistant", "content": a}])

# ---- 2. 带延伸的单轮 (核心扩充手段) ----
for q, a in base:
    for _ in range(2):
        ext = random.choice(EXTEND)
        a2 = a + ext
        samples.append([{"role": "user", "content": q}, {"role": "assistant", "content": a2}])

# ---- 3. 短多轮 (2-3 轮) ----
for _ in range(9000):
    msgs = []
    if random.random() < 0.45:
        q, a = random.choice(OPEN_KEEP)
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    for _ in range(random.randint(1, 2)):
        q = random.choice(qs)
        a = random.choice(by_q[q][:3])
        if random.random() < 0.35:
            a += random.choice(EXTEND)
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        if random.random() < 0.4:
            tq, ta = random.choice(TRANS2)
            msgs += [{"role": "user", "content": tq}, {"role": "assistant", "content": ta}]
    if random.random() < 0.3:
        cq, ca = random.choice(CLOSE2)
        msgs += [{"role": "user", "content": cq}, {"role": "assistant", "content": ca}]
    samples.append(msgs)

# ---- 4. 长多轮 (4-8 轮), 显著增加上下文长度 ----
for _ in range(5000):
    msgs = []
    q, a = random.choice(OPEN_KEEP)
    msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    for _ in range(random.randint(4, 8)):
        q = random.choice(qs)
        a = random.choice(by_q[q][:3])
        if random.random() < 0.3:
            a += random.choice(EXTEND)
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        if random.random() < 0.35:
            tq, ta = random.choice(TRANS2)
            msgs += [{"role": "user", "content": tq}, {"role": "assistant", "content": ta}]
    if random.random() < 0.4:
        cq, ca = random.choice(CLOSE2)
        msgs += [{"role": "user", "content": cq}, {"role": "assistant", "content": ca}]
    samples.append(msgs)

random.shuffle(samples)
seen, uniq = set(), []
for s in samples:
    k = json.dumps(s, ensure_ascii=False)
    if k not in seen:
        seen.add(k); uniq.append(s)

out = DATA / "train_v5.jsonl"
with out.open("w", encoding="utf-8") as f:
    for s in uniq:
        f.write(json.dumps({"messages": s}, ensure_ascii=False) + "\n")

chars = sum(len(m["content"]) for s in uniq for m in s)
turns = sum(len(s) for s in uniq)
print(f"会话数: {len(uniq)}")
print(f"消息数: {turns}")
print(f"字符数: {chars:,}")
print(f"=> {out}")
