# -*- coding: utf-8 -*-
"""
语料 v3 —— 修复 v2 的分布失衡问题

v2 的问题: 85% 的多轮样本以「你好」开头, 导致模型学到
          "看到你好就再输出你好" 这条捷径, 造成生成退化。

v3 的修复:
  1. 开场白大幅多样化 (20+ 种), 且每种的后续回答都不同
  2. 减少纯问候样本占比
  3. 保留 v2 的知识型单轮样本 (这部分质量是好的)
  4. 增加"同一问题 -> 多种回答"的多样性
"""
import json, random
from pathlib import Path

random.seed(7)
HERE = Path(__file__).parent
DATA = HERE / "data"

# 多样化的开场 (问, 答) —— 关键是每条答案都不同, 打破"你好->你好"的统计捷径
OPENERS = [
    ("你好", "你好！有什么可以帮你的吗？"),
    ("你好", "你好，很高兴见到你。"),
    ("你好", "嗨，你好呀。"),
    ("在吗", "在的，你说。"),
    ("在吗", "我在，有什么事吗？"),
    ("嗨", "嗨！今天过得怎么样？"),
    ("你好啊", "你好呀，有什么想聊的？"),
    ("早上好", "早上好！新的一天，希望你顺顺利利。"),
    ("晚上好", "晚上好！忙一天了，记得放松一下。"),
    ("下午好", "下午好，下午容易犯困，记得起来活动一下。"),
    ("哈喽", "哈喽！有什么我能帮忙的？"),
    ("喂", "哎，我在。你说吧。"),
    ("有人吗", "有的，我一直都在。"),
    ("我想问你个问题", "你问吧，我知道的都会尽量回答。"),
    ("帮我个忙", "可以，你说说看是什么事。"),
    ("我有个问题", "好的，你说，我们一起看看。"),
    ("聊聊呗", "好啊，想聊点什么？"),
    ("你现在有空吗", "有空，随时都可以。"),
    ("我来了", "欢迎回来，今天怎么样？"),
    ("好久不见", "是有一阵子了，最近还好吗？"),
]

# 过渡句 (问, 答)
TRANS = [
    ("嗯", "嗯，我在，你继续。"),
    ("然后呢", "然后你可以接着讲，我听着。"),
    ("还有吗", "还有一点，就是别急，慢慢来。"),
    ("我懂了", "好，那我们继续。"),
    ("明白了", "嗯，那我们说下一个。"),
    ("不太懂", "没关系，换个说法，你卡在哪一步？"),
    ("为什么", "原因可能有好几种，你是指哪方面？"),
    ("举个例子", "好比你学骑车，一开始怕摔，骑多了身体自然就记住平衡了。"),
    ("是这样吗", "按我了解的情况是这样，你要是有别的信息我们可以对一对。"),
    ("你说得对", "嗯，想通了就没那么难了。"),
    ("好吧", "嗯，那我们接着聊。"),
    ("等等", "好，不急，你想清楚再说。"),
    ("我明白了", "那就好，还有别的想问的吗？"),
    ("继续说", "好，我继续听着。"),
]

# 结束语 (问, 答)
CLOSERS = [
    ("谢谢你", "不客气，能帮上忙我很开心。"),
    ("多谢", "不用谢，随时找我。"),
    ("太感谢了", "别客气，这是应该的。"),
    ("再见", "再见，路上小心。"),
    ("拜拜", "拜拜，下次再聊。"),
    ("我先走了", "好的，有空再来。"),
    ("晚安", "晚安，好好休息。"),
    ("那就这样", "好，有需要随时找我。"),
    ("我走了", "好，慢走，注意安全。"),
    ("回头聊", "好，随时欢迎。"),
]


def main():
    # ---- 载入 v2 单轮知识样本 (不含那些以你好开头的多轮) ----
    knowledge = []
    with (DATA / "train_v2.jsonl").open(encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            # 只要 2~4 轮的单轮问答 (不包含对话连接词)
            if len(msgs) == 2 and msgs[0]["role"] == "user":
                knowledge.append(msgs)

    print(f"载入知识型单轮样本: {len(knowledge)}")

    # 按用户问题归类, 便于多轮时抽取
    by_q = {}
    for msgs in knowledge:
        by_q.setdefault(msgs[0]["content"], []).append(msgs[1]["content"])
    qs = list(by_q.keys())
    print(f"唯一问题数: {len(qs)}")

    samples = []

    # ---- 1. 全部单轮知识 ----
    samples.extend(knowledge)

    # ---- 2. 多样化开场 + 知识 的多轮 ----
    for _ in range(9000):
        msgs = []
        # 开场: 只用 40% 概率
        if random.random() < 0.4:
            q, a = random.choice(OPENERS)
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]

        # 中间: 1~3 个知识问答
        for _ in range(random.randint(1, 3)):
            q = random.choice(qs)
            a = random.choice(by_q[q])
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            # 过渡
            if random.random() < 0.45:
                tq, ta = random.choice(TRANS)
                msgs += [{"role": "user", "content": tq}, {"role": "assistant", "content": ta}]

        # 收尾: 30% 概率
        if random.random() < 0.3:
            cq, ca = random.choice(CLOSERS)
            msgs += [{"role": "user", "content": cq}, {"role": "assistant", "content": ca}]

        samples.append(msgs)

    # ---- 3. 纯问候单独样本 (让每种问候都有稳定的正确回答) ----
    for q, a in OPENERS:
        for _ in range(60):
            samples.append([{"role": "user", "content": q}, {"role": "assistant", "content": a}])

    # ---- 4. 长对话 (5~8 轮), 训练长上下文 ----
    for _ in range(2500):
        msgs = []
        q, a = random.choice(OPENERS)
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        for _ in range(random.randint(4, 7)):
            q = random.choice(qs)
            a = random.choice(by_q[q])
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            if random.random() < 0.3:
                tq, ta = random.choice(TRANS)
                msgs += [{"role": "user", "content": tq}, {"role": "assistant", "content": ta}]
        samples.append(msgs)

    random.shuffle(samples)
    seen, uniq = set(), []
    for s in samples:
        k = json.dumps(s, ensure_ascii=False)
        if k not in seen:
            seen.add(k); uniq.append(s)

    out = DATA / "train_v3.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in uniq:
            f.write(json.dumps({"messages": s}, ensure_ascii=False) + "\n")

    chars = sum(len(m["content"]) for s in uniq for m in s)
    # 统计：以"你好"开头的比例
    hi = sum(1 for s in uniq if s[0]["content"] == "你好")
    print(f"\n最终会话数: {len(uniq)}")
    print(f"总字符数: {chars:,}")
    print(f"以'你好'开头: {hi} ({hi/len(uniq)*100:.1f}%)  <- v2 是 85%")
    print(f"=> {out}")


if __name__ == "__main__":
    main()
