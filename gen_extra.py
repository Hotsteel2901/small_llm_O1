# -*- coding: utf-8 -*-
"""
补充语料: 用规则化生成大量「有逻辑」的问答, 显著扩充数据量。
思路不是灌垃圾, 而是把同一个知识点用多种句式和场景说出来，
让模型学到「问什么答什么」的稳定映射。
"""
import json, random, itertools
from pathlib import Path

random.seed(20261001)
OUT = Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

TEMPLATE_SETS = []


def add(name, q_tpls, a_tpls, ctx=None):
    TEMPLATE_SETS.append((name, q_tpls, a_tpls, ctx or [{}]))


# ---------- 1. 算术推理 ----------
arith_q, arith_a = [], []
for _ in range(600):
    a, b = random.randint(1, 99), random.randint(1, 99)
    op, sym = random.choice([("加", "+"), ("减", "-"), ("乘", "×")])
    if sym == "+":
        r = a + b
    elif sym == "-":
        a, b = max(a, b), min(a, b); r = a - b
    else:
        a, b = random.randint(2, 20), random.randint(2, 20); r = a * b
    qs = [f"{a} {sym} {b} 等于多少", f"帮我算一下 {a} {sym} {b}", f"{a} {op} {b}是多少"]
    ans = [
        f"{a} {sym} {b} 等于 {r}。",
        f"答案是 {r}。{a} {sym} {b} 的计算结果就是 {r}。",
        f"等于 {r}。",
    ]
    arith_q.append(random.choice(qs)); arith_a.append(random.choice(ans))
add("arith", arith_q, arith_a)

# ---------- 2. 单位换算 ----------
conv = []
units = [
    ("米", "厘米", 100), ("千米", "米", 1000), ("厘米", "毫米", 10),
    ("小时", "分钟", 60), ("分钟", "秒", 60), ("天", "小时", 24),
    ("千克", "克", 1000), ("吨", "千克", 1000), ("升", "毫升", 1000),
    ("年", "月", 12), ("周", "天", 7),
]
for u1, u2, k in units:
    for n in [1, 2, 3, 5, 7, 10, 20, 30, 60, 100]:
        r = n * k
        conv.append((f"{n}{u1}等于多少{u2}", f"{n}{u1} 等于 {r}{u2}，因为 1{u1} 等于 {k}{u2}。"))
        conv.append((f"{n}{u1}是多少{u2}", f"{n}{u1} 等于 {r}{u2}。"))
add("convert", [c[0] for c in conv], [c[1] for c in conv])

# ---------- 3. 比较大小 / 排序 ----------
cmp_pairs = []
for _ in range(400):
    a, b = random.randint(2, 999), random.randint(2, 999)
    if a == b:
        b += 1
    big, small = max(a, b), min(a, b)
    cmp_pairs.append((f"{a}和{b}哪个大", f"{big} 比 {small} 大。两个数比较，数值大的那个更大，所以是 {big}。"))
    cmp_pairs.append((f"{a}和{b}哪个小", f"{small} 比 {big} 小，所以较小的是 {small}。"))
add("compare", [c[0] for c in cmp_pairs], [c[1] for c in cmp_pairs])

# ---------- 4. 星期推算 ----------
WEEK = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
week_qa = []
for i, w in enumerate(WEEK):
    for d in [1, 2, 3, 4, 5, 6, 7]:
        nw = WEEK[(i + d) % 7]
        week_qa.append((f"今天是{w}，{d}天后是星期几", f"今天是{w}，再过 {d} 天就是{nw}。"))
        pw = WEEK[(i - d) % 7]
        week_qa.append((f"今天是{w}，{d}天前是星期几", f"今天是{w}，往前推 {d} 天是{pw}。"))
add("week", [w[0] for w in week_qa], [w[1] for w in week_qa])

# ---------- 5. 问答式常识 (定义类) ----------
DEFS = [
    ("什么是水", "水是由氢和氧组成的化合物，化学式是H2O，常温下是无色无味的液体，是生命离不开的物质。"),
    ("什么是光", "光是一种电磁波，我们能看见的那部分叫可见光。它沿直线传播，速度大约是每秒三十万公里。"),
    ("什么是声音", "声音是物体振动产生的波，通过空气等介质传播，传到耳朵里我们就听见了。真空中没有介质，声音传不过去。"),
    ("什么是时间", "时间是描述事件先后顺序和持续长短的量。它一直向前走，不会倒流，是我们理解世界变化的基本尺度。"),
    ("什么是重力", "重力是物体之间相互吸引的力。地球的引力把我们拉向地面，也让月亮绕着地球转。"),
    ("什么是温度", "温度表示物体冷热的程度。它实际上是分子运动剧烈程度的体现，分子动得越快，温度越高。"),
    ("什么是电", "电是电荷运动产生的现象。电荷流动起来形成电流，可以驱动各种设备工作。"),
    ("什么是空气", "空气是包围在地球周围的气体混合物，主要由氮气和氧气组成，还有少量二氧化碳、水蒸气等。"),
    ("什么是细胞", "细胞是生物体的基本结构和功能单位。除病毒外，所有生物都是由细胞组成的。"),
    ("什么是能量", "能量是物体做功的能力。它不会凭空产生也不会消失，只会从一种形式转变成另一种形式。"),
    ("什么是数据", "数据是对事物特征的记录，可以是数字、文字、图像等。经过整理和分析，数据能变成有用的信息。"),
    ("什么是算法", "算法是解决问题的一套明确步骤。给定输入，按照步骤执行，就能得到输出。"),
    ("什么是学习", "学习是通过经验让行为或理解发生持久改变的过程。它不只是记东西，也包括掌握方法。"),
    ("什么是习惯", "习惯是反复重复后变得自动化的行为。它一旦形成，做起来就不太费力。"),
    ("什么是情绪", "情绪是人对内外刺激产生的心理和生理反应，像开心、生气、害怕都是。它会影响我们的判断和行动。"),
    ("什么是记忆", "记忆是大脑把经历过的事情编码、储存、再提取出来的过程。重复和关联能让记忆更牢。"),
    ("什么是合作", "合作是几个人或几方为了共同目标一起出力。它需要沟通、分工和相互信任。"),
    ("什么是计划", "计划是事先想好要做什么、先做什么、什么时候做完。它能让行动更有方向，少走弯路。"),
    ("什么是效率", "效率是单位时间内完成的有效工作量。提高效率的关键是减少浪费在无关事情上的精力。"),
    ("什么是风险", "风险是可能发生的不利结果以及它的可能性大小。做事前评估风险，能帮我们提前准备。"),
]
def_qa = []
for term, d in DEFS:
    def_qa.append((term, d))
    def_qa.append((f"{term[2:]}是什么", d))
    def_qa.append((f"给我解释一下{term[2:]}", d))
    def_qa.append((f"简单说说{term[2:]}", d))
add("defs", [d[0] for d in def_qa], [d[1] for d in def_qa])

# ---------- 6. 步骤 / 怎么做 ----------
HOWTO = [
    ("怎么烧一壶水", ["把水壶洗干净装适量水", "放到灶上点火或插电加热", "等水沸腾冒泡后关火断电", "倒水时小心烫手"]),
    ("怎么做一份简单的蛋炒饭", ["先把鸡蛋打散", "热锅倒油把鸡蛋炒散盛出", "下米饭炒散", "把鸡蛋倒回锅里一起翻炒调味", "出锅装盘"]),
    ("怎么寄一个快递", ["把东西打包好并封牢", "在手机上选快递下单填地址", "等快递员上门取件", "保存单号方便查物流"]),
    ("怎么备份手机照片", ["打开相册找到云同步功能", "登录账号并开启自动备份", "等照片上传完成", "换手机时用同一账号登录就能恢复"]),
    ("怎么学会游泳", ["先熟悉水性，练习在水里憋气", "扶着池边练打腿", "用浮板练换气", "慢慢丢掉辅助自己游", "多练就会越来越顺"]),
    ("怎么整理房间", ["先把不要的东西挑出来扔掉", "把物品按类别分堆", "给每类东西找个固定的位置", "用完放回原处，保持住"]),
    ("怎么准备一场考试", ["先看考试范围列出要复习的知识点", "按重要程度排序", "制定每天的计划", "定期做真题检验", "考前留时间回顾错题"]),
    ("怎么开始跑步", ["先准备一双合适的跑鞋", "从快走和慢跑交替开始", "每次二十分钟左右别太拼", "每周跑三四次", "身体适应后再慢慢加量"]),
    ("怎么做一顿早餐", ["想好吃什么，比如鸡蛋牛奶面包", "把该加热的加热", "简单摆盘", "吃完顺手收拾", "整个过程十几分钟就够"]),
    ("怎么管理每个月的生活费", ["先记下固定支出一共有多少", "算出可自由支配的钱", "定一个大致预算", "大额消费前先想三天", "月底复盘哪里花多了"]),
]
ht_qa = []
for q, steps in HOWTO:
    ans = "可以这样一步步来：第一，" + steps[0] + "；第二，" + steps[1] + "；第三，" + steps[2] + "；最后，" + steps[3] + "。"
    if len(steps) > 4:
        ans = "分这几步：第一，" + "；接着，".join(steps[:-1]) + "；最后，" + steps[-1] + "。"
    ht_qa.append((q, ans))
    ht_qa.append((f"{q}呢", ans))
    ht_qa.append((f"请教我{q}", "好，我按顺序说：" + "；".join(steps) + "。"))
    ht_qa.append((f"{q}的步骤是什么", ans))
add("howto", [h[0] for h in ht_qa], [h[1] for h in ht_qa])


def main():
    convs = []
    for name, qs, as_, _ in TEMPLATE_SETS:
        assert len(qs) == len(as_), name
        for q, a in zip(qs, as_):
            if not q.strip() or not a.strip():
                continue
            convs.append([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
    print("补充样本数:", len(convs))

    # 追加到语料
    path = OUT / "corpus_extra.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps({"messages": c}, ensure_ascii=False) + "\n")
    chars = sum(len(m["content"]) for c in convs for m in c)
    print("补充字符数:", chars)

    # 合并
    all_samples = []
    for fn in ["corpus.jsonl", "corpus_extra.jsonl"]:
        with (OUT / fn).open(encoding="utf-8") as f:
            for line in f:
                all_samples.append(json.loads(line)["messages"])

    random.shuffle(all_samples)
    with (OUT / "train.jsonl").open("w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps({"messages": s}, ensure_ascii=False) + "\n")

    tc = sum(len(m["content"]) for s in all_samples for m in s)
    print(f"合并后会话数: {len(all_samples)}")
    print(f"合并后总字符数: {tc}")
    print(f"=> {OUT/'train.jsonl'}")


if __name__ == "__main__":
    main()
