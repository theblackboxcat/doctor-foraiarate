#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-side reference samples for the two registers the detector gets wrong.

Why these are hand-written here rather than pulled from a benchmark
-------------------------------------------------------------------
No public Chinese "modernist prose / poetry" AI-vs-human benchmark exists at
this size, and the commercial detectors do not publish per-register splits.
So the AI side is *generated* — the same compromise ``calib/ai_*/_meta.json``
already records for the general corpus (``generator: assistant-llm``).

What that costs, stated plainly: a detector tuned on these will separate
*this generator's* modernist pastiche from public-domain modernism. It is not
a claim about all LLMs. The honest reading of the numbers in
``eval_registers.py`` is "false positives on real 1920s–40s Chinese went down",
not "we detect AI poetry".

How the samples were written
----------------------------
Each one imitates what an LLM actually returns for 「写一段现代主义风格的短篇」
or 「写一首现代诗」: abstract nouns carrying the emotional load, a definitional
or universalising coda near the end, tidy antithesis, generic imagery with no
proper nouns, and a resolved ending. They were written **before** the new
features in ``register_zh.py`` were designed, so the feature set is not fitted
to individual strings here.

Run:
    python eval_set/make_ai_registers.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

AI_MODERNIST = {
"ai_mod_01": """雨停的时候，城市还没有醒。林知远站在窗前，看着玻璃上慢慢滑落的水痕。那些水痕彼此追逐，又彼此分开，像某种他无法命名的语言。他忽然想起许多年前的一个下午，也是这样的雨，也是这样的窗。时间在这一刻变得柔软，仿佛可以折叠，可以打开，可以重新来过。
他转身，屋子里很安静。桌上放着一只杯子，杯底剩了一点冷掉的茶。茶的颜色很深，深得像一个没有出口的梦。他想，也许所有的等待都是这样：你以为在等一个人，其实是在等自己变成另一个人。
窗外有车经过，灯光扫过天花板，又消失了。他坐回椅子上，把杯子端起来，喝了一口。茶是苦的，但他并不觉得苦。他只是觉得，有些东西已经过去了，而他还在这里。""",

"ai_mod_02": """她走过长廊的时候，光正从百叶窗的缝隙里落下来，一格一格地铺在地板上。她数着那些光，一、二、三，数到第七格的时候，她忘了自己在数什么。记忆是这样一种东西：它不按顺序来，它在你最不设防的时候突然出现，带着某个下午的气味，某句话的尾音。
母亲的声音从很远的地方传来，又像是从身体内部传来。她停下来，把手贴在墙上。墙是凉的，凉得让人安心。她想，如果时间可以停在这里就好了——不是因为这个瞬间有多美，而是因为她还没有准备好往前。
走廊尽头有一扇门。门是关着的。她知道推开它就会回到现实里，回到那些必须回答的问题里。所以她站着，站了很久，直到光从地板上退走。""",

"ai_mod_03": """村子在山的背面，一年里有半年照不到太阳。老周的房子在村口，屋顶的瓦片碎了几块，雨大的时候会漏水。他不修，说修了也一样会碎。
他每天做同样的事：早起，喂鸡，扫地，坐在门槛上抽烟。烟升上去，散在屋檐下面。他不看天，也不看路。他只是在等，等什么他自己也说不清。
有一年冬天，他的儿子回来过一次。儿子站在院子里，看着那几块碎瓦，说，爸，我帮你修吧。老周说，不用。儿子说，为什么不修。老周说，修了也一样。
儿子走的时候，老周送到村口。他没有说保重，也没有说常回来。他只是站在那里，看着儿子的背影一点点变小，最后变成一个点，然后连点也没有了。他转身回去，把门关上。门轴响了一声，像叹气。""",

"ai_mod_04": """地铁在隧道里行驶，车窗变成一面黑色的镜子。陈默看着镜子里的自己，觉得那个人有点陌生。他今年三十四岁，有一份还算体面的工作，一个不算太小的房子，和一些不太联系的朋友。他有时候会在深夜醒来，听见冰箱运转的声音，然后想：这就是我的生活吗。
这个念头一出现，就再也压不下去。它像一颗石子，投进了平静的水面，涟漪一圈一圈地扩散。他开始回忆，回忆那些被他放弃的选择：那封信他没有寄，那个城市他没有去，那个人他没有留。每一个没有都长成了一根刺，扎在他看不见的地方。
列车到站，门开了。他站起来，走出去。站台上人很多，每个人都低着头，看着手里的光。他忽然觉得，我们都是一样的：在黑暗里赶路，以为自己知道要去哪里。""",

"ai_mod_05": """旧书店在巷子的尽头，门口挂着一块褪色的木牌。推门进去，会听见铃铛响一声，然后是很浓的纸和灰尘的味道。老板坐在柜台后面，很少抬头。
我在最里面的一排架子上找到过一本一九八三年的诗集，纸已经发黄，边角卷起来。翻开的时候，有一张纸条掉出来，上面写着一行字：如果你读到这里，请替我记住这个下午。
我不知道写这张纸条的人是谁，也不知道那个下午发生了什么。但我把它夹回原来的那一页，把书放回原来的位置。有些东西不需要被带走，它们只需要有人看见过。
出门的时候，铃铛又响了一声。天已经暗了，巷子里的灯一盏一盏亮起来。我想，这大概就是所谓的相遇：短暂，偶然，然后各自回到自己的时间里。""",

"ai_mod_06": """她问，你还记得吗。
他说，记得什么。
她说，那年下雪。
他沉默了一会儿，说，记得。
她说，雪很大。
他说，嗯。
她说，我们在雪里走了很久。
他说，是。
她看着他，忽然笑了。她说，你什么都不记得了。
他说，我记得。
她说，那你说是哪一年。
他没有回答。窗外的雪又开始下了，一片一片，落在他看不见的地方。""",
}

AI_POETRY = {
"ai_poe_01": """夜的信

夜色是一封没有寄出的信
写满了，又擦掉
最后只剩下一个空白的地址

我坐在窗前
等一个字从纸上浮起来
它应该有重量
像石头落在水底

风穿过房间
把灯吹得晃了一下
我想说的那句话
就这样散了

天亮以前
我把信折好
放进抽屉最里面
和去年的雪放在一起""",

"ai_poe_02": """过程

雪落在屋顶上
也落在没有人的路上
它不解释自己
为什么选择这个时候来

我站在门口
看着世界一点点变白
像一张被反复擦过的纸
终于干净了

有些事就是这样
你等它的时候它不来
你转身的时候
它已经在身后站了很久""",

"ai_poe_03": """如果

如果时间可以倒流
我想回到那个下午
回到你说要走的那一刻
把门关上

可是时间不会倒流
它只会往前走
把所有的如果
都变成后来的样子

后来我学会了一个人吃饭
一个人看电影
一个人在雨天走很长的路
后来我学会了不问你

如果有一天你回来
会发现门是开着的
灯也是亮着的
只是我不在了""",

"ai_poe_04": """空房间

椅子还在原来的位置
杯子里的水已经干了
窗帘被风吹起来
又落下去

没有人知道
这里曾经住过谁
墙上的钉子
还留着一个洞

我推开门
阳光跟着进来
在地上画了一个方框
像一张没有人坐的桌子

我站了一会儿
然后关上门
把钥匙
放回口袋""",

"ai_poe_05": """地铁

地下三十米
一千个人同时沉默
列车进站的时候
风先到

我们被装进同一个铁盒子
彼此挨着
却谁也不看谁
每个人手里都有一小块光

隧道里没有星星
但我们都在许愿
关于明天
关于那个还没回消息的人

到站了
门开了
我们像水一样流出去
散进城市的缝隙里
明天这个时候
我们还会在这里""",

"ai_poe_06": """致

我想给你写一封信，用最旧的纸，最淡的墨。
我想告诉你，今年的槐花开得比去年早，巷口的修鞋摊换了人，那只总在墙头睡觉的猫不见了。
我想告诉你，我还是会在下雨的时候想起你，想起你把伞往我这边偏，自己半边肩膀都湿了。
可是我没有写。因为我知道，这些话你都听过，只是不是从我这里。
信纸在桌上放了三天，最后我把它折起来，夹进了一本书里。也许很多年以后，会有人在旧书摊上翻开它，看见这些没有寄出的话，然后想：这是谁写给谁的。
这样也很好。有些话不需要被听见，它们只需要被写下来。""",
}

AI_REALIST = {
# control: ordinary AI narrative prose. If the register work only helps
# modernism/poetry but breaks this, the change is a regression.
"ai_real_01": """派出所的电话是下午三点打来的。李建国正在车间里核对最后一批货，手上还沾着机油。他听完就往外跑，连工作服都没换。
到了医院，儿子已经在走廊的长椅上坐着，额头上贴了一块纱布。医生说没有大碍，观察一晚就可以回去。李建国想说点什么，张了张嘴，最后只问了一句：吃饭了没有。
儿子摇头。他下楼去买了一份炒饭，又买了一瓶水。回来的时候，儿子靠着墙睡着了。他把饭放在旁边，站着看了一会儿。
回去的路上，儿子一直没说话。快到楼下的时候，儿子忽然说，爸，我不是故意的。李建国说，我知道。儿子说，那你为什么不骂我。李建国说，骂你有什么用。""",
}


def main() -> int:
    manifest = {"note": "AI-side reference samples, hand-written to imitate LLM output",
                "generator": "assistant-llm", "items": []}
    for register, table in (("ai_modernist_prose", AI_MODERNIST),
                            ("ai_poetry", AI_POETRY),
                            ("ai_realist_prose", AI_REALIST)):
        d = HERE / register
        d.mkdir(parents=True, exist_ok=True)
        for sid, body in table.items():
            body = body.strip() + "\n"
            (d / (sid + ".txt")).write_text(body, encoding="utf-8")
            manifest["items"].append({
                "file": "%s/%s.txt" % (register, sid), "register": register,
                "label": "ai", "generator": "assistant-llm", "chars": len(body),
            })
    (HERE / "_manifest_ai.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("%d AI samples -> %s" % (len(manifest["items"]), HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
