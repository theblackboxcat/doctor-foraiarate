#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chinese-specific AI-writing heuristic features (v2.3).

Targets: 套话连接、排比骨架、政策腔、空泛主语/低具体名词。
Signals are ~0 on English-only text so language-gating can zero their weights.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Discourse / connector boilerplate common in LLM Chinese
ZH_DISCOURSE_BOILERPLATE = [
    "综上所述", "总而言之", "总的来说", "总体来看", "由此可见", "由此可知",
    "值得注意的是", "需要指出的是", "需要强调的是", "不难发现", "不难看出",
    "与此同时", "与此同时", "在此基础上", "在此背景下", "在此过程中",
    "进一步来说", "进一步而言", "更为重要的是", "更重要的是",
    "换句话说", "也就是说", "换言之", "归根结底", "从根本上说",
    "从某种意义上", "就目前而言", "不可否认", "毫无疑问", "毋庸置疑",
    "众所周知", "事实上", "实际上", "本质上", "客观来讲", "客观地说",
    "首先需要", "其次需要", "最后需要", "一方面", "另一方面",
    "不仅如此", "与此相应", "与此相关", "基于上述", "针对上述",
]

# Parallel / ordinal scaffolding beyond simple list markers
ZH_PARALLEL_PATTERNS = [
    r"首先[，,].{0,40}其次",
    r"其次[，,].{0,40}最后",
    r"第一[，,].{0,60}第二",
    r"第二[，,].{0,60}第三",
    r"一是.{0,40}二是",
    r"二是.{0,40}三是",
    r"不仅.{0,30}而且",
    r"不但.{0,30}而且",
    r"既要.{0,30}又要",
    r"既要.{0,30}也要",
    r"一方面.{0,40}另一方面",
]

ZH_PARALLEL_MARKERS = [
    "首先", "其次", "再次", "最后", "第一", "第二", "第三", "第四",
    "其一", "其二", "其三", "一是", "二是", "三是", "四是",
    "从…到…", # placeholder skipped in count
]

# Policy / officialese / corporate-China register
ZH_POLICY_REGISTER = [
    "高质量发展", "赋能", "数字化转型", "智能化", "新质生产力",
    "顶层设计", "闭环管理", "闭环", "落地见效", "落地生根", "开花结果",
    "系统观念", "统筹推进", "协同联动", "协同发展", "整体推进",
    "以人民为中心", "人民群众", "美好生活", "注入新动能", "新动能",
    "核心竞争力", "现代化体系", "治理体系", "治理能力", "基层治理",
    "提质增效", "降本增效", "攻坚克难", "久久为功", "笃行不怠",
    "真抓实干", "落细落实", "见行见效", "蹄疾步稳", "稳中求进",
    "守正创新", "破立并举", "标本兼治", "综合施策", "多措并举",
    "压实责任", "传导压力", "形成合力", "凝聚合力", "汇聚力量",
    "持续推进", "深入推进", "扎实推进", "全面推进", "加快推进",
    "着力提升", "着力加强", "切实提高", "不断增强", "持续提升",
    "重要举措", "关键举措", "有力举措", "重大部署", "战略部署",
    "工作格局", "制度体系", "长效机制", "体制机制", "保障机制",
    "抓手", "痛点", "堵点", "断点", "全覆盖", "全链条", "全流程",
    "场景化", "精准化", "精细化", "常态化", "长效化",
    "党建引领", "数字赋能", "科技赋能", "创新驱动", "双轮驱动",
    "内生动力", "外生动力", "发展动能", "增长极", "压舱石",
]

# Empty / vague subjects & obligation frames
ZH_EMPTY_SUBJECT_FRAMES = [
    "我们要", "我们应当", "我们需要", "我们必须", "我们应该",
    "人们应该", "人们需要", "大家要", "大家应当",
    "相关部门", "有关部门", "有关方面", "各方要", "各方应",
    "社会各界", "广大群众", "有关单位", "各级各部门",
    "必须坚持", "必须加强", "必须把", "应当坚持", "应当加强",
    "需要不断", "需要进一步", "要不断", "要进一步", "要切实",
    "要坚持", "要大力", "要全面", "要着力", "要深入",
]

# Concrete / sensory / everyday nouns (human alley/lit signal when dense)
ZH_CONCRETE = [
    "雨", "雪", "风", "雾", "霜", "露", "云", "雷", "闪电",
    "路", "街", "巷", "桥", "门", "窗", "墙", "屋顶", "楼梯", "楼道",
    "石板", "砖", "泥", "土", "沙", "灰尘", "脚印",
    "车", "自行车", "电车", "地铁", "公交", "船", "飞机",
    "醋", "盐", "糖", "油", "米", "面", "菜", "青菜", "肉", "鱼", "蛋",
    "茶", "酒", "水", "汤", "粥", "饭", "馒头", "包子",
    "口袋", "钱包", "钥匙", "伞", "帽子", "鞋", "袜子", "衣服", "被子", "床单",
    "桌子", "椅子", "床", "柜子", "抽屉", "杯子", "碗", "筷子", "刀", "锅",
    "手机", "电脑", "电视", "收音机", "灯", "灯泡", "电线",
    "手", "脚", "眼睛", "耳朵", "鼻子", "嘴", "脸", "头发", "牙齿", "指甲",
    "声音", "味道", "气味", "香味", "臭味", "光", "影子", "颜色",
    "猫", "狗", "鸟", "鸡", "鱼", "虫", "蝴蝶", "蚊子",
    "树", "花", "草", "叶子", "根", "枝", "果实", "种子",
    "河", "湖", "海", "山", "坡", "田", "地", "院子", "阳台",
    "店", "超市", "菜场", "饭店", "学校", "教室", "医院", "银行",
    "老板", "老板娘", "同学", "老师", "邻居", "阿姨", "叔叔",
    "妈妈", "爸爸", "奶奶", "爷爷", "孩子", "婴儿", "老人",
    "零钱", "硬币", "纸币", "票", "收据", "发票",
    "眼泪", "汗", "血", "疤", "伤口", "药",
    "烟", "火", "灰烬", "炭", "煤",
]

# Abstract / hollow nouns common in AI Chinese
ZH_ABSTRACT = [
    "发展", "提升", "优化", "完善", "加强", "推进", "落实", "统筹",
    "协同", "体系", "机制", "格局", "动能", "效能", "理念", "意识",
    "价值", "意义", "水平", "能力", "质量", "效果", "成果", "举措",
    "部署", "战略", "全局", "态势", "形势", "局面", "路径", "方向",
    "目标", "任务", "要求", "原则", "方针", "政策", "精神", "思想",
    "认识", "理解", "把握", "贯彻", "执行", "实施", "开展", "深化",
    "拓展", "创新", "转型", "升级", "改革", "开放", "建设", "治理",
    "服务", "保障", "支撑", "引领", "驱动", "赋能", "融合", "贯通",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def cjk_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return n / max(1, len(text))


def zh_feature_signals(text: str, n_sent: int = 1) -> Dict[str, float]:
    """
    Return raw 0..1 ZH-specific signals (higher = more AI-like).
    Near-zero on non-Chinese text.
    """
    if not text or cjk_char_ratio(text) < 0.12:
        return {
            "zh_discourse_boilerplate": 0.0,
            "zh_parallel_scaffold": 0.0,
            "zh_policy_register": 0.0,
            "zh_abstract_subject": 0.0,
        }

    n_sent = max(1, n_sent)
    cjk_n = max(1, sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff"))

    # 1) discourse boilerplate
    disc_hits = 0
    for phrase in ZH_DISCOURSE_BOILERPLATE:
        disc_hits += text.count(phrase)
    zh_discourse_boilerplate = _clamp(disc_hits / max(2.0, n_sent * 0.55))

    # 2) parallel scaffold
    par_hits = 0
    for pat in ZH_PARALLEL_PATTERNS:
        if re.search(pat, text):
            par_hits += 2  # pattern spans weigh more
    # ordinal starter density
    starters = 0
    for m in ("首先", "其次", "再次", "最后", "第一", "第二", "第三",
              "其一", "其二", "其三", "一是", "二是", "三是"):
        # count sentence-ish starts
        starters += len(re.findall(rf"(?:^|[。！？\n])\s*{m}", text))
        if text.lstrip().startswith(m):
            starters += 1
    # Unique ordinal variety bonus when ≥3 distinct
    present = sum(
        1 for m in ("首先", "其次", "最后", "第一", "第二", "第三", "一是", "二是", "三是")
        if m in text
    )
    par_score = par_hits + starters * 0.7 + max(0, present - 1) * 0.5
    zh_parallel_scaffold = _clamp(par_score / max(3.0, n_sent * 0.7))

    # 3) policy register
    pol_hits = 0
    for phrase in ZH_POLICY_REGISTER:
        pol_hits += text.count(phrase)
    zh_policy_register = _clamp(pol_hits / max(3.0, n_sent * 0.9))

    # 4) abstract subject / low concrete
    empty_hits = sum(text.count(p) for p in ZH_EMPTY_SUBJECT_FRAMES)
    # Longer concrete phrases first to avoid double-count bias? Use set of hits.
    concrete_hits = 0
    # Prefer multi-char concrete terms; count unique occurrences
    for w in sorted(ZH_CONCRETE, key=len, reverse=True):
        concrete_hits += text.count(w)
    abstract_hits = sum(text.count(w) for w in ZH_ABSTRACT)

    concrete_rate = _safe_div(concrete_hits, cjk_n / 20.0)  # per ~20 CJK chars
    abstract_rate = _safe_div(abstract_hits, cjk_n / 20.0)
    # High abstract + low concrete + empty subjects → AI-ish
    emptiness = _clamp(empty_hits / max(1.5, n_sent * 0.5))
    low_concrete = _clamp((1.2 - concrete_rate) / 1.2)
    high_abstract = _clamp(abstract_rate / 2.5)
    zh_abstract_subject = _clamp(0.40 * emptiness + 0.35 * low_concrete + 0.25 * high_abstract)

    return {
        "zh_discourse_boilerplate": zh_discourse_boilerplate,
        "zh_parallel_scaffold": zh_parallel_scaffold,
        "zh_policy_register": zh_policy_register,
        "zh_abstract_subject": zh_abstract_subject,
    }


FEATURE_META_ZH = {
    "zh_discourse_boilerplate": (
        "中文连接套话", "ZH discourse boilerplate",
        "综上所述/值得注意的是/与此同时/由此可见 等连接套话密度高 → 生成文常见。",
        "Dense connectors like 综上所述/值得注意的是/与此同时 are common in AI Chinese.",
    ),
    "zh_parallel_scaffold": (
        "中文排比/枚举骨架", "ZH parallel / ordinal scaffold",
        "首先…其次…最后 / 一是…二是…三是 / 不仅…而且 等整齐排比骨架。",
        "Ordinal/parallel scaffolding: 首先…其次…最后 / 一是…二是 / 不仅…而且.",
    ),
    "zh_policy_register": (
        "政策腔/公文腔", "ZH policy / officialese register",
        "高质量发展/赋能/顶层设计/落地见效/注入新动能 等政策空话。",
        "Policy/officialese lexicon: 高质量发展/赋能/顶层设计/落地见效 etc.",
    ),
    "zh_abstract_subject": (
        "空泛主语/低具体名词", "ZH abstract subject / low concreteness",
        "我们要/相关部门/各方 + 抽象名词多、感官/日常具体名词少。",
        "Vague subjects (我们要/相关部门) + abstract nouns, few concrete/sensory nouns.",
    ),
}

ZH_FEATURE_KEYS = tuple(FEATURE_META_ZH.keys())

DEFAULT_ZH_WEIGHTS: Dict[str, float] = {
    "zh_discourse_boilerplate": 0.07,
    "zh_parallel_scaffold": 0.06,
    "zh_policy_register": 0.09,
    "zh_abstract_subject": 0.06,
}

__all__ = [
    "DEFAULT_ZH_WEIGHTS",
    "FEATURE_META_ZH",
    "ZH_FEATURE_KEYS",
    "cjk_char_ratio",
    "zh_feature_signals",
]
