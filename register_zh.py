#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Register layer (v3.2): what kind of text is this, and what does "AI-flavoured"
mean *in that kind of text*.

The problem this solves
-----------------------
The 8 literary slots in ``literary_zh_features`` were designed against one
register: everyday/口語 fiction prose. On two other registers they mis-fire
systematically, and not because the patterns are wrong — because the same
surface shape means the opposite thing:

* **Modernist prose** repeats nouns, chops clauses, and drops into ultra-short
  nominal beats *on purpose* (「一株是枣树，还有一株也是枣树」). Every one of
  those is a hit for `lit_zh_noun_reemphasize` / `lit_zh_mech_scene` /
  `lit_zh_fake_terse_rhythm`. Measured on 鲁迅《影的告别》: Human Score 54,
  with `lit_zh_mech_scene` as the single largest contributor.
* **Poetry** has line breaks where prose has sentences. Sentence-length,
  comma-density, burstiness and clause-layering slots are then measuring a
  line-break artefact. Measured on 戴望舒《雨巷》: the tool cannot even decide
  whether it has enough evidence, and what evidence it does use is
  `compression_ai_like` on 44 characters.

So the layer does two separate things, and keeps them separate on purpose:

1. **A register router** (`poetry_probability`). One question only: *is this
   poetry?* — because that decides which calibration bucket the text is scored
   against, and which features are even defined. It deliberately does **not**
   try to answer "is this modernist", which is a judgement about intent that
   surface statistics cannot carry. Modernist prose stays in the ordinary
   Chinese bucket; what changes there is the *reference corpus*, not a router.
2. **Craft-tell features** (`register_feature_signals`). These are properties
   of *how the text handles its own meaning*, which is where AI literary
   writing and human modernist writing actually diverge — not in rhythm:

   * AI explains itself (「记忆是这样一种东西」/「这大概就是所谓的相遇」)
   * AI generalises at the end (「有些话不需要被听见」/「我们都是一样的」)
   * AI compares to abstractions (「像一个没有出口的梦」/「像某种他无法命名的语言」)
   * AI balances clauses into clean antithesis (「不是…而是…」)
   * AI keeps paragraphs and punctuation tidy

   A human modernist withholds meaning; that is the whole point of the
   register. These features are stated AI-ward (higher = more AI-like) so they
   plug into the existing calibration; whether they actually discriminate is
   decided by `disc` in ``calibration.json``, not by this docstring.

Honest boundary
---------------
The human side of the reference corpus is 1920s–40s public-domain Chinese,
because that is the only Chinese modernism that is legally available. A living
writer producing modernist prose in plain modern Chinese has no such reference
— for that writer this layer will *not* fix the false positive. What it fixes
is the case it can be measured on, and it says so in the UI.
"""
from __future__ import annotations

import re
from typing import Dict, List

from zh_features import cjk_char_ratio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _nonempty_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").split("\n") if ln.strip()]


_SENT_SPLIT = re.compile(r"[。！？…；;]+")
_END_PUNCT = "。！？…；;"


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def _n_cjk(text: str) -> int:
    return sum(1 for ch in text or "" if "\u4e00" <= ch <= "\u9fff")


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Modern-colloquial abstractions that carry the emotional load in LLM literary
# Chinese. Deliberately excludes 1920s literary vocabulary (寂寥/愁怨/彷徨),
# which is why it can separate the two eras at all — see the module docstring
# for what that costs.
ABSTRACT_BINOMES = (
    "时间", "记忆", "距离", "温度", "意义", "存在", "选择", "生活", "命运",
    "情绪", "状态", "关系", "瞬间", "永恒", "虚空", "温柔", "疼痛", "遗憾",
    "执念", "边界", "尽头", "深处", "沉默", "等待", "相遇", "告别", "遗忘",
    "回忆", "思念", "真相", "秘密", "答案", "世界", "生命", "灵魂", "自由",
    "岁月", "心事", "感受", "感觉", "本质", "孤独", "寂寞", "忧伤", "悲伤",
    "期待", "失望", "疲惫", "勇气", "欲望", "想象", "沉默", "空白",
)

# Concrete anchors: things you could photograph. A modernist text is dense with
# them even when it withholds meaning; AI literary prose floats above them.
CONCRETE_NOUNS = (
    "枣树", "瓦", "门槛", "烟", "雪", "伞", "丁香", "石板", "灯罩", "井",
    "墙", "窗", "门", "路", "树", "花", "草", "叶", "石头", "杯子", "椅子",
    "桌子", "纸", "笔", "书", "灯", "火", "灰", "土", "水", "河", "山",
    "天", "雨", "风", "霜", "雾", "车", "船", "桥", "街", "巷", "院子",
    "房", "屋顶", "楼", "楼梯", "手", "脚", "眼睛", "脸", "头发", "牙",
    "猫", "狗", "鸟", "鸡", "虫", "鱼", "刀", "绳", "线", "布", "衣",
    "碗", "锅", "煤", "炭", "钉子", "钟", "镜子", "抽屉", "钥匙", "口袋",
)

_CONNECTIVES = (
    "因为", "所以", "但是", "可是", "如果", "后来", "其实", "于是", "虽然",
    "然而", "而且", "然后", "只是", "不过", "因此", "尽管", "除非",
)

# ---------------------------------------------------------------------------
# 1) Register router — poetry or not
# ---------------------------------------------------------------------------


def poetry_probability(text: str) -> float:
    """
    0..1: how confident we are that this is a poem (line-broken verse) rather
    than prose. Only used to route calibration and to mask undefined features;
    it is *not* an AI signal.
    """
    t = text or ""
    lines = _nonempty_lines(t)
    if len(lines) < 4 or len(t) < 60:
        return 0.0

    sents = _sentences(t)
    n_sent = max(1, len(sents))
    lens = [len(re.sub(r"\s", "", ln)) for ln in lines]
    lens_sorted = sorted(lens)
    median = lens_sorted[len(lens_sorted) // 2]

    p = 0.0
    # (a) line length — verse lines are short and of comparable size
    if 3 <= median <= 18:
        p += 0.30
    elif median <= 26:
        p += 0.12

    # (b) lines that do not close with sentence punctuation: enjambment
    no_end = sum(1 for ln in lines if not ln.endswith(tuple(_END_PUNCT)))
    frac_no_end = no_end / len(lines)
    if frac_no_end >= 0.55:
        p += 0.30
    elif frac_no_end >= 0.30:
        p += 0.15

    # (c) few sentence terminators per 100 chars
    n_cjk = max(1, _n_cjk(t))
    end_punct = sum(t.count(c) for c in "。！？")
    if end_punct / n_cjk * 100 < 1.6:
        p += 0.20
    elif end_punct / n_cjk * 100 < 2.8:
        p += 0.10

    # (d) stanza structure: blank lines group the lines
    if re.search(r"\n\s*\n", t):
        p += 0.10

    # (e) lines per sentence — verse packs many lines into one sentence
    if len(lines) / n_sent >= 1.8:
        p += 0.10

    # (f) refutations. Dialogue/play formatting is not verse, however short the
    #     lines are: 《过客》 is a play and would otherwise route to poetry.
    dialogue = len(re.findall(r"(?:说|问|答|喊|道)[：:，,]", t))
    quotes = t.count("「") + t.count("“") + t.count("『")
    if dialogue >= 3 or quotes >= 3:
        p -= 0.35
    if re.search(r"^[\u4e00-\u9fff]{1,3}：", t, re.M):
        p -= 0.15

    return _clamp(p)


def register_of(text: str, threshold: float = 0.55) -> Dict[str, float]:
    p = poetry_probability(text)
    return {
        "poetry_p": round(p, 4),
        "is_poetry": 1.0 if p >= threshold else 0.0,
    }


# ---------------------------------------------------------------------------
# 2) Craft tells — AI-ward features (higher = more AI-like)
# ---------------------------------------------------------------------------

# The text explains its own meaning to the reader. This is the single most
# reliable separator: modernist writing withholds, LLM literary writing tutors.
_RE_GLOSS = [
    re.compile(r"(?:是|就是|大概是|也许是|正是|便是)这样(?:一种|一个)(?:东西|事情|感觉|状态|地方)"),
    re.compile(r"(?:这|那)(?:大概|也许|或许|正)?就是(?:所谓|我们说的|人们说的|所谓的)"),
    re.compile(r"所谓[\u4e00-\u9fff]{1,10}(?:不过是|就是|只是|无非)"),
    re.compile(r"与其说[\u4e00-\u9fff，,]{1,20}不如说"),
    re.compile(r"(?:我|他|她|你)(?:忽然|突然|终于|这才|这时|这才)?"
               r"(?:明白|懂得|意识到|发现|知道)了?[，,：:]\s*(?:原来|不过|其实)"),
    re.compile(r"(?:仿佛|好像|似乎)(?:是)?在(?:说|提醒|告诉|等待|诉说|讲述|暗示|问)"),
    re.compile(r"像(?:是在|在)(?:说|提醒|告诉|等待)"),
    re.compile(r"(?:在|于)这一刻(?:变得|显得|忽然)"),
    re.compile(r"(?:所有|一切)(?:的)?(?:等待|孤独|告别|相遇|爱|痛苦|思念)(?:都|也)"),
    re.compile(r"(?:某|一)种(?:无法|不能|说不清|难以|说不出的)[\u4e00-\u9fff]{0,4}的?[\u4e00-\u9fff]{1,4}"),
]

# The text generalises at the end. Human modernism ends on a concrete image or
# an unresolved beat; AI ends on a lesson.
_RE_CODA_UNIVERSAL = [
    # 从具体场景拔高成一条通则。允许句首有「他只是觉得」这类引导语——
    # 实测 LLM 很少让格言裸奔，总要挂一句「他想/她忽然明白」再上升。
    re.compile(r"(?:有些|所有|一切)(?:东西|话|事情|事|人|等待|孤独|告别)"
               r"(?:已经|不|只|终究|总是|都|也|需要|不必)"),
    re.compile(r"(?:每个|任何)(?:人|时候|瞬间)(?:都|也|终)"),
    re.compile(r"(?:都|终究|总是)(?:会|是|要)?(?:回到|归于|变成|成为|过去|结束|消失)"),
    re.compile(r"(?:这样|如此)(?:也)?(?:很好|就够了|就好|挺好)"),
    re.compile(r"(?:这就是|这大概就是|这便是|所谓)(?:生活|爱|成长|时间|命运|人生|孤独)"),
    re.compile(r"(?:剩下|留给|交给)(?:的)?(?:读|时间|后来|别人|自己)"),
    re.compile(r"(?:不需要|不必)(?:被)?(?:听见|看见|记住|理解|解释)"),
    re.compile(r"以为自己知道(?:要|该)(?:去|往)哪里"),
]

_RE_GENERIC_SIMILE = [
    re.compile(r"(?:像|仿佛|如同|好似)(?:某种|一种|一个|一场)(?:无法|不能|说不清|难以|没有|说不出)?"
               r"[\u4e00-\u9fff]{0,6}的?"
               r"(?:梦|语言|秘密|仪式|答案|伤口|宿命|命运|时间|记忆|告别|等待|回声|隐喻|寓言|符号|容器|迷宫|漩涡)"),
    re.compile(r"(?:像|仿佛|如同)(?:一颗|一块|一片|一张|一根)[\u4e00-\u9fff]{1,6}[，,]"
               r"(?:投进|落进|沉入|掉进|浮在)"),
    re.compile(r"(?:像|仿佛)(?:水|光|风|雪)一样(?:流|散|退|消|漫)"),
]

_RE_ANTITHESIS = [
    re.compile(r"不是[^。！？；\n]{1,16}[，,](?:而是|是)[^。！？；\n]{1,16}"),
    re.compile(r"并非[^。！？；\n]{1,16}[，,]?而是"),
    re.compile(r"与其说[^。！？；\n]{1,16}不如说"),
    re.compile(r"你以为[^。！？；\n]{1,16}[，,](?:其实|却|但)"),
    re.compile(r"既[^。！？；\n]{1,10}[，,]?又[^。！？；\n]{1,10}"),
]


def _per_sent(hits: float, n_sent: int, floor: float = 1.6) -> float:
    return _clamp(hits / max(floor, n_sent * 0.45))


def _sat(hits: float, k: float = 0.9) -> float:
    """Saturating map for *presence* features (codas), not density features.

    A definitional coda is a discrete move: one occurrence in a 500-character
    piece is already the tell, and dividing it by sentence count made it
    vanish (measured: 1 hit in a 14-sentence sample scored 0.16, i.e. the
    feature was silently dead). 1 hit → 0.59, 2 → 0.83, 3 → 0.93.
    """
    if hits <= 0:
        return 0.0
    return _clamp(1.0 - pow(2.718281828, -k * hits))


def _gloss_coda(text: str, n_sent: int) -> float:
    hits = 0.0
    for rx in _RE_GLOSS:
        hits += 1.0 * len(rx.findall(text))
    return _sat(hits, 0.85)


def _universal_coda(text: str, n_sent: int) -> float:
    sents = _sentences(text)
    if not sents:
        return 0.0
    tail = sents[-2:] if len(sents) >= 2 else sents
    score = 0.0
    for s in tail:
        s = s.strip()
        if not s:
            continue
        for rx in _RE_CODA_UNIVERSAL:
            if rx.search(s):
                score += 1.0
                break
    # A final simile is a softer version of the same move (the image is handed
    # over already interpreted).
    if re.search(r"(?:像|仿佛|如同)", sents[-1]):
        score += 0.4
    return _sat(score, 0.8)


def _abstract_density(text: str) -> float:
    n = max(1, _n_cjk(text))
    hits = 0
    for w in ABSTRACT_BINOMES:
        hits += text.count(w)
    rate = hits / n * 100.0            # per 100 CJK chars
    return _clamp((rate - 0.7) / 2.3)


def _concrete_density(text: str) -> float:
    n = max(1, _n_cjk(text))
    hits = 0
    for w in CONCRETE_NOUNS:
        hits += text.count(w)
    return hits / n * 100.0            # per 100 CJK chars (not clamped: used raw)


def _generic_simile(text: str, n_sent: int) -> float:
    hits = 0.0
    for rx in _RE_GENERIC_SIMILE:
        hits += len(rx.findall(text))
    return _per_sent(hits, n_sent, 2.2)


def _clean_antithesis(text: str, n_sent: int) -> float:
    hits = 0.0
    for rx in _RE_ANTITHESIS:
        hits += len(rx.findall(text))
    return _per_sent(hits, n_sent, 2.2)


def _para_uniformity(text: str) -> float:
    """Tidy paragraphs / stanzas of near-equal length are a generation artefact."""
    paras = [p for p in re.split(r"\n\s*\n|\n", text or "") if p.strip()]
    if len(paras) < 4:
        return 0.0
    lens = [len(re.sub(r"\s", "", p)) for p in paras]
    mean = sum(lens) / len(lens)
    if mean <= 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in lens) / len(lens)
    cv = (var ** 0.5) / mean
    return _clamp((0.52 - cv) / 0.52)


def _punct_monotony(text: str) -> float:
    """One terminator for everything (usually 。) reads as generated."""
    used = sum(1 for c in "。！？；…—" if c in (text or ""))
    return _clamp((3 - used) / 3.0)


# ---------------------------------------------------------------------------
# 3) Poetry-specific craft tells
# ---------------------------------------------------------------------------

_RE_PROSE_LINE = re.compile(
    r"(?:%s)" % "|".join(_CONNECTIVES)
)


def _poe_prose_line(text: str) -> float:
    """Lines that are really prose sentences with a newline in them."""
    lines = _nonempty_lines(text)
    if len(lines) < 3:
        return 0.0
    bad = 0
    for ln in lines:
        body = re.sub(r"\s", "", ln)
        if len(body) > 26 or ln.endswith(("。", "！", "？", "；")):
            bad += 1
        elif _RE_PROSE_LINE.search(ln):
            bad += 1
    return _clamp(bad / len(lines))


def _poe_explained_ending(text: str) -> float:
    lines = _nonempty_lines(text)
    if len(lines) < 4:
        return 0.0
    tail = lines[-max(2, len(lines) // 4):]
    hits = 0.0
    for ln in tail:
        if any(rx.search(ln) for rx in _RE_GLOSS):
            hits += 1.0
        elif any(rx.search(ln) for rx in _RE_CODA_UNIVERSAL):
            hits += 1.0
        elif _RE_PROSE_LINE.search(ln) and len(re.sub(r"\s", "", ln)) > 18:
            hits += 0.5
    return _clamp(hits / max(1.5, len(tail) * 0.6))


def _poe_abstract_line(text: str) -> float:
    """Stanzas of pure abstraction with nothing you could photograph."""
    lines = _nonempty_lines(text)
    if len(lines) < 3:
        return 0.0
    bare = 0
    for ln in lines:
        ab = sum(ln.count(w) for w in ABSTRACT_BINOMES)
        co = sum(ln.count(w) for w in CONCRETE_NOUNS)
        if ab >= 1 and co == 0:
            bare += 1
    return _clamp(bare / len(lines))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ZERO_SIGNALS: Dict[str, float] = {
    "mod_gloss_coda": 0.0,
    "mod_universal_coda": 0.0,
    "mod_abstract_binome": 0.0,
    "mod_generic_simile": 0.0,
    "mod_clean_antithesis": 0.0,
    "mod_para_uniformity": 0.0,
    "mod_punct_monotony": 0.0,
    "poe_prose_line": 0.0,
    "poe_explained_ending": 0.0,
    "poe_abstract_line": 0.0,
}


def register_feature_signals(text: str, n_sent: int = 1) -> Dict[str, float]:
    """
    AI-ward craft features (0..1, higher = more AI-like), plus register-neutral
    behaviour on non-Chinese text (all zeros).
    """
    if not text or cjk_char_ratio(text) < 0.12:
        return dict(ZERO_SIGNALS)

    n_sent = max(1, n_sent)
    is_poe = poetry_probability(text) >= 0.55

    out = {
        "mod_gloss_coda": _gloss_coda(text, n_sent),
        "mod_universal_coda": _universal_coda(text, n_sent),
        "mod_abstract_binome": _abstract_density(text),
        "mod_generic_simile": _generic_simile(text, n_sent),
        "mod_clean_antithesis": _clean_antithesis(text, n_sent),
        "mod_para_uniformity": _para_uniformity(text),
        "mod_punct_monotony": _punct_monotony(text),
        "poe_prose_line": 0.0,
        "poe_explained_ending": 0.0,
        "poe_abstract_line": 0.0,
    }
    if is_poe:
        out["poe_prose_line"] = _poe_prose_line(text)
        out["poe_explained_ending"] = _poe_explained_ending(text)
        out["poe_abstract_line"] = _poe_abstract_line(text)
    return out


def concrete_density(text: str) -> float:
    """Diagnostic only: concrete nouns per 100 CJK chars."""
    return round(_concrete_density(text), 3)


FEATURE_META_REG = {
    "mod_gloss_coda": (
        "自我解释/定义句", "Self-glossing / definitional coda",
        "「记忆是这样一种东西」「这大概就是所谓的相遇」——替读者把意思说出来。现代主义恰恰不说。",
        "Definitional frames (X是这样一种东西 / 这就是所谓的Y) that hand the reader the meaning.",
    ),
    "mod_universal_coda": (
        "结尾上升为格言", "Universalising ending",
        "末句从具体场景拔到「有些东西不需要…」「我们都是一样的」。真人现代主义收在具体物象或未解处。",
        "Final sentence generalises into a maxim; human modernism ends on a concrete or unresolved beat.",
    ),
    "mod_abstract_binome": (
        "抽象名词密度", "Abstract-noun density",
        "时间/记忆/距离/意义/孤独 一类抽象双音词占篇幅比高——情绪由名词承担，而非由事承担。",
        "Dense modern abstractions (时间/记忆/距离/意义/孤独) carrying the emotion instead of events.",
    ),
    "mod_generic_simile": (
        "抽象喻体比喻", "Abstraction-vehicle simile",
        "「像一个没有出口的梦」「像某种他无法命名的语言」——喻体本身是抽象概念。",
        "Similes whose vehicle is itself an abstraction (像一个没有出口的梦).",
    ),
    "mod_clean_antithesis": (
        "工整对偶", "Clean antithesis",
        "「不是…而是…」「你以为…其实…」成对出现，句子被修得太齐。",
        "Tidy paired constructions (不是…而是 / 你以为…其实) that sand the sentence smooth.",
    ),
    "mod_para_uniformity": (
        "段落长度过于整齐", "Uniform paragraph length",
        "段落/诗节字数变异小——生成文按节奏配额产出，真人写作长短相差很大。",
        "Paragraph/stanza lengths barely vary; human drafts swing much wider.",
    ),
    "mod_punct_monotony": (
        "句末标点单一", "Monotonous terminators",
        "几乎只用「。」收句；民国现代主义大量用「；」「——」「…」制造悬置。",
        "Almost only 。 closes sentences; period modernism leans on ；——… to suspend.",
    ),
    "poe_prose_line": (
        "分行散文（伪诗）", "Prose with line breaks",
        "诗行其实是完整散文句：带「因为/后来/如果」等连接词，或长度超过一行诗。",
        "Lines that are full prose sentences — connectives or over-long lines.",
    ),
    "poe_explained_ending": (
        "诗末解释", "Explained ending",
        "最后几行把前面的意象解释掉，而不是让它停在那里。",
        "The closing lines explain the image instead of leaving it standing.",
    ),
    "poe_abstract_line": (
        "纯抽象诗行", "Abstraction-only lines",
        "整行只有抽象名词，没有任何可拍下来的东西。",
        "Lines built entirely from abstractions, with nothing concrete to photograph.",
    ),
}

REG_FEATURE_KEYS = tuple(FEATURE_META_REG.keys())

DEFAULT_REG_WEIGHTS: Dict[str, float] = {
    "mod_gloss_coda": 0.10,
    "mod_universal_coda": 0.09,
    "mod_abstract_binome": 0.07,
    "mod_generic_simile": 0.06,
    "mod_clean_antithesis": 0.05,
    "mod_para_uniformity": 0.05,
    "mod_punct_monotony": 0.05,
    "poe_prose_line": 0.08,
    "poe_explained_ending": 0.07,
    "poe_abstract_line": 0.06,
}

# Features that are *undefined* on line-broken verse: they measure a sentence
# that the poem does not have. Masked to zero weight when the router says
# poetry. This is a validity fix, not a tuning knob — there is no threshold
# here that a corpus could disagree with.
POETRY_MASKED_FEATURES = (
    "burstiness_inv",
    "comma_density",
    "clause_layering_inv",
    "sentence_start_variety_inv",
    "window_burstiness_inv",
    "lit_zh_fake_terse_rhythm",
    "lit_zh_flat_dialogue",
    "lit_zh_mech_scene",
    "lit_zh_noun_reemphasize",
)

__all__ = [
    "ABSTRACT_BINOMES",
    "DEFAULT_REG_WEIGHTS",
    "FEATURE_META_REG",
    "POETRY_MASKED_FEATURES",
    "REG_FEATURE_KEYS",
    "ZERO_SIGNALS",
    "concrete_density",
    "poetry_probability",
    "register_feature_signals",
    "register_of",
]
