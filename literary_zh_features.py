#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Literary Chinese "AI-taste" heuristic features (v2.4).

Targets fiction/prose tells from the approved 8-class taxonomy
(literary_ai_taste_taxonomy_2026-10-25.md) — not officialese.

Precision-first: fire on stacked literary AI templates; stay quiet on
alley/oral human prose and English. Language-gated via CJK ratio.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from zh_features import cjk_char_ratio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _split_clauses(text: str) -> List[str]:
    """Split on Chinese/EN sentence + clause boundaries."""
    parts = re.split(r"[。！？…；;\n]+|(?<=[，,])", text)
    return [p.strip() for p in parts if p and p.strip()]


def _sent_units(text: str) -> List[str]:
    parts = re.split(r"[。！？…\n]+", text)
    return [p.strip() for p in parts if p and p.strip()]


# ---------------------------------------------------------------------------
# Class 1 — Fake terse rhythm
# ---------------------------------------------------------------------------

# Verb reduplication with 了: 张了张 / 想了想 / 看了看 / 点了点
_RE_LE_XX = re.compile(r"([\u4e00-\u9fff])了\1")

# Trailing stacked 的-tail after short reduplicated / forced adj: 直直的 / 空空的 / 慢慢的
_RE_ADJ_DE_TAIL = re.compile(
    r"(?:^|[，,。！？\s])"
    r"([\u4e00-\u9fff])\1的"            # XX的
    r"(?=[。！？…，,\s]|$)"
)

# One-idea forced into two clauses: short verb phrase ， short adj-的
# e.g. 烟升上去，直直的 — second clause must NOT start with 是/不/有/还
_RE_SPLIT_ONE_IDEA = re.compile(
    r"([\u4e00-\u9fff]{2,8})"
    r"[，,]"
    r"(?!是|不|有|还|又|就|却|并)"
    r"([\u4e00-\u9fff]{1,3}的)"
    r"(?=[。！？…]|$)"
)

# Bare overuse of 了 density (exclude 了吗/了吧 mild)
def _le_density_score(text: str, n_cjk: int) -> float:
    le_n = text.count("了")
    # Natural alley prose has some 了; flag only elevated density
    rate = le_n / max(1.0, n_cjk / 25.0)  # per ~25 CJK chars
    # Soft start: alley ~1-2 / 25chars ok; literary AI stacks more
    return _clamp((rate - 1.4) / 2.2)


def _fake_terse_rhythm(text: str, n_sent: int, n_cjk: int) -> float:
    le_xx = len(_RE_LE_XX.findall(text))
    adj = len(_RE_ADJ_DE_TAIL.findall(text))
    split = len(_RE_SPLIT_ONE_IDEA.findall(text))
    dens = _le_density_score(text, n_cjk)
    hits = 1.2 * le_xx + 1.4 * adj + 1.6 * split + 1.0 * dens
    # Precision: lone natural X了X (点了点头) without other terse cues → dampen
    if le_xx <= 1 and adj == 0 and split == 0 and dens < 0.35:
        hits *= 0.22
    return _clamp(hits / max(2.8, n_sent * 0.55))


# ---------------------------------------------------------------------------
# Class 2 — Linger / gaze templates
# ---------------------------------------------------------------------------

_RE_LINGER = [
    re.compile(r"站在(?:那儿|那里|这边|这边儿|这儿|这里)"),
    re.compile(r"在(?:那儿|那里|这儿|这里)站了"),
    re.compile(r"站了(?:一会儿|一会|很久|半天|许久)"),
    re.compile(r"在那儿站了"),
    re.compile(r"看了(?:一眼|一看|看)"),
    re.compile(r"看了看"),
    re.compile(r"看得(?:久|久了|久了些|很久)"),
    re.compile(r"望了(?:一眼|望)"),
    re.compile(r"盯了(?:一眼|盯)"),
]

def _linger_gaze(text: str, n_sent: int) -> float:
    hits = 0.0
    for rx in _RE_LINGER:
        hits += len(rx.findall(text))
    return _clamp(hits / max(1.6, n_sent * 0.35))


# ---------------------------------------------------------------------------
# Class 3 — Ought-to-connect but broken (weak)
# ---------------------------------------------------------------------------

# Short parallel 是X，是Y chops (那是名，是娘叫的)
_RE_SHI_SHI = re.compile(
    r"(?:^|[，,。！？\s])"
    r"(?:那)?是([\u4e00-\u9fff]{1,6})"
    r"[，,]"
    r"是([\u4e00-\u9fff]{1,10})"
)

# Micro-action then dialogue with hard break: ……。他说： / ……。她说：
_RE_ACTION_DIALOGUE_BREAK = re.compile(
    r"([\u4e00-\u9fff]{2,12}[。！？])"
    r"\s*"
    r"(?:他|她|我|那人|那个人|端[\u4e00-\u9fff]{0,2}|巫医)说[：:]"
)

def _broken_parallel(text: str, n_sent: int) -> float:
    hits = 0.0
    hits += 1.4 * len(_RE_SHI_SHI.findall(text))
    hits += 0.9 * len(_RE_ACTION_DIALOGUE_BREAK.findall(text))
    # Weak signal — require denser hits
    return _clamp(hits / max(2.2, n_sent * 0.45))


# ---------------------------------------------------------------------------
# Class 4 — Deliberate subject re-emphasis (noun repeat across comma)
# ---------------------------------------------------------------------------

_RE_NOUN_REPEAT = re.compile(
    r"(?:^|[。！？\n\s])"
    r"([\u4e00-\u9fff]{1,4})"
    r"([\u4e00-\u9fff]{0,8})"
    r"[，,]"
    r"\1"
    r"(?=[\u4e00-\u9fff。！？，,]|$)"
)

# Also: X拿到手里，X不叫
_RE_NOUN_REPEAT_LOOSE = re.compile(
    r"([\u4e00-\u9fff]{1,3})"
    r"([\u4e00-\u9fff]{1,6})"
    r"[，,]"
    r"\1"
    r"([\u4e00-\u9fff]{1,6})"
)

# Stopwords / function chars that should not count as "noun re-emphasis"
_NOUN_REPEAT_STOP = {
    "他", "她", "我", "你", "它", "这", "那", "有", "在", "是", "不", "也",
    "都", "就", "又", "还", "很", "更", "最", "被", "把", "让", "给", "从",
    "向", "对", "和", "与", "及", "或", "但", "而", "如果", "因为", "所以",
    "一个", "这个", "那个", "什么", "怎么", "没有", "不是", "可以", "自己",
}


def _noun_reemphasize(text: str, n_sent: int) -> float:
    hits = 0.0
    for m in _RE_NOUN_REPEAT.finditer(text):
        noun = m.group(1)
        if noun in _NOUN_REPEAT_STOP or len(noun) < 1:
            continue
        hits += 1.5
    for m in _RE_NOUN_REPEAT_LOOSE.finditer(text):
        noun = m.group(1)
        if noun in _NOUN_REPEAT_STOP:
            continue
        # Avoid double-count if already matched tight pattern
        if m.group(0) and _RE_NOUN_REPEAT.search(m.group(0)):
            continue
        hits += 1.2
    return _clamp(hits / max(1.8, n_sent * 0.4))


# ---------------------------------------------------------------------------
# Class 5 — 「不是A，是B」pseudo-poetic / 后半句旁白补课
# ---------------------------------------------------------------------------

_RE_BUSHI_SHI = re.compile(
    r"不是([\u4e00-\u9fff的了着过]{1,16})"
    r"[，,——–\-]"
    r"(?:.{0,4})?是"
)

# Half-sentence explanatory coda after period-like break:
# 铃舌是空的，响的不是风。 / 掌心里什么也没有，只有一道印子，是握久了…
_RE_EXPLAIN_CODA = re.compile(
    r"[，,]"
    r"(?:响的不是|不是.{1,8}的|"
    r"是(?:要让|为了|因为|握|压|叫|从)|"
    r"只有.{1,10}，是)"
)

def _bushi_shi(text: str, n_sent: int) -> float:
    hits = 0.0
    hits += 1.8 * len(_RE_BUSHI_SHI.findall(text))
    hits += 1.0 * len(_RE_EXPLAIN_CODA.findall(text))
    return _clamp(hits / max(1.8, n_sent * 0.4))


# ---------------------------------------------------------------------------
# Class 6 — Scene-less simile / vague 别人 (weak–medium)
# ---------------------------------------------------------------------------

_RE_SCENELESS = [
    re.compile(r"像(?:课堂上|考试时|开会时|演戏一样|电影里|书里写的)"),
    re.compile(r"像[\u4e00-\u9fff]{2,8}(?:要人|让人|叫人)"),
    re.compile(r"跟别人(?:说|描述|讲)"),
    re.compile(r"向别人(?:描述|说|讲)"),
    re.compile(r"对别人(?:描述|说)"),
    re.compile(r"试着跟别人"),
    re.compile(r"跟一个很熟的人说"),
    re.compile(r"描述不出来"),
]

def _sceneless_simile(text: str, n_sent: int) -> float:
    hits = 0.0
    for rx in _RE_SCENELESS:
        hits += len(rx.findall(text))
    return _clamp(hits / max(1.5, n_sent * 0.35))


# ---------------------------------------------------------------------------
# Class 7 — Flat dialogue tone / staccato (weak)
# ---------------------------------------------------------------------------

# Ultra-short dialogue beats: 有。你看不见。 / 不是累。 / 那就是年纪。
_RE_STACCATO_DIALOGUE = re.compile(
    r"(?:说[：:]|^|[。！？\n])\s*"
    r"([\u4e00-\u9fff]{1,4})[。！]"
    r"\s*"
    r"([\u4e00-\u9fff]{1,8})[。！]"
)

# Repeated「X说：短句」chain with no tone markers
_RE_SAY_CHAIN = re.compile(
    r"(?:他|她|我|那个人|那人)说[：:]([\u4e00-\u9fff的了不]{1,10})[。！]"
)

def _flat_dialogue(text: str, n_sent: int) -> float:
    hits = 0.0
    hits += 1.5 * len(_RE_STACCATO_DIALOGUE.findall(text))
    say_n = len(_RE_SAY_CHAIN.findall(text))
    if say_n >= 3:
        hits += 1.2 * (say_n - 2)
    # Special: bare「有。你看不见。」pattern
    if re.search(r"有。\s*你看不见。", text):
        hits += 1.8
    return _clamp(hits / max(2.0, n_sent * 0.4))


# ---------------------------------------------------------------------------
# Class 8 — Mechanical scene + fake narrative
# ---------------------------------------------------------------------------

# Ultra-short nominal copula sentences: 土是新翻的。人不在。
# Lookbehind so prior sentence-final punct is not consumed.
_RE_SHORT_NOUN_SENT = re.compile(
    r"(?:(?<=^)|(?<=[。！？\n]))\s*"
    r"([\u4e00-\u9fff]{1,4})"
    r"(?:是[\u4e00-\u9fff的了着过新翻]{1,8}|不在|开着|空了|还在)"
    r"(?=[。！])"
)

# X还在，Y还在 / X还在。Y还在
_RE_HAIZAI_PAIR = re.compile(
    r"([\u4e00-\u9fff]{1,6})还在"
    r"[\u4e00-\u9fff的了着过换了]{0,4}"
    r"[，,。]"
    r"([\u4e00-\u9fff]{1,6})还在"
)

# Time-jump + abstract Q: 仪式之后第七天…他是想问一件事
_RE_TIMEJUMP_Q = re.compile(
    r"(?:之后|以后)?第[一二三四五六七八九十\d]+天"
    r".{0,40}"
    r"(?:又去|再去|又来|是想问|想问一件事|还能不能)"
)

# Hollow abstract ask
_RE_ABSTRACT_ASK = re.compile(
    r"他是想问一件事|那个空[，,]?还能不能|还是空"
)

def _mech_scene(text: str, n_sent: int) -> float:
    hits = 0.0
    short_n = len(_RE_SHORT_NOUN_SENT.findall(text))
    # Fallback: count ultra-short sentence units matching N是…的 / N不在
    if short_n < 2:
        extra = 0
        for u in _sent_units(text):
            u = u.strip()
            if len(u) <= 12 and re.match(
                r"^[\u4e00-\u9fff]{1,4}(?:是[\u4e00-\u9fff的了着过新翻]{1,8}|不在|开着|空了|还在)$",
                u,
            ):
                extra += 1
        short_n = max(short_n, extra)
    # Reward clusters of ≥2 ultra-short scene beats
    if short_n >= 2:
        hits += 1.3 * short_n
    elif short_n == 1:
        hits += 0.4
    hits += 1.6 * len(_RE_HAIZAI_PAIR.findall(text))
    hits += 1.5 * len(_RE_TIMEJUMP_Q.findall(text))
    hits += 1.2 * len(_RE_ABSTRACT_ASK.findall(text))
    return _clamp(hits / max(2.2, n_sent * 0.4))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ZERO_SIGNALS: Dict[str, float] = {
    "lit_zh_fake_terse_rhythm": 0.0,
    "lit_zh_linger_gaze": 0.0,
    "lit_zh_broken_parallel": 0.0,
    "lit_zh_noun_reemphasize": 0.0,
    "lit_zh_bushi_shi": 0.0,
    "lit_zh_sceneless_simile": 0.0,
    "lit_zh_flat_dialogue": 0.0,
    "lit_zh_mech_scene": 0.0,
}


def literary_zh_feature_signals(text: str, n_sent: int = 1) -> Dict[str, float]:
    """
    Return raw 0..1 literary-ZH AI-taste signals (higher = more AI-like).
    Near-zero on non-Chinese text.
    """
    if not text or cjk_char_ratio(text) < 0.12:
        return dict(ZERO_SIGNALS)

    n_sent = max(1, n_sent)
    n_cjk = max(1, sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff"))

    return {
        "lit_zh_fake_terse_rhythm": _fake_terse_rhythm(text, n_sent, n_cjk),
        "lit_zh_linger_gaze": _linger_gaze(text, n_sent),
        "lit_zh_broken_parallel": _broken_parallel(text, n_sent),
        "lit_zh_noun_reemphasize": _noun_reemphasize(text, n_sent),
        "lit_zh_bushi_shi": _bushi_shi(text, n_sent),
        "lit_zh_sceneless_simile": _sceneless_simile(text, n_sent),
        "lit_zh_flat_dialogue": _flat_dialogue(text, n_sent),
        "lit_zh_mech_scene": _mech_scene(text, n_sent),
    }


FEATURE_META_LIT_ZH = {
    "lit_zh_fake_terse_rhythm": (
        "假短促节奏/了滥用", "Fake terse rhythm / 了 overuse",
        "张了张/想了想、直直的 的尾、一意拆两句（烟升上去，直直的）等假短促。",
        "Forced brevity: X了X reduplication, adj-的 tails, one idea split across two clauses.",
    ),
    "lit_zh_linger_gaze": (
        "伫立凝视套板", "Linger / gaze template",
        "站在那儿/站了很久/一会儿；看了一眼/看了看/看得久 等单一动作套板。",
        "Stock linger/gaze: 站在那儿/站了很久; 看了一眼/看了看/看得久.",
    ),
    "lit_zh_broken_parallel": (
        "该连却断的是X是Y", "Broken 是X，是Y parallel",
        "短截「是X，是Y」本该破折号却硬断；微动作后接对话节奏断裂（弱信号）。",
        "Short 是X，是Y chops that should connect; micro-action→dialogue rhythm break (weak).",
    ),
    "lit_zh_noun_reemphasize": (
        "主语刻意复提", "Deliberate noun re-emphasis",
        "逗号两侧同一名词复提（鸡…，鸡…）刻意强调导致节奏假。",
        "Same noun repeated across a comma (鸡…，鸡…) for forced emphasis.",
    ),
    "lit_zh_bushi_shi": (
        "不是A是B伪诗解释", "不是A，是B pseudo-poetic gloss",
        "「不是A，是B」旁白补课式解释，后半句替读者上课。",
        "Pseudo-poetic 不是A，是B gloss / half-sentence narrator tutoring.",
    ),
    "lit_zh_sceneless_simile": (
        "无场景比喻/空泛别人", "Scene-less simile / vague 别人",
        "像课堂上…；跟别人说/描述 — 未铺场景的空泛比附（弱–中）。",
        "Simile without scene (像课堂上…) or vague 别人 describe frame (weak–medium).",
    ),
    "lit_zh_flat_dialogue": (
        "对话语气平/短促", "Flat / staccato dialogue",
        "有。你看不见。 或多人「X说：短句」链同一语气（弱信号）。",
        "Staccato beats like 有。你看不见。 or flat X说：chains (weak).",
    ),
    "lit_zh_mech_scene": (
        "机械场景+假叙事", "Mechanical scene + fake narrative",
        "极短名词句连发；X还在，Y还在；想当然时间跳+抽象问答。",
        "Ultra-short noun beats; X还在 Y还在; unearned time-jump + abstract Q&A.",
    ),
}

LIT_ZH_FEATURE_KEYS = tuple(FEATURE_META_LIT_ZH.keys())

DEFAULT_LIT_ZH_WEIGHTS: Dict[str, float] = {
    "lit_zh_fake_terse_rhythm": 0.07,
    "lit_zh_linger_gaze": 0.07,
    "lit_zh_broken_parallel": 0.045,
    "lit_zh_noun_reemphasize": 0.06,
    "lit_zh_bushi_shi": 0.09,
    "lit_zh_sceneless_simile": 0.06,
    "lit_zh_flat_dialogue": 0.045,
    "lit_zh_mech_scene": 0.09,
}

__all__ = [
    "DEFAULT_LIT_ZH_WEIGHTS",
    "FEATURE_META_LIT_ZH",
    "LIT_ZH_FEATURE_KEYS",
    "ZERO_SIGNALS",
    "literary_zh_feature_signals",
]
