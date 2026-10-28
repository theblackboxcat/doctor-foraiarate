#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local AI-writing signal detector (heuristic).

Inspired by public TipType UX conventions (Human Score bands, sentence heat
thresholds, Clarity separate from AI score) and known AI-flavor linguistic
signals — but NOT TipType's proprietary model/API. Fully offline.

Honesty: this is a transparent feature-weighted heuristic, useful for editing
feedback. It is NOT authorship proof and will disagree with commercial detectors.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MIN_CHARS = 80          # was 100; short EN still warned
MIN_CHARS_ZH = 55       # CJK-heavy text: each char carries more meaning
HEAT_RED = 0.65
HEAT_YELLOW = 0.40

# v3.0 confidence / evidence gating
CONFIDENCE_INSUFFICIENT = 0.42   # below this the tool says "not enough evidence"
CONFIDENCE_LEN_FULL = 500        # chars at which length evidence saturates
PARA_TARGET_CHARS = 320          # paragraph-window size for mixed-authorship scans
PARA_MIN_CHARS = 120
MIXED_SPREAD = 0.40              # paragraph p spread that hints at mixed writing

# v3.1 Jev-style action bands (cost-aware policy; not a single Human Score cutoff)
ANN_FLOOR = 0.08            # feature must fire at least this raw value to annotate
ANN_FLAG_P = 0.65           # flag if p_ai ≥ this AND confidence ≥ ANN_FLAG_CONF
ANN_FLAG_CONF = 0.50
ANN_REVIEW_P = 0.40         # else review if p_ai ≥ this
ANN_FEATURE_KEYS_PREFIX = ("lit_zh_", "zh_", "marketing_", "mod_", "poe_")

_ENGINE = "local_heuristic_v3.2"

# Features that are undefined on line-broken verse (no sentences to measure).
# The poetry calibration bucket simply has no anchors for these, so they cannot
# contribute to a score there — `_weighted_z` skips keys without an anchor.
try:
    from register_zh import POETRY_MASKED_FEATURES  # noqa: E402
except Exception:                                       # noqa: BLE001
    POETRY_MASKED_FEATURES = ()
# Hybrid DNA gating (v2.1): only blend when heuristic is uncertain.
# Outside the band, DNA is skipped (weight≈0). dna_blend default lowered
# because 0.28 inverted Gettysburg (memorized classic) vs AI slop.
DNA_UNCERTAIN_LO = 35
DNA_UNCERTAIN_HI = 65
DNA_MIN_CHARS = 200
DNA_BLEND_DEFAULT = 0.14
DNA_MEMORIZATION_OVERLAP = 0.70
DNA_MEMORIZATION_HUMAN = 55

# Optional ONNX 3rd-vote blend (when models/onnx/ettin-68m present).
ONNX_BLEND_DEFAULT = 0.18

# ---------------------------------------------------------------------------
# Lexicons (EN + ZH marketing / template / discourse markers)
# ---------------------------------------------------------------------------

EN_ADVERBS = {
    "moreover", "furthermore", "additionally", "consequently", "therefore",
    "thus", "hence", "notably", "significantly", "effectively", "essentially",
    "fundamentally", "ultimately", "specifically", "particularly", "generally",
    "typically", "inherently", "substantially", "remarkably", "seamlessly",
    "holistically", "strategically", "comprehensively", "robustly",
    "consistently", "actively", "proactively", "deeply", "truly", "highly",
    "extremely", "incredibly", "absolutely", "clearly", "obviously",
    "certainly", "indeed", "simply", "merely", "literally", "actually",
    "basically", "virtually", "arguably", "interestingly", "importantly",
}

EN_MARKETING = {
    "leverage", "leverages", "leveraging", "optimize", "optimizes", "optimizing",
    "seamless", "seamlessly", "cutting-edge", "state-of-the-art", "innovative",
    "groundbreaking", "transformative", "holistic", "synergy", "synergies",
    "paradigm", "ecosystem", "unlock", "unlocks", "unprecedented", "delve",
    "delves", "delving", "myriad", "foster", "fosters", "empower", "empowers",
    "empowering", "streamline", "streamlines", "robust", "scalable",
    "best-in-class", "next-generation", "game-changer", "game-changing",
    "stakeholders", "landscape", "navigate", "navigating", "harness",
    "harnesses", "utilize", "utilizes", "facilitate", "facilitates",
    "enhance", "enhances", "boost", "boosts", "drive", "drives",
    "pivotal", "crucial", "vital", "essential", "paramount",
    "tapestry", "realm", "beacon", "testament", "underscore", "underscores",
    "intricate", "nuanced", "multifaceted", "ever-evolving", "rapidly",
}

EN_LIST_MARKERS = {
    "first", "second", "third", "fourth", "fifth", "finally", "lastly",
    "firstly", "secondly", "thirdly", "next", "then", "also",
}

EN_TRANSITIONS = {
    "in conclusion", "to summarize", "in summary", "overall", "in essence",
    "it is important to note", "it is worth noting", "needless to say",
    "as previously mentioned", "moving forward", "that being said",
    "with that in mind", "at the end of the day", "in today's",
    "in this regard", "on the other hand", "for instance", "for example",
}

EN_SUBORDINATORS = {
    "although", "though", "because", "since", "while", "whereas", "unless",
    "until", "if", "when", "whenever", "wherever", "whether", "after",
    "before", "once", "as", "than", "that", "which", "who", "whom", "whose",
}

ZH_ADVERBS_MARKERS = [
    "此外", "而且", "因此", "所以", "从而", "进而", "总之", "综上所述",
    "值得注意的是", "需要指出的是", "需要强调的是", "毫无疑问", "显而易见",
    "事实上", "本质上", "从根本上", "全面", "深度", "有效地", "显著地", "充分",
    "与此同时", "由此可见", "总而言之", "不难发现", "在此基础上",
]

ZH_MARKETING = [
    "赋能", "闭环", "抓手", "落地", "打通", "打法", "生态", "链路",
    "协同", "提质增效", "降本增效", "全链路", "数字化转型", "智能化",
    "场景化", "个性化", "一站式", "全方位", "多维度", "系统性",
    "前瞻性", "创新性", "突破性", "颠覆性", "无缝衔接", "深度融合",
    "价值主张", "核心竞争力", "最佳实践", "痛点", "机遇与挑战",
    "高质量发展", "新质生产力", "顶层设计", "落地见效", "注入新动能",
    "多措并举", "综合施策", "久久为功", "蹄疾步稳", "守正创新",
]

ZH_LIST_MARKERS = [
    "首先", "其次", "再次", "再次是", "最后", "第一", "第二", "第三",
    "第四", "第五", "其一", "其二", "其三", "一方面", "另一方面",
]

ZH_SUBORDINATORS = [
    "虽然", "尽管", "因为", "由于", "如果", "假如", "只要", "除非",
    "当", "在…时", "之后", "之前", "即使", "哪怕", "无论", "不管",
    "之所以", "之所以…是因为", "不仅", "不但", "既…又",
]

# v3.0: Chinese clause layering is NOT carried by explicit subordinators the
# way English is. Human Chinese prose ("昨晚下了点小雨，巷子里的石板路还是湿的。
# 我骑车去买醋…") can contain zero 虽然/因为 and still be deeply layered, which
# made clause_layering_inv peg at 1.000 on human Chinese and turn the slot into
# pure noise. Chinese layering is measured from clause density + paired
# connectives instead.
ZH_PAIRED_CONNECTIVES = [
    ("虽然", "但是"), ("虽然", "却"), ("尽管", "还是"), ("因为", "所以"),
    ("既然", "就"), ("不但", "而且"), ("不仅", "还"), ("一边", "一边"),
    ("既", "又"), ("越", "越"), ("一", "就"), ("与其", "不如"),
]

DEFAULT_WEIGHTS: Dict[str, float] = {
    # Shared EN/ZH (slightly trimmed in v2.3 to make room for ZH slots)
    "adverb_density": 0.07,
    "adverb_verb_proxy": 0.05,
    "unique_word_ratio_inv": 0.08,
    "sentence_start_variety_inv": 0.07,
    "burstiness_inv": 0.09,
    "list_template": 0.08,
    "comma_density": 0.05,
    "clause_layering_inv": 0.06,
    "marketing_inflation": 0.10,
    "compression_ai_like": 0.10,
    "window_burstiness_inv": 0.05,
    # v2.3 Chinese-specific (gated to ~0 weight on EN via apply_language_weights)
    "zh_discourse_boilerplate": 0.07,
    "zh_parallel_scaffold": 0.06,
    "zh_policy_register": 0.09,
    "zh_abstract_subject": 0.06,
    # v2.4 literary Chinese AI-taste (CJK-gated; precision-first)
    "lit_zh_fake_terse_rhythm": 0.07,
    "lit_zh_linger_gaze": 0.07,
    "lit_zh_broken_parallel": 0.045,
    "lit_zh_noun_reemphasize": 0.06,
    "lit_zh_bushi_shi": 0.09,
    "lit_zh_sceneless_simile": 0.06,
    "lit_zh_flat_dialogue": 0.045,
    "lit_zh_mech_scene": 0.09,
    # v3.2 register layer — craft tells that survive the modernist/poetry
    # registers where the rhythm slots above are ambiguous by design
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


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def cjk_ratio(text: str) -> float:
    """CJK Unified Ideographs fraction (chars)."""
    if not text:
        return 0.0
    n = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return n / max(1, len(text))


def min_chars_for(text: str) -> int:
    """Lower floor for CJK-heavy text; EN keeps MIN_CHARS."""
    if cjk_ratio(text) >= 0.45:
        return MIN_CHARS_ZH
    return MIN_CHARS


ZH_WEIGHT_KEYS = (
    "zh_discourse_boilerplate",
    "zh_parallel_scaffold",
    "zh_policy_register",
    "zh_abstract_subject",
    # v2.4 literary
    "lit_zh_fake_terse_rhythm",
    "lit_zh_linger_gaze",
    "lit_zh_broken_parallel",
    "lit_zh_noun_reemphasize",
    "lit_zh_bushi_shi",
    "lit_zh_sceneless_simile",
    "lit_zh_flat_dialogue",
    "lit_zh_mech_scene",
)


def apply_language_weights(weights: Dict[str, float], text: str) -> Dict[str, float]:
    """
    Zero ZH-only slots on EN text so they do not dilute EN feature mass
    after renormalization. On CJK-heavy text, keep ZH slots.
    """
    w = {k: float(v) for k, v in weights.items()}
    ratio = cjk_ratio(text)
    if ratio < 0.25:
        for k in ZH_WEIGHT_KEYS:
            if k in w:
                w[k] = 0.0
    s = sum(max(0.0, v) for v in w.values())
    if s > 0:
        w = {k: max(0.0, v) / s for k, v in w.items()}
    return w


LIT_ZH_WEIGHT_KEYS = (
    "lit_zh_fake_terse_rhythm",
    "lit_zh_linger_gaze",
    "lit_zh_broken_parallel",
    "lit_zh_noun_reemphasize",
    "lit_zh_bushi_shi",
    "lit_zh_sceneless_simile",
    "lit_zh_flat_dialogue",
    "lit_zh_mech_scene",
)


def apply_literary_activity_weights(
    weights: Dict[str, float], signals: Dict[str, float]
) -> Dict[str, float]:
    """
    Precision gate: if literary slots are collectively quiet, zero their
    weights so idle lit_* mass does not dilute officialese / alley scores.
    When any literary taste is active, keep weights (already language-gated).
    """
    lit_sum = sum(float(signals.get(k, 0.0) or 0.0) for k in LIT_ZH_WEIGHT_KEYS)
    w = {k: float(v) for k, v in weights.items()}
    if lit_sum < 0.22:
        for k in LIT_ZH_WEIGHT_KEYS:
            if k in w:
                w[k] = 0.0
        s = sum(max(0.0, v) for v in w.values())
        if s > 0:
            w = {k: max(0.0, v) / s for k, v in w.items()}
    return w


# ---------------------------------------------------------------------------
# Tokenization / sentence split (lightweight, EN+ZH)
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(
    r"(?<=[.!?。！？…])\s+|(?<=[.!?。！？])(?=[\"'”’)\]]?\s*[A-Z\u4e00-\u9fff])"
)
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]")
_EN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SYLLABLE_RE = re.compile(r"[aeiouy]+", re.I)


def split_sentences(text: str) -> List[Tuple[int, str]]:
    """Return list of (char_start, sentence_text)."""
    text = text.strip()
    if not text:
        return []
    parts: List[Tuple[int, str]] = []
    # Prefer punctuation-aware split; fall back to whole text
    chunks = re.split(r"(?<=[.!?。！？…])\s+", text)
    if len(chunks) == 1:
        # Also split on Chinese punctuation without space
        chunks = re.split(r"(?<=[。！？…])", text)
        chunks = [c for c in chunks if c and c.strip()]
    pos = 0
    for chunk in chunks:
        raw = chunk
        s = chunk.strip()
        if not s:
            pos += len(raw)
            # account for the separator consumed? re.split keeps delimiters only with capturing
            continue
        idx = text.find(s, pos)
        if idx < 0:
            idx = pos
        parts.append((idx, s))
        pos = idx + len(s)
    if not parts:
        parts = [(0, text)]
    return parts


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def en_words(text: str) -> List[str]:
    return [w.lower() for w in _EN_WORD.findall(text)]


def count_syllables_en(word: str) -> int:
    w = word.lower().strip(".,;:!?\"'")
    if not w:
        return 0
    if w.endswith("e") and len(w) > 2:
        w = w[:-1]
    groups = _SYLLABLE_RE.findall(w)
    return max(1, len(groups))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class FeatureHit:
    name: str
    name_zh: str
    name_en: str
    value: float          # raw 0..1-ish contribution signal
    weight: float         # user weight (normalised)
    contribution: float   # effective weight * calibrated z (toward AI)
    explanation_zh: str
    explanation_en: str
    # v3.0 calibration fields (all optional so older callers keep working)
    z: float = 0.0                  # calibrated score: 0 = human median, 1 = AI-ish
    anchor_lo: float = 0.0
    anchor_hi: float = 1.0
    disc: float = 0.0               # measured discriminability on calib/
    gain: float = 1.0               # applied gain (tanh-shaped; can be negative)
    effective_weight: float = 0.0   # weight * gain
    anchor_source: str = ""
    flipped: bool = False


@dataclass
class SentenceResult:
    text: str
    char_start: int
    length: int
    ai_probability: float
    heat: str  # "red" | "yellow" | "none"
    top_features: List[FeatureHit] = field(default_factory=list)
    percentile: float = -1.0   # v3.0: rank inside this document (0..1)


@dataclass
class DocumentResult:
    text: str
    char_count: int
    word_count: int
    sentence_count: int
    too_short: bool
    ai_probability: float
    human_score: int
    human_band_zh: str
    human_band_en: str
    clarity_score: int
    clarity_notes_zh: str
    clarity_notes_en: str
    features: List[FeatureHit]
    sentences: List[SentenceResult]
    highlight_policy: str
    weights_used: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)
    # v3.0
    confidence: float = 0.0
    insufficient_evidence: bool = False
    paragraphs: List[Dict[str, Any]] = field(default_factory=list)
    # v3.1 Jev-style parallel annotations (additive; does not replace scores)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def feat(f: FeatureHit) -> Dict[str, Any]:
            return {
                "name": f.name,
                "name_zh": f.name_zh,
                "name_en": f.name_en,
                "value": round(f.value, 4),
                "weight": round(f.weight, 4),
                "contribution": round(f.contribution, 4),
                "explanation_zh": f.explanation_zh,
                "explanation_en": f.explanation_en,
                "z": round(f.z, 4),
                "anchor_lo": round(f.anchor_lo, 4),
                "anchor_hi": round(f.anchor_hi, 4),
                "disc": round(f.disc, 4),
                "gain": round(f.gain, 4),
                "effective_weight": round(f.effective_weight, 4),
                "anchor_source": f.anchor_source,
                "flipped": f.flipped,
            }

        return {
            "text": self.text,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "too_short": self.too_short,
            "ai_probability": round(self.ai_probability, 4),
            "human_score": self.human_score,
            "human_band_zh": self.human_band_zh,
            "human_band_en": self.human_band_en,
            "clarity_score": self.clarity_score,
            "clarity_notes_zh": self.clarity_notes_zh,
            "clarity_notes_en": self.clarity_notes_en,
            "features": [feat(f) for f in self.features],
            "sentences": [
                {
                    "text": s.text,
                    "char_start": s.char_start,
                    "length": s.length,
                    "ai_probability": round(s.ai_probability, 4),
                    "heat": s.heat,
                    "percentile": round(s.percentile, 4) if s.percentile >= 0 else None,
                    "top_features": [feat(f) for f in s.top_features],
                }
                for s in self.sentences
            ],
            "highlight_policy": self.highlight_policy,
            "weights_used": self.weights_used,
            "confidence": round(self.confidence, 4),
            "insufficient_evidence": self.insufficient_evidence,
            "paragraphs": self.paragraphs,
            "annotations": self.annotations,
            "meta": self.meta,
            "disclaimer_zh": "本工具为本地启发式特征加权，非 TipType 专有模型/API，不能证明作者身份。",
            "disclaimer_en": "Local heuristic feature-weighted clone — not TipType's proprietary model/API; not authorship proof.",
        }


FEATURE_META = {
    "adverb_density": (
        "副词密度", "Adverb density",
        "高频抽象副词（moreover/furthermore/显著地等）常见于生成文。",
        "High density of abstract adverbs (moreover/furthermore/…) is common in AI prose.",
    ),
    "adverb_verb_proxy": (
        "副词-动词搭配代理", "Adverb–verb pairing proxy",
        "“副词+动作动词”套话（seamlessly leverages / 有效地赋能）抬高 AI 感。",
        "Stock adverb+verb pairings (seamlessly leverages / 有效地赋能) raise AI flavor.",
    ),
    "unique_word_ratio_inv": (
        "用词多样性不足", "Low unique-word ratio",
        "类型/形符比偏低 → 词汇重复、模板感强。",
        "Low type-token ratio → repetitive vocabulary / templated feel.",
    ),
    "sentence_start_variety_inv": (
        "句首/句根多样性不足", "Low sentence-start variety",
        "句首词重复多，像报告体/列表体骨架。",
        "Repeated sentence openings look like report/list skeletons.",
    ),
    "burstiness_inv": (
        "句长爆发力不足", "Low sentence-length burstiness",
        "句长过于均匀（低变异系数）是常见 LLM 统计痕迹。",
        "Overly even sentence lengths (low CV) are a common LLM statistical tell.",
    ),
    "list_template": (
        "列表模板", "List / ordinal template",
        "First/Second/Finally 或 首先/其次/最后 整齐枚举骨架。",
        "First/Second/Finally or 首先/其次/最后 ordinal scaffolding.",
    ),
    "comma_density": (
        "逗号密度偏高", "High comma density",
        "长串并列逗号、一气呵成的“流水句”常见于营销生成文。",
        "Dense comma stacking often appears in marketing-style generated prose.",
    ),
    "clause_layering_inv": (
        "从句层次不足", "Low clause layering",
        "缺少 although/because/虽然/因为 等从属层次，句式偏平。",
        "Few subordinators (although/because/虽然…) → flat clause structure.",
    ),
    "marketing_inflation": (
        "营销夸大词", "Marketing inflation lexicon",
        "leverage/seamless/赋能/闭环 等空心夸大词。",
        "Hollow hype lexicon: leverage/seamless/赋能/闭环 etc.",
    ),
    "compression_ai_like": (
        "压缩冗余/AI 前言亲和", "Compression AI-affinity",
        "自压缩比偏低、与 AI 套话前言交叉压缩更紧 → 统计冗余像生成文（离线 zlib/lzma）。",
        "Low self-compressibility / higher affinity to AI-flavored prelude (offline zlib/lzma).",
    ),
    "window_burstiness_inv": (
        "窗口压缩爆发力不足", "Low window-compression burstiness",
        "滑动窗压缩比过于均匀 → 缺少人类段落节奏变化。",
        "Overly even sliding-window compression ratios → flat paragraph rhythm.",
    ),
    "ollama_dna_overlap": (
        "Ollama 续写一致性", "Ollama rewrite-consistency (DNA-GPT-lite)",
        "截断后用本地模型续写，与原文后缀 n-gram 重合高 → 更像机器（需 Ollama）。",
        "Truncate→local regenerate; high n-gram overlap with original suffix → machine-like (needs Ollama).",
    ),
    "onnx_ai_prob": (
        "ONNX 分类器", "ONNX AI-text classifier",
        "可选本地 ONNX（英文 ettin-68m 或中文 aigc-zh）给出的 P(AI)；缺权重时跳过。",
        "Optional local ONNX (EN ettin-68m or ZH aigc-zh) P(AI); skipped if weights missing.",
    ),
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
    # ---- v3.2 register layer ------------------------------------------------
    "mod_gloss_coda": (
        "自我解释/定义句", "Self-glossing / definitional coda",
        "「记忆是这样一种东西」「这大概就是所谓的相遇」——替读者把意思说出来。",
        "Definitional frames that hand the reader the meaning instead of withholding it.",
    ),
    "mod_universal_coda": (
        "结尾上升为格言", "Universalising ending",
        "末句从具体场景拔到「有些东西不需要…」「我们都是一样的」。",
        "Final sentence generalises into a maxim instead of ending on an image.",
    ),
    "mod_abstract_binome": (
        "抽象名词密度", "Abstract-noun density",
        "时间/记忆/距离/意义/孤独 一类抽象双音词占比高，情绪由名词承担。",
        "Dense modern abstractions carry the emotion instead of events.",
    ),
    "mod_generic_simile": (
        "抽象喻体比喻", "Abstraction-vehicle simile",
        "「像一个没有出口的梦」「像某种他无法命名的语言」——喻体本身是抽象概念。",
        "Similes whose vehicle is itself an abstraction.",
    ),
    "mod_clean_antithesis": (
        "工整对偶", "Clean antithesis",
        "「不是…而是…」「你以为…其实…」成对出现，句子被修得太齐。",
        "Tidy paired constructions that sand the sentence smooth.",
    ),
    "mod_para_uniformity": (
        "段落长度过于整齐", "Uniform paragraph length",
        "段落/诗节字数变异小；真人写作长短相差很大。",
        "Paragraph/stanza lengths barely vary.",
    ),
    "mod_punct_monotony": (
        "句末标点单一", "Monotonous terminators",
        "几乎只用「。」收句；现代主义大量用「；」「——」「…」制造悬置。",
        "Almost only 。 closes sentences; modernism suspends with ；——….",
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


def load_weights(path: Optional[Path] = None) -> Dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if path is None:
        path = Path(__file__).with_name("weights.json")
    if path and Path(path).is_file():
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in weights and isinstance(v, (int, float)):
                        weights[k] = float(v)
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    # renormalize positive weights to sum ~1 for interpretability
    s = sum(max(0.0, w) for w in weights.values())
    if s > 0:
        weights = {k: max(0.0, v) / s for k, v in weights.items()}
    return weights


def save_weights(weights: Dict[str, float], path: Optional[Path] = None) -> Path:
    path = Path(path) if path else Path(__file__).with_name("weights.json")
    path.write_text(json.dumps(weights, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def human_band(score: int) -> Tuple[str, str]:
    if score < 20:
        return "高度疑似 AI", "Strong AI"
    if score < 40:
        return "疑似 AI", "Likely AI"
    if score < 60:
        return "人类与 AI 混合", "Mixed Human and AI"
    if score < 80:
        return "较符合人类写作", "Likely Human"
    return "高度符合人类写作", "Strong Human"


def heat_label(p: float) -> str:
    if p >= HEAT_RED:
        return "red"
    if p >= HEAT_YELLOW:
        return "yellow"
    return "none"


# ---------------------------------------------------------------------------
# v3.0 calibration plumbing
# ---------------------------------------------------------------------------

_CALIB_CACHE: Dict[str, Any] = {"obj": None, "loaded": False}


def get_calibration(path: Optional[Path] = None):
    """Load (and memoise) calibration.json. Returns None when absent."""
    if _CALIB_CACHE["loaded"] and path is None:
        return _CALIB_CACHE["obj"]
    try:
        from calibration import Calibration
        cal = Calibration.load(path)
    except Exception:  # noqa: BLE001 — calibration is optional by design
        cal = None
    if path is None:
        _CALIB_CACHE["obj"] = cal
        _CALIB_CACHE["loaded"] = True
    return cal


def reload_calibration(path: Optional[Path] = None):
    _CALIB_CACHE["obj"] = None
    _CALIB_CACHE["loaded"] = False
    return get_calibration(path)


def legacy_probability(acc: float) -> float:
    """v2.4 mapping — kept only as a fallback when calibration.json is absent."""
    return _clamp(0.08 + 0.92 * (1.0 - math.exp(-1.85 * acc)))


def paragraph_windows(text: str, target: int = PARA_TARGET_CHARS) -> List[Tuple[int, str]]:
    """
    Split into paragraph-sized windows for mixed-authorship scanning.
    Prefers real paragraph breaks, then packs sentences up to ``target`` chars.
    """
    text = (text or "").strip()
    if not text:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) >= 2:
        units = [(text.find(b), b) for b in blocks]
    else:
        units = split_sentences(text)
    windows: List[Tuple[int, str]] = []
    buf = ""
    start = 0
    for pos, u in units:
        if not buf:
            start = pos
        buf = (buf + " " + u).strip() if buf else u
        if len(buf) >= target:
            windows.append((start, buf))
            buf = ""
    if buf and len(buf) >= 40:
        windows.append((start, buf))
    elif buf and windows:
        s0, last = windows[-1]
        windows[-1] = (s0, (last + " " + buf).strip())
    return windows


def confidence_of(
    text_len: int,
    min_needed: int,
    mass_ratio: float,
    ai_p: float,
    provisional: bool,
    disagreement: float = 0.0,
) -> float:
    """
    How much the score deserves to be trusted — not how AI-like the text is.

    * length evidence: a 90-character note cannot support a verdict
    * mass_ratio: how much of the weight budget actually discriminates on calib/
    * separation: scores near 0.5 are genuinely ambiguous
    * provisional anchors (too few reference samples) discount everything
    * disagreement between voters (heuristic vs ONNX vs DNA) discounts too
    """
    len_f = _clamp((text_len - min_needed) / max(1.0, CONFIDENCE_LEN_FULL - min_needed))
    mass_f = _clamp(mass_ratio / 0.5)
    sep_f = _clamp(abs(ai_p - 0.5) * 2.0)
    conf = 0.18 + 0.82 * (0.38 * len_f + 0.32 * mass_f + 0.30 * sep_f)
    if provisional:
        conf *= 0.78
    if disagreement > 0:
        conf *= max(0.6, 1.0 - 0.5 * _clamp(disagreement))
    return _clamp(conf)


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------

def _feature_signals(text: str, sentences: List[Tuple[int, str]]) -> Dict[str, float]:
    """Return raw 0..1 signals (higher = more AI-like) for each feature key."""
    toks = tokenize(text)
    en = en_words(text)
    n_tok = max(1, len(toks))
    n_en = len(en)
    n_sent = max(1, len(sentences))
    lower_text = text.lower()

    # 1) adverb density (EN set + ZH markers)
    adv_hits = sum(1 for w in en if w in EN_ADVERBS)
    for m in ZH_ADVERBS_MARKERS:
        adv_hits += text.count(m)
    adverb_density = _clamp(_safe_div(adv_hits, max(n_en, n_tok) / 20.0))  # scale ~ per 20 tokens

    # 2) adverb-verb proxy: marketing verbs near -ly / ZH 地
    marketing_verbish = {
        "leverage", "leverages", "optimize", "optimizes", "enhance", "enhances",
        "unlock", "unlocks", "foster", "fosters", "empower", "empowers",
        "utilize", "utilizes", "facilitate", "facilitates", "streamline",
        "streamlines", "harness", "harnesses", "drive", "drives", "navigate",
        "navigating", "delve", "delves",
    }
    av_hits = 0
    for i, w in enumerate(en):
        if w.endswith("ly") or w in EN_ADVERBS:
            window = en[i : i + 3]
            if any(x in marketing_verbish for x in window):
                av_hits += 1
    av_hits += len(re.findall(r"有效地|深度地|全面|无缝|系统性地", text))
    adverb_verb_proxy = _clamp(av_hits / max(3.0, n_sent * 0.5))

    # 3) unique word ratio inverted
    types = set(t.lower() if t.isascii() else t for t in toks)
    ttr = _safe_div(len(types), n_tok, 1.0)
    # Human prose often ~0.45–0.7 for short windows; AI boilerplate lower
    unique_word_ratio_inv = _clamp((0.55 - ttr) / 0.35)

    # 4) sentence start variety inverted
    starts = []
    for _, s in sentences:
        ws = tokenize(s)
        if ws:
            starts.append(ws[0].lower() if ws[0].isascii() else ws[0])
    unique_starts = len(set(starts))
    start_variety = _safe_div(unique_starts, n_sent, 1.0)
    sentence_start_variety_inv = _clamp(1.0 - start_variety)

    # 5) burstiness (CV of sentence lengths) inverted
    lengths = [len(tokenize(s)) or len(s) for _, s in sentences]
    if len(lengths) >= 2:
        mean_l = sum(lengths) / len(lengths)
        var = sum((x - mean_l) ** 2 for x in lengths) / len(lengths)
        cv = math.sqrt(var) / mean_l if mean_l else 0.0
        # Human CV often 0.4–0.9+; very low CV → AI
        burstiness_inv = _clamp((0.45 - cv) / 0.45)
    else:
        burstiness_inv = 0.35  # single sentence: mild uncertainty

    # 6) list template
    list_hits = 0
    for _, s in sentences:
        first = (en_words(s)[:1] or [""])[0]
        if first in EN_LIST_MARKERS:
            list_hits += 1
        for m in ZH_LIST_MARKERS:
            if s.lstrip().startswith(m):
                list_hits += 1
                break
    # Also whole-doc First…Finally pattern
    ordinal_present = sum(
        1 for m in ("first", "second", "finally", "firstly", "secondly", "lastly")
        if re.search(rf"\b{m}\b", lower_text)
    )
    ordinal_present += sum(1 for m in ("首先", "其次", "最后", "第一", "第二", "第三") if m in text)
    list_template = _clamp(max(list_hits / max(3.0, n_sent * 0.4), ordinal_present / 4.0))

    # 7) comma density
    commas = text.count(",") + text.count("，")
    comma_density = _clamp(_safe_div(commas, max(1, n_tok / 8.0)) / 2.0)

    # 8) clause layering — invert: flat clause structure → AI
    if cjk_ratio(text) >= 0.25:
        # Chinese: clause density per sentence + paired connectives.
        cps_vals = []
        for _, s in sentences:
            cps_vals.append(1 + s.count("，") + s.count("；") + s.count("："))
        cps = sum(cps_vals) / max(1, len(cps_vals))
        pair_hits = 0
        for a, b in ZH_PAIRED_CONNECTIVES:
            if a in text and b in text:
                pair_hits += 1
        layering = 0.55 * min(1.0, cps / 2.5) + 0.45 * min(1.0, pair_hits / 3.0)
        clause_layering_inv = _clamp(1.0 - layering)
    else:
        sub_hits = sum(1 for w in en if w in EN_SUBORDINATORS)
        for m in ZH_SUBORDINATORS:
            if "…" in m:
                continue
            sub_hits += text.count(m)
        sub_rate = _safe_div(sub_hits, n_sent)
        clause_layering_inv = _clamp((1.2 - sub_rate) / 1.2)

    # 9) marketing inflation
    mkt = sum(1 for w in en if w in EN_MARKETING)
    for phrase in ("cutting-edge", "state-of-the-art", "best-in-class", "next-generation",
                   "game-changer", "game-changing", "ever-evolving"):
        mkt += lower_text.count(phrase)
    for m in ZH_MARKETING:
        mkt += text.count(m)
    for phrase in EN_TRANSITIONS:
        if phrase in lower_text:
            mkt += 1
    marketing_inflation = _clamp(mkt / max(4.0, n_sent * 0.8))

    # Offline compression bundle (no network / no Ollama)
    try:
        from compression_signals import compression_feature_bundle
        comp = compression_feature_bundle(text)
    except Exception:
        comp = {"compression_ai_like": 0.0, "window_burstiness_inv": 0.0}

    # v2.3 Chinese-specific heuristics (officialese)
    try:
        from zh_features import zh_feature_signals
        zh_sig = zh_feature_signals(text, n_sent=n_sent)
    except Exception:
        zh_sig = {
            "zh_discourse_boilerplate": 0.0,
            "zh_parallel_scaffold": 0.0,
            "zh_policy_register": 0.0,
            "zh_abstract_subject": 0.0,
        }

    # v2.4 literary Chinese AI-taste
    try:
        from literary_zh_features import literary_zh_feature_signals
        lit_sig = literary_zh_feature_signals(text, n_sent=n_sent)
    except Exception:
        lit_sig = {
            "lit_zh_fake_terse_rhythm": 0.0,
            "lit_zh_linger_gaze": 0.0,
            "lit_zh_broken_parallel": 0.0,
            "lit_zh_noun_reemphasize": 0.0,
            "lit_zh_bushi_shi": 0.0,
            "lit_zh_sceneless_simile": 0.0,
            "lit_zh_flat_dialogue": 0.0,
            "lit_zh_mech_scene": 0.0,
        }

    # v3.2 register layer (modernist prose + poetry craft tells)
    try:
        from register_zh import register_feature_signals
        reg_sig = register_feature_signals(text, n_sent=n_sent)
    except Exception:
        reg_sig = {}

    out = {
        "adverb_density": adverb_density,
        "adverb_verb_proxy": adverb_verb_proxy,
        "unique_word_ratio_inv": unique_word_ratio_inv,
        "sentence_start_variety_inv": sentence_start_variety_inv,
        "burstiness_inv": burstiness_inv,
        "list_template": list_template,
        "comma_density": comma_density,
        "clause_layering_inv": clause_layering_inv,
        "marketing_inflation": marketing_inflation,
        "compression_ai_like": float(comp.get("compression_ai_like", 0.0)),
        "window_burstiness_inv": float(comp.get("window_burstiness_inv", 0.0)),
        "zh_discourse_boilerplate": float(zh_sig.get("zh_discourse_boilerplate", 0.0)),
        "zh_parallel_scaffold": float(zh_sig.get("zh_parallel_scaffold", 0.0)),
        "zh_policy_register": float(zh_sig.get("zh_policy_register", 0.0)),
        "zh_abstract_subject": float(zh_sig.get("zh_abstract_subject", 0.0)),
    }
    for k, v in lit_sig.items():
        out[k] = float(v)
    for k, v in reg_sig.items():
        out[k] = float(v)
    return out


def _sentence_local_signals(sent: str) -> Dict[str, float]:
    """Per-sentence features (subset; window-level OK)."""
    sents = [(0, sent)]
    return _feature_signals(sent, sents)


def clarity_score(text: str, sentences: List[Tuple[int, str]]) -> Tuple[int, str, str]:
    """Separate readability-like clarity (0–100). Not an AI score."""
    en = en_words(text)
    n_sent = max(1, len(sentences))
    # Flesch-ish for English portion; for Chinese use avg chars/sentence heuristic
    if len(en) >= 20:
        words = len(en)
        syllables = sum(count_syllables_en(w) for w in en)
        asl = words / n_sent
        asw = syllables / words
        flesch = 206.835 - 1.015 * asl - 84.6 * asw
        # Map Flesch (~0–100) with mild clamp
        score = int(round(_clamp(flesch / 100.0) * 100))
        # Prefer mid-high clarity for moderately complex prose
        notes_zh = f"英文可读性近似 Flesch={flesch:.1f}；平均句长={asl:.1f} 词。"
        notes_en = f"English readability ≈ Flesch {flesch:.1f}; avg sentence length {asl:.1f} words."
    else:
        # Chinese / short: prefer 15–40 chars per sentence as clear
        lens = [len(s) for _, s in sentences] or [len(text)]
        avg = sum(lens) / len(lens)
        # Peak clarity around 25–40 chars (ZH) or mixed
        if 12 <= avg <= 55:
            score = 78
        elif avg < 12:
            score = 55
        else:
            score = int(round(_clamp(1.0 - (avg - 55) / 80.0) * 70 + 20))
        notes_zh = f"中文/短文启发式：平均句长 {avg:.0f} 字；清晰度与 AI 分独立。"
        notes_en = f"ZH/short heuristic: avg sentence {avg:.0f} chars; clarity is independent of AI score."

    # Penalize extreme marketing fog lightly on clarity only
    fog = sum(1 for w in en if w in EN_MARKETING)
    if fog >= 4:
        score = max(20, score - min(25, fog * 3))
        notes_zh += " 营销空词略降清晰度。"
        notes_en += " Marketing fog slightly lowers clarity."
    return int(_clamp(score / 100.0) * 100), notes_zh, notes_en


def _heuristic_human_score(
    signals: Dict[str, float],
    weights: Dict[str, float],
    lc=None,
) -> int:
    """
    Human score from offline features only (no ollama_dna_overlap).

    v3.0: uses the calibrated mapping when anchors are available, so the DNA
    gating band [35, 65] means the same thing as the scores shown in the UI
    (in v2.4 the two were computed by different curves).
    """
    sig = {k: v for k, v in signals.items() if k != "ollama_dna_overlap"}
    if lc is not None:
        try:
            from calibration import score_signals as _score_signals

            p = float(_score_signals(sig, weights, lc)["ai_probability"])
            return int(round((1.0 - _clamp(p)) * 100))
        except Exception:  # noqa: BLE001
            pass
    acc = 0.0
    wsum = 0.0
    for name, value in sig.items():
        if name not in FEATURE_META:
            continue
        w = float(weights.get(name, 0.0))
        acc += w * float(value)
        wsum += w
    if wsum <= 0:
        return 50
    ai_p = legacy_probability(acc)
    return int(round((1.0 - _clamp(ai_p)) * 100))


def effective_dna_blend(
    heuristic_human: int,
    overlap: float,
    base_blend: float = DNA_BLEND_DEFAULT,
    text_len: int = 0,
) -> Tuple[float, str]:
    """
    Decide DNA blend weight.

    Returns (blend, gate_reason).
      - outside uncertain band [35,65] → 0
      - text shorter than DNA_MIN_CHARS → 0
      - memorization FP: extremely high overlap on already-human-high text → ~0
      - else → base_blend (capped)
    """
    base = max(0.0, min(0.20, float(base_blend)))
    if text_len and text_len < DNA_MIN_CHARS:
        return 0.0, "skipped_short_text"
    if heuristic_human < DNA_UNCERTAIN_LO or heuristic_human > DNA_UNCERTAIN_HI:
        return 0.0, "skipped_outside_uncertain_band"
    # Classic / memorized text: model regenerates almost verbatim, but heuristic
    # already leans human — do not punish (Gettysburg-style false positive).
    if (
        overlap >= DNA_MEMORIZATION_OVERLAP
        and heuristic_human >= DNA_MEMORIZATION_HUMAN
    ):
        return 0.0, "memorization_downweight"
    return base, "blended"



def _slot_local_p(value: float, anc) -> float:
    """Map a raw feature value to a local AI-prob via its anchor (0≈human, 1≈AI).

    v3.2: honour the sign of ``g``. A slot the corpus marked ``flipped`` is one
    where *higher values went with human text* (measured: 主语刻意复提 on
    鲁迅《影的告别》 — the repetition is technique, not slop). Reporting
    ``p_ai = 1.00`` for such a slot contradicts the very calibration that
    produced it, and it showed up as a "很像" annotation on a document the tool
    had just scored 100/100.
    """
    if anc is None:
        return float(max(0.0, min(1.0, value)))
    z = anc.z(float(value))
    if float(getattr(anc, "g", 1.0) or 0.0) < 0:
        z = -z
    # squash z∈[-3,3] → (0,1); z=0 → ~0.18-ish via soft logistic around 0
    return 1.0 / (1.0 + math.exp(-1.4 * z))


def _slot_confidence(value: float, anc, mass_ratio: float = 0.5) -> float:
    """
    Concentration / evidence for one feature question — not max(p).

    Peaked when the value sits far from the idle/human floor relative to scale,
    and the slot has non-trivial gain. Idle near-zero firings → low confidence.
    """
    v = float(value)
    if v < ANN_FLOOR:
        return 0.0
    g = abs(float(getattr(anc, "g", 0.5) or 0.5)) if anc is not None else 0.5
    scale = max(float(getattr(anc, "scale", 0.5) or 0.5), 1e-3) if anc is not None else 0.5
    lo = float(getattr(anc, "lo", 0.0) or 0.0) if anc is not None else 0.0
    peaked = _clamp(abs(v - lo) / scale)
    # gain-weighted peak + mild mass prior
    conf = _clamp(0.15 + 0.55 * peaked * g + 0.20 * _clamp(mass_ratio) + 0.10 * _clamp(v))
    return conf


def action_band(p_ai: float, confidence: float) -> str:
    """Cost-aware policy band from (p, confidence) — Jev principle 4."""
    if p_ai >= ANN_FLAG_P and confidence >= ANN_FLAG_CONF:
        return "flag"
    if p_ai >= ANN_REVIEW_P:
        return "review"
    return "ignore"


def build_annotations(
    features: List["FeatureHit"],
    sentences: List["SentenceResult"],
    paragraphs: List[Dict[str, Any]],
    lc=None,
    mass_ratio: float = 0.5,
    doc_confidence: float = 0.5,
    doc_p_ai: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Typed parallel questions on shared text state (Jev principle 1).

    Each firing lit_zh_* / zh_* / marketing_* slot is an independent question
    returning its own p_ai + confidence + action. Sentence/paragraph heats are
    also first-class annotation labels so the rewrite UI can list them without
    conflating document Human Score.
    """
    out: List[Dict[str, Any]] = []
    anchors = getattr(lc, "anchors", {}) if lc is not None else {}

    for f in features:
        name = f.name
        # 前缀表原来只写在 docstring 里、判断处是硬编码的三项——于是 v3.2 加的
        # mod_* / poe_* 一律进不了 annotations[]，界面上「命中的成分」永远是 0 条，
        # 而句子那侧照样在报红。同一个事实两个说法，比没有标注更误导。
        if not name.startswith(ANN_FEATURE_KEYS_PREFIX):
            continue
        if float(f.value) < ANN_FLOOR:
            continue
        anc = anchors.get(name)
        # 这个口径里没有锚点 = 这个问题在本语域不成立（诗歌口径不量句长类槽）。
        # 早先的写法会把原始值直接当概率报出来（实测：一首诗报出「机械场景 100%」，
        # 而那个槽在诗歌口径里根本没被测量）。
        if lc is not None and anc is None:
            continue
        # 校准没量出方向的槽（|g|≈0）不该被当成一条"证据"报出来：
        # 它对总分的贡献是 0，对读者的影响却不是 0。
        if anc is not None and abs(float(anc.g)) < 0.05:
            continue
        p_ai = _slot_local_p(f.value, anc)
        # v3.2: 不再把文档级温度套到槽级概率上。
        # ``lc.temperature`` 是在**文档后验**上拟合的（拟合目标：让
        # human_score 的 ECE 变小），而槽级 p 是另一类量——它的校准对象是
        # "这个特征在此文档里的取值"，两者不是同一个随机变量。
        # 实测后果：鲁迅《秋夜》的 mod_para_uniformity 原始值只有 0.095，
        # 槽级 p=0.79 已经偏松，再过一遍 T=0.425 被抬到 0.958，
        # 于是一篇被判 98 分的散文里出现了"很像 AI 的段落整齐"。
        # 要校准槽级概率，得在槽级标签上单独拟合 T，不能借用文档的。
        conf = _slot_confidence(f.value, anc, mass_ratio=mass_ratio)
        act = action_band(p_ai, conf)
        # 作用域对齐：``kind="feature"`` 是**整篇的习惯统计**，和 Human Score
        # 是同一个作用域（句子/段落标注不是，它们的作用域是片段，本来就该
        # 和总分不一致）。同一个作用域的两个结论可以不同，但不该矛盾到要用户
        # 自己裁决。文档已经强判人类（p<0.20，即 Human Score ≥80）时把 flag
        # 降为 review：仍然告诉用户"这处习惯落在 AI 的频段上"，但不再说"很像"。
        # 实测触发点：鲁迅《秋夜》98 分 + 「段落长度过于整齐 很像」。
        if act == "flag" and float(doc_p_ai) < 0.20:
            act = "review"
        out.append(
            {
                "kind": "feature",
                "key": name,
                "name_zh": f.name_zh,
                "name_en": f.name_en,
                "p_ai": round(p_ai, 4),
                "confidence": round(conf, 4),
                "action": act,
                "value": round(float(f.value), 4),
                "span": None,
            }
        )

    for i, s in enumerate(sentences):
        if s.heat == "none" and s.ai_probability < ANN_REVIEW_P:
            continue
        # Sentence confidence: doc confidence × how far from 0.5 (concentration proxy)
        conf = _clamp(doc_confidence * (0.45 + 0.55 * abs(s.ai_probability - 0.5) * 2.0))
        act = action_band(s.ai_probability, conf)
        if act == "ignore" and s.heat == "none":
            continue
        # Heat can promote review even if band says ignore near threshold
        if s.heat in ("red", "yellow") and act == "ignore":
            act = "review"
        out.append(
            {
                "kind": "sentence",
                "key": f"sentence_{i}",
                "name_zh": f"句子#{i + 1}",
                "name_en": f"Sentence #{i + 1}",
                "p_ai": round(float(s.ai_probability), 4),
                "heat_score": round(float(s.ai_probability), 4),
                "confidence": round(conf, 4),
                "action": act if s.heat != "red" else ("flag" if act != "ignore" else "flag"),
                "span": {"char_start": s.char_start, "length": s.length},
                "preview": (s.text[:48] + "…") if len(s.text) > 48 else s.text,
            }
        )

    for i, p in enumerate(paragraphs or []):
        p_ai = float(p.get("ai_probability") or 0.0)
        if p_ai < ANN_REVIEW_P:
            continue
        conf = _clamp(doc_confidence * (0.40 + 0.60 * abs(p_ai - 0.5) * 2.0))
        act = action_band(p_ai, conf)
        out.append(
            {
                "kind": "paragraph",
                "key": f"paragraph_{i}",
                "name_zh": f"段落#{i + 1}",
                "name_en": f"Paragraph #{i + 1}",
                "p_ai": round(p_ai, 4),
                "heat_score": round(p_ai, 4),
                "confidence": round(conf, 4),
                "action": act,
                "span": {
                    "char_start": p.get("char_start"),
                    "char_end": p.get("char_end"),
                },
                "preview": p.get("preview") or "",
            }
        )

    # Stable order: flag → review → ignore; within that, higher p_ai first
    rank = {"flag": 0, "review": 1, "ignore": 2}
    out.sort(key=lambda d: (rank.get(d.get("action", "ignore"), 9), -float(d.get("p_ai") or 0)))
    return out


def analyze(
    text: str,
    weights: Optional[Dict[str, float]] = None,
    backend: str = "heuristic",
    ollama_host: str = "http://127.0.0.1:11434",
    ollama_model: str = "",
    ollama_timeout: float = 45.0,
    dna_blend: float = DNA_BLEND_DEFAULT,
    onnx_blend: float = ONNX_BLEND_DEFAULT,
    use_onnx: bool = True,
    highlight_mode: str = "max_band",
) -> DocumentResult:
    """
    backend:
      - heuristic: offline features only (default; recommended)
      - ollama_dna / hybrid: heuristic + optional Ollama DNA-GPT-lite overlap
        (experimental). DNA only blends when heuristic Human Score is in the
        uncertain band [35, 65]; memorized classics with extreme overlap are
        down-weighted. Prefer text models (qwen2:7b), never auto-pick VL.

    Optional ONNX 3rd vote: EN uses models/onnx/ettin-68m; CJK-heavy text uses
    optional models/onnx/aigc-zh (Eslzzyl/aigc-detector-zh-onnx). Missing weights
    → soft-fail; heuristic (incl. v2.3 ZH + v2.4 literary ZH features) still runs.
    """
    text = (text or "").strip()
    backend = (backend or "heuristic").strip().lower()
    if backend in ("ollama", "dna", "ollama_hybrid"):
        backend = "hybrid"
    weights = load_weights() if weights is None else dict(weights)
    # ensure all keys (including optional ollama feature weight)
    for k, v in DEFAULT_WEIGHTS.items():
        weights.setdefault(k, v)
    weights.setdefault("ollama_dna_overlap", 0.0)
    s = sum(max(0.0, w) for w in weights.values())
    if s > 0:
        weights = {k: max(0.0, float(v)) / s for k, v in weights.items()}
    # v3.0 note: v2.3/v2.4 zeroed ZH slots on EN by hand. Calibration now does it
    # from data — a slot with no human/AI separation gets gain≈0 and stops
    # counting, so the hand-written language gate is no longer applied.

    sentences = split_sentences(text)
    toks = tokenize(text)
    min_needed = min_chars_for(text)
    too_short = len(text) < min_needed

    signals = _feature_signals(text, sentences) if text else {k: 0.0 for k in DEFAULT_WEIGHTS}

    ollama_meta: Dict[str, Any] = {"used": False, "status": "skipped"}
    signals.setdefault("ollama_dna_overlap", 0.0)

    # --- v3.0 calibrated scoring -------------------------------------------
    calib = get_calibration()
    cjk = cjk_ratio(text)
    # v3.2 register router. One question only — "is this verse?" — because the
    # answer decides which bucket is legitimate, not which text is AI. It is
    # deliberately not a "is this modernist" classifier: that is a judgement
    # about intent, and surface statistics cannot carry it (register_zh.py).
    reg: Dict[str, Any] = {"poetry_p": 0.0, "is_poetry": 0.0}
    if text:
        try:
            from register_zh import register_of
            reg = register_of(text)
        except Exception:                                  # noqa: BLE001
            pass
    reg_bucket = "poetry" if reg.get("is_poetry") else None
    lc = calib.for_text(cjk, register=reg_bucket) if calib else None
    calib_meta: Dict[str, Any] = {
        "calibrated": bool(calib),
        "lang": getattr(lc, "lang", "en" if cjk < 0.25 else "zh"),
        "bucket": getattr(lc, "lang", ""),
        "register": {
            "poetry_p": reg.get("poetry_p", 0.0),
            "routed": bool(reg_bucket),
            "masked_features": (
                list(POETRY_MASKED_FEATURES) if reg_bucket else []
            ),
        },
        "provisional": bool(getattr(lc, "provisional", True)),
        "n_human": int(getattr(lc, "n_human", 0)),
        "n_ai": int(getattr(lc, "n_ai", 0)),
        "slope": float(getattr(lc, "slope", 0.0)),
        "center": float(getattr(lc, "center", 0.0)),
        "temperature": float(getattr(lc, "temperature", 1.0) or 1.0),
        "ece_before": getattr(lc, "ece_before", None),
        "ece_after": getattr(lc, "ece_after", None),
        "brier_before": getattr(lc, "brier_before", None),
        "brier_after": getattr(lc, "brier_after", None),
        "notes": list(getattr(lc, "notes", []) or []),
    }
    if lc is not None:
        from calibration import score_signals as _score_signals
        scored = _score_signals(signals, weights, lc)
        ai_probability = float(scored["ai_probability"])
        z_agg = float(scored["z_agg"])
        mass_ratio = float(scored["mass_ratio"])
        slot_info = {d["name"]: d for d in scored["slots"]}
        calib_meta["z_agg"] = round(z_agg, 4)
        calib_meta["mass_ratio"] = round(mass_ratio, 4)
    else:
        # No calibration.json: fall back to the v2.4 curve (and say so).
        acc = sum(weights.get(k, 0.0) * signals.get(k, 0.0) for k in DEFAULT_WEIGHTS)
        ai_probability = legacy_probability(acc)
        z_agg = acc
        mass_ratio = 1.0
        slot_info = {}
        calib_meta["fallback"] = "no calibration.json — run calibrate.py"
    heur_human = int(round((1.0 - _clamp(ai_probability)) * 100))
    p_heuristic = ai_probability

    if backend in ("hybrid", "ollama_dna") and text and not too_short:
        try:
            from backends.ollama_client import (
                dna_consistency_score,
                is_vision_or_embed_model,
                pick_model,
                probe,
            )
            st = probe(ollama_host)
            ollama_meta["status"] = "up" if st.available else "down"
            ollama_meta["models"] = st.models
            ollama_meta["probe_error"] = st.error
            ollama_meta["only_vl_available"] = bool(
                getattr(st, "only_vl_available", False)
            )
            ollama_meta["heuristic_human_score"] = heur_human

            model = (ollama_model or "").strip() or st.selected_model or pick_model(st.models)
            # Refuse auto / sticky VL unless user explicitly typed a VL name
            if model and is_vision_or_embed_model(model) and not (ollama_model or "").strip():
                model = pick_model(st.models)  # may be ""
            ollama_meta["selected_model"] = model

            if st.available and getattr(st, "only_vl_available", False) and not model:
                signals["ollama_dna_overlap"] = 0.0
                ollama_meta["fallback"] = "only_vl_models_pull_text_qwen2"
                ollama_meta["dna_gate"] = "skipped_only_vl"
            elif not (st.available and model):
                signals["ollama_dna_overlap"] = 0.0
                ollama_meta["fallback"] = "ollama_unavailable_using_heuristic"
                ollama_meta["dna_gate"] = "skipped_unavailable"
            else:
                # Pre-gate on heuristic band / length before expensive generate
                pre_blend, pre_reason = effective_dna_blend(
                    heur_human, overlap=0.0, base_blend=dna_blend, text_len=len(text)
                )
                if pre_reason in ("skipped_short_text", "skipped_outside_uncertain_band"):
                    signals["ollama_dna_overlap"] = 0.0
                    ollama_meta["used"] = False
                    ollama_meta["dna_gate"] = pre_reason
                    ollama_meta["dna_blend_applied"] = 0.0
                else:
                    dna = dna_consistency_score(
                        text, model=model, host=ollama_host, timeout=ollama_timeout
                    )
                    ollama_meta["dna"] = dna
                    overlap = float(dna.get("overlap") or 0.0)
                    blend, gate_reason = effective_dna_blend(
                        heur_human,
                        overlap=overlap,
                        base_blend=dna_blend,
                        text_len=len(text),
                    )
                    ollama_meta["dna_gate"] = gate_reason
                    ollama_meta["dna_blend_applied"] = blend
                    if dna.get("error"):
                        ollama_meta["used"] = False
                        signals["ollama_dna_overlap"] = 0.0
                    elif blend <= 1e-9:
                        ollama_meta["used"] = False
                        signals["ollama_dna_overlap"] = float(dna.get("ai_like") or 0.0)
                        # keep signal visible but weight 0
                        weights = {
                            k: v for k, v in weights.items() if k != "ollama_dna_overlap"
                        }
                        os = sum(max(0.0, v) for v in weights.values()) or 1.0
                        weights = {k: max(0.0, v) / os for k, v in weights.items()}
                        weights["ollama_dna_overlap"] = 0.0
                    else:
                        ollama_meta["used"] = True
                        signals["ollama_dna_overlap"] = float(dna.get("ai_like") or 0.0)
                        others = {
                            k: v for k, v in weights.items() if k != "ollama_dna_overlap"
                        }
                        os = sum(others.values()) or 1.0
                        scale = max(0.0, 1.0 - blend)
                        weights = {k: v / os * scale for k, v in others.items()}
                        weights["ollama_dna_overlap"] = blend
        except Exception as e:  # noqa: BLE001
            signals["ollama_dna_overlap"] = 0.0
            ollama_meta = {
                "used": False,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "heuristic_human_score": heur_human,
            }

    features: List[FeatureHit] = []
    for name, value in signals.items():
        if name not in FEATURE_META:
            continue
        w = weights.get(name, 0.0)
        zh, en, ez, ee = FEATURE_META[name]
        info = slot_info.get(name)
        if info is not None:
            z = info["z"]
            contrib = info["contribution"]
            gain = info["g"]
            eff_w = info["effective_weight"]
            a_lo = info["anchor_lo"]
            a_hi = info["anchor_hi"]
            disc = info["disc"]
            src = info["anchor_source"]
            flipped = info["flipped"]
        else:
            z = value
            contrib = w * value
            gain = 1.0
            eff_w = w
            a_lo, a_hi, disc, src, flipped = 0.0, 1.0, 0.0, "uncalibrated", False
        features.append(
            FeatureHit(
                name=name,
                name_zh=zh,
                name_en=en,
                value=value,
                weight=w,
                contribution=contrib,
                explanation_zh=ez,
                explanation_en=ee,
                z=z,
                anchor_lo=a_lo,
                anchor_hi=a_hi,
                disc=disc,
                gain=gain,
                effective_weight=eff_w,
                anchor_source=src,
                flipped=flipped,
            )
        )

    if too_short and text:
        # Uncertain — pull toward the middle, and the confidence term will say so
        ai_probability = 0.5 * ai_probability + 0.25
        p_heuristic = ai_probability
    if not text:
        ai_probability = 0.0
        p_heuristic = 0.0

    # --- Optional ONNX 3rd vote (soft-fail if missing) ---
    # EN → ettin-68m; CJK-heavy → optional aigc-zh (v2.3); missing → soft-fail.
    onnx_meta: Dict[str, Any] = {
        "used": False,
        "status": "skipped",
        "blend_applied": 0.0,
        "path": None,
    }
    if use_onnx and text and not too_short:
        try:
            from backends.onnx_classifier import (
                ONNX_BLEND_DEFAULT as _ONNX_B,
                blend_ai_probability,
                cjk_ratio as _cjk_ratio,
                probe as onnx_probe,
                probe_zh as onnx_probe_zh,
                score_text as onnx_score_text,
                score_text_zh as onnx_score_text_zh,
            )
            cjk = _cjk_ratio(text)
            onnx_meta["cjk_ratio"] = round(cjk, 4)
            heur_h = int(round((1.0 - _clamp(ai_probability)) * 100))
            onnx_meta["heuristic_human_before_onnx"] = heur_h

            scored = None
            if cjk >= 0.35:
                # Prefer Chinese ONNX; soft-fail if absent (heuristics already ZH-boosted)
                st_zh = onnx_probe_zh()
                onnx_meta["path"] = "zh"
                onnx_meta["model_dir"] = st_zh.model_dir
                onnx_meta["providers"] = st_zh.providers
                onnx_meta["status"] = st_zh.reason
                if st_zh.available:
                    scored = onnx_score_text_zh(text)
                else:
                    onnx_meta["status"] = f"zh_unavailable:{st_zh.reason}"
                    onnx_meta["used"] = False
            else:
                st = onnx_probe()
                onnx_meta["path"] = "en"
                onnx_meta["status"] = st.reason
                onnx_meta["model_dir"] = st.model_dir
                onnx_meta["providers"] = st.providers
                if st.available:
                    scored = onnx_score_text(text)
                else:
                    onnx_meta["status"] = st.reason

            if scored is not None:
                onnx_meta["score"] = {
                    k: scored.get(k)
                    for k in ("ok", "ai_prob", "human_prob", "error", "model", "logits")
                }
                if scored.get("ok"):
                    b = max(0.0, min(0.45, float(onnx_blend if onnx_blend is not None else _ONNX_B)))
                    # Protect classics / already-human heuristic (Gettysburg-style FP)
                    if heur_h >= 68 and float(scored["ai_prob"]) >= 0.55:
                        b = 0.0
                        onnx_meta["status"] = "skipped_already_human"
                    # Protect concrete ZH alley/lit when heuristic already human-high
                    if cjk >= 0.35 and heur_h >= 62 and float(scored["ai_prob"]) >= 0.60:
                        b = 0.0
                        onnx_meta["status"] = "skipped_already_human_zh"
                    pre = ai_probability
                    if b > 1e-9:
                        ai_probability = blend_ai_probability(pre, float(scored["ai_prob"]), b)
                        onnx_meta["used"] = True
                        onnx_meta["status"] = "blended"
                    else:
                        onnx_meta["used"] = False
                    onnx_meta["blend_applied"] = b
                    onnx_meta["ai_prob_before"] = round(pre, 4)
                    onnx_meta["ai_prob_onnx"] = round(float(scored["ai_prob"]), 4)
                    zh, en, ez, ee = FEATURE_META["onnx_ai_prob"]
                    features.append(
                        FeatureHit(
                            name="onnx_ai_prob",
                            name_zh=zh,
                            name_en=en,
                            value=float(scored["ai_prob"]),
                            weight=b,
                            contribution=b * float(scored["ai_prob"]),
                            explanation_zh=ez,
                            explanation_en=ee,
                        )
                    )
                else:
                    onnx_meta["status"] = scored.get("error") or onnx_meta.get("status")
        except Exception as e:  # noqa: BLE001
            onnx_meta = {
                "used": False,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "blend_applied": 0.0,
            }

    features.sort(key=lambda f: f.contribution, reverse=True)

    # --- voter disagreement (feeds confidence) ------------------------------
    votes: List[float] = [p_heuristic]
    if onnx_meta.get("used"):
        votes.append(float(onnx_meta.get("ai_prob_onnx") or 0.0))
    if ollama_meta.get("used"):
        votes.append(float(signals.get("ollama_dna_overlap") or 0.0))
    disagreement = (max(votes) - min(votes)) if len(votes) >= 2 else 0.0

    confidence = confidence_of(
        text_len=len(text),
        min_needed=min_needed,
        mass_ratio=mass_ratio,
        ai_p=ai_probability,
        provisional=bool(calib_meta.get("provisional", True)),
        disagreement=disagreement,
    )
    insufficient = bool(
        (text and (too_short or confidence < CONFIDENCE_INSUFFICIENT))
    )

    human_score = int(round((1.0 - _clamp(ai_probability)) * 100))
    band_zh, band_en = human_band(human_score)
    clarity, c_zh, c_en = clarity_score(text, sentences) if text else (0, "", "")

    # --- paragraph windows (mixed-authorship scan) --------------------------
    para_results: List[Dict[str, Any]] = []
    mixed_hint = False
    if text and len(text) >= PARA_MIN_CHARS * 2:
        windows = paragraph_windows(text)
        for w_start, chunk in windows:
            if len(chunk) < 60:
                continue
            w_sents = split_sentences(chunk)
            w_sig = _feature_signals(chunk, w_sents)
            if lc is not None:
                from calibration import score_signals as _score_signals
                w_p = float(_score_signals(w_sig, weights, lc)["ai_probability"])
            else:
                w_acc = sum(weights.get(k, 0.0) * w_sig.get(k, 0.0) for k in DEFAULT_WEIGHTS)
                w_p = legacy_probability(w_acc)
            para_results.append(
                {
                    "char_start": w_start,
                    "char_end": w_start + len(chunk),
                    "chars": len(chunk),
                    "ai_probability": round(w_p, 4),
                    "preview": (chunk[:60] + "…") if len(chunk) > 60 else chunk,
                }
            )
        if len(para_results) >= 2:
            ps = [p["ai_probability"] for p in para_results]
            spread = max(ps) - min(ps)
            mixed_hint = bool(spread >= MIXED_SPREAD)
            para_results.sort(key=lambda d: d["char_start"])

    # Per-sentence
    sent_results: List[SentenceResult] = []
    for start, sent in sentences:
        local = _sentence_local_signals(sent)
        if lc is not None:
            from calibration import score_signals as _score_signals
            sp = float(_score_signals(local, weights, lc)["ai_probability"])
        else:
            local_acc = sum(weights.get(k, 0.0) * local.get(k, 0.0) for k in DEFAULT_WEIGHTS)
            sp = legacy_probability(local_acc)
        # Blend with the document prior for stability (single sentences are noisy)
        sp = _clamp(0.62 * sp + 0.38 * ai_probability)
        top: List[FeatureHit] = []
        ranked = sorted(
            ((k, local[k], weights.get(k, 0.0) * local[k]) for k in local),
            key=lambda x: x[2],
            reverse=True,
        )
        for k, val, contrib in ranked[:4]:
            if contrib < 0.01 and val < 0.15:
                continue
            zh, en, ez, ee = FEATURE_META[k]
            top.append(
                FeatureHit(
                    name=k,
                    name_zh=zh,
                    name_en=en,
                    value=val,
                    weight=weights.get(k, 0.0),
                    contribution=contrib,
                    explanation_zh=ez,
                    explanation_en=ee,
                )
            )
        sent_results.append(
            SentenceResult(
                text=sent,
                char_start=start,
                length=len(sent),
                ai_probability=sp,
                heat="none",
                top_features=top,
            )
        )

    # --- sentence heat: absolute evidence AND within-document rank ----------
    # v2.4 used a single absolute threshold, so a clean human document could
    # still show a random red sentence and an all-AI document flagged nothing
    # special. v3.0 combines the calibrated probability with the sentence's
    # percentile inside this document, then gates the highlights.
    n_sent_res = len(sent_results)
    if n_sent_res:
        order = sorted(range(n_sent_res), key=lambda i: sent_results[i].ai_probability)
        percentile = {i: r / max(1, n_sent_res - 1) for r, i in enumerate(order)}
        for i, s in enumerate(sent_results):
            q = percentile[i]
            s.ai_probability = _clamp(0.60 * s.ai_probability + 0.40 * q)
            s.heat = heat_label(s.ai_probability)
        if ai_probability < 0.55:
            # Document leans human: only surface the few genuine outliers.
            k = max(1, math.ceil(0.08 * n_sent_res))
            keep = set(
                sorted(range(n_sent_res), key=lambda i: -sent_results[i].ai_probability)[:k]
            )
            for i, s in enumerate(sent_results):
                if i not in keep or percentile[i] < 0.80:
                    s.heat = "none"
        for i, s in enumerate(sent_results):
            s.percentile = round(percentile[i], 4)

    # Highlight policy (TipType UX convention: only the top severity band)
    has_red = any(s.heat == "red" for s in sent_results)
    has_yellow = any(s.heat == "yellow" for s in sent_results)
    if highlight_mode == "max_band":
        if has_red:
            for s in sent_results:
                if s.heat != "red":
                    s.heat = "none"
            policy = "max_band_red_only"
        elif has_yellow:
            policy = "max_band_yellow_only"
        else:
            policy = "no_highlight"
    else:
        policy = "all_bands"

    annotations = build_annotations(
        features=features,
        sentences=sent_results,
        paragraphs=para_results,
        lc=lc,
        mass_ratio=mass_ratio,
        doc_confidence=confidence,
        doc_p_ai=ai_probability,
    )

    return DocumentResult(
        text=text,
        char_count=len(text),
        word_count=len(toks),
        sentence_count=len(sentences),
        too_short=too_short,
        ai_probability=ai_probability,
        human_score=human_score,
        human_band_zh=band_zh,
        human_band_en=band_en,
        clarity_score=clarity,
        clarity_notes_zh=c_zh,
        clarity_notes_en=c_en,
        features=features,
        sentences=sent_results,
        highlight_policy=policy,
        weights_used=weights,
        confidence=round(confidence, 4),
        insufficient_evidence=insufficient,
        paragraphs=para_results,
        annotations=annotations,
        meta={
            "min_chars": min_needed,
            "min_chars_en": MIN_CHARS,
            "min_chars_zh": MIN_CHARS_ZH,
            "cjk_ratio": round(cjk_ratio(text), 4) if text else 0.0,
            "heat_red": HEAT_RED,
            "heat_yellow": HEAT_YELLOW,
            "engine": (
                _ENGINE if backend == "heuristic" else f"local_hybrid_v3.1:{backend}"
            ),
            "backend": backend,
            "not_tiptype": True,
            "ollama": ollama_meta,
            "onnx": onnx_meta,
            "calibration": calib_meta,
            "confidence": round(confidence, 4),
            "insufficient_evidence": insufficient,
            "voter_disagreement": round(disagreement, 4),
            "mixed_writing_hint": mixed_hint,
            "paragraphs": para_results,
        },
    )


def export_json(result: DocumentResult, path: Path) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    demo = (
        "Moreover, this innovative solution seamlessly leverages cutting-edge technology "
        "to optimize workflows, enhance productivity, and unlock unprecedented value for "
        "organizations navigating today's rapidly evolving digital landscape."
    )
    r = analyze(demo)
    print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
