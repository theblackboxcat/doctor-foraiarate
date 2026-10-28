#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline compression / window-burstiness signals (no ML weights).

Inspired by compression-based detectors (e.g. Uncanny-style zlib/lzma ratios)
and the classical idea that AI prose is more statistically redundant / uniform.
Not a copy of any proprietary detector.
"""
from __future__ import annotations

import lzma
import math
import zlib
from typing import Dict, List, Tuple


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _ratio(data: bytes, method: str) -> float:
    if not data:
        return 1.0
    if method == "zlib":
        out = zlib.compress(data, level=9)
    else:
        out = lzma.compress(data)
    return len(out) / max(1, len(data))


# Short bilingual "AI-flavored" vs "human diary" preludes for cross-compression.
# Cross-ratio idea: text that shares patterns with AI prelude compresses better
# when concatenated with it (relative to human prelude).
_AI_PRELUDE = (
    "Moreover, it is important to note that this innovative and comprehensive approach "
    "seamlessly leverages cutting-edge solutions to optimize workflows, enhance productivity, "
    "and unlock unprecedented value for stakeholders. Furthermore, organizations should "
    "carefully evaluate best practices and implement scalable frameworks across the landscape. "
    "值得注意的是，该方案能够全面赋能业务闭环，有效地提质增效，实现数字化转型与深度融合。"
).encode("utf-8")

_HUMAN_PRELUDE = (
    "I spilled coffee on the bus this morning and the guy next to me just sighed. "
    "Mom called twice; I didn't pick up the second time because I was still hunting "
    "for a dry napkin under the seat. Later the sky went that weird purple before rain. "
    "昨晚买醋的时候老板娘又念叨青菜贵，我只点头，口袋里的零钱还是湿的，撕开黏成一团。"
).encode("utf-8")


def compression_ai_like(text: str) -> float:
    """
    Higher ≈ more AI-like redundancy / affinity to AI-flavored prelude.
    Combines self-compression (lower ratio → more compressible → AI-ish)
    with cross-compression affinity vs AI vs human preludes.
    """
    raw = (text or "").encode("utf-8", errors="ignore")
    if len(raw) < 40:
        return 0.35

    z_self = _ratio(raw, "zlib")
    # Typical English prose zlib ratio ~0.4–0.7; very templated lower.
    # Map low ratio → high AI signal.
    self_signal = _clamp((0.62 - z_self) / 0.35)

    z_ai = _ratio(_AI_PRELUDE + raw, "zlib")
    z_hu = _ratio(_HUMAN_PRELUDE + raw, "zlib")
    # If concatenating with AI prelude compresses better (lower ratio) than with human,
    # affinity leans AI.
    cross = _clamp((z_hu - z_ai) / 0.08 + 0.5)

    # Mild lzma self check (sometimes separates better on longer text)
    lx = _ratio(raw, "lzma")
    lx_signal = _clamp((0.55 - lx) / 0.30)

    return _clamp(0.45 * self_signal + 0.35 * cross + 0.20 * lx_signal)


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return n / max(1, len(text))


def window_burstiness_inv(text: str, window: int = 0, step: int = 0) -> float:
    """
    Compression-ratio variance across sliding windows.
    Low variance (uniform compressibility) → AI-like → higher score.

    v3.0 fix: v2.4 slid an **80-byte** window. Chinese is ~3 bytes/char in
    UTF-8, so a Chinese window held ~27 characters against ~80 for English —
    two different measurements wearing one name. Measured, that made human
    Chinese diary prose score *higher* (0.86) than English AI slop (0.82).
    The window is now counted in **characters** and sized per language so both
    scripts see roughly the same amount of text per window.
    """
    text = (text or "").strip()
    if not text:
        return 0.0
    cjk = _cjk_ratio(text)
    if not window:
        # ~60 CJK chars vs ~200 latin chars carry a comparable amount of content
        window = 60 if cjk >= 0.25 else 200
    if not step:
        step = max(1, window // 2)
    if len(text) < window * 2:
        return 0.0

    ratios: List[float] = []
    for i in range(0, max(1, len(text) - window + 1), step):
        chunk = text[i : i + window]
        if len(chunk) < window // 2:
            break
        ratios.append(_ratio(chunk.encode("utf-8", errors="ignore"), "zlib"))
        if len(ratios) >= 24:
            break

    if len(ratios) < 3:
        return 0.0

    mean = sum(ratios) / len(ratios)
    var = sum((r - mean) ** 2 for r in ratios) / len(ratios)
    cv = math.sqrt(var) / mean if mean else 0.0
    # Human windows vary more; very low CV → AI. CJK compresses differently,
    # so the neutrality point is per-script.
    neutral = 0.055 if cjk >= 0.25 else 0.12
    return _clamp((neutral - cv) / neutral)


def compression_feature_bundle(text: str) -> Dict[str, float]:
    return {
        "compression_ai_like": compression_ai_like(text),
        "window_burstiness_inv": window_burstiness_inv(text),
    }
