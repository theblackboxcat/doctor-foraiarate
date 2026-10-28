#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature calibration layer (v3.0).

Why this exists
---------------
v2.4 summed *raw* feature values with hand-set weights and pushed the result
through ``p = 0.08 + 0.92*(1-exp(-1.85*acc))``. Measured on the built-in cases,
the raw weighted sum ``acc`` only ever landed in [0.13, 0.64], so the human
score was squeezed into roughly 28..72 no matter how strong the evidence was.
Two separate defects caused that:

1. **Raw values are not comparable across features.** Some slots idle at 0.0 on
   human prose, others never drop below 0.3 for anybody. Summing them lets the
   "always on" slots dominate and drowns the discriminative ones.
2. **Several slots carry no signal at all** (or point the wrong way on Chinese).
   Measured: ``window_burstiness_inv`` = 0.86 human / 0.82 AI slop;
   ``clause_layering_inv`` = 1.00 on human Chinese diary prose.

This module fixes both *from data* instead of by hand:

* every feature gets an anchor: ``lo`` = median over human samples,
  ``hi`` = median over AI samples, ``scale`` = IQR-based spread;
* the raw value is mapped to ``z = (v - lo) / scale`` so z≈0 means
  "typical human", z≈1 means "typical AI";
* each feature gets a **discriminability** ``g = tanh(disc)`` where
  ``disc = (median_ai - median_human) / spread``. A slot with no separation
  gets g≈0 and stops diluting the score; a slot that points the *wrong way*
  on your corpus gets a **negative** g, which is reported rather than hidden;
* the aggregate z is pushed through a logistic whose slope/intercept are pinned
  by two points: human median → 0.18, AI median → 0.82.

The result is a score that actually reaches the ends of the scale when the
evidence is strong, and that says "insufficient evidence" when it is not.

Anchors are per-language. Chinese-only slots have near-zero human *and* AI
values on English samples, so their ``disc`` collapses to 0 and they switch
themselves off — this replaces v2.4's hand-written language gating.

Honesty
-------
Calibration quality is bounded by the sample counts in ``calib/``. If a
language has fewer than ``MIN_SAMPLES_PER_SIDE`` human samples, the anchors for
that language are marked ``provisional`` and the detector lowers its confidence
instead of pretending the numbers are solid.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CALIB_FILE = "calibration.json"
MIN_SAMPLES_PER_SIDE = 5
DEFAULT_MIN_SCALE = 0.10          # floor for the z denominator
FLOOR_HUMAN_P = 0.18              # where the human median should land
TARGET_AI_P = 0.82                # where the AI median should land

# Slots that are language-agnostic enough to borrow English anchors when the
# Chinese human side is too small. Everything else falls back to a neutral
# anchor derived from the feature's own declared direction.
LANGUAGE_AGNOSTIC_SLOTS = (
    "adverb_density",
    "adverb_verb_proxy",
    "unique_word_ratio_inv",
    "sentence_start_variety_inv",
    "burstiness_inv",
    "list_template",
    "comma_density",
    "clause_layering_inv",
    "marketing_inflation",
    "compression_ai_like",
    "window_burstiness_inv",
)


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else 0.5 * float(s[mid - 1] + s[mid])


def _iqr(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> float:
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return float(s[lo])
        return float(s[lo] + (s[hi] - s[lo]) * (idx - lo))

    return max(0.0, q(0.75) - q(0.25))


@dataclass
class Anchor:
    """Per-feature calibration anchor."""

    lo: float = 0.0            # human median
    hi: float = 1.0            # ai median
    scale: float = 1.0         # z denominator
    disc: float = 0.0          # discriminability (ai - human) / spread
    g: float = 0.0             # applied gain = tanh(disc)
    n_human: int = 0
    n_ai: int = 0
    source: str = "measured"   # measured | en_fallback | neutral
    flipped: bool = False      # True when data says the slot points at humans

    def z(self, v: float) -> float:
        # Clamped: a slot whose reference spread is tiny would otherwise explode
        # into z=10 and dominate the aggregate on a single weird document.
        return max(-3.0, min(3.0, (float(v) - self.lo) / max(self.scale, 1e-6)))


@dataclass
class LangCalibration:
    """Anchors + logistic mapping for one language bucket."""

    lang: str = "en"
    anchors: Dict[str, Anchor] = field(default_factory=dict)
    z_human: float = 0.0
    z_ai: float = 1.0
    slope: float = 3.0
    center: float = 0.5
    n_human: int = 0
    n_ai: int = 0
    provisional: bool = False
    notes: List[str] = field(default_factory=list)
    # v3.1 post-hoc temperature scaling (Guo 2017). T>1 cools overconfidence.
    temperature: float = 1.0
    ece_before: Optional[float] = None
    ece_after: Optional[float] = None
    brier_before: Optional[float] = None
    brier_after: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lang": self.lang,
            "anchors": {k: asdict(v) for k, v in self.anchors.items()},
            "z_human": self.z_human,
            "z_ai": self.z_ai,
            "slope": self.slope,
            "center": self.center,
            "n_human": self.n_human,
            "n_ai": self.n_ai,
            "provisional": self.provisional,
            "notes": self.notes,
            "temperature": self.temperature,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LangCalibration":
        def _opt_float(key: str) -> Optional[float]:
            v = d.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        return LangCalibration(
            lang=d.get("lang", "en"),
            anchors={k: Anchor(**v) for k, v in (d.get("anchors") or {}).items()},
            z_human=float(d.get("z_human", 0.0)),
            z_ai=float(d.get("z_ai", 1.0)),
            slope=float(d.get("slope", 3.0)),
            center=float(d.get("center", 0.5)),
            n_human=int(d.get("n_human", 0)),
            n_ai=int(d.get("n_ai", 0)),
            provisional=bool(d.get("provisional", False)),
            notes=list(d.get("notes") or []),
            temperature=float(d.get("temperature", 1.0) or 1.0),
            ece_before=_opt_float("ece_before"),
            ece_after=_opt_float("ece_after"),
            brier_before=_opt_float("brier_before"),
            brier_after=_opt_float("brier_after"),
        )

    def probability(self, z: float) -> float:
        return 1.0 / (1.0 + math.exp(-self.slope * (z - self.center)))

    def apply_temperature(self, p: float) -> float:
        """Apply stored T in logit space. T=1 is a no-op; T>1 cools toward 0.5."""
        T = float(self.temperature or 1.0)
        if abs(T - 1.0) < 1e-9:
            return float(p)
        from prob_metrics import apply_temperature as _at
        return float(_at([p], T)[0])


@dataclass
class Calibration:
    version: str = "3.1"
    built_at: str = ""
    langs: Dict[str, LangCalibration] = field(default_factory=dict)
    corpus: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "built_at": self.built_at,
            "langs": {k: v.to_dict() for k, v in self.langs.items()},
            "corpus": self.corpus,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        p = Path(path) if path else Path(__file__).with_name(CALIB_FILE)
        p.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return p

    @staticmethod
    def load(path: Optional[Path] = None) -> Optional["Calibration"]:
        p = Path(path) if path else Path(__file__).with_name(CALIB_FILE)
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        try:
            return Calibration(
                version=str(data.get("version", "3.0")),
                built_at=str(data.get("built_at", "")),
                langs={k: LangCalibration.from_dict(v) for k, v in (data.get("langs") or {}).items()},
                corpus=data.get("corpus") or {},
            )
        except (TypeError, KeyError, ValueError):
            return None

    def for_text(self, cjk: float, register: Optional[str] = None) -> LangCalibration:
        """Pick the bucket for a text.

        ``register`` (v3.2) lets line-broken verse be scored against poetry
        anchors. It matters because the sentence-shaped features — burstiness,
        comma density, clause layering — are not *wrong* on a poem, they are
        undefined: there are no sentences to measure. Scoring verse against
        prose anchors therefore reports a confident number about the wrong
        quantity, which is exactly the failure this argument exists to stop.

        Falls back to the prose bucket when no poetry anchors are present, so
        an old ``calibration.json`` keeps working unchanged.
        """
        if register == "poetry" and "zh_poetry" in self.langs:
            return self.langs["zh_poetry"]
        if cjk >= 0.25 and "zh" in self.langs:
            return self.langs["zh"]
        return self.langs.get("en") or next(iter(self.langs.values()), LangCalibration())


# ---------------------------------------------------------------------------
# Building anchors
# ---------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _q(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    idx = p * (n - 1)
    lo_i = int(math.floor(idx))
    hi_i = int(math.ceil(idx))
    if lo_i == hi_i:
        return float(s[lo_i])
    return float(s[lo_i] + (s[hi_i] - s[lo_i]) * (idx - lo_i))


def _anchor_from_samples(
    h_vals: Sequence[float],
    a_vals: Sequence[float],
    conservative_flip: bool = False,
) -> Anchor:
    """
    lo  = median of the human side (what "typical human" looks like)
    hi  = 75th percentile of the AI side (a slot can be genuinely
          discriminative while firing on only a minority of AI texts — a bare
          median would zero those out, which is how ``list_template`` died in
          the first calibration run)

          Note on the ``hi == 0`` case: the ``zh`` bucket mixes 公文/议论文 with
          literary text, so a purely literary slot can have 75% zeros on the AI
          side and ``hi`` collapses to 0. ``scale`` then floors at 0.10, which
          is *not* the problem it looks like — ``max(hi - lo, 0.10)`` is the
          same whether ``hi`` is 0 or 0.085, so replacing ``hi`` with the AI
          mean here changes nothing except through ``disc``-adjacent paths.
          Tried it: it flipped ``realist_prose/luxun_guxiang_08`` from 0.399 to
          0.402 and cost the corpus its 0/112 false-positive record for zero
          visible gain. The real fix for the resulting over-eager slot
          probability lives in ``detector.build_annotations`` (scope-aware
          action band), not here.
    disc = (mean_ai - mean_human) / spread, spread floored at 0.08 so a flat
          both-sides slot cannot manufacture a huge discriminability
    """
    lo = _median(h_vals)
    hi = _q(a_vals, 0.75)
    spread = 0.5 * (_iqr(h_vals) + _iqr(a_vals))
    denom = max(spread, 0.08)
    disc = (_mean(a_vals) - _mean(h_vals)) / denom
    scale = max(abs(hi - lo), DEFAULT_MIN_SCALE)
    g = math.tanh(disc / 1.2)
    if conservative_flip and disc < 0:
        # A flipped slot is a real finding, but on a small corpus it is usually
        # corpus artefact rather than law. Halve it and let the UI show it.
        g *= 0.5
    return Anchor(
        lo=lo,
        hi=hi,
        scale=scale,
        disc=round(disc, 4),
        g=round(g, 4),
        n_human=len(h_vals),
        n_ai=len(a_vals),
        source="measured",
        flipped=bool(disc < 0),
    )


def prior_anchor(reason: str = "prior_provisional") -> Anchor:
    """
    Design-prior anchor for slots that have no usable reference corpus yet
    (e.g. Chinese-only slots before the user supplies Chinese human drafts).

    Keeps the slot's declared direction (0 = quiet, 1 = fully firing) instead of
    switching it off, which is what a measured-but-empty anchor would do.
    """
    return Anchor(
        lo=0.0,
        hi=0.5,
        scale=0.5,
        disc=1.0,
        g=0.60,
        n_human=0,
        n_ai=0,
        source=reason,
        flipped=False,
    )


def build_lang_calibration(
    lang: str,
    human_signals: Sequence[Dict[str, float]],
    ai_signals: Sequence[Dict[str, float]],
    fallback: Optional[LangCalibration] = None,
    feature_keys: Optional[Sequence[str]] = None,
) -> LangCalibration:
    """Compute anchors + logistic for one language."""
    keys = list(feature_keys) or sorted(
        set().union(*[set(s) for s in human_signals]) if human_signals else set()
    )
    provisional = (
        len(human_signals) < MIN_SAMPLES_PER_SIDE or len(ai_signals) < MIN_SAMPLES_PER_SIDE
    )
    notes: List[str] = []
    anchors: Dict[str, Anchor] = {}

    conservative_flip = (
        len(human_signals) < 2 * MIN_SAMPLES_PER_SIDE
        or len(ai_signals) < 2 * MIN_SAMPLES_PER_SIDE
    )
    for k in keys:
        h_vals = [float(s.get(k, 0.0)) for s in human_signals]
        a_vals = [float(s.get(k, 0.0)) for s in ai_signals]
        if not h_vals or not a_vals:
            anchors[k] = Anchor(source="neutral")
            continue
        if provisional and k in LANGUAGE_AGNOSTIC_SLOTS and fallback is not None and k in fallback.anchors:
            anchors[k] = Anchor(**{**asdict(fallback.anchors[k]), "source": "en_fallback"})
            continue
        if provisional:
            # Language-specific slot with too little human reference material:
            # fall back to the declared design prior rather than to a measured
            # anchor that is mostly zeros (which silently kills the slot).
            anchors[k] = prior_anchor("prior_provisional")
            continue
        anchors[k] = _anchor_from_samples(h_vals, a_vals, conservative_flip=conservative_flip)

    if provisional:
        notes.append(
            f"provisional: human={len(human_signals)} ai={len(ai_signals)} "
            f"(need >= {MIN_SAMPLES_PER_SIDE} per side); confidence is down-weighted"
        )
    if fallback is not None and provisional:
        notes.append("language-agnostic slots borrowed English anchors")

    # Aggregate z on both sides, then pin the logistic through the two medians.
    z_h, z_a = _aggregate_z_range(anchors, human_signals, ai_signals)
    if z_a - z_h < 1e-3:
        # Degenerate corpus: fall back to unit separation so the mapping stays finite.
        z_a = z_h + 1.0
        notes.append("degenerate separation on calibration corpus; using unit span")
    center = 0.5 * (z_h + z_a)
    slope = _solve_slope(FLOOR_HUMAN_P, TARGET_AI_P, z_h, z_a)

    return LangCalibration(
        lang=lang,
        anchors=anchors,
        z_human=round(z_h, 4),
        z_ai=round(z_a, 4),
        slope=round(slope, 4),
        center=round(center, 4),
        n_human=len(human_signals),
        n_ai=len(ai_signals),
        provisional=provisional,
        notes=notes,
    )


def _solve_slope(p_h: float, p_a: float, z_h: float, z_a: float) -> float:
    """Slope so that p(z_h)=p_h and p(z_a)=p_a (least-squares on the two points)."""
    l_h = math.log(p_h / (1.0 - p_h))
    l_a = math.log(p_a / (1.0 - p_a))
    dz = z_a - z_h
    if abs(dz) < 1e-6:
        return 3.0
    slope = (l_a - l_h) / dz
    # Center chosen so both points are hit as closely as possible.
    return max(0.4, min(12.0, slope))


def _weighted_z(anchors: Dict[str, Anchor], signals: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, float]:
    """Return (weighted mean z, total effective mass)."""
    num = 0.0
    mass = 0.0
    for k, v in signals.items():
        anc = anchors.get(k)
        if anc is None:
            continue
        w = float(weights.get(k, 0.0)) * float(anc.g)
        num += w * anc.z(float(v))
        mass += abs(w)
    if mass <= 1e-9:
        return 0.0, 0.0
    return num / mass, mass


def _aggregate_z_range(
    anchors: Dict[str, Anchor],
    human_signals: Sequence[Dict[str, float]],
    ai_signals: Sequence[Dict[str, float]],
) -> Tuple[float, float]:
    """Median aggregate z on the human side and on the AI side (unit weights)."""
    unit_w = {k: 1.0 for k in anchors}
    zs_h = [_weighted_z(anchors, s, unit_w)[0] for s in human_signals] or [0.0]
    zs_a = [_weighted_z(anchors, s, unit_w)[0] for s in ai_signals] or [1.0]
    return _median(zs_h), _median(zs_a)


def score_signals(
    signals: Dict[str, float],
    weights: Dict[str, float],
    lc: LangCalibration,
) -> Dict[str, Any]:
    """
    Map raw feature signals → calibrated AI probability + diagnostics.

    Returns dict with: z_agg, ai_probability, per-slot z/g/contribution,
    effective mass ratio (how much of the weight budget actually discriminates).
    """
    z_agg, mass = _weighted_z(lc.anchors, signals, weights)
    p_raw = lc.probability(z_agg)
    p = lc.apply_temperature(p_raw)
    total_w = sum(abs(float(weights.get(k, 0.0))) for k in lc.anchors)
    mass_ratio = mass / total_w if total_w > 1e-9 else 0.0
    slots = []
    for k, v in signals.items():
        anc = lc.anchors.get(k)
        if anc is None:
            continue
        w = float(weights.get(k, 0.0))
        eff = w * float(anc.g)
        slots.append(
            {
                "name": k,
                "value": round(float(v), 4),
                "z": round(anc.z(float(v)), 4),
                "anchor_lo": round(anc.lo, 4),
                "anchor_hi": round(anc.hi, 4),
                "disc": anc.disc,
                "g": anc.g,
                "weight": round(w, 4),
                "effective_weight": round(eff, 4),
                "contribution": round(eff * anc.z(float(v)), 4),
                "anchor_source": anc.source,
                "flipped": anc.flipped,
            }
        )
    slots.sort(key=lambda d: -abs(d["contribution"]))
    return {
        "z_agg": round(z_agg, 4),
        "ai_probability": p,
        "ai_probability_raw": round(p_raw, 6),
        "temperature": float(getattr(lc, "temperature", 1.0) or 1.0),
        "effective_mass": round(mass, 4),
        "mass_ratio": round(mass_ratio, 4),
        "slots": slots,
        "provisional": lc.provisional,
    }


__all__ = [
    "Anchor",
    "Calibration",
    "LangCalibration",
    "LANGUAGE_AGNOSTIC_SLOTS",
    "MIN_SAMPLES_PER_SIDE",
    "build_lang_calibration",
    "score_signals",
]
