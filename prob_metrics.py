#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probability calibration metrics (pure numpy / stdlib).

Inspired by Guo et al. 2017 (temperature scaling) and the equal-mass binning
used in reliability diagrams. No torch / netcal / sklearn required.

Probability ≠ confidence:
  - ``p`` is calibrated against outcomes (ECE / Brier / reliability).
  - Confidence elsewhere in this tool is distribution concentration / evidence
    mass — do not treat max(p) or |p-0.5| alone as confidence.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover — numpy is a project dep
    np = None  # type: ignore


def _as_float_lists(
    p: Sequence[float], y: Sequence[float]
) -> Tuple[List[float], List[float]]:
    ps = [float(x) for x in p]
    ys = [float(x) for x in y]
    if len(ps) != len(ys):
        raise ValueError(f"p/y length mismatch: {len(ps)} vs {len(ys)}")
    return ps, ys


def brier(p: Sequence[float], y: Sequence[float]) -> float:
    """Brier score (lower is better). Binary y in {0,1}, p = P(y=1)."""
    ps, ys = _as_float_lists(p, y)
    if not ps:
        return float("nan")
    if np is not None:
        pa = np.asarray(ps, dtype=float)
        ya = np.asarray(ys, dtype=float)
        return float(np.mean((pa - ya) ** 2))
    return sum((a - b) ** 2 for a, b in zip(ps, ys)) / len(ps)


def ece(p: Sequence[float], y: Sequence[float], bins: int = 10) -> float:
    """
    Expected Calibration Error with equal-width probability bins (Guo 2017).

    ECE = Σ_b (|B_b|/n) · |acc(B_b) − conf(B_b)|
    """
    ps, ys = _as_float_lists(p, y)
    n = len(ps)
    if n == 0 or bins < 1:
        return float("nan")
    if np is not None:
        pa = np.asarray(ps, dtype=float)
        ya = np.asarray(ys, dtype=float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        total = 0.0
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            if i == bins - 1:
                mask = (pa >= lo) & (pa <= hi)
            else:
                mask = (pa >= lo) & (pa < hi)
            m = int(mask.sum())
            if m == 0:
                continue
            conf = float(pa[mask].mean())
            acc = float(ya[mask].mean())
            total += (m / n) * abs(acc - conf)
        return float(total)
    # stdlib fallback
    edges = [i / bins for i in range(bins + 1)]
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        bucket_p, bucket_y = [], []
        for pv, yv in zip(ps, ys):
            if i == bins - 1:
                ok = lo <= pv <= hi
            else:
                ok = lo <= pv < hi
            if ok:
                bucket_p.append(pv)
                bucket_y.append(yv)
        m = len(bucket_p)
        if m == 0:
            continue
        conf = sum(bucket_p) / m
        acc = sum(bucket_y) / m
        total += (m / n) * abs(acc - conf)
    return total


def reliability_bins(
    p: Sequence[float],
    y: Sequence[float],
    bins: int = 10,
    equal_mass: bool = True,
) -> List[Dict[str, Any]]:
    """
    Reliability diagram data.

    equal_mass=True → quantile (equal-count) bins; else equal-width on [0,1].
    Each bin: {lo, hi, n, conf_mean, acc_mean, gap}.
    """
    ps, ys = _as_float_lists(p, y)
    n = len(ps)
    if n == 0 or bins < 1:
        return []
    order = sorted(range(n), key=lambda i: ps[i])
    ps_s = [ps[i] for i in order]
    ys_s = [ys[i] for i in order]
    out: List[Dict[str, Any]] = []

    if equal_mass:
        # Equal-count bins; last bin absorbs remainder.
        base, rem = divmod(n, bins)
        start = 0
        for b in range(bins):
            size = base + (1 if b < rem else 0)
            if size <= 0:
                continue
            end = start + size
            bp = ps_s[start:end]
            by = ys_s[start:end]
            conf = sum(bp) / size
            acc = sum(by) / size
            out.append(
                {
                    "bin": b,
                    "lo": round(bp[0], 4),
                    "hi": round(bp[-1], 4),
                    "n": size,
                    "conf_mean": round(conf, 4),
                    "acc_mean": round(acc, 4),
                    "gap": round(acc - conf, 4),
                }
            )
            start = end
        return out

    # Equal-width
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        bp, by = [], []
        for pv, yv in zip(ps, ys):
            if b == bins - 1:
                ok = lo <= pv <= hi
            else:
                ok = lo <= pv < hi
            if ok:
                bp.append(pv)
                by.append(yv)
        m = len(bp)
        if m == 0:
            out.append(
                {
                    "bin": b,
                    "lo": round(lo, 4),
                    "hi": round(hi, 4),
                    "n": 0,
                    "conf_mean": None,
                    "acc_mean": None,
                    "gap": None,
                }
            )
            continue
        conf = sum(bp) / m
        acc = sum(by) / m
        out.append(
            {
                "bin": b,
                "lo": round(lo, 4),
                "hi": round(hi, 4),
                "n": m,
                "conf_mean": round(conf, 4),
                "acc_mean": round(acc, 4),
                "gap": round(acc - conf, 4),
            }
        )
    return out


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, float(p)))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def apply_temperature(p: Sequence[float], T: float) -> List[float]:
    """
    Temperature scaling in logit space: p' = σ(logit(p) / T).

    T>1 cools overconfident probabilities toward 0.5;
    T<1 sharpens. T must be >0.
    """
    T = max(1e-3, float(T))
    return [_sigmoid(_logit(float(x)) / T) for x in p]


def fit_temperature(
    p: Sequence[float],
    y: Sequence[float],
    T_grid: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Fit a single temperature T by minimising NLL on (p, y).

    Uses a coarse→fine grid search (no scipy). Returns
    ``{T, nll, ece_before, ece_after, brier_before, brier_after, n}``.
    """
    ps, ys = _as_float_lists(p, y)
    n = len(ps)
    empty = {
        "T": 1.0,
        "nll": float("nan"),
        "ece_before": float("nan"),
        "ece_after": float("nan"),
        "brier_before": float("nan"),
        "brier_after": float("nan"),
        "n": n,
    }
    if n < 2:
        return empty

    def nll_at(T: float) -> float:
        calibrated = apply_temperature(ps, T)
        total = 0.0
        eps = 1e-6
        for pv, yv in zip(calibrated, ys):
            pv = min(1.0 - eps, max(eps, pv))
            total += -(yv * math.log(pv) + (1.0 - yv) * math.log(1.0 - pv))
        return total / n

    if T_grid is None:
        # Coarse log-spaced grid, then refine around the best.
        coarse = [0.5 * (1.15 ** i) for i in range(0, 25)]  # ~0.5 .. ~5.5
    else:
        coarse = [float(t) for t in T_grid if float(t) > 0]

    best_T = 1.0
    best_nll = nll_at(best_T)
    for T in coarse:
        v = nll_at(T)
        if v < best_nll:
            best_nll, best_T = v, T

    # Local refine ±20%
    refine = [best_T * (0.85 + 0.03 * i) for i in range(11)]
    for T in refine:
        if T <= 0:
            continue
        v = nll_at(T)
        if v < best_nll:
            best_nll, best_T = v, T

    before_ece = ece(ps, ys)
    before_brier = brier(ps, ys)
    after = apply_temperature(ps, best_T)
    after_ece = ece(after, ys)
    after_brier = brier(after, ys)

    return {
        "T": round(best_T, 4),
        "nll": round(best_nll, 6),
        "ece_before": round(before_ece, 4),
        "ece_after": round(after_ece, 4),
        "brier_before": round(before_brier, 4),
        "brier_after": round(after_brier, 4),
        "n": n,
    }


__all__ = [
    "brier",
    "ece",
    "reliability_bins",
    "fit_temperature",
    "apply_temperature",
]
