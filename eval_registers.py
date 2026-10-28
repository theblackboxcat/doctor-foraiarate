#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score the register evaluation set — before and after the v3.2 register layer.

    python eval_registers.py            # after: shipped calibration.json
    python eval_registers.py --before   # before: legacy anchors, legacy feature set
    python eval_registers.py --loo      # leave-one-out on the register buckets
    python eval_registers.py --detail   # per-file rows (worst first)

What "before" means, exactly
----------------------------
It rebuilds the v3.1 world: anchors from ``calib/`` only, the v3.1 feature list
(no ``mod_*`` / ``poe_*``), one Chinese bucket, no register router. That is not
a reconstruction from memory — it is the same code path the previous
``calibration.json`` was produced by, minus the files that did not exist then.

Reading the numbers
-------------------
``P(AI) ≥ 0.40`` is the tool's own "review" band and ``≥ 0.65`` its "flag" band
(``detector.ANN_REVIEW_P`` / ``ANN_FLAG_P``). For a human sample, either one is
a **false positive** — the thing this work exists to reduce. For an AI sample
they are detections.

A caveat that belongs next to the table: the AI side of this set was written by
the same assistant that wrote the features (see ``eval_set/make_ai_registers.py``).
So "detection went up" is weak evidence; "false positives went down" on
public-domain human text is the claim that actually holds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import detector  # noqa: E402
from calibration import Calibration, build_lang_calibration  # noqa: E402
from calibrate import _fit_lang_temperature, _signals_for  # noqa: E402

EVAL = ROOT / "eval_set"
CALIB = ROOT / "calib"

REVIEW_P = 0.40
FLAG_P = 0.65

# v3.1 feature list — what existed before the register layer. Kept as a literal
# so "before" cannot silently drift when DEFAULT_WEIGHTS grows.
V31_KEYS = [
    "adverb_density", "adverb_verb_proxy", "unique_word_ratio_inv",
    "sentence_start_variety_inv", "burstiness_inv", "list_template",
    "comma_density", "clause_layering_inv", "marketing_inflation",
    "compression_ai_like", "window_burstiness_inv",
    "zh_discourse_boilerplate", "zh_parallel_scaffold", "zh_policy_register",
    "zh_abstract_subject",
    "lit_zh_fake_terse_rhythm", "lit_zh_linger_gaze", "lit_zh_broken_parallel",
    "lit_zh_noun_reemphasize", "lit_zh_bushi_shi", "lit_zh_sceneless_simile",
    "lit_zh_flat_dialogue", "lit_zh_mech_scene",
]

REGISTERS = {
    "modernist_prose": ("modernist_prose", "ai_modernist_prose"),
    "poetry": ("poetry", "ai_poetry"),
    "realist_prose": ("realist_prose", "ai_realist_prose"),
}


def _load(folder: Path) -> List[Tuple[str, str]]:
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.glob("*.txt")):
        t = p.read_text(encoding="utf-8").strip()
        if len(t) >= 40:
            out.append((p.name, t))
    return out


def build_legacy() -> Calibration:
    """Rebuild the v3.1 calibration from calib/ only, with the v3.1 features."""
    langs = {}
    for lang, (hf, af) in (("en", ("human_en", "ai_en")), ("zh", ("human_zh", "ai_zh"))):
        def read(folder: Path) -> List[str]:
            if not folder.is_dir():
                return []
            return [p.read_text(encoding="utf-8").strip()
                    for p in sorted(folder.glob("*.txt"))
                    if len(p.read_text(encoding="utf-8").strip()) >= 40]

        h_sig = _signals_for(read(CALIB / hf))
        a_sig = _signals_for(read(CALIB / af))
        lc = build_lang_calibration(lang, h_sig, a_sig,
                                    fallback=langs.get("en"), feature_keys=V31_KEYS)
        _fit_lang_temperature(lc, h_sig, a_sig)
        langs[lang] = lc
    return Calibration(version="3.1", built_at="reconstructed", langs=langs)


def score_all(cal: Calibration) -> List[Dict]:
    """Run the real detector against a given calibration object."""
    detector._CALIB_CACHE["obj"] = cal
    detector._CALIB_CACHE["loaded"] = True
    rows: List[Dict] = []
    for register, (hdir, adir) in REGISTERS.items():
        for label, folder in (("human", hdir), ("ai", adir)):
            for name, text in _load(EVAL / folder):
                r = detector.analyze(text, use_onnx=False)
                rows.append({
                    "register": register,
                    "label": label,
                    "file": "%s/%s" % (folder, name),
                    "p": float(r.ai_probability),
                    "human_score": int(r.human_score),
                    "bucket": r.meta["calibration"].get("bucket"),
                    "poetry_p": r.meta["calibration"]["register"]["poetry_p"],
                    "confidence": float(r.confidence),
                    "insufficient": bool(r.insufficient_evidence),
                })
    detector.reload_calibration()
    return rows


def auc(human: List[float], ai: List[float]) -> float:
    if not human or not ai:
        return 0.5
    total = 0.0
    for h in human:
        for a in ai:
            total += 1.0 if a > h else (0.5 if a == h else 0.0)
    return total / (len(human) * len(ai))


def _med(xs: List[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def summarize(rows: List[Dict], title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print("%-16s %-6s %3s  %6s %6s %6s   %6s %6s   %s"
          % ("register", "label", "n", "medP", "minP", "maxP", "P>=.40", "P>=.65", "bucket"))
    for register in REGISTERS:
        for label in ("human", "ai"):
            sub = [r for r in rows if r["register"] == register and r["label"] == label]
            if not sub:
                continue
            ps = [r["p"] for r in sub]
            over = sum(1 for p in ps if p >= REVIEW_P) / len(ps)
            flag = sum(1 for p in ps if p >= FLAG_P) / len(ps)
            buckets = sorted({str(r["bucket"]) for r in sub})
            print("%-16s %-6s %3d  %6.3f %6.3f %6.3f   %5.0f%% %5.0f%%   %s"
                  % (register, label, len(sub), _med(ps), min(ps), max(ps),
                     over * 100, flag * 100, ",".join(buckets)))
    print("\n%-16s %s" % ("register", "AUC(human vs ai)"))
    for register in REGISTERS:
        h = [r["p"] for r in rows if r["register"] == register and r["label"] == "human"]
        a = [r["p"] for r in rows if r["register"] == register and r["label"] == "ai"]
        print("%-16s %.3f" % (register, auc(h, a)))

    fp = [r for r in rows if r["label"] == "human" and r["p"] >= REVIEW_P]
    print("\nfalse positives at P>=%.2f: %d / %d human samples"
          % (REVIEW_P, len(fp), sum(1 for r in rows if r["label"] == "human")))
    for r in sorted(fp, key=lambda d: -d["p"]):
        print("   %-40s p=%.3f  human=%3d  bucket=%s" %
              (r["file"], r["p"], r["human_score"], r["bucket"]))


def loo() -> None:
    """Leave-one-out on the register buckets: rebuild without the held-out item."""
    from calibrate import GROUPS, _read_group
    from register_zh import POETRY_MASKED_FEATURES
    from detector import DEFAULT_WEIGHTS
    from calibration import score_signals

    print("\n" + "=" * 78)
    print("leave-one-out on the register buckets (poetry + zh prose)")
    print("=" * 78)
    for lang in ("zh", "zh_poetry"):
        spec = GROUPS[lang]
        h_texts, h_names, _ = _read_group(spec["human"])
        a_texts, a_names, _ = _read_group(spec["ai"])
        h_sig = _signals_for(h_texts)
        a_sig = _signals_for(a_texts)
        keys = [k for k in DEFAULT_WEIGHTS if k not in spec["mask"]]
        unit = {k: 1.0 for k in keys}
        hp, ap, worst = [], [], []
        for i, sig in enumerate(h_sig):
            sub = [x for j, x in enumerate(h_sig) if j != i]
            lc = build_lang_calibration(lang, sub, a_sig, fallback=None, feature_keys=keys)
            p = float(score_signals(sig, unit, lc)["ai_probability"])
            hp.append(p)
            worst.append(("human", h_names[i], p))
        for i, sig in enumerate(a_sig):
            sub = [x for j, x in enumerate(a_sig) if j != i]
            lc = build_lang_calibration(lang, h_sig, sub, fallback=None, feature_keys=keys)
            p = float(score_signals(sig, unit, lc)["ai_probability"])
            ap.append(p)
            worst.append(("ai", a_names[i], p))
        print("  [%s] n_h=%d n_a=%d  LOO AUC=%.3f  human P in [%.3f, %.3f]  ai P in [%.3f, %.3f]"
              % (lang, len(hp), len(ap), auc(hp, ap), min(hp), max(hp), min(ap), max(ap)))
        for lab, name, p in sorted(worst, key=lambda x: x[2] if x[0] == "ai" else -x[2])[:4]:
            print("       worst %-5s %-38s p=%.3f" % (lab, name, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", action="store_true", help="score with reconstructed v3.1 calibration")
    ap.add_argument("--loo", action="store_true", help="leave-one-out on register buckets")
    ap.add_argument("--detail", action="store_true", help="per-file rows")
    args = ap.parse_args()

    if args.loo:
        loo()
        return 0

    if args.before:
        rows = score_all(build_legacy())
        summarize(rows, "BEFORE v3.2 — v3.1 anchors (calib/ only), v3.1 features, no router")
    else:
        cal = detector.reload_calibration()
        if cal is None:
            print("no calibration.json — run `python calibrate.py` first")
            return 1
        rows = score_all(cal)
        summarize(rows, "AFTER v3.2 — register buckets + craft-tell features")

    if args.detail:
        print("\n--- per file ---")
        for r in sorted(rows, key=lambda d: (d["register"], d["label"], -d["p"])):
            print("%-16s %-5s %-42s p=%.3f h=%3d bucket=%-9s poe=%.2f"
                  % (r["register"], r["label"], r["file"], r["p"], r["human_score"],
                     r["bucket"], r["poetry_p"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
