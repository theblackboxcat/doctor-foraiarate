#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optional ONNX AI-text classifier (3rd vote).

EN primary (Windows-laptop friendly, ~77 MB int8):
  Deckard618/textdet · ettin-68m
  https://huggingface.co/Deckard618/textdet

ZH optional (v2.3, ~98 MB INT8 BERT):
  Eslzzyl/aigc-detector-zh-onnx → models/onnx/aigc-zh/
  https://huggingface.co/Eslzzyl/aigc-detector-zh-onnx
  (based on yuchuantian/AIGC_detector_zhv3)

Soft-fails when model files / onnxruntime / tokenizers are missing.
No TipType API. No torch required for either path.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "onnx" / "ettin-68m"
DEFAULT_ZH_MODEL_DIR = ROOT / "models" / "onnx" / "aigc-zh"

REQUIRED_FILES = ("model.onnx", "tokenizer.json")

# Document-level blend into final AI probability when model is present.
ONNX_BLEND_DEFAULT = 0.18
ONNX_MIN_CHARS = 80
ONNX_MAX_LENGTH = 512

# From Deckard618 ettin-68m config.json calibration.offset
ETTIN_CALIB_OFFSET = 0.4767787903547287

_HF_REPO = "Deckard618/textdet"
_HF_SUBDIR = "ettin-68m"


@dataclass
class OnnxStatus:
    available: bool = False
    model_dir: str = ""
    reason: str = ""
    providers: List[str] = field(default_factory=list)
    label: str = "ettin-68m"
    approx_mb: int = 77


_SESSION_LOCK = threading.Lock()
_CACHED: Dict[str, Any] = {}  # key=str(model_dir) -> {session, tokenizer, ...}



def cjk_ratio(text: str) -> float:
    """Fraction of characters in CJK Unified Ideographs (+ compatibility)."""
    if not text:
        return 0.0
    n = 0
    for ch in text:
        o = ord(ch)
        if (
            0x4E00 <= o <= 0x9FFF
            or 0x3400 <= o <= 0x4DBF
            or 0xF900 <= o <= 0xFAFF
        ):
            n += 1
    return n / max(1, len(text))


def model_dir_ready(model_dir: Optional[Path] = None) -> Tuple[bool, Path, str]:
    """Return (ok, path, reason). Does not import onnxruntime."""
    path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not path.is_dir():
        return False, path, f"missing_dir:{path}"
    missing = [f for f in REQUIRED_FILES if not (path / f).is_file()]
    if missing:
        return False, path, f"missing_files:{','.join(missing)}"
    return True, path, "ok"


def probe(model_dir: Optional[Path] = None) -> OnnxStatus:
    """Cheap probe: files present + optional deps importable. Does not load weights."""
    ok, path, reason = model_dir_ready(model_dir)
    st = OnnxStatus(model_dir=str(path), reason=reason, available=False)
    if not ok:
        return st
    try:
        import onnxruntime as ort  # noqa: F401
        from tokenizers import Tokenizer  # noqa: F401
    except ImportError as e:
        st.reason = f"deps_missing:{e}"
        return st
    try:
        st.providers = list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        st.providers = ["CPUExecutionProvider"]
    st.available = True
    st.reason = "ready"
    return st


def _softmax2(a: float, b: float) -> Tuple[float, float]:
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    s = ea + eb
    return ea / s, eb / s


def _load(model_dir: Path) -> Dict[str, Any]:
    key = str(model_dir.resolve())
    with _SESSION_LOCK:
        if key in _CACHED:
            return _CACHED[key]
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tok.enable_truncation(max_length=ONNX_MAX_LENGTH)
        # Pad dynamically per encode; we build arrays ourselves.

        sess = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        inputs = {i.name for i in sess.get_inputs()}
        bundle = {
            "session": sess,
            "tokenizer": tok,
            "input_names": inputs,
            "np": np,
            "calib_offset": ETTIN_CALIB_OFFSET,
            "label": "Deckard618/textdet:ettin-68m",
        }
        _CACHED[key] = bundle
        return bundle


def score_text(
    text: str,
    model_dir: Optional[Path] = None,
    apply_calibration: bool = True,
) -> Dict[str, Any]:
    """
    Return dict:
      ok, ai_prob (0..1), human_prob, logits, error, model, model_dir
    Soft-fails (ok=False) instead of raising for missing deps/files.
    """
    text = (text or "").strip()
    out: Dict[str, Any] = {
        "ok": False,
        "ai_prob": 0.0,
        "human_prob": 1.0,
        "logits": None,
        "error": None,
        "model": "ettin-68m",
        "model_dir": "",
        "used": False,
    }
    if len(text) < ONNX_MIN_CHARS:
        out["error"] = "text_too_short"
        return out

    ok, path, reason = model_dir_ready(model_dir)
    out["model_dir"] = str(path)
    if not ok:
        out["error"] = reason
        return out

    try:
        bundle = _load(path)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"load_failed:{type(e).__name__}: {e}"
        return out

    try:
        np = bundle["np"]
        tok = bundle["tokenizer"]
        sess = bundle["session"]
        enc = tok.encode(text)
        ids = enc.ids[:ONNX_MAX_LENGTH]
        if not ids:
            out["error"] = "empty_tokens"
            return out
        # attention mask: 1 for real tokens
        mask = [1] * len(ids)
        # pad to at least 8 for some ORT kernels; model accepts variable length
        input_ids = np.asarray([ids], dtype=np.int64)
        attention_mask = np.asarray([mask], dtype=np.int64)
        feeds: Dict[str, Any] = {}
        names = bundle["input_names"]
        if "input_ids" in names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in names:
            feeds["attention_mask"] = attention_mask
        # Some exports also want token_type_ids
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        logits = sess.run(None, feeds)[0]
        # Expect [1, 2]
        flat = np.asarray(logits).reshape(-1).astype(float)
        if flat.size < 2:
            # single logit → sigmoid as P(AI)
            logit = float(flat[0])
            if apply_calibration:
                logit = logit - float(bundle.get("calib_offset") or 0.0)
            p_ai = 1.0 / (1.0 + math.exp(-logit))
            p_h = 1.0 - p_ai
            out.update(
                ok=True,
                used=True,
                ai_prob=float(p_ai),
                human_prob=float(p_h),
                logits=[logit],
                model=bundle["label"],
            )
            return out

        h_logit = float(flat[0])
        a_logit = float(flat[1])
        if apply_calibration:
            # Shift AI logit so 50% ≈ 5% FPR on MAGE human (per model card).
            a_logit = a_logit - float(bundle.get("calib_offset") or 0.0)
        p_h, p_ai = _softmax2(h_logit, a_logit)
        out.update(
            ok=True,
            used=True,
            ai_prob=float(p_ai),
            human_prob=float(p_h),
            logits=[h_logit, a_logit],
            model=bundle["label"],
        )
        return out
    except Exception as e:  # noqa: BLE001
        out["error"] = f"infer_failed:{type(e).__name__}: {e}"
        return out


def blend_ai_probability(
    heuristic_ai: float,
    onnx_ai: float,
    blend: float = ONNX_BLEND_DEFAULT,
) -> float:
    """Linear blend; blend clamped to [0, 0.45]."""
    b = max(0.0, min(0.45, float(blend)))
    h = max(0.0, min(1.0, float(heuristic_ai)))
    o = max(0.0, min(1.0, float(onnx_ai)))
    return (1.0 - b) * h + b * o



def probe_zh(model_dir: Optional[Path] = None) -> OnnxStatus:
    """Probe Chinese AIGC ONNX (aigc-zh). Soft-fails like probe()."""
    path = Path(model_dir) if model_dir else DEFAULT_ZH_MODEL_DIR
    st = probe(path)
    st.label = "aigc-zh"
    st.approx_mb = 98
    return st


def score_text_zh(
    text: str,
    model_dir: Optional[Path] = None,
    apply_calibration: bool = False,
) -> Dict[str, Any]:
    """
    Score with Chinese ONNX detector (BERT-base INT8).
    Logits [Human_Written, AI_Generated]; no ettin calibration offset.
    Soft-fails if missing.
    """
    path = Path(model_dir) if model_dir else DEFAULT_ZH_MODEL_DIR
    out = score_text(text, model_dir=path, apply_calibration=apply_calibration)
    if out.get("ok"):
        out["model"] = "Eslzzyl/aigc-detector-zh-onnx"
    out.setdefault("model", "aigc-zh")
    return out


__all__ = [
    "DEFAULT_MODEL_DIR",
    "DEFAULT_ZH_MODEL_DIR",
    "ONNX_BLEND_DEFAULT",
    "OnnxStatus",
    "blend_ai_probability",
    "cjk_ratio",
    "model_dir_ready",
    "probe",
    "probe_zh",
    "score_text",
    "score_text_zh",
]

