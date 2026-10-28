#!/usr/bin/env python3
"""Score a draft reply with local heuristic (onnx off). Prints one JSON line."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import analyze

def score(text: str) -> dict:
    r = analyze(text, backend="heuristic", use_onnx=False)
    anns = []
    for a in (getattr(r, "annotations", None) or [])[:12]:
        if isinstance(a, dict):
            anns.append({
                "id": a.get("id") or a.get("feature") or a.get("slot"),
                "action": a.get("action"),
                "p": a.get("p") or a.get("probability"),
                "confidence": a.get("confidence"),
            })
        else:
            anns.append({
                "id": getattr(a, "id", None) or getattr(a, "feature", None) or getattr(a, "slot", None),
                "action": getattr(a, "action", None),
                "p": getattr(a, "p", None) or getattr(a, "probability", None),
                "confidence": getattr(a, "confidence", None),
            })
    meta = r.meta if isinstance(getattr(r, "meta", None), dict) else dict(getattr(r, "meta", None) or {})
    p_ai = meta.get("p_ai")
    if p_ai is None:
        p_ai = meta.get("ai_probability") or meta.get("prob_ai")
    if p_ai is None and r.human_score is not None:
        p_ai = round(1.0 - (float(r.human_score) / 100.0), 4)
    return {
        "human_score": r.human_score,
        "p_ai": p_ai,
        "confidence": getattr(r, "confidence", None) or meta.get("confidence"),
        "engine": meta.get("engine"),
        "register": meta.get("register") or meta.get("zh_register") or meta.get("register_zh"),
        "flags": [a for a in anns if (a.get("action") or "") in ("flag", "review")],
        "n_chars": len(text),
        "meta_sample": {k: meta.get(k) for k in list(meta)[:20]},
    }

if __name__ == "__main__":
    text = sys.stdin.read() if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    if not text.strip():
        print(json.dumps({"error": "empty"}, ensure_ascii=False)); sys.exit(1)
    print(json.dumps(score(text), ensure_ascii=False, default=str))
