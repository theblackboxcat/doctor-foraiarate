#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3.0 backend — FastAPI REST API + static frontend.

    python server.py            # http://127.0.0.1:7860
    python server.py --port 8000 --no-browser

The Gradio app (app.py) still works; this is the separated front/back end.
Nothing here calls a remote service: analysis is local, Ollama is optional and
off by default, ONNX weights are optional.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
WEB_DIR = ROOT / "web"
EXPORT_DIR = ROOT / "exports"
EXPORT_DIR.mkdir(exist_ok=True)
HISTORY_FILE = EXPORT_DIR / "history.jsonl"

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from detector import (  # noqa: E402
    DEFAULT_WEIGHTS,
    MIN_CHARS,
    MIN_CHARS_ZH,
    analyze,
    cjk_ratio,
    load_weights,
    reload_calibration,
    save_weights,
)
# 引擎标识只有一个真相来源：detector._ENGINE。
# 这里原来是字面量 "local_heuristic_v3.1"，于是 v3.2 发版后 /api/meta 仍报 v3.1——
# 而前端标题、Cowrite 侧的 ai_annotate.available() 各自还有一份自己的版本号。
# 报 bug 的人会说"我在 v3.1 上遇到的"，实际跑的是 v3.2，排查从第一句就走偏。
from detector import _ENGINE as ENGINE_ID  # noqa: E402

app = FastAPI(title="Local AI-writing signal detector",
              version=ENGINE_ID.rsplit("_v", 1)[-1])


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = ""
    backend: str = "heuristic"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    weights: Optional[Dict[str, float]] = None
    use_onnx: bool = True
    use_whitebox: bool = False
    highlight_mode: str = "max_band"
    remember: bool = True


class WeightsPayload(BaseModel):
    weights: Dict[str, float]


class CalibUpload(BaseModel):
    group: str = "human_zh"      # human_en | ai_en | human_zh | ai_zh
    filename: str = "sample.txt"
    text: str = ""


class CompareRequest(BaseModel):
    before: str = ""
    after: str = ""
    weights: Optional[Dict[str, float]] = None
    use_onnx: bool = True


class ImportPayload(BaseModel):
    filename: str = "text.txt"
    content_base64: str = ""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _norm_weights(w: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not w:
        return None
    clean = {k: max(0.0, float(v)) for k, v in w.items() if k in DEFAULT_WEIGHTS}
    s = sum(clean.values())
    if s <= 0:
        return None
    return {k: v / s for k, v in clean.items()}


def _append_history(payload: Dict[str, Any]) -> None:
    try:
        with HISTORY_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _read_history(limit: int = 30) -> List[Dict[str, Any]]:
    if not HISTORY_FILE.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return rows[-limit:]


def _extract_text(raw: bytes, filename: str) -> str:
    """txt / md / (docx, pdf when the optional deps are installed)."""
    name = (filename or "").lower()
    if name.endswith(".docx"):
        try:
            import docx  # type: ignore
            import io
            d = docx.Document(io.BytesIO(raw))
            return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"读取 .docx 失败（可选依赖 python-docx 未装？）：{type(e).__name__}: {e}")
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore
            import io
            r = PdfReader(io.BytesIO(raw))
            return "\n\n".join((p.extract_text() or "") for p in r.pages)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"读取 .pdf 失败（可选依赖 pypdf 未装？）：{type(e).__name__}: {e}")
    for enc in ("utf-8", "utf-8-sig", "gbk", "big5", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    idx = WEB_DIR / "index.html"
    if not idx.is_file():
        return PlainTextResponse("web/index.html missing", status_code=500)
    return FileResponse(idx)


@app.get("/api/meta")
def api_meta():
    return {
        "engine": ENGINE_ID,
        "min_chars": MIN_CHARS,
        "min_chars_zh": MIN_CHARS_ZH,
        "features": [
            {"name": k, "default_weight": v} for k, v in DEFAULT_WEIGHTS.items()
        ],
        "not_tiptype": True,
        "disclaimer_zh": "本地启发式 + 校准，非 TipType 专有模型/API，不能证明作者身份。",
    }


@app.get("/api/weights")
def api_get_weights():
    return {"weights": load_weights(), "defaults": DEFAULT_WEIGHTS}


@app.post("/api/weights")
def api_set_weights(payload: WeightsPayload):
    w = _norm_weights(payload.weights)
    if not w:
        raise HTTPException(400, "权重为空或非法")
    path = save_weights(w)
    return {"ok": True, "path": str(path), "weights": w}


@app.get("/api/calibration")
def api_calibration():
    from calibration import Calibration
    cal = Calibration.load()
    if cal is None:
        return {"present": False, "message": "没有 calibration.json，正在使用 v2.4 回退曲线。请跑 calibrate.py。"}
    out = {
        "present": True,
        "version": cal.version,
        "built_at": cal.built_at,
        "langs": {},
        "corpus": cal.corpus,
    }
    for lang, lc in cal.langs.items():
        out["langs"][lang] = {
            "n_human": lc.n_human,
            "n_ai": lc.n_ai,
            "provisional": lc.provisional,
            "slope": lc.slope,
            "center": lc.center,
            "temperature": getattr(lc, "temperature", 1.0),
            "ece_before": getattr(lc, "ece_before", None),
            "ece_after": getattr(lc, "ece_after", None),
            "brier_before": getattr(lc, "brier_before", None),
            "brier_after": getattr(lc, "brier_after", None),
            "notes": lc.notes,
            "anchors": [
                {
                    "name": k,
                    "lo": a.lo,
                    "hi": a.hi,
                    "scale": a.scale,
                    "disc": a.disc,
                    "g": a.g,
                    "source": a.source,
                    "flipped": a.flipped,
                }
                for k, a in sorted(lc.anchors.items(), key=lambda kv: -abs(kv[1].g))
            ],
        }
    return out


@app.post("/api/calibrate")
def api_recalibrate():
    """Rebuild calibration.json from calib/ (runs in a thread-safe way)."""
    lock = threading.Lock()
    with lock:
        try:
            import calibrate as calib_mod
            cal, _ = calib_mod.build(write=True)
            reload_calibration()
            return {
                "ok": True,
                "built_at": cal.built_at,
                "langs": {k: {"n_human": v.n_human, "n_ai": v.n_ai, "provisional": v.provisional}
                          for k, v in cal.langs.items()},
            }
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"重算失败：{type(e).__name__}: {e}")


@app.get("/api/calibration/validate")
def api_validate():
    try:
        import calibrate as calib_mod
        return {"ok": True, "loo": calib_mod.leave_one_out(write_report=False)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/api/calib/upload")
def api_calib_upload(payload: CalibUpload):
    allowed = {"human_en", "ai_en", "human_zh", "ai_zh"}
    if payload.group not in allowed:
        raise HTTPException(400, f"group 必须是 {sorted(allowed)} 之一")
    text = (payload.text or "").strip()
    if len(text) < 40:
        raise HTTPException(400, "样本太短（至少 40 字）")
    folder = ROOT / "calib" / payload.group
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch for ch in payload.filename if ch.isalnum() or ch in "._-")
    safe = safe or "sample.txt"
    if not safe.endswith(".txt"):
        safe += ".txt"
    (folder / safe).write_text(text + "\n", encoding="utf-8")
    return {"ok": True, "path": str(folder / safe),
            "hint": "放好样本后点「重算校准」；人类侧 ≥5 条后中文锚点才不再是 provisional。"}


@app.get("/api/probe")
def api_probe(ollama_host: str = "http://127.0.0.1:11434"):
    out: Dict[str, Any] = {"ollama": {"available": False}, "onnx": {"en": False, "zh": False}}
    try:
        from backends.ollama_client import probe
        st = probe(ollama_host)
        out["ollama"] = {
            "available": st.available,
            "host": st.host,
            "models": st.models[:20],
            "selected_model": st.selected_model,
            "only_vl_available": getattr(st, "only_vl_available", False),
            "error": st.error,
        }
    except Exception as e:  # noqa: BLE001
        out["ollama"]["error"] = f"{type(e).__name__}: {e}"
    try:
        from backends.onnx_classifier import DEFAULT_MODEL_DIR, DEFAULT_ZH_MODEL_DIR, probe, probe_zh
        se, sz = probe(), probe_zh()
        out["onnx"] = {
            "en": {"available": se.available, "dir": str(DEFAULT_MODEL_DIR), "reason": se.reason},
            "zh": {"available": sz.available, "dir": str(DEFAULT_ZH_MODEL_DIR), "reason": sz.reason},
        }
    except Exception as e:  # noqa: BLE001
        out["onnx"]["error"] = f"{type(e).__name__}: {e}"
    return out


@app.post("/api/import")
def api_import(payload: ImportPayload):
    """Decode an uploaded file (txt/md/docx/pdf) into plain text.

    Sent as base64 JSON on purpose — that keeps python-multipart out of the
    dependency list, so `pip install -r requirements.txt` stays small.
    """
    import base64

    try:
        raw = base64.b64decode(payload.content_base64 or "")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"base64 解码失败：{e}")
    if not raw:
        raise HTTPException(400, "空文件")
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(400, "文件过大（>8MB）")
    text = _extract_text(raw, payload.filename)
    return {"ok": True, "filename": payload.filename, "chars": len(text), "text": text}


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest):
    text = (req.text or "").strip()
    result = analyze(
        text,
        weights=_norm_weights(req.weights),
        backend=req.backend or "heuristic",
        ollama_host=req.ollama_host or "http://127.0.0.1:11434",
        ollama_model=(req.ollama_model or "").strip(),
        use_onnx=bool(req.use_onnx),
        highlight_mode=req.highlight_mode or "max_band",
    )
    data = result.to_dict()

    # Optional white-box signal (needs Ollama; off by default)
    wb: Dict[str, Any] = {"used": False}
    if req.use_whitebox and text and result.char_count >= 200:
        try:
            from backends.ollama_whitebox import whitebox_signals
            wb = whitebox_signals(
                text,
                host=req.ollama_host or "http://127.0.0.1:11434",
                model=(req.ollama_model or "").strip(),
            )
        except Exception as e:  # noqa: BLE001
            wb = {"used": False, "error": f"{type(e).__name__}: {e}"}
    data["whitebox"] = wb

    data["analyzed_at"] = datetime.now().isoformat(timespec="seconds")
    if req.remember and text:
        _append_history(
            {
                "at": data["analyzed_at"],
                "chars": result.char_count,
                "human_score": result.human_score,
                "confidence": round(result.confidence, 3),
                "preview": text[:80],
            }
        )
    return JSONResponse(content=data)


@app.post("/api/compare")
def api_compare(req: CompareRequest):
    """Before/after revision comparison — did the AI flavour drop?"""
    w = _norm_weights(req.weights)
    a = analyze(req.before or "", weights=w, use_onnx=bool(req.use_onnx))
    b = analyze(req.after or "", weights=w, use_onnx=bool(req.use_onnx))
    feats_a = {f.name: f for f in a.features}
    feats_b = {f.name: f for f in b.features}
    deltas = []
    for name in DEFAULT_WEIGHTS:
        ca = float(getattr(feats_a.get(name), "contribution", 0.0) or 0.0)
        cb = float(getattr(feats_b.get(name), "contribution", 0.0) or 0.0)
        if abs(cb - ca) > 1e-4:
            deltas.append({"name": name, "before": round(ca, 4), "after": round(cb, 4),
                           "delta": round(cb - ca, 4)})
    deltas.sort(key=lambda d: d["delta"])
    return {
        "before": {
            "human_score": a.human_score,
            "ai_probability": round(a.ai_probability, 4),
            "confidence": round(a.confidence, 4),
            "chars": a.char_count,
        },
        "after": {
            "human_score": b.human_score,
            "ai_probability": round(b.ai_probability, 4),
            "confidence": round(b.confidence, 4),
            "chars": b.char_count,
        },
        "delta_human_score": b.human_score - a.human_score,
        "feature_deltas": deltas[:10],
    }


@app.post("/api/export")
def api_export(payload: Dict[str, Any]):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EXPORT_DIR / f"report_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "path": str(path)}


@app.get("/api/history")
def api_history():
    return {"items": _read_history()}


@app.delete("/api/history")
def api_clear_history():
    try:
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
    except OSError:
        pass
    return {"ok": True}


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    import uvicorn
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
