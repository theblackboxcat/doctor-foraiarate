#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the optional ONNX AI-text weights for local_ai_detector.

EN default: Deckard618/textdet ettin-68m (~77 MB int8) → models/onnx/ettin-68m/
ZH optional: Eslzzyl/aigc-detector-zh-onnx (~98 MB INT8) → models/onnx/aigc-zh/

  python download_onnx_model.py
  python download_onnx_model.py --model ettin-150m
  python download_onnx_model.py --model aigc-zh      # Chinese detector

Requires: pip install huggingface_hub
(or: pip install -r requirements-onnx.txt)

No TipType. No credentials required for public HF repos.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Union

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "models" / "onnx"

# files: list of str (same remote/local name under subdir) OR (remote, local) tuples
MODELS = {
    "ettin-68m": {
        "repo": "Deckard618/textdet",
        "hf_prefix": "ettin-68m",
        "dest_subdir": "ettin-68m",
        "files": [
            "model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ],
        "approx_mb": 77,
        "note": "Recommended EN 3rd-vote for Windows laptop (int8, onnxruntime only).",
    },
    "ettin-150m": {
        "repo": "Deckard618/textdet",
        "hf_prefix": "ettin-150m",
        "dest_subdir": "ettin-150m",
        "files": [
            "model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ],
        "approx_mb": 168,
        "note": "Larger Ettin variant; still ONNX int8.",
    },
    "aigc-zh": {
        "repo": "Eslzzyl/aigc-detector-zh-onnx",
        "hf_prefix": "",
        "dest_subdir": "aigc-zh",
        "files": [
            ("onnx/model_quantized.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
            ("tokenizer_config.json", "tokenizer_config.json"),
        ],
        "approx_mb": 98,
        "note": "Chinese AIGC detector (BERT-base INT8). Used when CJK≥35%. Soft-fail if missing.",
    },
}


def _normalize_files(files: list) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in files:
        if isinstance(item, tuple):
            out.append((item[0], item[1]))
        else:
            out.append((item, item))
    return out


def download(model_key: str = "ettin-68m", force: bool = False) -> Path:
    if model_key not in MODELS:
        raise SystemExit(f"Unknown model {model_key!r}. Choose from: {', '.join(MODELS)}")
    cfg = MODELS[model_key]
    dest = OUT_ROOT / cfg["dest_subdir"]
    dest.mkdir(parents=True, exist_ok=True)

    onnx_path = dest / "model.onnx"
    if onnx_path.is_file() and not force:
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"Already present: {onnx_path} ({size_mb:.1f} MB). Use --force to re-download.")
        return dest

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Missing huggingface_hub. Install:\n"
            "  pip install huggingface_hub\n"
            "or:\n"
            "  pip install -r requirements-onnx.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)

    prefix = cfg.get("hf_prefix") or ""
    print(
        f"Downloading {cfg['repo']}"
        + (f"/{prefix}" if prefix else "")
        + f" (~{cfg['approx_mb']} MB)…\n"
        f"  → {dest}\n"
        f"  {cfg['note']}"
    )
    for remote_name, local_name in _normalize_files(cfg["files"]):
        rel = f"{prefix}/{remote_name}" if prefix else remote_name
        local = hf_hub_download(repo_id=cfg["repo"], filename=rel)
        target = dest / local_name
        shutil.copy2(local, target)
        print(f"  ok {local_name} ({target.stat().st_size} bytes)")

    print(f"Done. Model ready at: {dest}")
    print("Then: pip install onnxruntime tokenizers")
    print("Detector soft-fails without these files; heuristic still works.")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Download ONNX AI-text detector weights")
    ap.add_argument(
        "--model",
        default="ettin-68m",
        choices=list(MODELS),
        help="Which ONNX bundle (default: ettin-68m ~77MB EN; aigc-zh ~98MB ZH)",
    )
    ap.add_argument("--force", action="store_true", help="Re-download even if present")
    args = ap.parse_args()
    download(args.model, force=args.force)


if __name__ == "__main__":
    main()
