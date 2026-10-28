#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Ollama HTTP client for optional DNA-GPT-style rewrite consistency.

Ollama (as of logprobs release notes) exposes logprobs for *generated* tokens,
not full prompt-echo perplexity. So we use black-box regeneration:
  truncate text → generate continuation → n-gram overlap with original suffix.
High overlap → more machine-like (DNA-GPT idea; Yang et al. 2023/2024).

Never calls TipType. Graceful when Ollama is down.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 45
# Text chat models only — never auto-pick VL / vision / embed.
PREFERRED_MODELS = (
    "qwen2:7b",
    "qwen2.5:7b",
    "qwen2.5:3b",
    "qwen2.5:1.5b",
    "qwen2:latest",
    "qwen2",
    "llama3.2:3b",
    "llama3.2:1b",
    "phi3:mini",
    "gemma2:2b",
    "mistral:7b",
    "deepseek-r1:8b",
    "english_chat:latest",
)


def is_vision_or_embed_model(name: str) -> bool:
    """True for VL / vision / embedding models that must not be auto-selected."""
    n = (name or "").lower()
    if not n:
        return True
    # *-vl*, *vision*, *embed* (also qwen2.5vl without hyphen)
    if ("vl" in n) or ("vision" in n) or ("embed" in n):
        return True
    # Common multimodal chat tags that lack an explicit "vl" token
    for marker in (
        "llava",
        "bakllava",
        "minicpm-v",
        "moondream",
        "clip",
        "preprocessor",
    ):
        if marker in n:
            return True
    return False


@dataclass
class OllamaStatus:
    available: bool
    host: str
    models: List[str] = field(default_factory=list)
    error: str = ""
    selected_model: str = ""
    only_vl_available: bool = False


def _get(url: str, timeout: float = 3.0) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(url: str, payload: Dict[str, Any], timeout: float = DEFAULT_TIMEOUT) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def probe(host: str = DEFAULT_HOST, timeout: float = 2.5) -> OllamaStatus:
    host = host.rstrip("/")
    try:
        tags = _get(f"{host}/api/tags", timeout=timeout)
        models = []
        for m in tags.get("models") or []:
            name = m.get("name") or m.get("model") or ""
            if name:
                models.append(name)
        selected = pick_model(models)
        only_vl = bool(models) and not selected and all(
            is_vision_or_embed_model(m) for m in models
        )
        return OllamaStatus(
            True, host, models, "", selected, only_vl_available=only_vl
        )
    except Exception as e:  # noqa: BLE001 — soft fail for UX
        return OllamaStatus(False, host, [], f"{type(e).__name__}: {e}", "")


def pick_model(models: Sequence[str], preferred: Optional[str] = None) -> str:
    """Pick a text chat model. Never auto-selects VL / vision / embed models.

    If ``preferred`` is an exact installed name (even VL), that exact name wins.
    Otherwise only text models are considered. Returns "" when no text model exists.
    """
    if not models:
        return ""
    names = list(models)
    text_models = [n for n in names if not is_vision_or_embed_model(n)]

    if preferred:
        pref = preferred.strip()
        if pref:
            # Exact match (allow explicit VL if user typed it)
            for n in names:
                if n == pref or n.lower() == pref.lower():
                    return n
            # Substring / tag match — text models only
            pool = text_models or []
            for n in pool:
                if pref.lower() in n.lower() or n.lower().startswith(pref.split(":")[0].lower()):
                    return n

    pool = text_models
    if not pool:
        return ""

    lower_map = {n.lower(): n for n in pool}
    for pref in PREFERRED_MODELS:
        key = pref.lower()
        if key in lower_map:
            return lower_map[key]

    # Prefix match on model base (e.g. qwen2:7b for pref qwen2), text only.
    # Require base equality or base.startswith(pref_base) with care:
    # pref "qwen2.5" must not match unrelated families; VL already filtered out.
    for pref in PREFERRED_MODELS:
        pref_base = pref.split(":")[0].lower()
        exact_base = [n for n in pool if n.lower().split(":")[0] == pref_base]
        if exact_base:
            # Prefer larger / named tags stably: sort so :7b-ish comes before :latest if both
            return sorted(exact_base, key=lambda x: (0 if "7b" in x.lower() else 1, x.lower()))[0]
        prefix_hits = [
            n for n in pool
            if n.lower().split(":")[0].startswith(pref_base)
            and not is_vision_or_embed_model(n)
        ]
        if prefix_hits:
            return sorted(prefix_hits, key=lambda x: (0 if "7b" in x.lower() else 1, x.lower()))[0]

    return pool[0]


def _tokenize_ngrams(text: str) -> List[str]:
    # EN words + ZH chars as tokens
    toks = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]", text.lower())
    return toks


def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    ta, tb = _tokenize_ngrams(a), _tokenize_ngrams(b)
    if len(ta) < n or len(tb) < n:
        # fall back to unigram Jaccard
        sa, sb = set(ta), set(tb)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)
    ga = {tuple(ta[i : i + n]) for i in range(len(ta) - n + 1)}
    gb = {tuple(tb[i : i + n]) for i in range(len(tb) - n + 1)}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def split_prefix_suffix(text: str, prefix_ratio: float = 0.55) -> Tuple[str, str]:
    text = text.strip()
    if len(text) < 80:
        mid = max(1, len(text) // 2)
        return text[:mid], text[mid:]
    # Prefer sentence boundary near ratio
    target = int(len(text) * prefix_ratio)
    # search for punctuation near target
    window = text[max(0, target - 40) : min(len(text), target + 40)]
    best = target
    for m in re.finditer(r"[.!?。！？]\s*", text):
        pos = m.end()
        if abs(pos - target) < abs(best - target) and 0.35 * len(text) < pos < 0.75 * len(text):
            best = pos
    return text[:best].strip(), text[best:].strip()


def generate_continuation(
    prefix: str,
    model: str,
    host: str = DEFAULT_HOST,
    num_predict: int = 180,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Continue writing after prefix; temperature 0 for stability."""
    prompt = (
        "Continue the following text naturally. Output ONLY the continuation, "
        "no quotes, no preamble.\n\n" + prefix
    )
    # Chinese-friendly alternate if mostly CJK
    cjk = len(re.findall(r"[\u4e00-\u9fff]", prefix))
    if cjk >= max(8, len(prefix) // 4):
        prompt = "请自然续写下文，只输出续写内容，不要解释、不要加引号。\n\n" + prefix

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": num_predict,
        },
    }
    resp = _post(f"{host.rstrip('/')}/api/generate", payload, timeout=timeout)
    return (resp.get("response") or "").strip()


def dna_consistency_score(
    text: str,
    model: str,
    host: str = DEFAULT_HOST,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Returns dict with:
      overlap (0..1), ai_like (0..1 mapped), model, prefix_len, suffix_len, error?
    Higher ai_like = more DNA-GPT machine-like (continuation matches original suffix).
    """
    prefix, suffix = split_prefix_suffix(text)
    meta: Dict[str, Any] = {
        "model": model,
        "prefix_len": len(prefix),
        "suffix_len": len(suffix),
        "overlap": 0.0,
        "ai_like": 0.0,
        "continuation_preview": "",
        "error": "",
    }
    if len(suffix) < 20 or len(prefix) < 20:
        meta["error"] = "text_too_short_for_dna"
        meta["ai_like"] = 0.35
        return meta
    try:
        cont = generate_continuation(prefix, model=model, host=host, timeout=timeout)
        meta["continuation_preview"] = cont[:240]
        # Compare continuation to original suffix (truncate to similar length)
        cut = min(len(suffix), max(len(cont), 40))
        ov3 = ngram_overlap(cont[: cut + 40], suffix[: cut + 40], n=3)
        ov2 = ngram_overlap(cont[: cut + 40], suffix[: cut + 40], n=2)
        overlap = 0.6 * ov3 + 0.4 * ov2
        meta["overlap"] = round(overlap, 4)
        # Map: overlap 0.05 → ~0.2 AI; 0.25 → ~0.7; 0.4+ → ~0.9
        ai_like = max(0.0, min(1.0, (overlap - 0.04) / 0.32))
        meta["ai_like"] = round(ai_like, 4)
    except Exception as e:  # noqa: BLE001
        meta["error"] = f"{type(e).__name__}: {e}"
        meta["ai_like"] = 0.0
    return meta
