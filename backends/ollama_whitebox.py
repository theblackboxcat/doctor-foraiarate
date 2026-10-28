#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optional white-box signals via Ollama log-probabilities (GLTR / LRR flavoured).

Why this exists
---------------
The heuristic layer only sees surface features. The strongest public open-source
detector family instead asks a language model how predictable the text is:

* GLTR (Gehrmann et al. 2019) — the *rank* of each real token in the model's
  candidate list; machine text crowds into rank 1.
* LRR (Su et al. 2023) — log-rank plus log-probability together.
* DetectGPT / Fast-DetectGPT — curvature / conditional-probability deviation.

Ollama exposes ``logprobs`` + ``top_logprobs`` (0..20) on **/api/generate**, but
only for tokens it *generates*, not for a scored prompt. So we use the echo
trick: ask the model to reproduce the passage verbatim at temperature 0 and read
the log-probs of the tokens it emits. If the reproduction is faithful, those
log-probs are a good stand-in for "how predictable is this passage".

Faithfulness is measured, not assumed: the echoed tokens are re-joined and
compared to the source. If coverage < 0.7 the result is flagged
``unfaithful`` and must not be trusted.

NOT CALIBRATED, NOT VERIFIED IN THIS SANDBOX
--------------------------------------------
This module was written against the documented Ollama API; it has not been run
against a live Ollama server here. Treat its numbers as an uncalibrated extra
vote until you have checked them on your own texts. It is off by default.
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 120
TOP_LOGPROBS = 10
FAITHFUL_MIN = 0.70


def _post(url: str, payload: Dict[str, Any], timeout: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _echo_prompt(text: str) -> str:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text)) / max(1, len(text))
    if cjk >= 0.25:
        return (
            "下面是一段需要被逐字复制的文本。请原样输出这段文字，"
            "不要翻译、不要总结、不要添加或删改任何内容，也不要加引号或说明。\n\n"
            + text
        )
    return (
        "Copy the following passage verbatim, character for character. "
        "Output ONLY the passage — no quotes, no commentary, no translation, "
        "no summary.\n\n" + text
    )


def _estimate_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    rest = max(0, len(text) - cjk)
    return int(cjk * 1.3 + rest / 3.6) + 40


def _normalise(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def echo_logprobs(
    text: str,
    model: str,
    host: str = DEFAULT_HOST,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Ask Ollama to echo ``text`` and return per-token logprob records."""
    payload = {
        "model": model,
        "prompt": _echo_prompt(text),
        "stream": False,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": min(2048, _estimate_tokens(text)),
        },
    }
    resp = _post(f"{host.rstrip('/')}/api/generate", payload, timeout)
    records: List[Dict[str, Any]] = resp.get("logprobs") or []
    echoed = resp.get("response") or ""
    out: Dict[str, Any] = {
        "ok": bool(records),
        "model": model,
        "records": records,
        "echoed": echoed,
        "error": "" if records else "no logprobs returned (older Ollama?)",
    }
    return out


def _stats_from_records(records: List[Dict[str, Any]]) -> Dict[str, float]:
    logprobs: List[float] = []
    ranks: List[int] = []
    entropies: List[float] = []
    top1 = 0
    for rec in records:
        lp = rec.get("logprob")
        if isinstance(lp, (int, float)):
            logprobs.append(float(lp))
        tops = rec.get("top_logprobs") or []
        if tops:
            ordered = sorted(tops, key=lambda x: -float(x.get("logprob", -99.0)))
            # rank of the actually emitted token inside the returned candidate list
            tok = rec.get("token", "")
            rank = next(
                (i + 1 for i, t in enumerate(ordered) if t.get("token") == tok),
                len(ordered) + 1,
            )
            ranks.append(rank)
            if rank == 1:
                top1 += 1
            # entropy over the returned candidates (renormalised); the long tail
            # is missing so this is a lower bound
            lps = [float(t.get("logprob", -20.0)) for t in ordered]
            m = max(lps)
            exps = [math.exp(x - m) for x in lps]
            z = sum(exps) or 1.0
            ps = [e / z for e in exps]
            entropies.append(-sum(p * math.log(p + 1e-12) for p in ps))
    n = max(1, len(records))
    return {
        "n_tokens": len(records),
        "mean_logprob": (sum(logprobs) / len(logprobs)) if logprobs else 0.0,
        "mean_rank": (sum(ranks) / len(ranks)) if ranks else 0.0,
        "mean_entropy": (sum(entropies) / len(entropies)) if entropies else 0.0,
        "top1_frac": top1 / n,
    }


def whitebox_signals(
    text: str,
    host: str = DEFAULT_HOST,
    model: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Return dict with token-level predictability stats + a rough ``ai_like``.

    ``ai_like`` is an UNCALIBRATED heuristic blend (higher = more predictable =
    more machine-like). It is deliberately reported alongside its inputs rather
    than folded silently into the Human Score.
    """
    out: Dict[str, Any] = {"used": False, "host": host, "model": model}
    text = (text or "").strip()
    if not text:
        out["error"] = "empty text"
        return out
    if not model:
        try:
            from backends.ollama_client import pick_model, probe
            st = probe(host)
            if not st.available:
                out["error"] = f"ollama_unavailable:{st.error}"
                return out
            model = st.selected_model or pick_model(st.models)
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
            return out
        if not model:
            out["error"] = "no_text_model"
            return out
    out["model"] = model

    try:
        echo = echo_logprobs(text, model=model, host=host, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    if not echo.get("ok"):
        out["error"] = echo.get("error") or "echo failed"
        return out

    stats = _stats_from_records(echo["records"] or [])
    src = _normalise(text)
    got = _normalise(echo.get("echoed", ""))
    # Faithful-prefix coverage: how much of the source the echo reproduced.
    coverage = 0.0
    if src:
        i = 0
        for ch in got:
            if i < len(src) and ch == src[i]:
                i += 1
        coverage = i / len(src)
    faithful = coverage >= FAITHFUL_MIN

    out.update(stats)
    out["echo_coverage"] = round(coverage, 4)
    out["faithful"] = bool(faithful)
    out["used"] = bool(faithful)

    # Rough, uncalibrated mapping — reported, never silently blended.
    mlp = stats["mean_logprob"]
    t1 = stats["top1_frac"]
    ent = stats["mean_entropy"]
    s = 0.0
    s += 0.45 * max(0.0, min(1.0, (mlp + 3.0) / 3.0))      # -3 nats ≈ very predictable
    s += 0.35 * max(0.0, min(1.0, (t1 - 0.45) / 0.45))
    s += 0.20 * max(0.0, min(1.0, (2.6 - ent) / 2.6))
    out["ai_like"] = round(max(0.0, min(1.0, s)), 4)
    out["note"] = (
        "GLTR/LRR-flavoured echo scoring; uncalibrated, off by default, "
        "and only meaningful when echo_coverage is high."
    )
    if not faithful:
        out["note"] = f"echo unfaithful (coverage={coverage:.2f}) — stats unreliable"
    return out


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "The cat sat on the mat and looked at the rain."
    print(json.dumps(whitebox_signals(t), ensure_ascii=False, indent=2))
