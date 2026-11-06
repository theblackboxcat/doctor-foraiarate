# -*- coding: utf-8 -*-
"""Honesty / reverse-consensus helpers (aidetector.com as negative example).

Do not change scoring. Only surface voter split + process-evidence hints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Display-only. Same 0.3 Gradio (app.py) and web/app.js already use for the
# "意见不一致" note. Not a scoring threshold; do not retune it here.
EFFECTIVE_SPREAD_DISAGREE = 0.3

# Pair note treats these hint levels as "look at process evidence first".
_PROCESS_FIRST_LEVELS = frozenset({"disagreement", "single_voter"})


def _f(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def build_voter_split(meta: Dict[str, Any] | None, ai_probability: float | None = None) -> List[Dict[str, Any]]:
    """Per-detector (and observer) AI probabilities for UI/API.

    Counted voters participate in meta.voter_disagreement / blend.
    Observers ran but were not mixed into the final score — still shown.
    """
    meta = meta or {}
    out: List[Dict[str, Any]] = []

    onnx = meta.get("onnx") if isinstance(meta.get("onnx"), dict) else {}
    ollama = meta.get("ollama") if isinstance(meta.get("ollama"), dict) else {}

    # Heuristic posterior: prefer pre-ONNX if present, else final ai_probability.
    heur = _f(onnx.get("ai_prob_before"))
    if heur is None:
        heur = _f(ai_probability)
    if heur is None:
        # last resort: invert human_before if present
        hb = _f(onnx.get("heuristic_human_before_onnx"))
        if hb is not None:
            heur = max(0.0, min(1.0, 1.0 - hb / 100.0))
    if heur is not None:
        out.append(
            {
                "id": "heuristic",
                "label_zh": "启发式",
                "ai_prob": round(heur, 4),
                "role": "counted",
                "note_zh": "主路径后验",
            }
        )

    ox_prob = _f(onnx.get("ai_prob_onnx"))
    if ox_prob is None:
        score = onnx.get("score") if isinstance(onnx.get("score"), dict) else {}
        ox_prob = _f(score.get("ai_prob")) if score.get("ok") else None
    if ox_prob is not None:
        counted = bool(onnx.get("used"))
        status = str(onnx.get("status") or "")
        out.append(
            {
                "id": "onnx",
                "label_zh": "ONNX",
                "ai_prob": round(ox_prob, 4),
                "role": "counted" if counted else "observer",
                "note_zh": ("已混入终分" if counted else f"旁证未混入（{status or 'n/a'}）"),
            }
        )

    dna = ollama.get("dna") if isinstance(ollama.get("dna"), dict) else {}
    dna_prob = _f(dna.get("ai_like"))
    if dna_prob is None:
        dna_prob = _f(ollama.get("ai_like"))
    if dna_prob is not None and (ollama.get("used") or dna):
        counted = bool(ollama.get("used"))
        out.append(
            {
                "id": "ollama_dna",
                "label_zh": "DNA/Ollama",
                "ai_prob": round(dna_prob, 4),
                "role": "counted" if counted else "observer",
                "note_zh": "已混入终分" if counted else f"旁证未混入（{ollama.get('status') or 'n/a'}）",
            }
        )

    return out


def observer_spread(split: List[Dict[str, Any]]) -> float:
    probs = [_f(v.get("ai_prob")) for v in split]
    probs = [p for p in probs if p is not None]
    if len(probs) < 2:
        return 0.0
    return round(max(probs) - min(probs), 4)


def build_process_evidence_hint(
    meta: Dict[str, Any] | None,
    split: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Non-authoritative guidance. Never a verdict."""
    meta = meta or {}
    split = split if split is not None else build_voter_split(meta)
    counted_dis = _f(meta.get("voter_disagreement")) or 0.0
    obs_dis = observer_spread(split)
    effective = max(counted_dis, obs_dis)

    roles = {v.get("role") for v in split}
    has_observer = "observer" in roles
    n = len(split)

    if n < 2:
        level = "single_voter"
        zh = (
            "当前只有一路有效票。单检不是判决；高风险场景（成绩/处分/雇佣）请优先看版本历史、"
            "口头核对与作业设计等过程证据，本分数只做改稿线索。"
        )
    elif effective > EFFECTIVE_SPREAD_DISAGREE:
        level = "disagreement"
        zh = (
            f"分票极差约 {effective:.2f}（含未混入终分的旁证票时一并计入展示）。"
            "分歧是信号：请提高过程证据权重；混合分 / Human Score 不是作者标签。"
            "不要把「多检一致」的外部营销（含 ~0% 假阳）搬到本仓库。"
        )
        if has_observer and counted_dis < obs_dis:
            zh += " 注意：有旁证票未混入终分，界面仍应看见分歧。"
        zh += (
            "高分歧场景请先看版本历史/口头核对/作业设计等过程证据，再看 Human Score；分数不是作者标签。"
        )
    else:
        level = "agreement_not_verdict"
        zh = (
            f"分票接近（极差约 {effective:.2f}）。票面接近仍然不是作者证明；"
            "高风险场景仍以过程证据优先，本仓库分数只回答成品表面是否像公共风格签名。"
        )

    if level == "agreement_not_verdict":
        recommend_process_first = False
        priority = "style_signal_only"
    else:
        # disagreement and single_voter: process evidence before the score.
        recommend_process_first = True
        priority = "process_evidence"

    return {
        "level": level,
        "zh": zh,
        "recommend_process_first": recommend_process_first,
        "priority": priority,
        "counted_disagreement": round(counted_dis, 4),
        "observer_spread": obs_dis,
        "effective_spread": effective,
        "anti_consensus_marketing": True,
    }


def build_process_note(
    before_hint: Dict[str, Any] | None,
    after_hint: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """One line for a before/after pair. Does not rescore.

    ``any_disagreement`` is true when either side's ``level`` is
    ``disagreement`` or ``single_voter``. ``recommend_process_first`` is the
    OR of the two sides' flags. The ``zh`` line says whether this pair should
    elevate process evidence. It is not an authorship verdict.
    """
    before_hint = before_hint if isinstance(before_hint, dict) else {}
    after_hint = after_hint if isinstance(after_hint, dict) else {}
    levels = (str(before_hint.get("level") or ""), str(after_hint.get("level") or ""))
    any_disagreement = any(level in _PROCESS_FIRST_LEVELS for level in levels)
    recommend = bool(before_hint.get("recommend_process_first")) or bool(
        after_hint.get("recommend_process_first")
    )
    if recommend or any_disagreement:
        zh = (
            "前后至少一侧是高分歧或单票：请先看版本历史、口头核对、作业设计等过程证据，"
            "再看 Human Score；分数不是作者标签。"
        )
    else:
        zh = (
            "前后分票接近，不必因分歧抬高过程证据；票面接近仍然不是作者证明，分数不是作者标签。"
        )
    return {
        "any_disagreement": any_disagreement,
        "recommend_process_first": recommend,
        "zh": zh,
    }


def attach_honesty_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    split = build_voter_split(meta, data.get("ai_probability"))
    hint = build_process_evidence_hint(meta, split)
    data["voter_split"] = split
    data["process_evidence_hint"] = hint
    return data
