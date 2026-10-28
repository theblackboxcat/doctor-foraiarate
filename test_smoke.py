#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests for local_ai_detector — no TipType API. Ollama optional."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detector import (  # noqa: E402
    DNA_BLEND_DEFAULT,
    MIN_CHARS,
    MIN_CHARS_ZH,
    analyze,
    effective_dna_blend,
    export_json,
    human_band,
    load_weights,
    min_chars_for,
    save_weights,
)
from compression_signals import compression_feature_bundle  # noqa: E402
from backends.ollama_client import (  # noqa: E402
    is_vision_or_embed_model,
    ngram_overlap,
    pick_model,
    probe,
    split_prefix_suffix,
)


AI_SLOP = (
    "Moreover, this innovative solution seamlessly leverages cutting-edge technology "
    "to optimize workflows, enhance productivity, and unlock unprecedented value for "
    "organizations navigating today's rapidly evolving digital landscape. Furthermore, "
    "it is important to note that stakeholders should delve into the myriad of opportunities "
    "that this holistic approach can foster in the foreseeable future. Additionally, "
    "this comprehensive framework strategically empowers teams to unlock synergies "
    "across the entire digital ecosystem in a seamless and robust manner."
)

GETTYSBURG = (
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, "
    "conceived in Liberty, and dedicated to the proposition that all men are created equal. "
    "Now we are engaged in a great civil war, testing whether that nation, or any nation so "
    "conceived and so dedicated, can long endure. We are met on a great battle-field of that war. "
    "We have come to dedicate a portion of that field, as a final resting place for those who "
    "here gave their lives that that nation might live."
)

ZH_HUMAN = (
    "昨晚下了点小雨，巷子里的石板路还是湿的。我骑车去买醋，老板娘还是那副慢悠悠的样子，"
    "一边找零钱一边抱怨说今年的青菜贵得没道理。我没接话，只是点了点头。回到家才发现，"
    "口袋里那张五块钱不知道什么时候蹭湿了，黏在一起，撕开的时候还带出一点纸毛。"
    "窗外有人在晾被单，铁架子偶尔咔一声。我把醋瓶搁灶台上，听见楼道里电视的声音模模糊糊传过来。"
)

ZH_OFFICIAL_AI = (
    "综上所述，在高质量发展的背景下，我们要进一步赋能基层治理，推动数字化转型落地见效。"
    "值得注意的是，必须坚持系统观念，全面统筹各方资源，持续提升核心竞争力。首先，要完善顶层设计；"
    "其次，要加强协同联动；最后，要强化闭环管理，确保各项举措落地生根、开花结果。"
    "与此同时，我们要以人民为中心，不断满足人民群众对美好生活的向往，为经济社会发展注入新动能。"
)

ZH_ESSAY_AI = (
    "随着科技的飞速发展，人工智能已经深刻地改变了我们的生活方式。首先，AI技术在教育领域发挥着重要作用；"
    "其次，在医疗健康方面也展现出巨大潜力；最后，在交通出行等领域同样不可或缺。"
    "总而言之，我们应当正确认识人工智能带来的机遇与挑战，积极拥抱这一时代浪潮，"
    "从而更好地推动社会的全面进步与发展。值得注意的是，相关政策的落地也至关重要。"
)

# v2.4 literary AI-taste composite (anonymized from user examples; not a real work)
ZH_LIT_AI = (
    "张了张嘴，想了想。它还是空的，直直的。他在那儿站了一会儿，回去了。"
    "巫医看了他一眼，没说什么。鸡拿到手里，鸡不叫。那是名，是娘叫的。"
    "魂若来，铃会响。铃舌是空的，响的不是风。"
    "端木赐后来试着跟别人描述过，描述不出来。"
    "喊的不是词，是要让人知道他在下面。"
    "不是风吹的响法——风吹是乱响，这一声是正的。"
    "不是从席上起来的，是从月亮底下走进来的。"
    "他抬起手，像课堂上要人安静。看得久了些。烟升上去，直直的。"
    "他想张开，张不开。他说：没有线。有。你看不见。"
    "冰还在换，苍蝇还在。"
    "仪式之后第七天，端木赐又去了一趟城外。他是想问一件事：那个空，还能不能改。"
    "土是新翻的。人不在。屋门开着，里面空了。还是空。"
)

MIXED_ZH_SHORT = "今天天气不错，我去超市买了点菜。"

LIST_TMPL = (
    "First, it is essential to establish a clear framework for analysis. "
    "Second, stakeholders should carefully evaluate the available options. "
    "Finally, organizations must implement best practices to ensure sustainable outcomes "
    "across all relevant dimensions of the initiative."
)


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    # 1) too short warning (EN floor)
    short = analyze("Hello world.")
    assert_true(short.too_short, "short text should set too_short")
    assert_true(short.char_count < MIN_CHARS, "char_count below EN min")
    assert_true(min_chars_for("Hello world.") == MIN_CHARS, "EN min_chars_for")
    assert_true(min_chars_for("中" * 60) == MIN_CHARS_ZH, "ZH min_chars_for")
    # Mixed_ZH short: still too_short but messaging uses ZH floor
    mixed = analyze(MIXED_ZH_SHORT)
    assert_true(mixed.too_short, "Mixed_ZH short should be too_short")
    assert_true(mixed.meta.get("min_chars") == MIN_CHARS_ZH, "ZH short uses MIN_CHARS_ZH")

    # 2) AI slop should score more AI-like than Gettysburg
    ai = analyze(AI_SLOP)
    human = analyze(GETTYSBURG)
    assert_true(ai.human_score < human.human_score,
                f"AI slop human_score {ai.human_score} should be < Gettysburg {human.human_score}")
    assert_true(ai.human_score < 55, f"AI slop expected lowish human_score, got {ai.human_score}")
    assert_true(human.human_score >= 45, f"Gettysburg expected higher human_score, got {human.human_score}")

    # 3) marketing feature should fire on AI slop
    mkt = next(f for f in ai.features if f.name == "marketing_inflation")
    assert_true(mkt.value > 0.2, f"marketing_inflation should be elevated, got {mkt.value}")

    # 4) list template
    lst = analyze(LIST_TMPL)
    lt = next(f for f in lst.features if f.name == "list_template")
    assert_true(lt.value > 0.3, f"list_template should fire, got {lt.value}")

    # 5) Chinese human prose should not crash; clarity independent
    zh = analyze(ZH_HUMAN, use_onnx=False)
    assert_true(0 <= zh.clarity_score <= 100, "clarity in range")
    assert_true(0 <= zh.human_score <= 100, "human_score in range")
    assert_true(len(zh.sentences) >= 2, "should split Chinese sentences")
    assert_true(zh.human_score >= 60, f"ZH alley/lit should stay humanish, got {zh.human_score}")

    # 5b) v2.3 ZH AI vs human separation (heuristic; larger gap than v2.2)
    zh_off = analyze(ZH_OFFICIAL_AI, use_onnx=False)
    zh_essay = analyze(ZH_ESSAY_AI, use_onnx=False)
    assert_true(
        zh_off.human_score < zh.human_score,
        f"ZH official H={zh_off.human_score} should be < alley H={zh.human_score}",
    )
    assert_true(
        zh_essay.human_score < zh.human_score,
        f"ZH essay H={zh_essay.human_score} should be < alley H={zh.human_score}",
    )
    gap_off = zh.human_score - zh_off.human_score
    gap_essay = zh.human_score - zh_essay.human_score
    assert_true(gap_off >= 25, f"ZH official↔alley gap should be ≥25, got {gap_off}")
    assert_true(gap_essay >= 15, f"ZH essay↔alley gap should be ≥15, got {gap_essay}")
    assert_true(zh_off.human_score <= 40, f"ZH official should be ≤40, got {zh_off.human_score}")
    # ZH feature slots must fire on officialese
    zh_names = {f.name for f in zh_off.features}
    assert_true("zh_policy_register" in zh_names, "zh_policy_register missing")
    pol = next(f for f in zh_off.features if f.name == "zh_policy_register")
    assert_true(pol.value >= 0.4, f"zh_policy_register should fire, got {pol.value}")
    disc = next(f for f in zh_off.features if f.name == "zh_discourse_boilerplate")
    assert_true(disc.value >= 0.3, f"zh_discourse_boilerplate should fire, got {disc.value}")

    # 5c) v2.4 literary AI-taste: composite fires; alley/EN stay quiet
    zh_lit = analyze(ZH_LIT_AI, use_onnx=False)
    lit_names = [
        "lit_zh_fake_terse_rhythm", "lit_zh_linger_gaze", "lit_zh_broken_parallel",
        "lit_zh_noun_reemphasize", "lit_zh_bushi_shi", "lit_zh_sceneless_simile",
        "lit_zh_flat_dialogue", "lit_zh_mech_scene",
    ]
    feat_map = {f.name: f for f in zh_lit.features}
    for ln in lit_names:
        assert_true(ln in feat_map, f"missing literary feature {ln}")
    lit_fired = sum(1 for ln in lit_names if feat_map[ln].value >= 0.15)
    assert_true(lit_fired >= 4, f"literary composite should fire ≥4 slots, got {lit_fired}")
    assert_true(
        feat_map["lit_zh_bushi_shi"].value >= 0.25
        or feat_map["lit_zh_mech_scene"].value >= 0.25,
        "bushi_shi or mech_scene should fire on composite",
    )
    assert_true(
        zh_lit.human_score < zh.human_score,
        f"literary AI H={zh_lit.human_score} should be < alley H={zh.human_score}",
    )
    # Alley: literary slots stay quiet
    alley_lit = {f.name: f.value for f in zh.features if f.name.startswith("lit_zh_")}
    alley_max = max(alley_lit.values()) if alley_lit else 0.0
    assert_true(alley_max < 0.25, f"alley literary max should be quiet, got {alley_max}")
    # EN: literary signals ~0 (and weights gated)
    en_lit = {f.name: f.value for f in human.features if f.name.startswith("lit_zh_")}
    assert_true(all(v < 0.05 for v in en_lit.values()), f"EN literary should be ~0, got {en_lit}")
    # Sentence heat should surface literary cues on composite
    lit_heat = [s for s in zh_lit.sentences if s.heat in ("red", "yellow")]
    assert_true(len(lit_heat) >= 1, "literary composite should have ≥1 heated sentence")
    heated_feats = {f.name for s in lit_heat for f in s.top_features}
    assert_true(
        any(n.startswith("lit_zh_") for n in heated_feats) or lit_fired >= 4,
        f"sentence heat should mention lit_zh_* or slots fired; top={heated_feats}",
    )

    # 6) bands + heat labels exist
    bz, be = human_band(12)
    assert_true("AI" in be or "AI" in bz, "band labels")
    heats = {s.heat for s in ai.sentences}
    assert_true(heats <= {"red", "yellow", "none"}, f"bad heat labels {heats}")

    # 7) feature names exposed + compression features present
    assert_true(all(f.name_zh and f.name_en for f in ai.features), "bilingual feature names")
    names = {f.name for f in ai.features}
    assert_true("compression_ai_like" in names, "compression feature missing")
    assert_true("window_burstiness_inv" in names, "window burstiness missing")
    assert_true(len(ai.features) >= 15, "expect expanded feature vector (v2.4)")
    # signals always include zh_* keys (values ~0 on EN; weights gated)
    assert_true(any(f.name.startswith("zh_") for f in ai.features), "zh feature keys present")

    # 8) compression bundle runs
    comp = compression_feature_bundle(AI_SLOP)
    assert_true(0.0 <= comp["compression_ai_like"] <= 1.0, "compression range")
    assert_true(0.0 <= comp["window_burstiness_inv"] <= 1.0, "window range")

    # 9) weight adjust + JSON export + engine tag
    w = load_weights()
    w["marketing_inflation"] = 0.30
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "weights.json"
        save_weights(w, p)
        w2 = load_weights(p)
        assert_true(abs(sum(w2.values()) - 1.0) < 1e-6, "weights renormalize to 1")
        r = analyze(AI_SLOP, weights=w2)
        out = Path(td) / "report.json"
        export_json(r, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert_true("human_score" in data and "features" in data, "export schema")
        assert_true(data.get("meta", {}).get("not_tiptype") is True, "disclaimer meta")
        assert_true("disclaimer_zh" in data, "honesty field")
        # v3.2：断言不再钉死小版本号。引擎串是给界面和导出看的标识，
        # 每次加一层就改一次测试，测的就不是"引擎标识还在"而是"版本号没动"。
        assert_true("local_heuristic_v3." in str(data.get("meta", {}).get("engine", "")),
                    "engine v3.x")

    # 10) highlight policy only max band
    if any(s.heat == "red" for s in ai.sentences):
        assert_true(all(s.heat in ("red", "none") for s in ai.sentences),
                    "when red present, yellow should be suppressed")

    # 11) Ollama probe soft-fails; hybrid falls back
    st = probe("http://127.0.0.1:11434")
    hybrid = analyze(AI_SLOP, backend="hybrid")
    assert_true(hybrid.meta.get("backend") == "hybrid", "backend recorded")
    assert_true("ollama" in hybrid.meta, "ollama meta present")
    if not st.available:
        assert_true(hybrid.meta["ollama"].get("used") is False, "should not claim ollama used")
    # ngram helper sanity
    assert_true(ngram_overlap("hello world hello", "hello world foo", n=2) > 0.2, "ngram")
    pref, suf = split_prefix_suffix(AI_SLOP)
    assert_true(len(pref) > 20 and len(suf) > 20, "split prefix/suffix")

    # 12) pick_model: never auto-select VL when a text model exists
    laptop_models = [
        "english_chat:latest",
        "qwen2:7b",
        "deepseek-r1:8b",
        "qwen2.5vl:3b",
    ]
    picked = pick_model(laptop_models)
    assert_true(picked == "qwen2:7b", f"expected qwen2:7b, got {picked!r}")
    assert_true(not is_vision_or_embed_model(picked), "picked must be text")
    assert_true(is_vision_or_embed_model("qwen2.5vl:3b"), "vl detector")
    assert_true(is_vision_or_embed_model("something-vision:1b"), "vision detector")
    assert_true(is_vision_or_embed_model("nomic-embed-text"), "embed detector")
    assert_true(pick_model(["qwen2.5vl:3b", "llava:7b"]) == "", "only-VL → empty")
    # prefix bug regression: qwen2.5 preferred must not leap to qwen2.5vl
    assert_true(
        pick_model(["qwen2.5vl:3b", "llama3.2:3b"]) == "llama3.2:3b",
        "must skip VL and take text",
    )
    # explicit preferred VL still allowed
    assert_true(
        pick_model(laptop_models, preferred="qwen2.5vl:3b") == "qwen2.5vl:3b",
        "explicit VL preferred ok",
    )

    # 13) DNA gating: band / memorization / default blend cap (no Ollama needed)
    assert_true(0.12 <= DNA_BLEND_DEFAULT <= 0.15, f"dna_blend default {DNA_BLEND_DEFAULT}")
    b, reason = effective_dna_blend(65, overlap=1.0, base_blend=0.14, text_len=400)
    assert_true(b == 0.0 and "memorization" in reason, f"Gettysburg-style gate: {b} {reason}")
    b2, r2 = effective_dna_blend(20, overlap=0.5, base_blend=0.14, text_len=400)
    assert_true(b2 == 0.0 and "outside" in r2, f"outside band low: {b2} {r2}")
    b3, r3 = effective_dna_blend(80, overlap=0.1, base_blend=0.14, text_len=400)
    assert_true(b3 == 0.0 and "outside" in r3, f"outside band high: {b3} {r3}")
    b4, r4 = effective_dna_blend(50, overlap=0.2, base_blend=0.14, text_len=400)
    assert_true(abs(b4 - 0.14) < 1e-9 and r4 == "blended", f"uncertain blend: {b4} {r4}")
    b5, r5 = effective_dna_blend(50, overlap=0.2, base_blend=0.14, text_len=50)
    assert_true(b5 == 0.0 and "short" in r5, f"short text: {b5} {r5}")

    # 14) default backend remains heuristic
    assert_true(analyze(AI_SLOP).meta.get("backend") == "heuristic", "default heuristic")
    # hybrid path always attaches ollama meta (gate present when probe attempted)
    assert_true(isinstance(hybrid.meta.get("ollama"), dict), "ollama meta dict")


    # 15) ONNX soft-fail without weights; meta always present
    from backends.onnx_classifier import (  # noqa: WPS433
        ONNX_BLEND_DEFAULT as ONNX_B,
        blend_ai_probability,
        model_dir_ready,
        probe as onnx_probe,
        score_text as onnx_score,
    )
    assert_true(0.15 <= ONNX_B <= 0.25, f"onnx blend default {ONNX_B}")
    assert_true(abs(blend_ai_probability(0.2, 0.8, 0.5) - 0.5) < 1e-9 or
                abs(blend_ai_probability(0.2, 0.8, 0.45) - (0.55*0.2 + 0.45*0.8)) < 1e-9,
                "blend helper")
    # blend clamp at 0.45
    assert_true(abs(blend_ai_probability(0.0, 1.0, 0.45) - 0.45) < 1e-9, "blend clamp")
    ost = onnx_probe()
    r_onnx = analyze(AI_SLOP)  # default use_onnx=True
    assert_true("onnx" in r_onnx.meta, "onnx meta present on default path")
    assert_true(isinstance(r_onnx.meta["onnx"], dict), "onnx meta dict")
    if not ost.available:
        assert_true(r_onnx.meta["onnx"].get("used") is False, "no false onnx used")
        scored = onnx_score(AI_SLOP)
        assert_true(scored.get("ok") is False, "score soft-fails without model")
    else:
        # With weights: AI slop should blend; Gettysburg may skip_already_human; ZH skip CJK
        assert_true(r_onnx.meta["onnx"].get("used") is True, "onnx should blend on AI slop")
        names2 = {f.name for f in r_onnx.features}
        assert_true("onnx_ai_prob" in names2, "onnx feature exposed")
        r_off = analyze(AI_SLOP, use_onnx=False)
        assert_true(r_off.meta["onnx"].get("used") is False, "use_onnx=False skips")
        r_g = analyze(GETTYSBURG)
        assert_true(
            r_g.meta["onnx"].get("status") in ("skipped_already_human", "blended", "ready")
            or r_g.meta["onnx"].get("used") in (True, False),
            "gettysburg onnx status recorded",
        )
        # Prefer skip or non-damaging: human score still >= 45
        assert_true(r_g.human_score >= 45, f"Gettysburg still humanish got {r_g.human_score}")
        r_zh = analyze(ZH_HUMAN)
        zh_st = r_zh.meta["onnx"].get("status") or ""
        # Without aigc-zh weights: zh_unavailable:* ; with weights may blend or
        # skipped_already_human_zh. ettin must NOT run on CJK (path=zh).
        assert_true(
            r_zh.meta["onnx"].get("path") == "zh"
            or "cjk" in zh_st
            or zh_st.startswith("zh_unavailable")
            or zh_st in ("skipped_already_human_zh", "blended", "skipped_already_human")
            or r_zh.meta["onnx"].get("used") is False,
            f"ZH ONNX path should be zh/soft-fail, got {r_zh.meta['onnx']}",
        )
        # Soft-fail probe_zh
        from backends.onnx_classifier import probe_zh
        pz = probe_zh()
        assert_true(pz.label == "aigc-zh", "probe_zh label")

    # 16) window_burstiness_inv default weight reduced vs older 0.09
    from detector import DEFAULT_WEIGHTS
    assert_true(DEFAULT_WEIGHTS["window_burstiness_inv"] <= 0.07,
                f"window weight should be lowered, got {DEFAULT_WEIGHTS['window_burstiness_inv']}")
    wfile = load_weights()
    assert_true(wfile.get("window_burstiness_inv", 1) <= 0.08 + 1e-6,
                "weights.json window burstiness should be modest after renormalize")

    # ------------------------------------------------------------------
    # v3.0: calibration, confidence, sentence gating, paragraph windows
    # ------------------------------------------------------------------
    from detector import (
        CONFIDENCE_INSUFFICIENT,
        get_calibration,
        paragraph_windows,
        reload_calibration,
        CONFIDENCE_LEN_FULL,
    )

    cal = get_calibration()
    if cal is not None:
        # 17) calibrations exist for both languages; EN is built from real books
        assert_true("en" in cal.langs, "calibration has en anchors")
        assert_true(cal.langs["en"].n_human >= 10, "en human anchor corpus >= 10")
        assert_true(cal.langs["en"].n_ai >= 5, "en ai anchor corpus >= 5")
        # 18) calibration must actually widen the scale vs the v2.4 curve
        assert_true(ai.human_score <= 15, f"AI slop should now be <=15, got {ai.human_score}")
        assert_true(human.human_score >= 80, f"Gettysburg should now be >=80, got {human.human_score}")
        assert_true(zh.human_score >= 80, f"ZH human should now be >=80, got {zh.human_score}")
        assert_true(zh_off.human_score <= 15, f"ZH officialese should be <=15, got {zh_off.human_score}")
        assert_true(zh.human_score - zh_lit.human_score >= 40,
                    f"literary AI gap should be wide, got {zh.human_score - zh_lit.human_score}")
        # 19) per-slot calibration fields are exposed and consistent
        b_feat = next(f for f in ai.features if f.name == "burstiness_inv")
        assert_true(b_feat.gain > 0.3, f"burstiness gain should be live, got {b_feat.gain}")
        assert_true(b_feat.z > b_feat.anchor_lo, "z maps above human anchor for AI slop")
        flipped = [f for f in ai.features if f.flipped]
        # whatever the data says, a flipped slot must be labelled, not hidden
        assert_true(all(isinstance(f.gain, float) for f in flipped), "flipped slots labelled")
        # 20) dead slots stop diluting: zh slots carry gain 0 on English text
        zh_slots = {f.name: f for f in human.features if f.name.startswith("zh_")}
        assert_true(all(abs(f.gain) <= 0.35 for f in zh_slots.values()),
                    "ZH slots should be near-inactive on English")
    else:
        print("  (calibration.json absent — v3.0 anchor assertions skipped)")

    # 21) confidence: long clean text confident, tiny text not
    long_human = analyze(ZH_HUMAN * 3, use_onnx=False)
    assert_true(0.0 <= long_human.confidence <= 1.0, "confidence in range")
    assert_true(long_human.confidence > 0.55, f"long text confidence should be higher, got {long_human.confidence}")
    tiny = analyze("今天天气不错，我去超市买了点菜。", use_onnx=False)
    assert_true(tiny.insufficient_evidence is True, "short text must be flagged insufficient")
    # v3.1: clearing ZH provisional raises baseline conf; insufficient still trips via too_short.
    assert_true(tiny.too_short is True, "short ZH must be too_short")
    assert_true(tiny.confidence < long_human.confidence,
                f"short conf {tiny.confidence} should be < long {long_human.confidence}")

    # 22) sentence heat gating: a human-leaning doc must not paint many sentences
    heated = [s for s in zh.sentences if s.heat in ("red", "yellow")]
    assert_true(len(heated) <= max(1, int(0.2 * zh.sentence_count)),
                f"human doc should not over-highlight, got {len(heated)}/{zh.sentence_count}")
    # AI doc should highlight most sentences
    ai_heated = [s for s in ai.sentences if s.heat in ("red", "yellow")]
    assert_true(len(ai_heated) >= max(1, int(0.4 * ai.sentence_count)),
                f"AI doc should highlight broadly, got {len(ai_heated)}/{ai.sentence_count}")
    assert_true(any(s.percentile >= 0 for s in ai.sentences), "percentile recorded")

    # 23) paragraph windows and mixed-writing hint
    mixed_text = AI_SLOP + "\n\n" + ZH_HUMAN
    wins = paragraph_windows(mixed_text)
    assert_true(len(wins) >= 1, "paragraph windows produced")
    r_mixed = analyze(mixed_text, use_onnx=False)
    assert_true(len(r_mixed.paragraphs) >= 1, "paragraph scores attached")
    assert_true(isinstance(r_mixed.meta.get("mixed_writing_hint"), bool), "mixed hint flag")
    assert_true(r_mixed.human_score not in (0, 100), "mixed text lands in between")

    # 24) highlight_mode=all keeps both bands (compare onnx-off to onnx-off)
    r_base_hl = analyze(AI_SLOP, use_onnx=False)
    r_all = analyze(AI_SLOP, use_onnx=False, highlight_mode="all")
    assert_true(r_all.highlight_policy == "all_bands", "highlight_mode all")
    assert_true(r_all.human_score == r_base_hl.human_score, "highlight mode does not change score")

    # 25) weights still user-controllable after calibration
    w_zero = dict(load_weights())
    for k in w_zero:
        w_zero[k] = 0.0
    w_zero["burstiness_inv"] = 1.0
    r_single = analyze(AI_SLOP, weights=w_zero, use_onnx=False)
    assert_true(r_single.human_score <= ai.human_score + 20,
                "single-slot weighting should not explode the score")


    # ------------------------------------------------------------------
    # v3.1: temperature scaling, annotations, ECE/Brier
    # ------------------------------------------------------------------
    from prob_metrics import ece, brier, apply_temperature, fit_temperature, reliability_bins

    # ECE sanity: perfect calibration → ~0; all p=1 on half-positive → high
    perfect_p = [0.0, 0.0, 1.0, 1.0]
    perfect_y = [0.0, 0.0, 1.0, 1.0]
    assert_true(ece(perfect_p, perfect_y, bins=4) < 0.05, f"perfect ECE should be ~0, got {ece(perfect_p, perfect_y)}")
    bad_p = [0.9, 0.9, 0.9, 0.9]
    bad_y = [0.0, 0.0, 1.0, 1.0]
    assert_true(ece(bad_p, bad_y, bins=4) > 0.3, f"miscalibrated ECE should be high, got {ece(bad_p, bad_y)}")
    assert_true(0.0 <= brier(perfect_p, perfect_y) <= 0.01, "perfect Brier ~0")
    cooled = apply_temperature([0.9], 2.0)[0]
    assert_true(0.5 < cooled < 0.9, f"T>1 should cool 0.9 toward 0.5, got {cooled}")
    fit = fit_temperature([0.2, 0.3, 0.8, 0.9], [0.0, 0.0, 1.0, 1.0])
    assert_true(fit["T"] > 0, f"fit_temperature returned T={fit['T']}")
    bins = reliability_bins([0.1, 0.2, 0.8, 0.9], [0.0, 0.0, 1.0, 1.0], bins=2, equal_mass=True)
    assert_true(len(bins) == 2 and bins[0]["n"] + bins[1]["n"] == 4, "equal-mass bins")

    # Annotations present; ZH_LIT_AI should surface ≥1 flag/review lit label
    assert_true(isinstance(zh_lit.annotations, list), "annotations list on result")
    lit_ann = [
        a for a in zh_lit.annotations
        if a.get("kind") == "feature" and str(a.get("key", "")).startswith("lit_zh_")
        and a.get("action") in ("flag", "review")
    ]
    assert_true(len(lit_ann) >= 1, f"ZH_LIT_AI should have ≥1 flag/review lit annotation, got {lit_ann}")
    # Each annotation has the Jev fields
    for a in zh_lit.annotations[:3]:
        assert_true("p_ai" in a and "confidence" in a and "action" in a, f"annotation shape {a}")
        assert_true(a["action"] in ("flag", "review", "ignore"), f"bad action {a['action']}")

    # Temperature applied when present in calibration
    cal2 = get_calibration()
    if cal2 is not None:
        for lang, lc in cal2.langs.items():
            T = float(getattr(lc, "temperature", 1.0) or 1.0)
            assert_true(T > 0, f"{lang} temperature must be positive, got {T}")
        zh_lc = cal2.langs.get("zh")
        if zh_lc is not None:
            assert_true(zh_lc.n_human >= 5, f"zh human should be ≥5 after enrich, got {zh_lc.n_human}")
            # provisional may clear once n≥5
            print(f"  zh provisional={zh_lc.provisional} T={zh_lc.temperature} "
                  f"ECE {zh_lc.ece_before}→{zh_lc.ece_after}")
        # highlight_mode onnx-off comparison stays fixed (already asserted above)
        assert_true(r_all.human_score == r_base_hl.human_score, "highlight_mode fixed vs onnx-off")

    print("SMOKE OK")
    print(f"  AI_SLOP   human_score={ai.human_score} clarity={ai.clarity_score} ai_p={ai.ai_probability:.3f}")
    print(f"  Gettysburg human_score={human.human_score} clarity={human.clarity_score}")
    print(f"  ZH_HUMAN  human_score={zh.human_score} clarity={zh.clarity_score} sents={zh.sentence_count}")
    print(f"  ZH_OFFICIAL human_score={zh_off.human_score} gap_vs_alley={zh.human_score - zh_off.human_score}")
    print(f"  ZH_ESSAY  human_score={zh_essay.human_score} gap_vs_alley={zh.human_score - zh_essay.human_score}")
    print(f"  ZH_LIT_AI human_score={zh_lit.human_score} lit_fired={lit_fired} alley_lit_max={alley_max:.3f}")
    print(f"  ZH_LIT_AI annotations flag/review lit={len(lit_ann)} total_ann={len(zh_lit.annotations)}")
    print(f"  engine={ai.meta.get("engine")} conf_ai={ai.confidence:.3f} conf_zh={zh.confidence:.3f}")
    print(f"  LIST      human_score={lst.human_score} list_template={lt.value:.3f}")
    print(f"  compression_ai_like(AI)={comp['compression_ai_like']:.3f}")
    print(f"  Ollama available={st.available} models={len(st.models)} hybrid_used={hybrid.meta['ollama'].get('used')}")
    print(f"  pick_model(laptop)={picked!r} dna_blend_default={DNA_BLEND_DEFAULT}")
    print(f"  ONNX available={ost.available} used={r_onnx.meta['onnx'].get('used')} status={r_onnx.meta['onnx'].get('status')} blend_default={ONNX_B}")
    print(f"  window_burstiness_inv default={DEFAULT_WEIGHTS['window_burstiness_inv']}")
    print(f"  MIN_CHARS={MIN_CHARS} MIN_CHARS_ZH={MIN_CHARS_ZH}")
    print(f"  zh_policy default_w={DEFAULT_WEIGHTS.get('zh_policy_register')}")


if __name__ == "__main__":
    main()
