#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local AI-writing signal detector — Gradio desktop/localhost GUI.

Run (offline after pip install):
  pip install -r requirements.txt
  python app.py

Opens http://127.0.0.1:7860 by default. No TipType API calls.
Optional: Ollama at http://127.0.0.1:11434 for DNA-GPT-lite hybrid.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr

from detector import (
    DEFAULT_WEIGHTS,
    MIN_CHARS,
    MIN_CHARS_ZH,
    analyze,
    load_weights,
    save_weights,
)

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights.json"
EXPORT_DIR = ROOT / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

HEAT_CSS = {
    "red": "background:#fecaca;border-bottom:3px solid #dc2626;",
    "yellow": "background:#fef08a;border-bottom:3px solid #ca8a04;",
    "none": "",
}

WEIGHT_KEYS = [
    "adverb_density",
    "adverb_verb_proxy",
    "unique_word_ratio_inv",
    "sentence_start_variety_inv",
    "burstiness_inv",
    "list_template",
    "comma_density",
    "clause_layering_inv",
    "marketing_inflation",
    "compression_ai_like",
    "window_burstiness_inv",
    "zh_discourse_boilerplate",
    "zh_parallel_scaffold",
    "zh_policy_register",
    "zh_abstract_subject",
    "lit_zh_fake_terse_rhythm",
    "lit_zh_linger_gaze",
    "lit_zh_broken_parallel",
    "lit_zh_noun_reemphasize",
    "lit_zh_bushi_shi",
    "lit_zh_sceneless_simile",
    "lit_zh_flat_dialogue",
    "lit_zh_mech_scene",
]


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def weights_from_sliders(*vals) -> Dict[str, float]:
    raw = {k: float(v) for k, v in zip(WEIGHT_KEYS, vals)}
    s = sum(max(0.0, v) for v in raw.values()) or 1.0
    return {k: max(0.0, v) / s for k, v in raw.items()}


def build_highlight_html(result) -> str:
    if not result.text:
        return "<p style='color:#888'>（粘贴文本后点「分析」）</p>"
    parts: List[str] = []
    cursor = 0
    ordered = sorted(result.sentences, key=lambda s: s.char_start)
    for s in ordered:
        if s.char_start > cursor:
            parts.append(_esc(result.text[cursor : s.char_start]))
        style = HEAT_CSS.get(s.heat, "")
        title = f"AI≈{s.ai_probability:.2f} | {s.heat}"
        if s.top_features:
            title += " | " + ", ".join(f"{f.name_zh}" for f in s.top_features[:3])
        inner = _esc(s.text)
        if style:
            parts.append(
                f'<span title="{_esc(title)}" style="{style}padding:1px 2px;border-radius:3px">{inner}</span>'
            )
        else:
            parts.append(inner)
        cursor = s.char_start + s.length
    if cursor < len(result.text):
        parts.append(_esc(result.text[cursor:]))
    legend = (
        "<div style='margin-bottom:8px;font-size:13px;color:#444'>"
        "<b>高亮策略</b>（与 TipType 前端约定一致：只标当前文档最高严重档）："
        " <span style='background:#fecaca;padding:0 6px'>红 ≥0.65</span> "
        " <span style='background:#fef08a;padding:0 6px'>黄 ≥0.40</span> "
        f" · policy=<code>{result.highlight_policy}</code>"
        "</div>"
    )
    return legend + (
        "<div style='line-height:1.75;font-size:15px;white-space:pre-wrap;"
        "font-family:Georgia,'Noto Serif SC',serif;padding:12px;"
        "border:1px solid #e5e7eb;border-radius:8px;background:#fff'>"
        + "".join(parts)
        + "</div>"
    )


def build_score_md(result) -> str:
    warn = ""
    if result.too_short:
        floor = result.meta.get("min_chars", MIN_CHARS)
        cjk = result.meta.get("cjk_ratio", 0)
        if cjk and float(cjk) >= 0.45:
            warn = (
                f"\n\n⚠️ 中文文本短于建议下限 **{floor}** 字"
                f"（英文建议 ≥{MIN_CHARS}；中文 ≥{MIN_CHARS_ZH}）。"
                "短句分数不稳定，请多贴几句再判。"
            )
        else:
            warn = (
                f"\n\n⚠️ 文本短于建议下限 **{floor}** 字符"
                f"（EN≥{MIN_CHARS} / ZH≥{MIN_CHARS_ZH}），分数不稳定。"
            )
    ollama = result.meta.get("ollama") or {}
    ollama_line = ""
    if result.meta.get("backend") in ("hybrid", "ollama_dna"):
        dna = ollama.get("dna") or {}
        gate = ollama.get("dna_gate") or ""
        blend_applied = ollama.get("dna_blend_applied")
        model = ollama.get("selected_model") or "—"
        if ollama.get("used"):
            ollama_line = (
                f"\n\n🦙 Ollama DNA（实验）：model=`{model}` · "
                f"overlap={dna.get('overlap')} · ai_like={dna.get('ai_like')} · "
                f"blend={blend_applied} · gate=`{gate}`"
            )
        else:
            why = (
                ollama.get("fallback")
                or gate
                or ollama.get("status")
                or ollama.get("error")
                or "n/a"
            )
            ollama_line = (
                f"\n\n🦙 Ollama DNA（实验）未混入：model=`{model}` · {why}；"
                "以启发式为准。"
            )
            if ollama.get("only_vl_available"):
                ollama_line += " ⚠️ 仅有 VL 模型，请 `ollama pull qwen2:7b`。"
    onnx = result.meta.get("onnx") or {}
    onnx_line = ""
    if onnx.get("used"):
        onnx_line = (
            f"\n\n🧠 ONNX 第三票（path={onnx.get('path') or '?'}）："
            f"model=`{(onnx.get('score') or {}).get('model') or 'onnx'}` · "
            f"P(AI)={onnx.get('ai_prob_onnx')} · blend={onnx.get('blend_applied')} · "
            f"before={onnx.get('ai_prob_before')}"
        )
    else:
        st = onnx.get("status") or "n/a"
        opath = onnx.get("path") or ""
        hint = "EN: download_onnx_model.py · ZH: --model aigc-zh"
        onnx_line = (
            f"\n\n🧠 ONNX 第三票未混入：{st}"
            + (f" · path={opath}" if opath else "")
            + f"（缺权重软失败；{hint}）。"
        )
    calib = result.meta.get("calibration") or {}
    conf = getattr(result, "confidence", 0.0) or 0.0
    conf_line = f"\n\n**置信度 {conf:.2f}**"
    if getattr(result, "insufficient_evidence", False):
        conf_line += " ⚠️ **证据不足**（文本过短或特征结论分散，分数仅作参考）"
    if calib.get("provisional"):
        conf_line += (
            f"\n\n⚠️ 校准 `{calib.get('lang')}` 侧样本不足"
            f"（human={calib.get('n_human')} / ai={calib.get('n_ai')}），锚点为临时先验，置信度已下调。"
            "把稿子放进 `calib/human_zh/` 后跑 `python calibrate.py` 可显著改善。"
        )
    if not calib.get("calibrated", True):
        conf_line += "\n\n没有 `calibration.json`，正在使用 v2.4 回退曲线（`python calibrate.py` 修复）。"
    return f"""
### Human Score：**{result.human_score}** / 100
**{result.human_band_zh}** · *{result.human_band_en}*

AI probability ≈ `{result.ai_probability:.3f}`  
（约定：`Human Score ≈ round((1 − ai_probability) × 100)`）{conf_line}

### Clarity：**{result.clarity_score}** / 100
*{result.clarity_notes_zh}*  
*{result.clarity_notes_en}*

字符 {result.char_count} · 词元 {result.word_count} · 句子 {result.sentence_count}  
引擎 `<code>{result.meta.get('engine')}</code>` · backend=`{result.meta.get('backend')}`
{warn}{ollama_line}{onnx_line}

> 本工具为**本地启发式（分位校准）+ 可选 ONNX + 可选 Ollama**，**不是** TipType 专有模型/API；分数是编辑信号，**不能证明作者身份**。
"""


def build_features_table(result) -> str:
    rows = [
        "| # | 特征 Feature | 值 value | 权重 w | 贡献 contrib | 说明 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for i, f in enumerate(result.features, 1):
        rows.append(
            f"| {i} | **{f.name_zh}**<br><code>{f.name}</code> / {f.name_en} "
            f"| {f.value:.3f} | {f.weight:.3f} | {f.contribution:.3f} "
            f"| {f.explanation_zh}<br>*{f.explanation_en}* |"
        )
    return "\n".join(rows)


def build_sentence_md(result) -> str:
    lines = ["| # | heat | AI p | 句子 | 顶层特征 |", "|---:|:---:|---:|---|---|"]
    for i, s in enumerate(result.sentences, 1):
        if s.heat == "none" and s.ai_probability < 0.35:
            continue
        feats = "; ".join(f"{f.name_zh}({f.value:.2f})" for f in s.top_features[:3]) or "—"
        preview = s.text if len(s.text) <= 120 else s.text[:117] + "…"
        lines.append(
            f"| {i} | {s.heat} | {s.ai_probability:.2f} | {_esc(preview)} | {feats} |"
        )
    if len(lines) == 2:
        lines.append("| — | — | — | （无黄/红句子） | — |")
    return "\n".join(lines)


def refresh_ollama(host: str) -> str:
    try:
        from backends.ollama_client import is_vision_or_embed_model, probe
        st = probe(host or "http://127.0.0.1:11434")
        if not st.available:
            return f"❌ Ollama 不可用：`{st.host}` · {st.error}\n将自动回退启发式。"
        models = ", ".join(st.models[:12]) or "(无模型)"
        more = f" …共{len(st.models)}个" if len(st.models) > 12 else ""
        selected = st.selected_model or "—"
        warn = ""
        if getattr(st, "only_vl_available", False) or (
            st.models and not st.selected_model
        ):
            warn = (
                "\n⚠️ **仅有 VL/vision/embed 模型，无法自动用于 DNA。**"
                "请拉取文本模型：`ollama pull qwen2:7b`（或 qwen2.5 文本版，不要 vl）。"
            )
        elif st.selected_model and is_vision_or_embed_model(st.selected_model):
            warn = "\n⚠️ 当前选中疑似视觉模型；hybrid DNA 需要文本模型。"
        vl_notes = [
            m for m in st.models if is_vision_or_embed_model(m)
        ]
        if vl_notes and st.selected_model:
            warn += f"\n（已忽略 VL：{', '.join(vl_notes[:4])}）"
        return (
            f"✅ Ollama OK · `{st.host}`\n"
            f"**已选文本模型：** `{selected}`\n"
            f"模型：{models}{more}\n"
            f"建议：`ollama pull qwen2:7b`（文本；勿用 qwen2.5vl）"
            f"{warn}"
        )
    except Exception as e:  # noqa: BLE001
        return f"❌ 探测失败：{type(e).__name__}: {e}"



def refresh_onnx() -> str:
    try:
        from backends.onnx_classifier import (
            DEFAULT_MODEL_DIR,
            DEFAULT_ZH_MODEL_DIR,
            probe,
            probe_zh,
        )
        st = probe()
        st_zh = probe_zh()
        lines = []
        if st.available:
            lines.append(
                f"✅ EN ONNX 就绪 · `{st.model_dir}` · providers={st.providers}"
            )
        else:
            lines.append(
                f"⚠️ EN ONNX 未就绪：{st.reason}\n"
                f"  预期：`{DEFAULT_MODEL_DIR}` · "
                "`python download_onnx_model.py`（~77MB ettin-68m）"
            )
        if st_zh.available:
            lines.append(
                f"✅ ZH ONNX 就绪 · `{st_zh.model_dir}`（CJK≥35% 时使用）"
            )
        else:
            lines.append(
                f"⚠️ ZH ONNX 未就绪：{st_zh.reason}\n"
                f"  预期：`{DEFAULT_ZH_MODEL_DIR}` · "
                "`python download_onnx_model.py --model aigc-zh`（~98MB）\n"
                "  缺权重时中文仍走 v2.3/v2.4 启发式（套话/政策腔 + 文学八槽），软失败。"
            )
        lines.append("无任何 ONNX 权重时仍可纯启发式运行；blend 默认约 0.18。")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"❌ 探测失败：{type(e).__name__}: {e}"

def run_analyze(
    text: str,
    backend: str,
    ollama_host: str,
    ollama_model: str,
    *slider_vals,
) -> Tuple[str, str, str, str, str]:
    weights = weights_from_sliders(*slider_vals)
    result = analyze(
        text or "",
        weights=weights,
        backend=backend or "heuristic",
        ollama_host=ollama_host or "http://127.0.0.1:11434",
        ollama_model=(ollama_model or "").strip(),
    )
    report = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    return (
        build_score_md(result),
        build_highlight_html(result),
        build_features_table(result),
        build_sentence_md(result),
        report,
    )


def do_save_weights(*slider_vals) -> str:
    weights = weights_from_sliders(*slider_vals)
    path = save_weights(weights, WEIGHTS_PATH)
    return f"已保存权重到 `{path}`（下次启动会加载；已归一化）。\n```json\n{json.dumps(weights, ensure_ascii=False, indent=2)}\n```"


def do_export_json(report_json: str) -> str:
    if not report_json.strip():
        return "没有可导出的报告，请先分析。"
    try:
        data = json.loads(report_json)
    except json.JSONDecodeError:
        return "报告 JSON 无效。"
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EXPORT_DIR / f"report_{stamp}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return f"已导出：`{path}`"


def load_slider_defaults() -> List[float]:
    w = load_weights(WEIGHTS_PATH)
    return [float(w.get(k, DEFAULT_WEIGHTS[k])) for k in WEIGHT_KEYS]


EXAMPLES = [
    [
        "Moreover, this innovative solution seamlessly leverages cutting-edge technology "
        "to optimize workflows, enhance productivity, and unlock unprecedented value for "
        "organizations navigating today's rapidly evolving digital landscape. Furthermore, "
        "it is important to note that stakeholders should delve into the myriad of opportunities "
        "that this holistic approach can foster in the foreseeable future."
    ],
    [
        "Four score and seven years ago our fathers brought forth on this continent, a new nation, "
        "conceived in Liberty, and dedicated to the proposition that all men are created equal. "
        "Now we are engaged in a great civil war, testing whether that nation, or any nation so "
        "conceived and so dedicated, can long endure."
    ],
    [
        "昨晚下了点小雨，巷子里的石板路还是湿的。我骑车去买醋，老板娘还是那副慢悠悠的样子，"
        "一边找零钱一边抱怨说今年的青菜贵得没道理。我没接话，只是点了点头。回到家才发现，"
        "口袋里那张五块钱不知道什么时候蹭湿了，黏在一起，撕开的时候还带出一点纸毛。"
    ],
    [
        "First, it is essential to establish a clear framework for analysis. "
        "Second, stakeholders should carefully evaluate the available options. "
        "Finally, organizations must implement best practices to ensure sustainable outcomes "
        "across all relevant dimensions of the initiative."
    ],
    [
        "综上所述，在高质量发展的背景下，我们要进一步赋能基层治理，推动数字化转型落地见效。"
        "值得注意的是，必须坚持系统观念。首先，要完善顶层设计；其次，要加强协同联动；"
        "最后，要强化闭环管理，为经济社会发展注入新动能。"
    ],
]


def build_ui() -> gr.Blocks:
    defaults = load_slider_defaults()
    labels = [
        "adverb_density 副词密度",
        "adverb_verb_proxy 副词-动词",
        "unique_word_ratio_inv 用词多样性不足",
        "sentence_start_variety_inv 句首多样性不足",
        "burstiness_inv 句长爆发力不足",
        "list_template 列表模板",
        "comma_density 逗号密度",
        "clause_layering_inv 从句层次不足",
        "marketing_inflation 营销夸大词",
        "compression_ai_like 压缩AI亲和",
        "window_burstiness_inv 窗口压缩爆发力不足",
        "zh_discourse_boilerplate 中文连接套话 (v2.3)",
        "zh_parallel_scaffold 中文排比骨架 (v2.3)",
        "zh_policy_register 政策腔 (v2.3)",
        "zh_abstract_subject 空泛主语/低具体 (v2.3)",
        "lit_zh_fake_terse_rhythm 假短促/了滥用 (v2.4)",
        "lit_zh_linger_gaze 伫立凝视套板 (v2.4)",
        "lit_zh_broken_parallel 是X是Y断裂 (v2.4)",
        "lit_zh_noun_reemphasize 主语刻意复提 (v2.4)",
        "lit_zh_bushi_shi 不是A是B伪诗 (v2.4)",
        "lit_zh_sceneless_simile 无场景比喻/别人 (v2.4)",
        "lit_zh_flat_dialogue 对话平/短促 (v2.4)",
        "lit_zh_mech_scene 机械场景+假叙事 (v2.4)",
    ]
    with gr.Blocks(
        title="本地 AI 写作信号检测器 v3.0",
        theme=gr.themes.Soft(primary_hue="rose", secondary_hue="slate"),
        css=".gradio-container {max-width: 1100px !important;}",
    ) as demo:
        gr.Markdown(
            """
# 本地 AI 写作信号检测器 v3.0
**Local AI-writing signal detector** · 猫猫专用桌面/本机版

吸收 TipType 公开 UX 约定（Human Score 分档、句级红/黄热力、Clarity 独立），
用**透明可调特征** + **离线压缩信号** + **可选 ONNX 分类器（第三票）** + **可选 Ollama 续写一致性（实验）**。  
默认后端 **heuristic**（推荐）。**不调用 TipType API，不是其专有模型。**
"""
        )
        with gr.Row():
            with gr.Column(scale=3):
                text_in = gr.Textbox(
                    label="粘贴文本 Paste text（EN≥80 / 中文≥55 字）",
                    lines=12,
                    placeholder="在此粘贴英文或中文草稿…",
                )
                with gr.Row():
                    backend = gr.Radio(
                        choices=["heuristic", "hybrid"],
                        value="heuristic",
                        label="后端 backend（默认 heuristic；hybrid=实验性 DNA，仅不确定区间混入）",
                    )
                with gr.Row():
                    ollama_host = gr.Textbox(
                        value="http://127.0.0.1:11434",
                        label="Ollama host",
                        scale=3,
                    )
                    ollama_model = gr.Textbox(
                        value="",
                        label="模型名（空=自动选文本模型；勿填 VL）",
                        placeholder="qwen2:7b",
                        scale=2,
                    )
                    btn_ollama = gr.Button("探测 Ollama", scale=1)
                    btn_onnx = gr.Button("探测 ONNX", scale=1)
                ollama_status = gr.Markdown("点击「探测 Ollama」检查本机服务。")
                onnx_status = gr.Markdown("点击「探测 ONNX」检查 EN `ettin-68m` / ZH `aigc-zh` 权重。")
                with gr.Row():
                    btn = gr.Button("分析 Analyze", variant="primary")
                    btn_export = gr.Button("导出 JSON 报告")
                    btn_save_w = gr.Button("保存当前权重")
                gr.Examples(examples=EXAMPLES, inputs=[text_in], label="示例")
            with gr.Column(scale=2):
                score_md = gr.Markdown(label="分数")
                export_status = gr.Markdown()

        highlight = gr.HTML(label="句级高亮")
        with gr.Tabs():
            with gr.Tab("特征表 Features（公开，TipType UI 不展示名）"):
                feat_md = gr.Markdown()
            with gr.Tab("句子明细 Sentences"):
                sent_md = gr.Markdown()
            with gr.Tab("权重 Weights（可调）"):
                gr.Markdown("拖动后点「分析」生效；「保存当前权重」写入 `weights.json`。")
                sliders = []
                for lab, val in zip(labels, defaults):
                    sliders.append(gr.Slider(0, 0.4, value=val, step=0.01, label=lab))
            with gr.Tab("原始 JSON"):
                report_json = gr.Code(language="json", label="report")

        btn.click(
            run_analyze,
            inputs=[text_in, backend, ollama_host, ollama_model, *sliders],
            outputs=[score_md, highlight, feat_md, sent_md, report_json],
        )
        btn_save_w.click(do_save_weights, inputs=sliders, outputs=[export_status])
        btn_export.click(do_export_json, inputs=[report_json], outputs=[export_status])
        btn_ollama.click(refresh_ollama, inputs=[ollama_host], outputs=[ollama_status])
        btn_onnx.click(refresh_onnx, inputs=[], outputs=[onnx_status])

        gr.Markdown(
            """
---
**分档**：&lt;20 高度疑似 AI · &lt;40 疑似 AI · &lt;60 混合 · &lt;80 较人类 · ≥80 高度人类  

**v2 新增**：`compression_ai_like` / `window_burstiness_inv`（纯离线）；可选 Ollama DNA-GPT-lite 续写重合。  
**v2.1 DNA**：默认请用 **heuristic**；hybrid 为实验性，仅在 Human Score 约 35–65 时混入，且自动避开 VL/记忆型误伤。请用 `qwen2:7b`（文本），不要 `qwen2.5vl`。  
**v2.2 ONNX**：可选第三票 `Deckard618/textdet` ettin-68m（~77MB EN）。  
**v2.3 中文**：套话/排比/政策腔/空泛主语启发式；**v2.4/v3.0 文学 AI 味**八槽（假短促、伫立凝视、是X是Y、名词复提、不是A是B、无场景比喻、平对话、机械场景）；可选 `aigc-zh` ONNX；CJK 下限 55 字；EN ONNX 仍不跑 CJK。  
**v3.0 判定**：逐特征分位校准（锚点 + 判别力增益）+ 置信度/证据不足 + 句级分位与绝对阈值双条件 + 段落窗口混写检测。
**前端**：`python server.py` 启动前后端分离版（FastAPI + 原生前端，支持文件导入 / 校准管理 / A-B 对比）。  
引擎：`detector.py` local_heuristic_v3.0 · 默认 heuristic · 完全离线。
"""
        )
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
