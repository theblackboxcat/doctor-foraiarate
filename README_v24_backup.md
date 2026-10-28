# 本地 AI 写作信号检测器 v2.4（猫猫桌面版）

吸收 TipType 检测器的**公开 UX 约定**与已知「AI 味」语言学/统计信号，做成可离线运行的本地软件。  
**这不是 TipType 专有模型，也不调用 TipType API。** 分数是启发式编辑信号，**不能证明作者是谁**。

默认请用 **heuristic（纯启发式）**。可选：

1. **ONNX 第三票**：
   - **英文** `models/onnx/ettin-68m`（~77MB）— CJK&lt;35% 时混入
   - **中文** `models/onnx/aigc-zh`（~98MB，可选）— CJK≥35% 时混入；缺文件**软失败**，v2.3/v2.4 中文启发式照常
2. **Ollama DNA**（`hybrid`，实验性）：截断→续写→n-gram 重合；无合适文本模型时回退启发式。

## 功能

- **Human Score**（0–100）与分档文案（中英）
- **Clarity**（清晰度，与 AI 分独立）
- **句级红/黄高亮**（≥0.65 红 / ≥0.40 黄；只标当前文档最高严重档）
- **公开特征名 + 贡献**（含 v2 压缩信号、**v2.3 中文四槽**、**v2.4 文学 AI 味八槽**；ONNX 启用时多一项）
- **可调权重**（滑条 → 保存 `weights.json`）
- **导出 JSON 报告**到 `exports/`
- **后端**：`heuristic`（**默认推荐**） / `hybrid`（实验性 DNA）
- **短文本**：英文建议 ≥80 字；中文（CJK≥45%）≥55 字，短于下限会提示但不崩溃

## Windows 安装与运行

1. 安装 [Python 3.10+](https://www.python.org/downloads/)（勾选 *Add Python to PATH*）
2. 解压本文件夹，例如 `C:\Users\你\Downloads\local_ai_detector`
3. 命令提示符 / PowerShell：

```bat
cd C:\Users\你\Downloads\local_ai_detector
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

4. 浏览器打开 `http://127.0.0.1:7860`

### （可选）ONNX 第三票 — Windows 笔记本友好

```bat
pip install -r requirements-onnx.txt
python download_onnx_model.py
REM 中文可选第二路（~98MB）：
python download_onnx_model.py --model aigc-zh
REM 可选更大英文：python download_onnx_model.py --model ettin-150m
python test_smoke.py
python app.py
```

| 模型 | 目录 | 何时用 | 来源 |
|------|------|--------|------|
| ettin-68m ~77MB | `models\onnx\ettin-68m\` | 英文 / 低 CJK | [Deckard618/textdet](https://huggingface.co/Deckard618/textdet) |
| aigc-zh ~98MB | `models\onnx\aigc-zh\` | CJK≥35% | [Eslzzyl/aigc-detector-zh-onnx](https://huggingface.co/Eslzzyl/aigc-detector-zh-onnx)（yuchuantian/AIGC_detector_zhv3 INT8） |

GUI「探测 ONNX」可分别看 EN/ZH 是否就绪。缺权重时分析日志写未混入原因，**不影响启发式**。

门控：启发式已偏人类时，若 ONNX 仍偏 AI，可跳过（保护 Gettysburg / 巷子文）。ettin 本身偏英语，故 CJK 重文本改走 aigc-zh（有则用，无则纯启发式）。

> 其他备选（未自动下载）：[houtini-ai/ai-detect](https://github.com/houtini-ai/ai-detect) light ≈126MB；全尺寸 desklib DeBERTa ≈1.7GB 默认不拉。

### （可选，实验性）Ollama hybrid

```bat
ollama pull qwen2:7b
REM 不要用视觉版 qwen2.5vl / *-vl* / *vision* / *embed*
```

> **推荐日常：heuristic +（有的话）对应语种 ONNX。** hybrid DNA 在经典背诵文上仍可能误伤；v2.1+ 已门控。

仅跑核心检测：

```bat
python detector.py
python test_smoke.py
```

### 从 box 拷到 Windows（覆盖桌面旧版）

- 目录：`/workspace/local_ai_detector/`
- 压缩包：`/workspace/local_ai_detector_win.zip`（**不含** Linux `.venv`，**不含**已下载的大 ONNX 权重——请在 Windows 上跑 `download_onnx_model.py`）

**覆盖步骤（笔记本）：**

1. 用 Cursor **CopyFromBox** 把 `/workspace/local_ai_detector_win.zip` 拷到本机（如 `Downloads\`）
2. 解压，**覆盖**旧的 `local_ai_detector` 文件夹（可先备份自己的 `weights.json` / `exports/`）
3. 删除旧 `.venv` 后重建：
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   REM 可选 ONNX：
   pip install -r requirements-onnx.txt
   python download_onnx_model.py
   python download_onnx_model.py --model aigc-zh
   python test_smoke.py
   python app.py
   ```

## 检测原理（短）

研究笔记：`/workspace/ai_detect_research/`（`notes_zh.md` · `sources.md` · `onnx_notes_2026-10-25.md` · `zh_upgrade_2026-10-25.md` · **`literary_ai_taste_taxonomy_2026-10-25.md`**）

| 信号 | 直觉 |
|------|------|
| 风格/模板特征 | 列表骨架、营销空词、句首重复等 |
| **v2.3 中文专用** | 连接套话、排比骨架、政策腔、空泛主语/低具体名词 |
| **v2.4 文学 AI 味** | 假短促/了滥用、伫立凝视、是X是Y断裂、名词复提、不是A是B、无场景比喻、平对话、机械场景 |
| Burstiness / 压缩比 | 人类句长与可压缩性更「起伏」；AI 更均匀、更冗余 |
| ONNX 分类器（可选） | EN ettin / ZH aigc-zh 的监督分类 P(AI) |
| DNA-GPT 续写一致性 | 截断后让模型续写，与原文后缀重合高 → 更像机器 |
| 困惑度 / 曲率 | 真 PPL 需白盒 logprob；本工具默认不做 |

**会变不准**：短文本、同义/递归改写、人机混写、跨模型。**请勿当学术诚信铁证。**

## 分数怎么算

```
Human Score ≈ round((1 − clamp(ai_probability, 0, 1)) × 100)
```

`ai_probability` = 特征加权启发式（+ 可选 ONNX 线性混入 + 可选 Ollama DNA），不是 TipType 线上模型。

| Human Score | 中文 | English |
|---|---|---|
| &lt;20 | 高度疑似 AI | Strong AI |
| 20–39 | 疑似 AI | Likely AI |
| 40–59 | 人类与 AI 混合 | Mixed |
| 60–79 | 较符合人类写作 | Likely Human |
| ≥80 | 高度符合人类写作 | Strong Human |

建议：**英文 ≥80 字符**；**中文 ≥55 字**（更稳建议更长）。

### 特征（越高越偏 AI）

| 内部名 | 说明 |
|---|---|
| `adverb_density` 等 | 原语言学/模板特征 |
| `compression_ai_like` / `window_burstiness_inv` | v2 压缩信号 |
| `zh_discourse_boilerplate` | **v2.3** 综上所述/值得注意的是… |
| `zh_parallel_scaffold` | **v2.3** 首先其次最后 / 一是二是 |
| `zh_policy_register` | **v2.3** 高质量发展/赋能/顶层设计… |
| `zh_abstract_subject` | **v2.3** 空泛主语 + 低具体名词 |
| `lit_zh_*`（八槽） | **v2.4** 文学 AI 味：假短促、伫立凝视、是X是Y、名词复提、不是A是B、无场景比喻、平对话、机械场景 |
| `onnx_ai_prob` | 可选 ONNX P(AI) |
| `ollama_dna_overlap` | 可选/实验 Ollama 续写重合 |

英文文本上 ZH / 文学槽权重自动置 0（不稀释 EN）。详见 `literary_ai_taste_taxonomy_2026-10-25.md`。

## 诚实边界

- 未使用 TipType 专有权重/模型/API。
- 与 GPTZero、Originality、TipType 等结果会不一致——正常。
- 中文分数以 **v2.3/v2.4 启发式**为主（公文四槽 + 文学八槽）；`aigc-zh` ONNX 为可选增强。
- 适合自己改稿看「哪里像模板」，不适合给人贴标签。

## 文件一览

```
local_ai_detector/
  app.py                    # Gradio GUI
  detector.py               # 核心 v2.4
  zh_features.py            # 中文公文/套话特征 (v2.3)
  literary_zh_features.py   # 文学 AI 味八槽 (v2.4)
  compression_signals.py
  backends/ollama_client.py
  backends/onnx_classifier.py   # EN + ZH probe/score
  download_onnx_model.py    # ettin-68m / 150m / aigc-zh
  models/onnx/              # 下载后放权重（默认不随 zip）
  weights.json
  requirements.txt
  requirements-onnx.txt
  test_smoke.py
  README_ZH.md
  exports/
```

引擎标识：`local_heuristic_v2.4` / `local_hybrid_v2.4:*`

## 版本摘要

- **v2**：压缩特征 + Ollama hybrid 骨架  
- **v2.1**：DNA 不确定区间门控、避开 VL、记忆型降权；默认 heuristic  
- **v2.2**：可选 ONNX 第三票（ettin-68m）；`window_burstiness_inv` 略降  
- **v2.3**：中文四槽启发式 + 可选 aigc-zh ONNX；CJK 短文下限 55；语言门控权重  
- **v2.4**：文学中文 AI 味八槽（`literary_zh_features.py`）；精度优先，巷子口语/EN 安静  
