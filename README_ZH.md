# 本地 AI 写作信号检测器 v3.2（猫猫桌面版）

对标公开的 AIGC 检测思路（[GLTR](https://arxiv.org/abs/1906.04043) / LRR / DetectGPT / Fast-DetectGPT / Binoculars / DNA-GPT / GPTZero 的 UX 约定；中文对照语料常见于 [HC3-Chinese](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)）与开源实现，
做成**可离线、可解释、可自己重校准**的本机工具。

**不是 TipType 专有模型，不调用任何远端 API。** 分数是改稿线索，**不能证明作者是谁**。

---

## 1. v3.2 改了什么：语域分层

v3.0 解决了"映射压平"，v3.1 解决了"一个总分说不出理由"。v3.2 解决的是另一类问题：
**同一套特征在不同语域里意义相反。**

| 层 | v3.1 | v3.2 |
|---|---|---|
| 校准桶 | `en` / `zh` | `en` / `zh` / **`zh_poetry`** |
| 语域路由 | 无（只按 CJK 占比分中英） | **`register_zh.poetry_probability`** —— 只回答"这是不是分行文本"，决定用哪个桶、哪些特征在本语域根本不成立 |
| 中文人类参照 | 1 条手稿 + 填充样本（≈5 条） | **110 条**（1920s–40s 公有领域中文，`eval_set/`） |
| 文学特征 | 8 槽（日常口语小说语域） | 8 槽 + **7 个工艺特征 `mod_*`** + **3 个诗歌特征 `poe_*`** |
| 诗歌上的句长类特征 | 照算，测的是分行伪影 | **不设锚点**（`POETRY_MASKED_FEATURES`）——定义域修复，不是调参 |
| 标注清单 | 硬编码三个前缀，`mod_*`/`poe_*` 进不来 | 前缀常量生效；**校准没量出方向的槽不报**（`|g|<0.05`）；**本语域无锚点的槽不报**；**文档强判人类时槽级不报"很像"** |
| 槽级概率 | 借用文档级温度 T | **不再借用**——T 是在文档后验上拟合的，套到槽级会把 0.79 抬成 0.958 |
| 温度校准 | 小样本上 T 可退到 0.425 | `n<30 且 T<0.8` → **钳到 0.8**，notes 里写明原因 |
| 引擎标识 | `local_heuristic_v3.1` | `local_heuristic_v3.2` |
| 版本号来源 | `server.py` 硬编码 `"local_heuristic_v3.1"`、前端标题写死 `v3.1`、Cowrite 侧另读一份 —— 三处会各自漂移 | **单一来源** `detector._ENGINE`；`/api/meta` 从它读，前端启动时从 `/api/meta` 同步标题与角标 |

### 三条修法，各自解决不同的问题

1. **扩充人类参照**——把鲁迅《野草》《呐喊》、萧红《呼兰河传》、戴望舒、闻一多、徐志摩放进 `zh` 桶的人类侧。
   旧锚点代表的是"某一个人的日记"，不是"人类中文散文"。这是**参照系**问题。
2. **新增工艺特征**——AI 文学写作与人类现代主义的真正分野不在节奏，在**如何处理自己的意义**：
   自我解释、结尾上升为格言、喻体抽象、对偶工整、段落整齐、标点单调。
   这些属性在散文和诗里都成立，所以它们能跨语域迁移。这是**特征集**问题。
3. **诗歌单独建桶**——"诗没有句子可量"。句长、逗号密度、burstiness、分句层次在诗上测的是**分行本身**。
   与其调参，不如承认这些槽在诗里没有定义。这是**定义域**问题。

---

## 2. 为什么现代主义与诗歌是误判重灾区（先看数据）

v3.1 在 112 条公有领域人类样本上的实测：

| 语域 | 人类 P 中位 | 人类 P≥0.40 占比 | AUC（人 vs AI） |
|---|---:|---:|---:|
| 现代主义散文（37 条） | 0.296 | **32%** | **0.378** |
| 诗歌（8 条） | 0.103 | 0% | 0.958 |
| 现实主义散文（67 条） | 0.406 | **51%** | **0.373** |

AUC 低于 0.5 —— 不是"判别力弱"，是**方向反了**。最惨的几条：

```
realist_prose/xiaohong_hulanhe_1_18.txt  p=0.995  human=  1   ← 萧红被判 1 分
realist_prose/xiaohong_hulanhe_2_17.txt  p=0.961  human=  4
modernist_prose/luxun_sihou_01.txt       p=0.934  human=  7   ← 鲁迅《死后》
realist_prose/luxun_guxiang_02.txt       p=0.891  human= 11
```

**原因不是模式写错了。** 现代主义散文故意重复名词、故意切碎分句、故意落进极短的名词节拍
（「一株是枣树，还有一株也是枣树」）——这每一件事都是 `lit_zh_noun_reemphasize` /
`lit_zh_mech_scene` / `lit_zh_fake_terse_rhythm` 的命中条件。旧逻辑把这些**当成套板**，
但在鲁迅那里它们是**技法**。

数据自己给出了判决：重建校准后，`lit_zh_mech_scene` 的 `g` 从 +0.4 变成 **−0.109（FLIP）**，
`lit_zh_noun_reemphasize` 变成 **−0.779（FLIP）**——在 110 条人类中文上，这两个槽**偏向人类**。

---

## 3. 语域层怎么做的（`register_zh.py`）

### 3.1 路由只问一个问题

```python
poetry_probability(text) -> 0..1
```

它**只**回答"这是不是分行文本/诗"，用来（a）选校准桶、（b）屏蔽未定义特征。
它**刻意不做**"这是不是现代主义"——那是关于意图的判断，表层统计扛不住。
现代主义散文留在普通中文桶，那里改变的是**参照语料**，不是路由器。

打分规则（都是可读的，没有训练）：

| 线索 | 加分 |
|---|---:|
| 行长中位数 ∈ [3, 18] | +0.30 |
| 行末无句末标点比例 ≥ 0.55 | +0.30 |
| 每百字句末标点 < 1.6 | +0.20 |
| 有空行分节 | +0.10 |
| 行数/句数 ≥ 1.8 | +0.10 |
| **反驳**：对话标记 ≥3 或引号 ≥3 | −0.35 |
| **反驳**：行首「X：」 | −0.15 |

最后两条是必须的：鲁迅《过客》是**剧本**，分行、有「：」、有大量短行。
不加反驳项它会被路由成诗，然后按诗歌口径去量——比不路由更糟。

阈值 0.55。低于阈值的文本走普通中文桶。

### 3.2 十个新特征

七个工艺特征（`mod_*`，两种语域都算）：

| 键 | 抓什么 | AI 的样子 |
|---|---|---|
| `mod_gloss_coda` | 自我解释式收尾 | 「记忆是这样一种东西」「这大概就是所谓的相遇」 |
| `mod_universal_coda` | 结尾上升为格言 | 「有些话不需要被听见」「我们都是一样的」 |
| `mod_abstract_binome` | 抽象双音词密度 | 时间/记忆/距离/意义/孤独 扎堆 |
| `mod_generic_simile` | 喻体是抽象物 | 「像一个没有出口的梦」「像某种他无法命名的语言」 |
| `mod_clean_antithesis` | 工整对偶 | 「不是…而是…」「既…又…」成对出现 |
| `mod_para_uniformity` | 段落长度过于整齐 | 每段都差不多长 |
| `mod_punct_monotony` | 标点单调 | 全篇只有逗号和句号 |

三个诗歌特征（`poe_*`，只在路由到诗歌时算）：

| 键 | 抓什么 |
|---|---|
| `poe_prose_line` | 分行散文：把散文按 12 字切断，行内语法完整 |
| `poe_explained_ending` | 诗末解释：最后两行把前面的意象"翻译"一遍 |
| `poe_abstract_line` | 纯抽象诗行：整行没有可拍照之物 |

`ABSTRACT_BINOMES` 词表**故意排除**了「寂寥 / 愁怨 / 彷徨」这类 1920s 文学词——
它们在现代诗里是语料特征，不是 AI 特征。保留它们会把戴望舒判成 AI。

### 3.3 诗歌桶里被屏蔽的特征

```python
POETRY_MASKED_FEATURES = (
    "burstiness_inv", "comma_density", "clause_layering_inv",
    "sentence_start_variety_inv", "window_burstiness_inv",
    "lit_zh_fake_terse_rhythm", "lit_zh_flat_dialogue",
    "lit_zh_mech_scene", "lit_zh_noun_reemphasize",
)
```

这九个槽在 `zh_poetry` 桶里**没有锚点**（`calibrate.py` 建桶时按 `mask` 跳过）。
后果是双重的：它们不参与聚合，也**不会出现在标注清单里**——
`build_annotations()` 里加了 `if lc is not None and anc is None: continue`。

之前的行为是把原始特征值当概率报出来（诗里 `lit_zh_mech_scene=1.0` → "机械场景 100%"），
那是最坏的一种错：**用一个在本语域无定义的量，给出一个看起来很确定的结论。**

### 3.4 温度钳制

诗歌桶只有 14 条样本且两类几乎完全可分，NLL 的极小值在 T→0，
会拟合出"我永远很确定"。这不是校准，是过拟合。

```python
if len(ps) < 30 and T < 0.8:
    T = 0.8   # 并在 notes 里写明：separable small sample drives T→0
```

`zh_poetry` 的最终 T=0.8，ECE 0.1981→0.1527。这个 ECE 不好看，但它是**诚实的**。

---

## 4. v3.0 的分位校准（仍然有效）

对 v2.4 跑内置用例，加权和 `acc` 只落在 **0.13–0.64**，而映射 `p = 0.08 + 0.92(1−e^(−1.85·acc))` 在这一段只覆盖 0.28–0.72。
于是无论证据多强，Human Score 都被压在 **28–72**：AI 营销腔 37、葛底斯堡演说 68、真人中文散文 72。**判别力不是被特征拖累，是被映射压平。**

第二组数据更说明问题（v2.4 原始特征值）：

| 特征 | 人类散文 | AI 营销腔 | 判定 |
|---|---:|---:|---|
| `burstiness_inv` | 0.000 | 0.800 | 有效 |
| `marketing_inflation` | 0.000 | 1.000 | 有效 |
| `window_burstiness_inv` | **0.856** | **0.815** | 无判别力（甚至反向） |
| `clause_layering_inv` | **1.000** | 1.000 | 中文满格误伤 |
| `comma_density` | 0.243 | 0.278 | 近似常数 |
| `sentence_start_variety_inv` | 0.333 | 0.000 | 方向可疑 |
| `unique_word_ratio_inv` | 0.000 | 0.000 | 从未触发 |

两个具体 bug：

1. **`window_burstiness_inv` 按字节切窗**（80 字节）。中文 UTF-8 约 3 字节/字，于是中文窗口只有 ~27 字、英文 ~80 字——同一名字下测的是两件事。改按**字符**切窗并分语言定尺度（中文 60 字 / 英文 200 字符），中性点也分脚本。
2. **`clause_layering_inv` 用显式从属连词衡量中文层次**。"昨晚下了点小雨，巷子里的石板路还是湿的。我骑车去买醋……"没有"虽然/因为"，但层次很深——旧逻辑给它 1.000 满格。改中文用**分句密度 + 成对关联词**（虽然…但是 / 一边…一边 / 越…越）。

### 校准机制（`calibration.py`）

每条参考样本先算特征，然后对每个特征取三个数：

| 量 | 含义 |
|---|---|
| `lo` | 人类侧**中位数**——"典型人类长什么样" |
| `hi` | AI 侧**上四分位**——故意不用中位数：稀疏特征（如 `list_template`）只有少数 AI 文本触发，中位数会把它们判死 |
| `disc` | `(mean_AI − mean_H) / spread`，spread 用 IQR 且下限 0.10，防止"两边都平"的特征凭空造出巨大判别力 |

派生：

```
z     = clamp((v − lo) / max(hi − lo, 0.10), −3, 3)   # 0 = 人类中位，1 = AI 偏高段
g     = tanh(disc / 1.2)                               # 可正可负；≈0 表示该特征没用
z_agg = Σ(w·g·z) / Σ|w·g|                              # 分母是"有效质量"，不是原始权重
p     = sigmoid(slope·(z_agg − center))                # 人类中位→0.18，AI 中位→0.82
```

关键副作用（这正是想要的）：**语言门控不用手写了**。中文专有槽在英文样本上人类/AI 两侧都是 0，`disc=0`，增益自动归零。
v3.2 的诗歌屏蔽用的是同一个机制——只是把"靠数据自然归零"换成"按定义直接跳过"，
因为诗歌桶的样本量不足以让数据自己说话。

---

## 5. v3.1 Jev 对齐（仍然有效）

吸收 [Jev](https://github.com/) 类标注系统的原则（**不调用远端 API，全部本地**），把「标注」做成改稿辅助，而不是商业准确率宣言：

1. **共享文本上的并行问题**：同一段文字是共享状态；每个 AI 味特征槽 / 句段是独立问题，各自返回 `p_ai` + `confidence` + `action`，而不是一个不透明总分。
2. **概率 ≠ 置信度**：`ai_probability` 相对标签做温度校准；`confidence` 是证据量 / 分布集中度（长度、有效质量、投票分歧），**不要**和 `max(p)` 混为一谈。
3. **事后校准 + 正规分数**：对校准集拟合温度 `T`（logit 空间，`T>1` 冷却过自信），报告 ECE / Brier / 等质量 reliability bins（纯 numpy，见 `prob_metrics.py`）。
4. **按代价的行动阈值**：标注带 `flag` / `review` / `ignore`（默认：flag 若 p≥0.65 且 conf≥0.5；review 若 p≥0.4），不是单一 Human Score 砍一刀。
5. **隔离**：`lit_zh_*` / `mod_*` / `poe_*` 特征标注与句级 heat 并列出现在 `annotations[]`，互不吞并。

v3.2 补了三条过滤/约束，因为"能报出来"和"该报出来"是两件事：

```python
# 1) 本语域没有锚点 = 这个问题在这里不成立
if lc is not None and anc is None: continue
# 2) 校准没量出方向（|g|≈0）= 它对总分贡献是 0，对读者的影响不是 0
if anc is not None and abs(float(anc.g)) < 0.05: continue
# 3) 作用域对齐：文档已强判人类时，槽级不报"很像"
if act == "flag" and float(doc_p_ai) < 0.20: act = "review"
```

第 3 条值得单独说。`kind="feature"` 是**整篇的习惯统计**，和 Human Score 是**同一个作用域**；
句子/段落标注不是——它们的作用域是片段，本来就该和总分不一致。
同一个作用域的两个结论可以不同，但不该矛盾到要用户自己裁决。
实测触发点：鲁迅《秋夜》得分 98，却因为 `mod_para_uniformity=0.095` 报出「段落长度过于整齐 · 很像」。
现在它降为「可疑」，而 `ai_mod_01`（得分 1）的同一条仍然是「很像」——不对称是有意的。

另外 `_slot_local_p()` 现在尊重 `g` 的符号。之前忽略它，导致《影的告别》得分 100 的同时
列出「主语刻意复提 p_ai=1.0（很像）」——那个槽的 `disc=-1.252`，数据明明说它偏向人类。

还有一处更隐蔽的：**文档级温度不再套到槽级概率上**。
`lc.temperature` 是在文档后验上拟合的（目标是把 `human_score` 的 ECE 压下去），
而槽级 p 是另一类量，它的校准对象是"这个特征在此文档里的取值"。
借用文档的 T=0.425 会把槽级 p=0.79 抬到 0.958——一个 0.095 的特征值被说成"几乎确定"。
要校准槽级概率，得在槽级标签上单独拟合 T，不能借用文档的。

API / UI：

- 分析结果新增 `annotations: [{kind, key, name_zh, name_en, p_ai, confidence, action, span?}]`
- `calibration.json` → `langs.<lang>.temperature`（及 ece/brier before/after）、`langs.<lang>.notes`
- `calib_meta.register = {poetry_p, routed, masked_features}` —— 界面可以显示"这段被当成诗来量了"
- 前端「标注」页列出 flag/review；校准页显示 T / ECE / Brier；分数卡区分「AI概率」与「置信度」

---

## 6. 分数怎么读

```
Human Score ≈ round((1 − ai_probability) × 100)
```

| Human Score | 中文 | English |
|---|---|---|
| <20 | 高度疑似 AI | Strong AI |
| 20–39 | 疑似 AI | Likely AI |
| 40–59 | 人类与 AI 混合 | Mixed |
| 60–79 | 较符合人类写作 | Likely Human |
| ≥80 | 高度符合人类写作 | Strong Human |

**置信度**（`confidence`）回答的是另一个问题：这个分数值不值得信。

- 文本长度（≥500 字饱和）
- 有效质量占比（多少权重落在真有判别力的特征上）
- 距离 0.5 的分离度
- 校准是否 provisional
- 启发式 / ONNX / DNA 三个投票分歧

`confidence < 0.42` 或短于下限 → **`insufficient_evidence = true`**，UI 明示"证据不足"。这是 v3.0 最重要的诚实改动：v2.4 在 90 字上也会给一个理直气壮的分数。

**总分高 ≠ 没有 AI 成分。** v3.2 的标注清单是**文内相对排名**：
一段整体很像人写的文字里，仍可能有一两处特征突出。界面的 headline 现在会明说这一点，
不再出现"看不出明显的 AI 成分"和"1 处成分命中套板"并列的自我矛盾。

---

## 7. 安装与运行（Windows）

```bat
REM 1) 一次
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
REM 可选：.docx/.pdf 导入
pip install -r requirements-import.txt

REM 2) 启动（首次会自动 calibrate）
run_windows.bat
```

打开 `http://127.0.0.1:7860`。

| 入口 | 说明 |
|---|---|
| `run_windows.bat` | v3.0+ 前后端分离版（推荐） |
| `run_calibrate.bat` | 重算校准 + 留出法验证 |
| `run_gradio_legacy.bat` | 旧 Gradio 界面（需 `pip install gradio`） |
| `run_smoke.bat` | 回归测试 |
| `python calibrate.py --report` | 命令行看判别力表（三个桶都会打印） |
| `python eval_registers.py` | 语域分桶评测（v3.2 新增） |
| `python eval_registers.py --before` | 用 v3.1 的世界重跑同一批样本，做前后对比 |
| `python eval_registers.py --loo` | 语域桶留一法 |
| `python eval_registers.py --detail` | 逐条样本（看漏检/误报具体是哪一篇） |
| `python test_smoke.py` | 命令行回归 |

### 重抓语料

```bat
python eval_set\fetch_registers.py    REM 从 zh.wikisource.org 抓公有领域中文
python eval_set\make_ai_registers.py  REM 重写 AI 侧对照样本
python calibrate.py --report          REM 重算三个桶
```

`fetch_registers.py` 抓的是 `action=parse&variant=zh-hans`，会先整块剥 `<style>/<script>`，
再用 `CSS_JUNK` 正则和 ASCII 符号占比过滤排版残留。
**这一步不能省**——早期版本把 `.mw-parser-output` 的 CSS 卷进了正文，中文样本 `cjk_ratio`
掉到 0.03，检测器把它们全判成英文桶，得到一份"看起来好多了"的假数据。

### 可选增强

```bat
REM ONNX 第三票（EN ~77MB / ZH ~98MB）
pip install -r requirements-onnx.txt
python download_onnx_model.py
python download_onnx_model.py --model aigc-zh

REM Ollama 黑盒 DNA（hybrid 后端）+ 白盒 logprobs（需 logprobs 支持的 Ollama）
ollama pull qwen2:7b
```

---

## 8. 验证结果（真实跑出来的）

### 8.1 语域分桶：v3.1 → v3.2

同一批样本（人类 112 条公有领域 / AI 13 条手写），只换判定层：

| 语域 | v3.1 人类 P 中位 | v3.2 人类 P 中位 | v3.1 P≥.40 | v3.2 P≥.40 | v3.1 AUC | v3.2 AUC |
|---|---:|---:|---:|---:|---:|---:|
| 现代主义散文 | 0.296 | **0.029** | 32% | **0%** | 0.378 | **0.964** |
| 诗歌 | 0.103 | 0.106 | 0% | **0%** | 0.958 | **1.000** |
| 现实主义散文 | 0.406 | **0.026** | 51% | **0%** | 0.373 | **1.000** |

**误报：46/112 → 0/112**（P≥0.40 口径）。

AI 侧没有被牺牲掉：

| 语域 | v3.2 AI P 中位 | AI 最低 | P≥.40 | P≥.65 |
|---|---:|---:|---:|---:|
| 现代主义散文（6 条） | 0.922 | 0.052 | 83% | 83% |
| 诗歌（6 条） | 0.839 | 0.648 | 100% | 83% |
| 现实主义散文（1 条） | 0.711 | 0.711 | 100% | 100% |

### 8.2 留一法（`eval_registers.py --loo`）

| 桶 | n（人/AI） | LOO AUC | 人类 P 区间 | AI P 区间 |
|---|---|---:|---|---|
| `zh` | 110 / 23 | **0.979** | 0.050 – 0.471 | 0.195 – 0.990 |
| `zh_poetry` | 8 / 6 | **1.000** | 0.177 – 0.378 | 0.706 – 0.935 |

`zh` 桶最差的人类样本是 `realist_prose/luxun_guxiang_08.txt`（0.471）——刚好在 review 线 0.40 之上，
这是刻意的：宁可让人看见一条"可疑"，也不要把它压成 0 然后假装没问题。

### 8.3 `zh` 桶的判别力表（节选，`python calibrate.py --report`）

```
marketing_inflation                0.000   0.125   2.446   0.967
zh_policy_register                 0.000   0.000   1.570   0.864
lit_zh_linger_gaze                 0.000   0.220   1.536   0.857
burstiness_inv                     0.000   0.527   1.387   0.820
lit_zh_noun_reemphasize            0.000   0.000  -1.252  -0.779  <-FLIP
mod_clean_antithesis               0.000   0.065   1.091   0.721
mod_gloss_coda                     0.000   0.000   1.067   0.711
mod_punct_monotony                 0.000   0.667   1.056   0.707
mod_abstract_binome                0.000   0.414   1.010   0.687
mod_para_uniformity                0.000   0.000   0.979   0.673
mod_universal_coda                 0.000   0.000   0.906   0.638
mod_generic_simile                 0.000   0.000   0.741   0.549
lit_zh_flat_dialogue               0.000   0.000   0.138   0.114
lit_zh_mech_scene                  0.000   0.000  -0.132  -0.109  <-FLIP
```

七个新 `mod_*` 槽的 `g` 全部落在 **0.55–0.72**，没有一个被数据判死——它们不是凑数的。
同时两个旧文学槽被数据**反向否定**（`<FLIP`），这是预期的：它们衡量的正是现代主义会故意做的事。

### 8.4 `zh_poetry` 桶

```
mod_punct_monotony                 0.167   1.000   4.417   0.999
mod_universal_coda                 0.000   0.481   3.924   0.997
mod_para_uniformity                0.000   0.353   3.280   0.992
mod_abstract_binome                0.000   0.404   1.156   0.746
poe_abstract_line                  0.000   0.104   0.495   0.391
poe_prose_line                     0.312   0.044  -0.653  -0.248  <-FLIP
```

`poe_prose_line` 翻成负号，原因是人类诗（闻一多《死水》式的长行）比 AI 诗**更长**——
AI 诗倾向于短行、碎行。这个槽的方向由数据定，不由直觉定。

### 8.5 标注清单的实测行为

同一套代码，人类样本与 AI 样本的动作档位是**故意不对称**的：

```
poetry/dai_yuxiang_00.txt          score=89  bucket=zh_poetry
  整体不像生成文（89）；但有 2 处成分、1 句在文内相对突出——这是相对排名，不是判定。
     poe_prose_line      可疑  p=0.587   分行散文（伪诗）
     mod_punct_monotony  可疑  p=0.570   句末标点单一

modernist_prose/luxun_qiuye_00.txt score=98  bucket=zh
  整体不像生成文（98）；但有 2 处成分、1 句在文内相对突出——这是相对排名，不是判定。
     mod_para_uniformity 可疑  p=0.791   段落长度过于整齐
     mod_punct_monotony  可疑  p=0.668   句末标点单一

ai_modernist_prose/ai_mod_01.txt   score=1   bucket=zh
  很像生成文（1）：5 处成分命中了套板。
     mod_generic_simile  很像  p=0.985   抽象喻体比喻
     mod_gloss_coda      很像  p=0.985   自我解释/定义句
     mod_universal_coda  很像  p=0.985   结尾上升为格言
     mod_clean_antithesis 很像 p=0.985   工整对偶
     mod_punct_monotony  很像  p=0.802   句末标点单一
```

注意《秋夜》那条 `mod_para_uniformity` 的 **p 仍然是 0.791**（比 0.65 的 flag 线高），
只是动作被降成了「可疑」。数字不撒谎，动作讲代价——这两件事在 v3.1 就被分开了，
v3.2 只是让动作尊重作用域。

### 8.6 试过但撤掉的改动

`zh` 桶的 AI 侧混了公文/议论文，纯文学槽的上四分位常被拖到 0（`mod_gloss_coda` 的 `hi=0.000`，
而 AI 中位数是 0.286）。看着像 bug，试过把 `hi` 改成 `max(q75, mean_ai)`：
**误报从 0/112 变成 1/112**（`realist_prose/luxun_guxiang_08` 从 0.399 抬到 0.402），
AUC 0.964→0.959，而显示层毫无变化——因为 `scale = max(hi − lo, 0.10)` 在地板附近两种写法结果相同。
收益为零、代价为真，已撤回。`_anchor_from_samples` 的 docstring 里留了这段记录，免得下次有人再试一遍。

### 8.7 回归测试

```
python test_smoke.py  →  SMOKE OK
  AI_SLOP     human_score=2   ai_p=0.979
  Gettysburg  human_score=96
  ZH_HUMAN    human_score=92  clarity=78
  ZH_OFFICIAL human_score=0   gap_vs_alley=92
  ZH_LIT_AI   human_score=7   lit_fired=7  annotations(flag/review)=5
  engine=local_heuristic_v3.2
```

---

## 9. 竞品特征对照

### 9.0 与 [humanize-chinese](https://github.com/swaylq/humanize-chinese)

对照只列**双方文档或公开代码能直接核对**的维度。下表**不报**准确率 / AUC /「谁更强」——那些数字依赖各自校准集与测法，不能跨仓库横比。

| 维度 | humanize-chinese（其 README 自述） | 本仓库 `local_heuristic_v3.2` |
|---|---|---|
| 产品重心 | 检测 + **离线改写 / 降 AIGC**（六段式流水线） | **只做检测与改稿线索**；不内置 humanizer |
| 运行形态 | 纯标准库 CLI，零依赖；可选 CiLin | 本机 FastAPI + 启发式；可选 ONNX / Ollama |
| 句长变异 | 句长 **CV** + 短句占比；在 **HC3-Chinese** 对照上报告 Cohen's d | `burstiness_inv` / `window_burstiness_inv` 等；锚点来自本仓库 `calibration.json`（`en` / `zh` / `zh_poetry`） |
| 字符 trigram | 有：字符级 3-gram 困惑度（`ngram_freq_cn*.json`） | **主路径无** trigram / 困惑度表 |
| GLTR 风格 rank 分桶 | 有：按 n-gram 频率做 top-k 分桶（对标 [GLTR](https://arxiv.org/abs/1906.04043)） | **主路径无**；可选 Ollama 白盒 logprobs 为实验项，默认关 |
| 校准叙事 | 阈值在 **[HC3-Chinese](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)**（短问答）上校准；另有 academic / novel 等 scene | **不用 HC3 作锚点**；人类侧以公有领域文学/散文 + 自备 `calib/human_*`；AI 侧为助手生成样本（已标注） |
| 语域 / 场景 | `--scene general \| academic \| novel`（及 auto） | `register_zh.poetry_probability` → `zh` / `zh_poetry`；诗歌桶屏蔽句长类未定义特征 |
| 中文「工艺 / 套板」特征 | 规则层（模板句、高频词等）+ 统计层 + LR ensemble | 显式槽：`lit_zh_*`、`mod_*`、`poe_*`，经 `disc`/`g` 决定方向与是否入标注 |
| 输出形态 | 0–100 分；可接改写 | Human Score + `confidence` + `annotations[]`（flag / review / ignore） |
| 可复现自检 | 其仓库内 HC3 回归脚本 | `python test_smoke.py`、`python eval_registers.py` |

**怎么读这张表**

- humanize-chinese 把「测到」和「改掉」绑在同一工具链；本仓库刻意停在「指出可疑处」，避免把改写本身当成洗稿方案。
- 双方都提到句长起伏，但**实现与校准集不同**：一边是 HC3 问答上的 CV；一边是本机分位锚点 + 诗歌定义域屏蔽。不能拿一边的 d 值去解释另一边的分数。
- GLTR / HC3 是公开参照，不是本引擎的运行时依赖。本仓库对标其**问题设定**（哪类表面统计会骗人），不是复刻其公式。

### 9.1 反向借鉴：共识判决 vs 分歧信号

这一节借用 [aidetector.com](https://aidetector.com/) 的公开页（2026）来收紧本仓库已经写下的边界。对方口号只出现在引号里，不当本工具的 slogan。下面的百分比是**该站营销表或论文摘要里的话**。不跨站横比准确率，也不外推成中文文学检测的承诺。

**多票是信息，四检不是判决。** 首页原话是 “One detector is a guess. Four is a verdict.” 同页营销表写着：单检假阳约 1.3–5.0%，两检须一致约 0.36%，三检 triad 0.0073%，四检 “~0%”；另一张卡片把两检写成约 0.56%。页内注明这些假阳率 “derived from Hyatt et al. (2025)”，并称四检是在三检方法上再加一路信号。英雄区另标 “99.1% Accuracy Rate”。这些都是该站的营销表述。Hyatt 等人讨论的是降低一类 STEM 短文的假阳，不能外推成本仓库中文文学语域的准确率。本仓库若同时给出启发式、ONNX、DNA，读法固定为：分歧和一致都是信息；票面一致也不能定责。

论文场景以可打开的机构记录为准：[Using aggregated AI detector outcomes to eliminate false positives in STEM student writing](https://asu.elsevierpure.com/en/publications/using-aggregated-ai-detector-outcomes-to-eliminate-false-positive/)（Hyatt, Bienenstock, Firetto, Woods, Comus，*Advances in Physiology Education* 49(2):486–495，2025，DOI `10.1152/advan.00235.2024`）。摘要写的是本科解剖与生理课上的事实型短文（细胞膜如何选择透过；完成调查的参与者 n=190；检测子集为随机抽出的 n=50 名参与者的人写与 AI 写答卷，送四个检测器）。最佳检测器一侧的假阳约 1.3%，人类评分者约 5.0%；摘要称聚合后假阳 “nearly 0%” / “nearly zero”，用途是告知教师。那不是中文文学，也不是本仓库的校准集。期刊全文页在本次核对时直接请求返回 HTTP 403，正文不挂该链。检索到的结果节比营销表细：含 Copyleaks 的联合假阳写成 0%（图注写明这是观察数据上的计算，不是数学绝对零）；不含 Copyleaks 的 GPTZero–Detect GPT–Originality 三元组为 0.0073%；把 Detect GPT 并进四个检测器的平均后，假阳与假阴升到 5.5±4.2% 与 15.3±9.3%。首页的四检 “~0%” 对不上「四个检测器取平均」，也不是本工具的承诺。同站博客 [What are “False-Positives” in AI Writing Detection?](https://aidetector.com/blog/false-positives) 另写 348 篇（174 篇 AI + 174 篇人类）以及 Originality.ai 97.98%、DetectGPT 69.70%。那是博客自己的句子，和上面的摘要不是同一组样本，引用时分开。

**分歧是信号；混合分仍然只是改稿线索。** 首页还有一句 “Disagreement is the signal.”，后半句是 “When all four detectors flag a piece as AI, you can trust it.” 本仓库只保留前半的含义：不一致时提高过程证据的权重。字段以 `detector.py` 为准：

- `meta.voter_disagreement` 是已投票后端的极差，`max(votes) − min(votes)`。票始终含启发式后验；只有 `meta.onnx.used` 时才加入 ONNX 的 `ai_prob`；只有 `meta.ollama.used` 时才加入 DNA 的 `ai_like`（写在信号名 `ollama_dna_overlap` 上）。不足两票时该值为 0，表示没有第二票，不表示各检测器一致。
- 这个极差乘进 `confidence`（见 §6），并可能把文档打成 `insufficient_evidence`（`confidence < 0.42` 或短于下限）。它本身不是作者标签。网页在 `meta.voter_disagreement > 0.3` 时提示「启发式 / ONNX / DNA 意见不一致」（`web/app.js`）。
- 分析响应另返回 `voter_split` 与 `process_evidence_hint`。`voter_split` 列出每一路概率，包括已经算出但没有混进终分的 ONNX/DNA 旁证票；`process_evidence_hint.effective_spread` 取计入极差与旁证极差的较大者，用来提示先看过程证据。网页分票表会标出这些观察票。计分与 `voter_disagreement` 进入 `confidence` 的方式不变。
- 后端打开时，ONNX / DNA 仍可能被混合进 `ai_probability`。混合后的 Human Score 仍是改稿线索。分歧变大时去看过程证据，不要把混合结果当成作者标签。
- 句级和文档级各看各的作用域。`sentences[].heat` 取 `red` / `yellow` / `none`；`annotations[]` 里 `kind` 为 `sentence` 或 `paragraph` 的条目作用域是片段，`kind` 为 `feature` 的条目与 Human Score 同属整篇（§5）。片段与总分不一致时，同样去看过程证据，而不是把两边平均成一个作者结论。每条标注仍带自己的 `p_ai`、`confidence`、`action`（`flag` / `review` / `ignore`）。

**高风险场景先看过程，检测分只做改稿线索。** 上面那篇假阳博客在介绍多检的同时，把 Google Docs 版本历史、当场口头核对、要求引用到页码、按个人经验重设计作业写成办法。成绩、处分、雇佣这类场景里，这些过程证据优先；本仓库的分数只回答「成品表面哪里像公共风格签名、值不值得自己再改」。这与 §10.3「过程证据 ≠ 成稿判决」是同一条边界。博客 URL 留在这里，当作连检测商的公开说明也把过程证据写进建议的外部旁证。旁证不能给成稿分数授权。

首页 FAQ「Can AI detection be fooled?」的原文是 “While no tool is infallible…”；同页准确率问答另写 “over 90%”。两句都只属于该站。本次核对没有在首页或常见路径上找到该站自营的 Text Humanizer（`/humanizer`、`/ai-humanizer`、`/text-humanizer` 均为 HTTP 404）；博客在评测别家的 humanizer。本仓库继续不内置 humanizer。

| 维度 | aidetector.com（公开页，2026） | 本仓库 `local_heuristic_v3.2` |
|---|---|---|
| 产品重心 | 云端多检共识：内置检测免费，Originality.ai / Winston AI / GPTZero 按积分加开；另售抄袭、拼写、事实检查。句级色块、分句置信、PDF 报告。主叙事是聚合降假阳。不跟本仓库比准确率 | 离线启发式，槽级可解释。ONNX / DNA 可选，分歧进 `meta.voter_disagreement` 并下调 `confidence`。无内置 humanizer。多票只提供信息，拒绝作者判决 |

---

## 10. 诚实边界

怎么读分数，以本节为准。本仓库说的「AI 味」按 PASTA 的意思，近似后训练留下的公共风格签名；**检测器分数 ≠ 作者证明**。出处与用法在 §10.3。政策条件、过程证书与标量分数见 §10.4。

### 10.1 这一层不解决的问题（重要）

`register_zh.py` 的 docstring 里写得很直白，这里原样重复：

> The human side of the reference corpus is 1920s–40s public-domain Chinese.
> A living writer producing modernist prose in plain modern Chinese has no such
> reference — for that writer this layer will *not* fix the false positive.

翻译过来：**这次修的是"鲁迅们"的误判，不是"你"的误判。**
参照语料是 1920s–40s 的公有领域中文（中国著作权为作者终生 +50 年，所以只能取到这里）。
如果你用现代白话写现代主义散文，你的句法比鲁迅更接近当代语料，
那么你在锚点上的位置介于两边之间——这一层**帮不上忙**，甚至可能更糟。

唯一的修法是把你自己的稿子放进 `calib/human_zh/`，重跑 `calibrate.py`。
这也是为什么校准面板一直留着上传样本的入口。

### 10.2 两个已知的具体漏检

`eval_registers.py --detail` 里有两条 AI 样本没被抓住，**没有去调参追它们**：

**`ai_mod_06`（p=0.050, human=95）** —— 纯对话体：

```
「她说，那年下雪。」
「他说，记得。」
```

极短对答。这正是旧 `lit_zh_flat_dialogue` 想抓的形状，但该槽在 110 条人类中文上
`g=0.114`，接近无效——因为**人类的现代主义也这么写**（鲁迅《过客》全是短对答）。
对话体是 AI 与现代主义的**共同形状**，靠表层统计分不开。这是特征集的天花板，不是 bug。

**`ai_poe_06`（p=0.648, poe=0.10）** —— 书信体「致 XX」：

行首是称呼而非「X：」，`poetry_probability` 只给 0.10，没进诗歌桶，按散文量。
好在散文桶也给了 0.648（勉强过 review 线）。要修就得给"书信体"加一路由，
但那会把真正的书信散文也拖进诗歌桶——代价大于收益，所以留作已知边界。

### 10.3 使用边界（与行业共识对齐）

语气对齐 [Jisc / National Centre for AI：AI detection and assessment — an update for 2026（2026-10-15）](https://nationalcentreforai.jiscinvolve.org/wp/2026/10/15/ai-detection-and-assessment-an-update-for-2026/)：检测在进步，但问题本身更复杂；**更好的「检出 AI 参与」≠ 更好的「不当使用证据」**。

本仓库说的「AI 味」（套板、助手腔、特征槽点名的那层规律）不是作者指纹。按 [PASTA](https://aclanthology.org/2026.emnlp-main.708/)（Post-training Alignment Signature Targeted Ablation；Anand et al., EMNLP 2026，[arXiv:2605.30526](https://arxiv.org/abs/2605.30526)）的意思，它近似于**后训练留下的公共风格签名**（commodity / public-style signature）：对齐会把生成从人类语料风格推向一种更窄、跨模型可共享、检测器更容易看见的文体。PASTA 把这层规律当成可测量的后训练签名，而不是某一个人的笔迹。因此 **检测器分数 ≠ 作者证明**：高分只说明表面统计更靠近这层公共腔调，不能证明「这段是模型写的」；低分也不能证明「这段是人写的」。

助手改写还会把不同作者的文风收拢。[Malik 与 Awan](https://arxiv.org/abs/2608.00926) 把改写后作者归属准确率的下降定义为 Idiolect Erasure Rate（IER），并把「既难归因到人、又可能躲过现行检测器」称为 double erasure。风格往检测器放行的样子收敛时，检测会把剩下的差异反向定义成「像人」。本仓库没有测量 IER。低分和被改写抹平的表面一样，都不能当作者证明。多检分歧与高风险场景见 §9.1。

- **人机混写（mixed authorship）很难。** 学生/作者与模型共同改同一篇，已越来越常见。本工具的段落窗口与标注清单最多给出「哪一段相对突出」，**不能**把混写拆成可上法庭的作者比例，也**不能**区分「整段生成」与「人类起草后润色」。
- **过程证据 ≠ 成稿判决。** 键击回放、文档历史、过程监控（若你另用别的工具）说明的是写作过程；本仓库只看**成品文本的表面统计**。两套信号不可互换，更不能单凭其中一套定责。成绩、处分、雇佣场景里，版本历史、口头核对、作业设计优先于检测分；检测分只做改稿线索（§9.1）。政策如何改检测目标、证书能证明什么，见 §10.4。
- **改稿线索 ≠ 作者身份证明。** Human Score / `annotations[]` 回答的是「哪里像上述公共风格签名、值不值得自己再改」；**不回答**「谁写的」。禁止把本机分数当作指控、成绩处罚或解雇依据。
- **AI 侧参考样本由本助手生成**（`calib/ai_*/_meta.json` 里标 `generator: assistant-llm`）。检测**同源**文本时会偏极端。换一个模型的输出，分数可能没那么极端——这是校准集偏差，不是"真实世界准确率"。
- **`eval_set/` 的 AI 侧只有 13 条，是手写的**（`make_ai_registers.py`）。它们写在新特征设计**之前**，所以特征集不是照着这些字符串拟合的——但 13 条也不足以支撑"AI 文学写作"这个类别的统计结论。上面所有 AUC 都是**小样本 AUC**。
- **`zh_poetry` 桶的 ECE 是 0.1527**，比 `zh` 的 0.0489 差很多。14 条样本能拟合出的东西就这么多。诗歌的分数请当"提示"看，别当"概率"看。
- **白盒模块未经真机验证**。`backends/ollama_whitebox.py` 按 Ollama 文档实现（echo + logprobs），本沙箱没有 Ollama，没跑通过。默认关闭，开了也要先在你的机器上核对 `echo_coverage`。
- `sentence_start_variety_inv` / `comma_density` 在英文校准集上是 **FLIP**（数据说它们偏人类）。小样本下可能是语料现象，UI 标了 FLIP，别当规律用。
- 与 GPTZero / Turnitin / 朱雀 / Originality 会不一致——正常。人机混写、同义改写、自动 humanizer 都会让信号衰减或失效。不一致时提高过程证据的权重（§9.1），不要把多票混合结果当成作者标签。
- 适合**自己改稿**看"哪句像模板"，不适合给人贴标签。

### 10.4 设计注记：政策条件下的 AI 使用，过程证据 vs 成稿判决

本节只登记读法。计分、校准和 `honesty_signals.py` 都不动。

命运不是「用不用」一道选择题。同一段模型参与，依角色、任务和已经写明的规则，可以是允许的协助，也可以是禁止的委托。分数是公摊的风格信号（§10.3：后训练留下的公共风格签名）。承重在可追溯的过程，以及这段过程是否对齐那条规则。检测器分数仍不是作者判决。

**检测目标是角色与任务上的政策合规。** [Policy-Conditioned AI-Use Detection](https://arxiv.org/abs/2609.38427)（Diaz-Rodriguez 与 Jia）把场所规则收成检测的显式输入：要估的是某角色、某任务的流程是否符合已声明的规则，而不是成稿是否出自模型。报告给出假设、证据、校准情形与不确定性，代替 “AI detected” 这类判决。摘要还写到：在同行评审里，即便操作点很强，在看似合理的违规率下，被标出的合规作者仍可能多于违规者。本仓库的 Human Score 不接收政策输入，也不出具那种报告；它仍只回答成品表面是否靠近公共腔调。

**证书是过程证据。** [Humanly](https://arxiv.org/abs/2607.21758)（Zhu et al.）把一次会话封成证书，内容是活动日志、回放、键入，以及当时的环境配置。论文的局限写明：证书记录平台内观察到的活动，不是作者判决；不能证明意图，也不能排除每一条平台外渠道。页可见性只说明工作区被藏起或重新可见，不指明离开时用了哪个标签页、网站、应用或设备。抄写外部生成、口授，或经另一台设备送入，都不在证书里。这与 §10.3「过程证据 ≠ 成稿判决」是同一条。本仓库不记录过程，也不签发证书。

**标量分数是弱的部署界面。** [When AI-Text Detectors Become Evidence](https://aclanthology.org/2026.findings-emnlp.1539/)（AGENT-XTrust；Li et al., Findings of EMNLP 2026）把「按分类器评测，部署时人却把输出当作要检查、信任、争辩或据以升级的证据」称为 classification–evidence gap。标量分数藏起判断依据，招致过度信任，也可能把假阳变成指控。他们主张把 AI 文本检测评成带证据的决策支持，而不是作者证明。摘要里一项受控、固定刺激的用户研究写的是：分到完整 AGENT-XTrust 系统与界面包的参与者，决策准确率比仅分数包高 19.8 个点，比分数加解释包高 11.3 个点；在各条件下检测器判错的案例上，过度信任分别低 17.8 与 18.3 个点。这些差只属于那项研究的包对照，不是本仓库的准确率。

旁注：[PeerPrism](https://arxiv.org/abs/2604.14513)（Sadeghian et al., SIGIR 2026）把审稿拆成观点来源与表面文本。摘要称若干方法在人类对完全合成的二元任务上准确率高，到混合体制——观点来自人、表面文本由模型生成——预测急剧分歧，检测器经常给出互相矛盾的分类。分歧仍是信息，不是作者判决（§9.1）。

---

## 11. 文件一览

```
local_ai_detector/
  server.py                  # v3.0 后端（FastAPI）
  web/index.html styles.css app.js   # 前端（无构建）
  detector.py                # 判定内核 v3.2
  calibration.py             # 校准层（锚点 / 增益 / logistic / 语域选桶）
  calibrate.py               # 重算 + 判别力表 + 留出法验证（三个桶）
  calibration.json           # 校准产物（可重算，别手改）
  calibration.json.bak_v31   # v3.1 的旧校准，供 eval_registers.py --before 对照
  register_zh.py             # v3.2 语域层：诗歌路由 + 10 个工艺/诗歌特征
  zh_features.py             # 中文公文四槽 (v2.3)
  literary_zh_features.py    # 文学 AI 味八槽 (v2.4)
  compression_signals.py     # 压缩信号（v3.0 修正切窗）
  eval_registers.py          # v3.2 语域分桶评测（--before / --loo / --detail）
  eval_set/
    modernist_prose/         # 鲁迅《野草》逐篇（37 条，公有领域）
    poetry/                  # 戴望舒 / 闻一多 / 徐志摩（8 条）
    realist_prose/           # 鲁迅《呐喊》+ 萧红《呼兰河传》（67 条）
    ai_modernist_prose/      # 6 条手写 AI 现代主义
    ai_poetry/               # 6 条手写 AI 诗
    ai_realist_prose/        # 1 条（对照组）
    fetch_registers.py       # 从 zh.wikisource.org 重抓
    make_ai_registers.py     # 重写 AI 侧样本
    _manifest_human.json _manifest_ai.json
  calib/
    human_en/                # 公有领域英文（Gutenberg #1342/#1661/#98/#205）
    human_zh/                # 你自己的中文手稿（放这里！）
    ai_en/  ai_zh/           # 本助手生成的 AI 样本（已标注）
    _make_ai_samples.py      # 重新生成 AI 侧样本
  backends/ollama_client.py  # 黑盒 DNA 续写
  backends/ollama_whitebox.py# 白盒 echo-logprobs（实验，默认关）
  backends/onnx_classifier.py
  app.py                     # legacy Gradio
  weights.json               # 你的权重（可在 UI 里改）
  README_ZH.md  README_v24_backup.md
```

引擎标识：`local_heuristic_v3.2` / `local_hybrid_v3.2:*`

---

## 12. 版本摘要

- **v2**：压缩特征 + Ollama hybrid 骨架
- **v2.1**：DNA 不确定区间门控、避开 VL、记忆型降权
- **v2.2**：ONNX 第三票（ettin-68m）
- **v2.3**：中文四槽启发式 + 可选 aigc-zh ONNX
- **v2.4**：文学中文 AI 味八槽
- **v3.0**：逐特征分位校准 + 判别力增益 + 置信度/证据不足 + 句级分位双条件 + 段落窗口混写检测 + 前后端分离 + 文件导入 / A-B 对比 / 校准面板 + 白盒 logprobs（实验）
- **v3.1**：Jev 式并行标注（`annotations[]`）+ 温度校准 / ECE / Brier + 按代价的行动阈值（flag / review / ignore）
- **v3.2**：语域分层——诗歌路由（`register_zh.py`）、`zh_poetry` 校准桶、9 个句长类特征在诗里不设锚点、7 个工艺特征 `mod_*` + 3 个诗歌特征 `poe_*`、中文人类参照扩到 110 条公有领域语料、标注清单按"本语域是否有定义"过滤、小样本温度钳制。现代主义散文误报 46/112 → 0/112

---

## 13. 发行包

```bat
cd local_ai_detector_win
python make_release.py            REM 出包到 ../dist/local_ai_detector_v3.2_win.zip
python make_release.py --no-zip   REM 只铺开目录，便于先看一眼
```

发包是脚本不是手工拷贝，因为手工拷贝有两种必然的失效方式：

1. **漏文件**——少一个 `register_zh.py` 或 `calibration.json.bak_v31`，
   `eval_registers.py --before` 不会报错，只会安静地少一个桶；
2. **带垃圾**——`.venv`(514M) + `models/onnx`(176M) 会让包从 0.36M 变成 690M。

`make_release.py` 里 `TOP_FILES` 是**白名单**（新增文件默认不进包，逼你做一次判断），
每个排除项都写了理由。包内附 `VERSION.txt`（含三个桶的样本数）和 `SHA256SUMS.txt`。
随包还带 `发行说明.md`——装法、自检命令、期望数字、诚实边界（含人机混写 / 过程≠成稿 / 「AI 味」≈ 后训练公共风格签名 / 检测器分数 ≠ 作者证明 / 多票分歧不是作者判决）。

包里**不含** ONNX 权重与 Ollama。它们只是"第三票"，缺了会软失败
（`probe()` 返回 `available=False`），主路径不受影响。
