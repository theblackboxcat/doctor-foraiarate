# 本地 AI 写作信号检测器

v3.2，引擎 `local_heuristic_v3.2`。离线的中文/英文 AI 写作信号检测工具：给的是改稿线索，不是作者证明。不调用远端 API，不上传文字，不需要账号。

完整说明见 [README_ZH.md](README_ZH.md)（含 §9 竞品特征对照、§9.1 反向借鉴：共识判决 vs 分歧信号、§10 诚实边界；边界语气对齐 [Jisc 2026-10-15 AI detection update](https://nationalcentreforai.jiscinvolve.org/wp/2026/10/15/ai-detection-and-assessment-an-update-for-2026/)）。装法、自检和边界见 [发行说明.md](发行说明.md)。

## 反向借鉴（短）

多票是分歧或一致的信息，不是作者判决。公开站 [aidetector.com](https://aidetector.com/) 的「四检 ≈ 0 误报」是该站营销表，对应的论文场景是 Hyatt 等人的本科解剖生理事实型短文，不能外推成本仓库的中文文学准确率。启发式 / ONNX / DNA 不一致时看 `meta.voter_disagreement`（它压低 `confidence`），并优先采用过程证据。全文见 [README_ZH.md §9.1](README_ZH.md)。

## The same boundary in English

Multiple votes are information about agreement or disagreement. They are not an authorship verdict. The public “four detectors ≈ 0% false positives” table on [aidetector.com](https://aidetector.com/) is that site’s marketing. The underlying study is Hyatt et al.’s undergraduate anatomy-and-physiology fact essays, and it is not an accuracy promise for this detector on Chinese literary text. When the heuristic, ONNX, and DNA votes disagree, read `meta.voter_disagreement` (it lowers `confidence`) and prefer process evidence. Full write-up: [README_ZH.md §9.1](README_ZH.md).

## 安装与运行（Windows / Python 3.10+）

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
run_windows.bat
```

打开 `http://127.0.0.1:7860`。`calibration.json` 已随仓库提供，首次启动不会重算。

自检：`python test_smoke.py`（期望最后一行 `SMOKE OK`）。

## 不在仓库里的东西

ONNX 权重（`models/onnx/*/model.onnx`）不入库。需要第三票时：

```bat
pip install -r requirements-onnx.txt
python download_onnx_model.py
python download_onnx_model.py --model aigc-zh
```

也可以用 `download_onnx.bat`。没有这些权重时主路径仍可用，ONNX 探测会软失败。分词器、`models/onnx/README.txt` 和各目录的 `.gitkeep` 已在仓库中。
