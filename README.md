# 本地 AI 写作信号检测器

v3.2，引擎 `local_heuristic_v3.2`。离线的中文/英文 AI 写作信号检测工具：给的是改稿线索，不是作者证明。不调用远端 API，不上传文字，不需要账号。

完整说明见 [README_ZH.md](README_ZH.md)。装法、自检和边界见 [发行说明.md](发行说明.md)。

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
