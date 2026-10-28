Place downloaded ONNX weights here (see download_onnx_model.py).

EN (default 3rd vote):
  models/onnx/ettin-68m/{model.onnx,tokenizer.json}
  python download_onnx_model.py

ZH (optional, used when CJK≥35%):
  models/onnx/aigc-zh/{model.onnx,tokenizer.json}
  python download_onnx_model.py --model aigc-zh

Zip packages omit large model blobs; heuristic runs without them.
