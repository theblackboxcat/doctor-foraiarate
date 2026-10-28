@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
  echo Create venv first: run_windows.bat once, or python -m venv .venv
  pause
  exit /b 1
)
pip install -r requirements-onnx.txt
python download_onnx_model.py
echo.
echo Optional Chinese ONNX (~98MB):
echo   python download_onnx_model.py --model aigc-zh
echo.
echo Done. Re-run app.py / run_windows.bat. Probe ONNX in the GUI.
pause
