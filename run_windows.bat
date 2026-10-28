@echo off
REM v3.0 launcher: starts the FastAPI + web frontend on http://127.0.0.1:7860
REM Legacy Gradio UI: run `python app.py` instead.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate.bat
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
REM First run: build calibration.json from calib/ (safe to re-run any time)
if not exist calibration.json (
  python calibrate.py
)
python server.py
pause
