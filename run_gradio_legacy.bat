@echo off
REM v3.0 legacy Gradio UI (kept for the old muscle memory)
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
pip install "gradio>=4.44.0,<6"
if not exist calibration.json python calibrate.py
python app.py
pause
