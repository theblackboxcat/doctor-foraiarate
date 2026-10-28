@echo off
REM Rebuild calibration.json from calib/ and show the discriminability table
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python calibrate.py --report
echo.
echo Leave-one-out validation:
python calibrate.py --validate
pause
