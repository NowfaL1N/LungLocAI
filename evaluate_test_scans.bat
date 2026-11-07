@echo off
REM Score the website on the test scans (detection + cancer classification)
cd /d %~dp0
if not exist .venv\Scripts\python.exe (echo Run run.bat once first to install everything. & pause & exit /b 1)
.venv\Scripts\python tools\evaluate.py --scans test_scans --annotations test_scans\annotations.csv
echo.
echo Summary saved to test_scans\evaluation_summary.txt
pause
