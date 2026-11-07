@echo off
REM Start LungLocAI on http://localhost:8000 (Windows)
title LungLocAI server
cd /d %~dp0

if exist .venv\Scripts\python.exe goto run

echo Looking for Python...
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY (python -c "import sys" >nul 2>nul && set PY=python)
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set PY="%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set PY="%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\python.exe" set PY="%LOCALAPPDATA%\anaconda3\python.exe"
if not defined PY if exist "C:\ProgramData\anaconda3\python.exe" set PY="C:\ProgramData\anaconda3\python.exe"
if not defined PY if exist "C:\ProgramData\Anaconda3\python.exe" set PY="C:\ProgramData\Anaconda3\python.exe"
if not defined PY (
  echo Python 3.9+ was not found. Install it from https://www.python.org/downloads/ and run this file again.
  pause
  exit /b 1
)
echo Using %PY%
echo Creating a private Python environment and installing packages (first run only, a few minutes)...
%PY% -m venv .venv || (echo Could not create the environment. & pause & exit /b 1)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || (echo Package install failed - check your internet connection. & pause & exit /b 1)

:run
echo.
echo ==========================================================
echo   LungLocAI is starting. Open  http://localhost:8000
echo   Keep this window open. Close it to stop the website.
echo ==========================================================
start "" /min cmd /c "timeout /t 10 >nul & start http://localhost:8000"
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
