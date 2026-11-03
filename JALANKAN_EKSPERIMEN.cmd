@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto ready
where py >nul 2>nul
if errorlevel 1 goto pythonfallback
py -3.12 -m venv .venv
if not errorlevel 1 goto ready
:pythonfallback
python -m venv .venv
if errorlevel 1 (
  echo Pasang Python 3.12 dari https://www.python.org/downloads/ lalu buka CMD baru.
  exit /b 1
)
:ready
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" scripts\download_dataset.py
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" scripts\run_project.py
exit /b %errorlevel%
