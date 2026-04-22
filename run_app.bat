@echo off
setlocal
cd /d "%~dp0"
title Aptamer QSAR Tool - Local Launcher

echo ============================================================
echo  Aptamer QSAR Tool
echo  Local Windows Launcher
echo ============================================================

set "PY="
for %%V in (3.13 3.12 3.11 3.10) do (
  py -%%V -c "import sys" >nul 2>nul
  if not errorlevel 1 if not defined PY set "PY=py -%%V"
)
if not defined PY set "PY=python"

if not exist ".venv\Scripts\python.exe" (
  echo Creating local Python environment...
  %PY% -m venv .venv
)

set "VENV_PY=.venv\Scripts\python.exe"
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.txt

echo Starting Aptamer QSAR Tool at http://localhost:8503 ...
start "" http://localhost:8503
"%VENV_PY%" -m streamlit run app.py --server.port 8503 --server.headless true --browser.gatherUsageStats false
pause
