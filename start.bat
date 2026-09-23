@echo off
cd /d "%~dp0"
python --version >nul 2>&1 || (echo [ERROR] Python not found on PATH. & pause & exit /b 1)
echo ==================================================
echo   POI Governance Chain  (Laya + LLM)
echo   http://127.0.0.1:8000     Ctrl+C to stop
echo ==================================================
echo.
start /b cmd /c "ping 127.0.0.1 -n 4 >nul & start http://127.0.0.1:8000"
python run.py
