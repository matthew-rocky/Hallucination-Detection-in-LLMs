@echo off
setlocal

set "ROOT=%~dp0"
set "FRONTEND=%ROOT%frontend"

if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%ROOT%.venv\Scripts\python.exe"
) else (
    set "PYTHON_CMD=python"
)

echo Starting Hallucination Detection Studio...
echo.

start "Hallucination Detection Studio - FastAPI" /D "%ROOT%" cmd /k ""%PYTHON_CMD%" -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"

if exist "%FRONTEND%node_modules" (
    start "Hallucination Detection Studio - Next.js" /D "%FRONTEND%" cmd /k "npm run dev"
) else (
    start "Hallucination Detection Studio - Next.js" /D "%FRONTEND%" cmd /k "npm install && npm run dev"
)

echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:3000

endlocal