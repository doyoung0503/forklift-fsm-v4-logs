@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

py -3 -u preflight_v4.py
if errorlevel 1 (
    echo.
    echo Preflight failed. Fix the reported items before running FSM v4.
    pause
    exit /b 1
)

echo.
echo Check CAN_ENABLED in depth_cam\calib\fsm_v4\config.py before starting.
echo CAN ON starts SEARCH with an in-place right turn; CAN OFF is monitor-only.
choice /C YN /N /M "Start FSM v4 now? [Y/N]: "
if errorlevel 2 exit /b 0

cd /d "%~dp0depth_cam"
py -3 -u main_rec_v4.py
set "fsm_exit=%errorlevel%"
echo.
echo FSM v4 exited with code %fsm_exit%.
pause
exit /b %fsm_exit%
