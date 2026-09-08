@echo off
setlocal
cd /d "%~dp0"
python serve_v4.py
if errorlevel 1 pause
