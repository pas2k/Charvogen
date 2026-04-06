@echo off

if not defined PYTHON (set PYTHON="%LocalAppData%\Microsoft\WindowsApps\py.exe")
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

%PYTHON% -m gui.backend
