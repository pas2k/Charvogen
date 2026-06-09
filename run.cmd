@echo off

if not defined PYTHON (set PYTHON="%LocalAppData%\Microsoft\WindowsApps\py.exe")
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")
if not defined VIRTUAL_ENV (set "VIRTUAL_ENV=%~dp0venv")

%PYTHON% -m gui.backend
