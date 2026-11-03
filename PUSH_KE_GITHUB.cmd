@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\push_to_github.ps1"
if errorlevel 1 (
  echo.
  echo Proses belum selesai. Baca pesan kesalahan di atas.
  exit /b 1
)
endlocal
