@echo off
title Proxemix
cd /d "%~dp0"

echo.
echo   Proxemix — starting the demo on the plaza footage.
echo.
echo   Your browser will open in a few seconds at http://127.0.0.1:8000
echo   Leave this window open while you use it. Closing it stops the app.
echo.

start "" /b cmd /c "timeout /t 5 >nul & start "" http://127.0.0.1:8000"
py -m proxemix.serve --bundle data/plaza_bundle --port 8000

echo.
echo   Proxemix has stopped.
pause
