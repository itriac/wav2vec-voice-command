@echo off
chcp 65001 >nul
title 離線語音指令系統 — wav2vec2
echo.
echo  ================================================
echo   離線語音指令系統  (wav2vec 2.0)
echo  ================================================
echo.

where uv >nul 2>&1
if errorlevel 1 (
    echo  [提示] 找不到 uv，正在安裝...
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo  [錯誤] uv 安裝失敗，請手動安裝：https://docs.astral.sh/uv/
        pause
        exit /b 1
    )
    echo.
)

echo  正在同步依賴套件（首次需下載 PyTorch 等套件）...
uv sync
if errorlevel 1 (
    echo  [錯誤] 依賴安裝失敗
    pause
    exit /b 1
)

echo.
echo  啟動伺服器於 http://127.0.0.1:3001
echo  首次執行將下載 wav2vec2 模型 (~1.2 GB)
echo  Ctrl+C 停止伺服器
echo.
uv run server.py
pause
