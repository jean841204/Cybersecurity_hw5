@echo off
REM AI 文本檢測器 - Windows 快速啟動腳本

echo =======================================
echo 🤖 AI 文本檢測器 - 啟動中...
echo =======================================
echo.

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤：未找到 Python
    echo 請先安裝 Python 3.8 或以上版本
    pause
    exit /b 1
)

echo ✅ Python 版本：
python --version
echo.

REM 檢查虛擬環境
if not exist "venv" (
    echo 📦 未找到虛擬環境，正在創建...
    python -m venv venv
    echo ✅ 虛擬環境創建完成
    echo.
)

REM 啟動虛擬環境
echo 🔄 啟動虛擬環境...
call venv\Scripts\activate.bat

REM 檢查依賴
echo 📋 檢查依賴套件...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo 📥 安裝依賴套件（首次運行需要幾分鐘）...
    pip install -r requirements.txt
    echo ✅ 依賴套件安裝完成
) else (
    echo ✅ 依賴套件已安裝
)
echo.

REM 啟動應用
echo =======================================
echo 🚀 正在啟動 Streamlit 應用...
echo =======================================
echo.
echo 💡 提示：
echo   - 首次使用需下載模型（約 500MB）
echo   - 應用會自動在瀏覽器中開啟
echo   - 按 Ctrl+C 停止應用
echo.

streamlit run app.py
