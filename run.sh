#!/bin/bash

# AI 文本檢測器 - 快速啟動腳本

echo "======================================="
echo "🤖 AI 文本檢測器 - 啟動中..."
echo "======================================="
echo ""

# 檢查 Python 是否安裝
if ! command -v python3 &> /dev/null; then
    echo "❌ 錯誤：未找到 Python 3"
    echo "請先安裝 Python 3.8 或以上版本"
    exit 1
fi

echo "✅ Python 版本："
python3 --version
echo ""

# 檢查虛擬環境
if [ ! -d "venv" ]; then
    echo "📦 未找到虛擬環境，正在創建..."
    python3 -m venv venv
    echo "✅ 虛擬環境創建完成"
    echo ""
fi

# 啟動虛擬環境
echo "🔄 啟動虛擬環境..."
source venv/bin/activate

# 檢查依賴
echo "📋 檢查依賴套件..."
if ! pip show streamlit > /dev/null 2>&1; then
    echo "📥 安裝依賴套件（首次運行需要幾分鐘）..."
    pip install -r requirements.txt
    echo "✅ 依賴套件安裝完成"
else
    echo "✅ 依賴套件已安裝"
fi
echo ""

# 啟動應用
echo "======================================="
echo "🚀 正在啟動 Streamlit 應用..."
echo "======================================="
echo ""
echo "💡 提示："
echo "  - 首次使用需下載模型（約 500MB）"
echo "  - 應用會自動在瀏覽器中開啟"
echo "  - 按 Ctrl+C 停止應用"
echo ""

streamlit run app.py
