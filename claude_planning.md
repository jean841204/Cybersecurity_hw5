# AI 文本檢測器開發規劃

## 專案概述
使用 Streamlit 開發一個 AI 文本檢測器，用於辨識文字內容是否由 AI 生成。類似於之前的 spam 郵件檢測專案，但目標是檢測 AI 生成的文本。

## 參考網站分析
- **網站**: JustDone AI Detector
- **核心功能**:
  - 文本輸入與檢測
  - 詳細分析報告
  - AI 生成內容標記
  - 信心度評分

## 技術棧
- **前端框架**: Streamlit
- **程式語言**: Python 3.8+
- **機器學習**:
  - Scikit-learn（如果訓練自己的模型）
  - Transformers（Hugging Face 預訓練模型）
  - 可選：OpenAI API / GPTZero API
- **數據處理**: Pandas, NumPy
- **文本處理**: NLTK, spaCy
- **視覺化**: Matplotlib, Plotly

## 專案檔案結構
```
hw5/
├── app.py                  # Streamlit 主應用程式
├── requirements.txt        # 依賴套件
├── claude_planning.md      # 本規劃文件
├── models/                 # 模型目錄
│   ├── detector_model.pkl  # 訓練好的模型
│   └── vectorizer.pkl      # 文本向量化器
├── data/                   # 數據目錄（可選）
│   ├── train/             # 訓練數據
│   └── test/              # 測試數據
├── utils/                  # 工具函數
│   ├── __init__.py
│   ├── text_analyzer.py    # 文本分析工具
│   └── model_loader.py     # 模型載入工具
└── README.md              # 專案說明
```

## 開發階段規劃

### 階段 1: 環境設置與基礎架構 (預計 1 天)
**任務清單:**
1. 創建專案目錄結構
2. 安裝必要的 Python 套件
3. 建立 requirements.txt
4. 設置 Streamlit 基礎框架

**具體步驟:**
- 初始化虛擬環境: `python -m venv venv`
- 安裝核心套件: `streamlit`, `pandas`, `numpy`
- 創建簡單的 Streamlit 測試頁面

### 階段 2: 選擇檢測方法 (預計 1 天)
**方案 A: 使用預訓練模型（推薦初學者）**
- Hugging Face 的 `roberta-base-openai-detector`
- GPTZero API
- OpenAI Text Classifier

**方案 B: 訓練自定義模型**
- 收集 AI 生成與人類撰寫的文本數據集
- 使用傳統 ML 方法（如之前的 spam 檢測）
- 特徵工程：詞頻、句子長度、困惑度等

**方案 C: 混合方法**
- 結合多個檢測指標
- 使用規則基礎 + ML 模型

**建議**: 先從方案 A 開始，使用 Hugging Face 預訓練模型

### 階段 3: 核心功能開發 (預計 2-3 天)

#### 3.1 文本輸入介面
- 文本框輸入（支援大段文字）
- 文件上傳功能（.txt, .docx）
- 字數限制與驗證

#### 3.2 AI 檢測邏輯
```python
# 核心功能架構
def detect_ai_text(text):
    """
    檢測文本是否由 AI 生成

    Returns:
        - ai_probability: float (0-1)
        - confidence: str (High/Medium/Low)
        - details: dict (詳細分析)
    """
    pass
```

#### 3.3 分析特徵（參考 spam 檢測經驗）
- **統計特徵**:
  - 平均句子長度
  - 詞彙多樣性（Type-Token Ratio）
  - 標點符號使用頻率

- **語言特徵**:
  - 困惑度（Perplexity）
  - 連貫性分數
  - 重複短語檢測

- **AI 特徵**:
  - 過度正式或完美的語法
  - 缺乏個人風格
  - 典型 AI 用語檢測

### 階段 4: 結果呈現介面 (預計 1-2 天)

#### 4.1 檢測結果展示
- **機率儀表板**:
  - 使用 Streamlit Progress Bar 或 Gauge Chart
  - 顯示 AI 生成機率 (0-100%)

- **信心等級**:
  - 🟢 高信心度 (>80%)
  - 🟡 中等信心度 (50-80%)
  - 🔴 低信心度 (<50%)

#### 4.2 詳細報告
- 句子級別分析（標記可疑句子）
- 文本統計摘要
- 視覺化圖表（詞頻、句長分佈等）

#### 4.3 比較功能
- 顯示與典型 AI/人類文本的差異
- 可下載檢測報告（PDF/CSV）

### 階段 5: 優化與測試 (預計 1-2 天)

#### 5.1 性能優化
- 快取機制（`@st.cache_data`）
- 批次處理大文件
- 異步處理（如果需要）

#### 5.2 UI/UX 改進
- 響應式設計
- 載入動畫
- 錯誤處理與用戶提示
- 多語言支援（中英文）

#### 5.3 測試
- 單元測試核心函數
- 使用已知 AI 生成文本測試
- 使用人類撰寫文本測試
- 邊界案例測試

### 階段 6: 部署與文檔 (預計 1 天)

#### 6.1 部署選項
- Streamlit Cloud（最簡單）
- Heroku
- Docker 容器化

#### 6.2 文檔撰寫
- README.md 使用說明
- 程式碼註釋
- API 文檔（如果有）

## 詳細實作步驟

### Step 1: 創建 Streamlit 基礎應用
```python
# app.py 基本結構
import streamlit as st

st.set_page_config(page_title="AI 文本檢測器", page_icon="🤖")

st.title("🤖 AI 文本檢測器")
st.write("檢測文字內容是否由 AI 生成")

# 側邊欄
with st.sidebar:
    st.header("設定")
    sensitivity = st.slider("檢測靈敏度", 0, 100, 50)

# 主要內容區
text_input = st.text_area("請輸入要檢測的文字", height=200)

if st.button("開始檢測"):
    if text_input:
        # 檢測邏輯
        st.success("檢測完成！")
    else:
        st.warning("請輸入文字")
```

### Step 2: 整合檢測模型

**選項 1: 使用 Hugging Face 模型**
```python
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("text-classification",
                   model="roberta-base-openai-detector")

detector = load_model()

def detect_ai(text):
    result = detector(text)
    return result[0]
```

**選項 2: 特徵工程方法（類似 spam 檢測）**
```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def extract_features(text):
    features = {
        'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()),
        'avg_sentence_length': len(text.split()) / len(text.split('.')),
        'punctuation_ratio': sum(c in '.,!?' for c in text) / len(text),
        # ... 更多特徵
    }
    return features

def predict(text):
    features = extract_features(text)
    # 使用訓練好的模型預測
    probability = model.predict_proba([features])[0][1]
    return probability
```

### Step 3: 視覺化結果
```python
import plotly.graph_objects as go

def show_result(probability):
    # 建立儀表板
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        title = {'text': "AI 生成機率"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"}
            ],
        }
    ))
    st.plotly_chart(fig)

    # 解釋結果
    if probability > 0.8:
        st.error("⚠️ 此文本很可能由 AI 生成")
    elif probability > 0.5:
        st.warning("⚡ 此文本可能包含 AI 生成內容")
    else:
        st.success("✅ 此文本很可能由人類撰寫")
```

## 數據集建議

### 公開數據集
1. **HC3 Dataset** - 人類與 ChatGPT 比較數據集
2. **GPT-wiki-intro** - Wikipedia 介紹與 GPT-3 生成比較
3. **TuringBench** - 圖靈測試數據集
4. **自行收集**:
   - 人類文本：新聞文章、部落格、論壇
   - AI 文本：ChatGPT、Claude、GPT-3 生成內容

### 數據標註
- 標籤：`human` (0) vs `ai` (1)
- 平衡數據集（50/50 分佈）
- 訓練/測試分割：80/20

## 關鍵技術挑戰

### 挑戰 1: 模型準確度
- **問題**: AI 模型不斷進化，檢測難度增加
- **解決方案**:
  - 定期更新訓練數據
  - 使用多個檢測器投票機制
  - 結合統計與深度學習方法

### 挑戰 2: 混合文本
- **問題**: 人類編輯過的 AI 文本或 AI 輔助寫作
- **解決方案**:
  - 句子級別分析
  - 顯示機率範圍而非絕對判斷
  - 標記可疑段落

### 挑戰 3: 不同語言與領域
- **問題**: 中文、技術文檔、學術論文檢測差異大
- **解決方案**:
  - 針對不同領域訓練專門模型
  - 用戶可選擇文本類型

## 評估指標

### 模型評估
- **準確率 (Accuracy)**: 整體正確率
- **精確率 (Precision)**: 預測為 AI 中真正是 AI 的比例
- **召回率 (Recall)**: 實際 AI 文本被正確識別的比例
- **F1 分數**: 精確率與召回率的調和平均
- **ROC-AUC**: 模型區分能力

### 目標指標
- 準確率 > 85%
- F1 分數 > 0.80
- 處理時間 < 2 秒（1000 字文本）

## 擴展功能（選做）

### 進階功能
1. **批次檢測**: 上傳多個文件同時檢測
2. **API 介面**: 提供 REST API 供其他應用使用
3. **瀏覽器擴充功能**: Chrome/Firefox 插件
4. **歷史記錄**: 保存檢測歷史
5. **文本改寫建議**: 如果檢測為 AI，提供人性化改寫建議
6. **多模型比較**: 顯示不同檢測模型的結果
7. **學習模式**: 用戶可標註結果幫助模型學習

### 整合功能
- Turnitin API
- Grammarly 整合
- Google Docs 插件

## 時程規劃總結

| 階段 | 任務 | 預計時間 | 優先級 |
|------|------|----------|--------|
| 1 | 環境設置與基礎架構 | 0.5-1 天 | 高 |
| 2 | 選擇檢測方法 | 1 天 | 高 |
| 3 | 核心功能開發 | 2-3 天 | 高 |
| 4 | 結果呈現介面 | 1-2 天 | 高 |
| 5 | 優化與測試 | 1-2 天 | 中 |
| 6 | 部署與文檔 | 1 天 | 中 |

**總預計時間**: 6.5-10 天

## 學習資源

### 推薦教學
1. **Streamlit 官方文檔**: https://docs.streamlit.io/
2. **Hugging Face Transformers**: https://huggingface.co/docs/transformers/
3. **AI 文本檢測研究論文**:
   - "DetectGPT: Zero-Shot Machine-Generated Text Detection"
   - "GLTR: Statistical Detection and Visualization of Generated Text"

### 相關專案參考
- GPTZero: https://gptzero.me/
- AI Text Classifier (OpenAI)
- Writer AI Content Detector

## 下一步行動

### 立即開始
1. ✅ 創建專案規劃文件（本文件）
2. ⬜ 設置開發環境
3. ⬜ 安裝必要套件
4. ⬜ 選擇檢測方法並進行小規模測試
5. ⬜ 開發 Streamlit 基礎介面

### 決策點
**需要決定的事項:**
- [ ] 使用預訓練模型還是自訓練模型？
- [ ] 是否需要支援文件上傳？
- [ ] 檢測結果要多詳細（簡單/中等/詳盡）？
- [ ] 是否需要中英文雙語支援？
- [ ] 是否需要保存檢測歷史？

---

## 備註
- 本規劃基於參考網站 JustDone AI Detector 的功能分析
- 可根據實際開發進度調整各階段時間
- 建議採用敏捷開發，先完成 MVP（最小可行產品）再逐步增加功能
- 參考之前的 spam 檢測專案經驗，可重用數據處理和模型訓練流程

**最後更新**: 2025-11-30
