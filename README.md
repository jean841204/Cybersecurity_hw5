# 🤖 AI 文本檢測器

使用 Streamlit 和預訓練模型開發的 AI 文本檢測應用，能夠辨識文字內容是否由 AI 生成。

## 📋 專案簡介

本專案是中興大學資訊安全課程作業，參考 [JustDone AI Detector](https://justdone.com/ai-detector) 的功能，實作一個能夠檢測文本是否由 ChatGPT、Claude 等 AI 工具生成的應用程式。

### 主要功能

- ✅ AI 文本檢測（使用預訓練模型）
- ✅ 詳細統計分析（詞頻、句長、詞彙多樣性等）
- ✅ 視覺化結果呈現（機率儀表板）
- ✅ 文件上傳支援（.txt 格式）
- ✅ 快速與詳細兩種檢測模式
- ✅ 執行時間優化（快取機制、文本長度限制）

### 技術特點

- **模型快取**：使用 `@st.cache_resource` 確保模型只載入一次
- **結果快取**：使用 `@st.cache_data` 快取相同文本的檢測結果
- **長度限制**：自動截斷過長文本避免處理時間過長
- **進度顯示**：即時顯示檢測進度
- **響應式介面**：清晰的 UI/UX 設計

## 🚀 快速開始

### 環境需求

- Python 3.8 或以上
- pip（Python 套件管理器）
- 網路連接（首次使用需下載模型）

### 安裝步驟

1. **克隆或下載專案**
```bash
cd /Users/jessica/Desktop/NCHU/Cybersecurity/hw5
```

2. **建立虛擬環境（建議）**
```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

3. **安裝依賴套件**
```bash
pip install -r requirements.txt
```

4. **運行應用程式**
```bash
streamlit run app.py
```

5. **開啟瀏覽器**

應用會自動在瀏覽器中開啟，預設地址：`http://localhost:8501`

## 📖 使用說明

### 基本使用

1. **輸入文本**
   - 方式一：直接在文本框中輸入或貼上文字
   - 方式二：上傳 .txt 文件

2. **調整設定（側邊欄）**
   - 設置最大處理字數（100-2000 詞）
   - 選擇檢測模式（快速/詳細）

3. **開始檢測**
   - 點擊「開始檢測」按鈕
   - 等待分析完成（通常 1-3 秒）

4. **查看結果**
   - AI 生成機率（0-100%）
   - 判定結果（AI 生成 / 人類撰寫）
   - 信心等級（高/中/低）
   - 詳細統計分析（詳細模式）

### 結果解讀

| AI 機率 | 判定 | 建議 |
|---------|------|------|
| > 80% | 高度可疑 | 很可能由 AI 生成，建議深入調查 |
| 50-80% | 中度可疑 | 可能包含 AI 內容或 AI 輔助寫作 |
| < 50% | 低度可疑 | 更像人類撰寫，AI 生成可能性低 |

### 性能優化建議

1. **控制文本長度**
   - 建議 100-800 詞以內
   - 過長文本會自動截斷

2. **使用快取**
   - 重複檢測相同文本會使用快取結果
   - 速度提升 10-100 倍

3. **首次使用**
   - 第一次運行需下載模型（約 500MB）
   - 下載時間視網路速度而定（3-10 分鐘）
   - 後續使用無需重新下載

## 📁 專案結構

```
hw5/
├── app.py                      # Streamlit 主應用程式
├── requirements.txt            # Python 依賴套件
├── claude_planning.md          # 專案規劃文件
├── README.md                   # 本文件
├── .gitignore                  # Git 忽略文件
├── .streamlit/
│   └── config.toml            # Streamlit 配置
├── utils/
│   ├── __init__.py
│   ├── model_loader.py        # 模型載入與預測
│   └── text_analyzer.py       # 文本統計分析
├── models/                     # 模型快取目錄（自動生成）
└── data/                       # 數據目錄（選用）
```

## 🔧 技術細節

### 使用的模型

- **模型名稱**：`Hello-SimpleAI/chatgpt-detector-roberta`
- **模型類型**：RoBERTa-base fine-tuned
- **訓練數據**：ChatGPT 生成文本 vs 人類撰寫文本
- **準確度**：約 85-90%

### 核心技術棧

| 技術 | 版本 | 用途 |
|------|------|------|
| Streamlit | 1.28+ | Web 應用框架 |
| Transformers | 4.35+ | 模型載入與推理 |
| PyTorch | 2.0+ | 深度學習後端 |
| Plotly | 5.17+ | 互動式圖表 |
| NLTK | 3.8+ | 文本處理 |

### 優化機制

1. **模型快取**
```python
@st.cache_resource
def load_detector_model():
    # 模型只載入一次，存在記憶體中
```

2. **結果快取**
```python
@st.cache_data
def predict_ai_text(text):
    # 相同文本的結果會被快取
```

3. **文本長度限制**
```python
max_length = 512  # tokens
max_words = 800   # words
```

## ⚠️ 注意事項

### 限制說明

1. **準確度限制**
   - 檢測準確度約 85-90%
   - 無法 100% 確定文本來源
   - 混合文本（人類編輯過的 AI 文本）可能誤判

2. **技術限制**
   - 主要針對英文文本訓練
   - 中文檢測可能準確度較低
   - 新版 AI 模型可能繞過檢測

3. **性能限制**
   - 首次使用需下載 500MB 模型
   - 長文本處理時間較長
   - 建議單次檢測不超過 2000 詞

### 使用建議

- ✅ 作為初步篩選工具
- ✅ 結合其他證據綜合判斷
- ✅ 關注高機率（>80%）結果
- ❌ 不應作為唯一判斷依據
- ❌ 不應用於正式的學術審查

## 🐛 常見問題

### Q1: 首次運行很慢怎麼辦？
A: 首次使用需要從 Hugging Face 下載模型（約 500MB），請耐心等待。下載完成後會自動快取，後續使用會很快。

### Q2: 出現 "模型載入失敗" 怎麼辦？
A:
1. 檢查網路連接
2. 確認防火牆沒有阻擋 Hugging Face 網站
3. 嘗試使用 VPN
4. 手動下載模型（見進階設定）

### Q3: 檢測速度太慢怎麼辦？
A:
1. 減少文本長度（建議 < 800 詞）
2. 使用「快速模式」
3. 降低最大字數限制
4. 確保使用了快取（重複檢測相同文本）

### Q4: 中文檢測準確嗎？
A: 模型主要針對英文訓練，中文檢測準確度可能較低。建議：
- 對中文文本謹慎使用
- 結合其他方法判斷
- 可以試試混合語言的長文本

### Q5: 如何提高準確度？
A:
- 提供足夠長度的文本（建議 > 100 詞）
- 使用詳細模式查看統計分析
- 結合 AI 指標綜合判斷
- 對比多個段落的檢測結果

## 📊 測試範例

### 範例 1: 明顯的 AI 文本
```
Artificial intelligence has revolutionized numerous industries.
Moreover, it continues to evolve at an unprecedented pace.
Furthermore, the implications of this technology are far-reaching.
Additionally, researchers are constantly developing new applications.
```
**預期結果**：AI 機率 > 85%

### 範例 2: 人類撰寫文本
```
I went to the store yesterday. Got some milk and eggs.
Forgot to buy bread though... typical me lol.
The cashier was super friendly, made my day better!
```
**預期結果**：AI 機率 < 30%

## 🔮 未來改進

- [ ] 支援更多語言（中文、日文等）
- [ ] 批次檢測多個文件
- [ ] 句子級別的詳細分析
- [ ] 導出檢測報告（PDF）
- [ ] API 介面
- [ ] 歷史記錄功能
- [ ] 更多模型選擇

## 📚 參考資料

- [JustDone AI Detector](https://justdone.com/ai-detector) - 功能參考
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - 模型文檔
- [Streamlit Documentation](https://docs.streamlit.io/) - Streamlit 官方文檔
- [RoBERTa Model](https://arxiv.org/abs/1907.11692) - RoBERTa 論文

## 👨‍💻 開發者

- **學校**：國立中興大學 (NCHU)
- **課程**：資訊安全 (Cybersecurity)
- **作業**：HW5 - AI 文本檢測器

## 📝 授權

本專案僅用於教育目的。

---

**最後更新**：2025-11-30

如有任何問題或建議，歡迎聯繫！
