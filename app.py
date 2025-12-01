"""
AI 文本檢測器 - Streamlit 應用程式
檢測文字內容是否由 AI 生成

優化要點：
1. 使用 @st.cache_resource 快取模型（只載入一次）
2. 使用 @st.cache_data 快取檢測結果
3. 限制文本長度避免處理時間過長
4. 提供進度指示
5. 分段處理長文本
"""

import streamlit as st
import plotly.graph_objects as go
import time
from utils.model_loader import (
    load_detector_model,
    predict_ai_text,
    chunk_text,
    batch_predict,
    get_model_info
)
from utils.text_analyzer import (
    analyze_text_features,
    get_ai_indicators,
    get_confidence_color,
    format_percentage
)

# 頁面配置
st.set_page_config(
    page_title="AI 文本檢測器",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-text {
        color: #ff6b6b;
        font-weight: bold;
    }
    .success-text {
        color: #51cf66;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 預載入模型（在頁面開啟時就載入，不等用戶點擊按鈕）
@st.cache_resource
def initialize_app():
    """初始化應用，預載入模型"""
    tokenizer, model = load_detector_model()
    return tokenizer, model

# 執行預載入
tokenizer, model = initialize_app()

# 標題
st.markdown('<div class="main-header">🤖 AI 文本檢測器</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">檢測文字內容是否由 AI 生成</div>', unsafe_allow_html=True)

# 顯示模型狀態
if tokenizer and model:
    st.success("✅ 模型已就緒，可以開始檢測！")
else:
    st.error("❌ 模型載入失敗，請重新整理頁面")

# 側邊欄設定
with st.sidebar:
    st.header("⚙️ 設定")

    # 顯示模型資訊
    st.markdown("---")
    st.subheader("🤖 使用的模型")
    model_info = get_model_info()

    with st.expander("查看模型詳情", expanded=False):
        st.markdown(f"""
        **模型名稱：** {model_info['name']}

        **模型類型：** {model_info['type']}

        **模型大小：** {model_info['size']}

        **訓練數據：** {model_info['training_data']}

        **準確度：** {model_info['accuracy']}

        **說明：** {model_info['description']}

        **完整路徑：** `{model_info['full_name']}`
        """)

    st.markdown("---")
    st.subheader("文本長度限制")
    max_words = st.slider(
        "最大字數（避免處理過慢）",
        min_value=100,
        max_value=2000,
        value=800,
        step=100,
        help="較長的文本會被截斷以提升檢測速度"
    )

    st.markdown("---")
    st.subheader("檢測模式")
    detection_mode = st.radio(
        "選擇模式",
        ["快速模式", "詳細模式"],
        help="快速模式：只進行 AI 檢測\n詳細模式：額外顯示文本統計分析"
    )

    st.markdown("---")
    st.subheader("📊 使用統計")
    if 'detection_count' not in st.session_state:
        st.session_state.detection_count = 0
    st.metric("總檢測次數", st.session_state.detection_count)

    st.markdown("---")
    st.info("""
    **使用提示：**
    - 首次使用需下載模型（約 500MB）
    - 建議文本長度 100-800 字
    - 過長文本會自動截斷
    - 結果會被快取以加速重複查詢
    """)

# 主要內容區
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 輸入文本")

    # 文本輸入方式選擇
    input_method = st.radio(
        "選擇輸入方式",
        ["直接輸入", "上傳檔案"],
        horizontal=True
    )

    text_input = ""

    if input_method == "直接輸入":
        text_input = st.text_area(
            "請輸入要檢測的文字",
            height=300,
            placeholder="在此輸入或貼上文字...",
            help=f"最多處理 {max_words} 個英文單詞"
        )
    else:
        uploaded_file = st.file_uploader(
            "上傳文字檔案",
            type=['txt'],
            help="支援 .txt 格式"
        )
        if uploaded_file is not None:
            try:
                text_input = uploaded_file.read().decode('utf-8')
                st.success(f"成功讀取檔案！字數：{len(text_input.split())} 詞")
                with st.expander("查看檔案內容"):
                    st.text(text_input[:500] + "..." if len(text_input) > 500 else text_input)
            except Exception as e:
                st.error(f"檔案讀取失敗：{str(e)}")

    # 顯示字數統計
    if text_input:
        word_count = len(text_input.split())
        char_count = len(text_input)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("字數", f"{word_count} 詞")
        with col_b:
            st.metric("字元數", f"{char_count} 字元")
        with col_c:
            if word_count > max_words:
                st.metric("處理字數", f"{max_words} 詞", delta=f"-{word_count - max_words}",
                         delta_color="inverse")
                st.warning(f"⚠️ 文本過長，將只分析前 {max_words} 個詞")
            else:
                st.metric("處理字數", f"{word_count} 詞", delta="全部")

with col2:
    st.subheader("ℹ️ 關於此工具")
    st.markdown("""
    此工具使用預訓練的機器學習模型來檢測文本是否由 AI 生成。

    **檢測原理：**
    - 分析文本的語言模式
    - 比對 AI 生成特徵
    - 計算 AI 生成機率

    **限制說明：**
    - 準確度約 85-90%
    - 無法 100% 確定
    - 混合文本可能誤判
    - 持續更新中

    **適用場景：**
    - 學術論文檢查
    - 作業原創性審核
    - 內容真實性驗證
    """)

# 檢測按鈕
st.markdown("---")
if st.button("🔍 開始檢測", type="primary", use_container_width=True):
    if not text_input or len(text_input.strip()) < 10:
        st.error("❌ 請輸入至少 10 個字元的文本")
    else:
        # 模型已在頁面載入時預載入，直接使用
        if tokenizer and model:
            try:
                # 開始計時
                start_time = time.time()

                # 進度條
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 步驟 1: 文本預處理
                status_text.text("步驟 1/3: 文本預處理...")
                progress_bar.progress(20)

                # 限制文本長度
                words = text_input.split()
                if len(words) > max_words:
                    text_to_analyze = ' '.join(words[:max_words])
                else:
                    text_to_analyze = text_input

                # 步驟 2: AI 檢測
                status_text.text("步驟 2/3: AI 檢測分析...")
                progress_bar.progress(50)

                result = predict_ai_text(tokenizer, model, text_to_analyze)

                # 步驟 3: 額外分析（詳細模式）
                if detection_mode == "詳細模式":
                    status_text.text("步驟 3/3: 文本特徵分析...")
                    progress_bar.progress(80)
                    features = analyze_text_features(text_to_analyze)
                    indicators = get_ai_indicators(features)
                else:
                    features = None
                    indicators = []

                progress_bar.progress(100)
                status_text.text("✅ 檢測完成！")

                # 計算耗時
                elapsed_time = time.time() - start_time

                # 更新統計
                st.session_state.detection_count += 1

                # 顯示結果
                time.sleep(0.5)  # 短暫延遲以顯示完成狀態
                progress_bar.empty()
                status_text.empty()

                st.markdown("---")
                st.markdown("## 📊 檢測結果")

                if result:
                    # 主要結果顯示
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        # 儀表板
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result['ai_probability'] * 100,
                            title={'text': "AI 生成機率", 'font': {'size': 24}},
                            number={'suffix': "%", 'font': {'size': 48}},
                            gauge={
                                'axis': {'range': [None, 100], 'tickwidth': 1},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "lightyellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 80
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.markdown("### 判定結果")
                        if result['is_ai']:
                            st.markdown('<p class="warning-text">⚠️ 可能是 AI 生成</p>',
                                      unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="success-text">✅ 可能是人類撰寫</p>',
                                      unsafe_allow_html=True)

                        st.markdown(f"""
                        **信心等級:** {result['confidence']}

                        **詳細機率:**
                        - AI: {format_percentage(result['ai_probability'])}
                        - 人類: {format_percentage(result['human_probability'])}
                        """)

                    with col3:
                        st.markdown("### 性能指標")
                        st.metric("處理時間", f"{elapsed_time:.2f} 秒")
                        st.metric("分析字數", len(text_to_analyze.split()))
                        st.metric("速度", f"{len(text_to_analyze.split())/elapsed_time:.0f} 詞/秒")

                    # 判定原因與評比指標
                    st.markdown("---")
                    st.markdown("### 🔍 判定原因與評比指標")

                    col_reason1, col_reason2 = st.columns(2)

                    with col_reason1:
                        st.markdown("#### 📌 檢測指標")
                        if 'indicators' in result and result['indicators']:
                            for indicator in result['indicators']:
                                st.info(f"🎯 {indicator}")
                        else:
                            st.info("🎯 基於 RoBERTa 模型的語言模式分析")

                        st.markdown(f"""
                        **機率差距：** {result.get('probability_difference', 0):.2%}

                        {
                            "（差距大，判定明確）" if result.get('probability_difference', 0) > 0.5
                            else "（差距中等）" if result.get('probability_difference', 0) > 0.2
                            else "（差距小，較不確定）"
                        }
                        """)

                    with col_reason2:
                        st.markdown("#### 💡 為什麼這樣判定？")
                        if 'reasons' in result and result['reasons']:
                            for i, reason in enumerate(result['reasons'], 1):
                                st.markdown(f"{i}. {reason}")
                        else:
                            st.markdown("基於模型訓練的數百萬個樣本進行判斷")

                    # 評比標準說明
                    st.markdown("---")
                    st.markdown("### 📊 評比標準")

                    with st.expander("模型如何判定 AI 文本？", expanded=True):
                        st.markdown("""
                        **RoBERTa 模型的評比機制：**

                        1. **語言模式分析**
                           - 分析句子結構和詞彙選擇
                           - 檢測典型的 AI 生成模式
                           - 識別過於完美或規律的語法

                        2. **上下文連貫性**
                           - 評估段落之間的邏輯連接
                           - 檢測轉折詞的使用頻率
                           - 分析句子長度的一致性

                        3. **詞彙特徵**
                           - 詞彙多樣性分析
                           - 專業術語使用頻率
                           - 常見 AI 用語檢測

                        4. **寫作風格**
                           - 識別人類寫作的不規則性
                           - 檢測情感表達方式
                           - 分析個人化語言特徵

                        **機率計算：**
                        - 模型輸出兩個機率：AI 機率 vs 人類機率
                        - 使用 Softmax 函數標準化結果
                        - 較高機率決定最終判定
                        - 機率差距反映判定信心度

                        **信心等級定義：**
                        - **高信心**（>85%）：模型非常確定
                        - **中信心**（65-85%）：模型較為確定
                        - **低信心**（<65%）：模型不太確定，可能是邊界案例
                        """)

                    # 結果解釋
                    st.markdown("---")
                    st.markdown("### 📝 結果解釋")

                    if result['ai_probability'] > 0.8:
                        st.error("""
                        **高度可疑（>80%）**

                        此文本具有強烈的 AI 生成特徵，很可能由 ChatGPT、Claude 或其他 AI 工具生成。

                        建議：
                        - 仔細審查文本內容
                        - 要求提供寫作過程證明
                        - 進行面談確認理解程度
                        """)
                    elif result['ai_probability'] > 0.5:
                        st.warning("""
                        **中度可疑（50-80%）**

                        此文本可能包含 AI 生成內容，或者受到 AI 工具的輔助。

                        建議：
                        - 結合其他證據判斷
                        - 關注具體段落內容
                        - 考慮是否為 AI 輔助寫作
                        """)
                    else:
                        st.success("""
                        **低度可疑（<50%）**

                        此文本更像是人類自然撰寫，AI 生成的可能性較低。

                        註記：
                        - 不排除高品質 AI 或人工編輯過的內容
                        - 建議綜合其他因素判斷
                        """)

                    # 詳細模式：顯示統計分析
                    if detection_mode == "詳細模式" and features:
                        st.markdown("---")
                        st.markdown("### 📈 文本統計分析")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("總字數", features['word_count'])
                            st.metric("句子數", features['sentence_count'])

                        with col2:
                            st.metric("平均句長", f"{features['avg_sentence_length']} 詞")
                            st.metric("平均詞長", f"{features['avg_word_length']} 字元")

                        with col3:
                            st.metric("詞彙多樣性", features['vocabulary_diversity'])
                            st.metric("標點符號比", f"{features['punctuation_ratio']:.3f}")

                        with col4:
                            st.metric("句長變異", features['sentence_variance'])
                            st.metric("轉折詞比", f"{features['transition_words_ratio']:.3f}")

                        # AI 指標
                        if indicators:
                            st.markdown("#### 🚨 AI 寫作指標")
                            for indicator in indicators:
                                st.warning(f"• {indicator}")
                        else:
                            st.success("✅ 未發現明顯的 AI 寫作指標")

                    # 免責聲明
                    st.markdown("---")
                    st.info("""
                    **⚠️ 免責聲明**

                    此工具基於機器學習模型，檢測結果僅供參考，不應作為唯一判斷依據。
                    AI 技術不斷進化，檢測準確度無法達到 100%。建議結合其他方法綜合判斷。
                    """)

                else:
                    st.error("❌ 檢測失敗，請重試")

            except Exception as e:
                st.error(f"❌ 檢測過程發生錯誤：{str(e)}")
                st.info("請嘗試縮短文本長度或重新整理頁面")

        else:
            st.error("❌ 模型載入失敗，請檢查網路連接並重新整理頁面")

# 頁尾
st.markdown("---")
model_info = get_model_info()
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🎓 NCHU Cybersecurity - AI Text Detector</p>
    <p>Powered by Hugging Face Transformers & Streamlit</p>
    <p style='font-size: 0.8rem;'>使用模型：{model_info['name']} ({model_info['full_name']})</p>
</div>
""", unsafe_allow_html=True)
