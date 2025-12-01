"""
AI 文本檢測模型載入工具
使用 Streamlit 快取機制優化載入時間
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 模型配置信息
MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"
MODEL_INFO = {
    "name": "ChatGPT Detector RoBERTa",
    "full_name": MODEL_NAME,
    "type": "RoBERTa-base",
    "size": "約 500 MB",
    "training_data": "ChatGPT 生成文本 vs 人類撰寫文本",
    "accuracy": "85-90%",
    "description": "基於 RoBERTa 的 AI 文本檢測模型，專門訓練用於識別 ChatGPT 生成的內容"
}


def get_model_info():
    """
    獲取模型資訊

    Returns:
        dict: 模型資訊字典
    """
    return MODEL_INFO


@st.cache_resource(show_spinner="正在載入 AI 檢測模型... (首次載入需要 1-2 分鐘)")
def load_detector_model():
    """
    載入預訓練的 AI 文本檢測模型
    使用 @st.cache_resource 確保模型只載入一次

    Returns:
        tuple: (tokenizer, model)
    """
    try:
        # 使用較小的模型以提升速度
        # 可選模型：
        # 1. "roberta-base-openai-detector" - OpenAI 官方檢測器（較大）
        # 2. "Hello-SimpleAI/chatgpt-detector-roberta" - 較小且快速
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        # 設置為評估模式
        model.eval()

        return tokenizer, model

    except Exception as e:
        st.error(f"模型載入失敗: {str(e)}")
        st.info("提示：首次使用需要下載模型文件（約 500MB），請確保網路連接正常")
        return None, None


@st.cache_data(show_spinner="正在分析文本...")
def predict_ai_text(_tokenizer, _model, text, max_length=512):
    """
    預測文本是否由 AI 生成
    使用 @st.cache_data 快取相同文本的結果

    Args:
        _tokenizer: tokenizer 物件（_ 前綴表示不要 hash 此參數）
        _model: model 物件（_ 前綴表示不要 hash 此參數）
        text: 要檢測的文本
        max_length: 最大文本長度（避免過長導致處理緩慢）

    Returns:
        dict: {
            'is_ai': bool,
            'ai_probability': float,
            'human_probability': float,
            'confidence': str
        }
    """
    if not _tokenizer or not _model:
        return None

    try:
        # 文本預處理：限制長度以提升速度
        if len(text.split()) > max_length:
            # 只取前 max_length 個詞
            text = ' '.join(text.split()[:max_length])

        # Tokenize 輸入
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # 預測（不計算梯度以節省記憶體和時間）
        with torch.no_grad():
            outputs = _model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # 解析結果
        human_prob = predictions[0][0].item()
        ai_prob = predictions[0][1].item()

        # 判斷信心等級
        max_prob = max(human_prob, ai_prob)
        if max_prob > 0.85:
            confidence = "高"
        elif max_prob > 0.65:
            confidence = "中"
        else:
            confidence = "低"

        # 生成判定原因說明
        reasons = []
        indicators = []

        # 分析機率分佈
        prob_diff = abs(ai_prob - human_prob)

        if ai_prob > human_prob:
            # AI 判定的原因
            if ai_prob > 0.9:
                reasons.append("模型對 AI 生成內容有極高信心（>90%）")
                indicators.append("強 AI 語言模式")
            elif ai_prob > 0.8:
                reasons.append("檢測到明顯的 AI 生成特徵（80-90%）")
                indicators.append("明顯 AI 特徵")
            elif ai_prob > 0.65:
                reasons.append("文本具有一定 AI 特徵（65-80%）")
                indicators.append("部分 AI 特徵")
            else:
                reasons.append("輕微的 AI 特徵，但不確定（50-65%）")
                indicators.append("輕微 AI 傾向")

            if prob_diff > 0.5:
                reasons.append("AI 與人類機率差距大，判定較明確")
            elif prob_diff < 0.2:
                reasons.append("機率接近，可能是混合內容或邊界案例")

        else:
            # 人類判定的原因
            if human_prob > 0.9:
                reasons.append("模型對人類撰寫有極高信心（>90%）")
                indicators.append("強人類寫作風格")
            elif human_prob > 0.8:
                reasons.append("檢測到明顯的人類寫作特徵（80-90%）")
                indicators.append("明顯人類特徵")
            elif human_prob > 0.65:
                reasons.append("文本具有一定人類寫作特徵（65-80%）")
                indicators.append("部分人類特徵")
            else:
                reasons.append("輕微的人類特徵，但不確定（50-65%）")
                indicators.append("輕微人類傾向")

            if prob_diff > 0.5:
                reasons.append("人類與 AI 機率差距大，判定較明確")
            elif prob_diff < 0.2:
                reasons.append("機率接近，可能是 AI 輔助寫作或高品質 AI 內容")

        return {
            'is_ai': ai_prob > human_prob,
            'ai_probability': ai_prob,
            'human_probability': human_prob,
            'confidence': confidence,
            'reasons': reasons,
            'indicators': indicators,
            'probability_difference': prob_diff
        }

    except Exception as e:
        st.error(f"預測過程發生錯誤: {str(e)}")
        return None


def chunk_text(text, max_words=400):
    """
    將長文本分割成多個片段以加速處理

    Args:
        text: 原始文本
        max_words: 每個片段的最大字數

    Returns:
        list: 文本片段列表
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks


@st.cache_data
def batch_predict(_tokenizer, _model, chunks):
    """
    批次處理多個文本片段

    Args:
        _tokenizer: tokenizer 物件
        _model: model 物件
        chunks: 文本片段列表

    Returns:
        list: 每個片段的預測結果
    """
    results = []

    for chunk in chunks:
        result = predict_ai_text(_tokenizer, _model, chunk)
        if result:
            results.append(result)

    return results
