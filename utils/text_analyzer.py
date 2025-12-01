"""
文本分析工具
提供額外的統計特徵分析
"""

import re
from collections import Counter
import streamlit as st


@st.cache_data
def analyze_text_features(text):
    """
    分析文本的統計特徵

    Args:
        text: 要分析的文本

    Returns:
        dict: 包含各種統計特徵的字典
    """
    # 基本統計
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 字數和句數
    word_count = len(words)
    sentence_count = len(sentences)

    # 平均句長
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # 平均詞長
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0

    # 詞彙多樣性（Type-Token Ratio）
    unique_words = len(set(words))
    ttr = unique_words / word_count if word_count > 0 else 0

    # 標點符號統計
    punctuation = re.findall(r'[,.!?;:]', text)
    punctuation_count = len(punctuation)
    punctuation_ratio = punctuation_count / len(text) if len(text) > 0 else 0

    # 常見 AI 指標
    # 1. 過度使用的轉折詞
    transition_words = [
        'however', 'moreover', 'furthermore', 'additionally',
        'consequently', 'therefore', 'thus', 'hence'
    ]
    transition_count = sum(1 for word in words if word.lower() in transition_words)
    transition_ratio = transition_count / word_count if word_count > 0 else 0

    # 2. 句子長度變異性（人類寫作通常句長變化較大）
    sentence_lengths = [len(s.split()) for s in sentences]
    if len(sentence_lengths) > 1:
        avg_len = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((x - avg_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        sentence_variance = variance ** 0.5
    else:
        sentence_variance = 0

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': round(avg_sentence_length, 2),
        'avg_word_length': round(avg_word_length, 2),
        'vocabulary_diversity': round(ttr, 3),
        'punctuation_ratio': round(punctuation_ratio, 4),
        'transition_words_ratio': round(transition_ratio, 4),
        'sentence_variance': round(sentence_variance, 2)
    }


def get_ai_indicators(features):
    """
    根據文本特徵判斷 AI 寫作的指標

    Args:
        features: analyze_text_features 返回的特徵字典

    Returns:
        list: AI 寫作指標列表
    """
    indicators = []

    # 1. 句子長度過於一致
    if features['sentence_variance'] < 3:
        indicators.append("句子長度變化小（AI 傾向）")

    # 2. 詞彙多樣性過高
    if features['vocabulary_diversity'] > 0.7:
        indicators.append("詞彙多樣性極高（可能是 AI）")

    # 3. 過度使用轉折詞
    if features['transition_words_ratio'] > 0.05:
        indicators.append("過度使用轉折詞（AI 特徵）")

    # 4. 平均句長過於標準
    if 15 <= features['avg_sentence_length'] <= 25:
        indicators.append("句長過於標準（15-25 詞）")

    # 5. 標點符號使用過於規律
    if 0.03 <= features['punctuation_ratio'] <= 0.05:
        indicators.append("標點符號使用規律（AI 傾向）")

    return indicators


def highlight_sentences(text, chunk_results):
    """
    根據檢測結果標記可疑句子

    Args:
        text: 原始文本
        chunk_results: 片段檢測結果列表

    Returns:
        str: 標記後的 HTML 文本
    """
    sentences = re.split(r'([.!?]+)', text)
    highlighted = []

    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            # 簡單的標記邏輯：根據整體 AI 機率
            # 這裡可以根據實際需求改進
            highlighted.append(sentence)

    return ''.join(highlighted)


def get_confidence_color(confidence):
    """
    根據信心等級返回對應顏色

    Args:
        confidence: 信心等級字串

    Returns:
        str: 顏色代碼
    """
    colors = {
        "高": "#28a745",  # 綠色
        "中": "#ffc107",  # 黃色
        "低": "#dc3545"   # 紅色
    }
    return colors.get(confidence, "#6c757d")


def format_percentage(value):
    """
    格式化百分比顯示

    Args:
        value: 0-1 之間的數值

    Returns:
        str: 格式化的百分比字串
    """
    return f"{value * 100:.2f}%"
