# 情感分析器 - 分析文本情感并生成情绪因子

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

try:
    import jieba
    import jieba.analyse
except ImportError:
    jieba = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)

class SentimentLabel(Enum):
    """情感标签枚举"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

@dataclass
class SentimentResult:
    """情感分析结果"""
    score: float  # 情感分数 [-1, 1]
    label: SentimentLabel  # 情感标签
    confidence: float  # 置信度 [0, 1]
    method: str  # 分析方法
    details: Dict[str, Any]  # 详细信息

@dataclass
class EmotionResult:
    """情绪分析结果"""
    emotions: Dict[str, float]  # 情绪类型 -> 强度
    dominant_emotion: str  # 主导情绪
    intensity: float  # 情绪强度

class SentimentAnalyzer:
    """情感分析器
    
    支持多种情感分析方法：
    1. 基于词典的情感分析
    2. 基于机器学习的情感分析
    3. 混合方法
    
    专门针对金融文本进行优化
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get('method', 'hybrid')  # lexicon, ml, hybrid
        
        # 初始化情感词典
        self._init_sentiment_lexicons()
        
        # 初始化机器学习模型
        if self.method in ['ml', 'hybrid']:
            self._init_ml_models()
        
        # 初始化金融领域特定规则
        self._init_financial_rules()
        
        logger.info(f"情感分析器初始化完成，方法: {self.method}")
    
    def _init_sentiment_lexicons(self):
        """初始化情感词典"""
        # 正面词汇
        self.positive_words = {
            # 涨跌相关
            '上涨', '涨幅', '涨停', '大涨', '暴涨', '飙升', '攀升', '走高', '冲高',
            '突破', '创新高', '新高', '历史新高', '强势', '强劲', '坚挺',
            
            # 业绩相关
            '盈利', '利润', '增长', '增收', '营收', '业绩', '超预期', '亮眼',
            '优秀', '出色', '卓越', '杰出', '领先', '第一', '冠军',
            
            # 投资相关
            '买入', '增持', '推荐', '看好', '乐观', '积极', '利好', '机会',
            '投资', '价值', '潜力', '前景', '发展', '扩张', '布局',
            
            # 市场相关
            '活跃', '热门', '火爆', '抢手', '追捧', '青睐', '关注', '受益',
            '复苏', '回暖', '企稳', '反弹', '反转', '转机'
        }
        
        # 负面词汇
        self.negative_words = {
            # 涨跌相关
            '下跌', '跌幅', '跌停', '大跌', '暴跌', '重挫', '下滑', '走低', '跳水',
            '破位', '创新低', '新低', '历史新低', '弱势', '疲软', '低迷',
            
            # 业绩相关
            '亏损', '损失', '下降', '减少', '萎缩', '业绩', '低于预期', '惨淡',
            '糟糕', '恶化', '困难', '挑战', '压力', '危机', '风险',
            
            # 投资相关
            '卖出', '减持', '回避', '看空', '悲观', '消极', '利空', '威胁',
            '抛售', '套现', '撤资', '退出', '谨慎', '担忧', '忧虑',
            
            # 市场相关
            '低迷', '冷清', '萧条', '衰退', '下行', '恶化', '动荡', '不稳',
            '震荡', '调整', '回调', '修正', '波动', '不确定'
        }
        
        # 中性词汇
        self.neutral_words = {
            '平稳', '稳定', '持平', '维持', '保持', '观望', '等待', '关注',
            '分析', '研究', '评估', '考虑', '讨论', '会议', '公告', '发布'
        }
        
        # 程度副词权重
        self.degree_words = {
            '非常': 2.0, '极其': 2.0, '十分': 1.8, '相当': 1.6, '很': 1.5,
            '比较': 1.2, '较': 1.2, '稍': 0.8, '略': 0.8, '有点': 0.8,
            '不太': -0.5, '不': -1.0, '没有': -1.0, '未': -1.0
        }
        
        # 否定词
        self.negation_words = {'不', '没', '无', '非', '未', '否', '别', '勿'}
    
    def _init_ml_models(self):
        """初始化机器学习模型"""
        try:
            if pipeline is not None:
                # 使用预训练的中文情感分析模型
                model_name = self.config.get(
                    'sentiment_model', 
                    'uer/roberta-base-finetuned-chinanews-chinese'
                )
                
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name
                )
                
                logger.info(f"情感分析模型加载成功: {model_name}")
            else:
                self.sentiment_pipeline = None
                logger.warning("transformers未安装，无法使用机器学习模型")
        
        except Exception as e:
            logger.error(f"情感分析模型加载失败: {str(e)}")
            self.sentiment_pipeline = None
    
    def _init_financial_rules(self):
        """初始化金融领域特定规则"""
        # 金融关键词权重调整
        self.financial_keywords = {
            # 高权重正面词
            '涨停': 3.0, '暴涨': 2.5, '大涨': 2.0, '强势': 1.8,
            '超预期': 2.0, '利好': 1.5, '买入': 1.8, '推荐': 1.5,
            
            # 高权重负面词
            '跌停': -3.0, '暴跌': -2.5, '大跌': -2.0, '弱势': -1.8,
            '低于预期': -2.0, '利空': -1.5, '卖出': -1.8, '回避': -1.5,
            
            # 风险相关
            '风险': -1.2, '危机': -2.0, '警告': -1.5, '担忧': -1.0,
            '不确定': -1.0, '波动': -0.8, '调整': -0.5
        }
        
        # 数值模式（涨跌幅等）
        self.number_patterns = [
            (r'涨(?:幅|了)?\s*(\d+(?:\.\d+)?)%', 1.0),  # 涨幅
            (r'跌(?:幅|了)?\s*(\d+(?:\.\d+)?)%', -1.0), # 跌幅
            (r'增长\s*(\d+(?:\.\d+)?)%', 0.8),          # 增长
            (r'下降\s*(\d+(?:\.\d+)?)%', -0.8),         # 下降
        ]
    
    async def analyze_sentiment(self, text: str, context: Dict[str, Any] = None) -> SentimentResult:
        """分析文本情感
        
        Args:
            text: 待分析文本
            context: 上下文信息
            
        Returns:
            情感分析结果
        """
        try:
            results = []
            
            # 基于词典的分析
            if self.method in ['lexicon', 'hybrid']:
                lexicon_result = await self._analyze_by_lexicon(text)
                results.append(lexicon_result)
            
            # 基于机器学习的分析
            if self.method in ['ml', 'hybrid'] and self.sentiment_pipeline:
                ml_result = await self._analyze_by_ml(text)
                results.append(ml_result)
            
            # 融合结果
            final_result = self._combine_results(results, text)
            
            # 应用金融领域调整
            final_result = self._apply_financial_adjustment(final_result, text)
            
            logger.debug(f"情感分析完成: {final_result.label.value}, 分数: {final_result.score:.3f}")
            return final_result
            
        except Exception as e:
            logger.error(f"情感分析失败: {str(e)}")
            return SentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                method='error',
                details={'error': str(e)}
            )
    
    async def _analyze_by_lexicon(self, text: str) -> SentimentResult:
        """基于词典的情感分析"""
        # 分词
        if jieba:
            words = list(jieba.cut(text))
        else:
            # 简单分词
            words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', text)
        
        sentiment_score = 0.0
        word_count = 0
        details = {'positive_words': [], 'negative_words': [], 'neutral_words': []}
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # 检查程度副词
            degree_weight = 1.0
            if i > 0 and words[i-1] in self.degree_words:
                degree_weight = self.degree_words[words[i-1]]
            
            # 检查否定词
            negation = False
            if i > 0 and words[i-1] in self.negation_words:
                negation = True
            elif i > 1 and words[i-2] in self.negation_words:
                negation = True
            
            # 计算词汇情感分数
            word_score = 0.0
            if word in self.positive_words:
                word_score = 1.0
                details['positive_words'].append(word)
            elif word in self.negative_words:
                word_score = -1.0
                details['negative_words'].append(word)
            elif word in self.neutral_words:
                word_score = 0.0
                details['neutral_words'].append(word)
            elif word in self.financial_keywords:
                word_score = self.financial_keywords[word]
                if word_score > 0:
                    details['positive_words'].append(word)
                else:
                    details['negative_words'].append(word)
            
            # 应用程度和否定修饰
            if word_score != 0.0:
                word_score *= degree_weight
                if negation:
                    word_score *= -1
                
                sentiment_score += word_score
                word_count += 1
            
            i += 1
        
        # 处理数值模式
        for pattern, weight in self.number_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    value = float(match)
                    # 根据数值大小调整情感强度
                    if value > 10:
                        sentiment_score += weight * 2.0
                    elif value > 5:
                        sentiment_score += weight * 1.5
                    else:
                        sentiment_score += weight * 1.0
                    word_count += 1
                except ValueError:
                    continue
        
        # 归一化分数
        if word_count > 0:
            sentiment_score = sentiment_score / word_count
        
        # 限制在[-1, 1]范围内
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # 确定标签
        label = self._score_to_label(sentiment_score)
        
        # 计算置信度
        confidence = min(abs(sentiment_score) + 0.3, 1.0) if word_count > 0 else 0.3
        
        return SentimentResult(
            score=sentiment_score,
            label=label,
            confidence=confidence,
            method='lexicon',
            details=details
        )
    
    async def _analyze_by_ml(self, text: str) -> SentimentResult:
        """基于机器学习的情感分析"""
        try:
            # 限制文本长度
            max_length = self.config.get('max_text_length', 512)
            if len(text) > max_length:
                text = text[:max_length]
            
            # 调用模型
            result = self.sentiment_pipeline(text)
            
            # 解析结果
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            label_text = result.get('label', 'NEUTRAL').upper()
            confidence = result.get('score', 0.5)
            
            # 映射标签到分数
            if 'POSITIVE' in label_text or 'POS' in label_text:
                score = confidence
                label = SentimentLabel.POSITIVE if confidence < 0.8 else SentimentLabel.VERY_POSITIVE
            elif 'NEGATIVE' in label_text or 'NEG' in label_text:
                score = -confidence
                label = SentimentLabel.NEGATIVE if confidence < 0.8 else SentimentLabel.VERY_NEGATIVE
            else:
                score = 0.0
                label = SentimentLabel.NEUTRAL
            
            return SentimentResult(
                score=score,
                label=label,
                confidence=confidence,
                method='ml',
                details={'model_output': result}
            )
            
        except Exception as e:
            logger.error(f"机器学习情感分析失败: {str(e)}")
            return SentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                method='ml_error',
                details={'error': str(e)}
            )
    
    def _combine_results(self, results: List[SentimentResult], text: str) -> SentimentResult:
        """融合多个分析结果"""
        if not results:
            return SentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                method='none',
                details={}
            )
        
        if len(results) == 1:
            return results[0]
        
        # 加权平均
        weights = {
            'lexicon': 0.6,
            'ml': 0.4
        }
        
        total_score = 0.0
        total_weight = 0.0
        combined_details = {}
        methods = []
        
        for result in results:
            weight = weights.get(result.method, 0.5)
            total_score += result.score * weight * result.confidence
            total_weight += weight * result.confidence
            combined_details[result.method] = result.details
            methods.append(result.method)
        
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        
        # 计算综合置信度
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        return SentimentResult(
            score=final_score,
            label=self._score_to_label(final_score),
            confidence=avg_confidence,
            method='_'.join(methods),
            details=combined_details
        )
    
    def _apply_financial_adjustment(self, result: SentimentResult, text: str) -> SentimentResult:
        """应用金融领域特定调整"""
        # 检查是否包含强烈的金融信号
        strong_signals = ['涨停', '跌停', '暴涨', '暴跌', '重大利好', '重大利空']
        
        for signal in strong_signals:
            if signal in text:
                if '涨' in signal or '利好' in signal:
                    result.score = max(result.score, 0.8)
                    if result.score > 0.8:
                        result.label = SentimentLabel.VERY_POSITIVE
                elif '跌' in signal or '利空' in signal:
                    result.score = min(result.score, -0.8)
                    if result.score < -0.8:
                        result.label = SentimentLabel.VERY_NEGATIVE
                
                result.confidence = min(result.confidence + 0.2, 1.0)
                break
        
        return result
    
    def _score_to_label(self, score: float) -> SentimentLabel:
        """将分数转换为标签"""
        if score >= 0.6:
            return SentimentLabel.VERY_POSITIVE
        elif score >= 0.2:
            return SentimentLabel.POSITIVE
        elif score <= -0.6:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    async def analyze_emotion(self, text: str) -> EmotionResult:
        """分析文本情绪"""
        # 简单的情绪分析实现
        emotions = {
            'joy': 0.0,      # 喜悦
            'anger': 0.0,    # 愤怒
            'fear': 0.0,     # 恐惧
            'sadness': 0.0,  # 悲伤
            'surprise': 0.0, # 惊讶
            'trust': 0.0,    # 信任
            'anticipation': 0.0, # 期待
            'disgust': 0.0   # 厌恶
        }
        
        # 情绪关键词
        emotion_keywords = {
            'joy': ['高兴', '兴奋', '喜悦', '开心', '乐观', '积极'],
            'anger': ['愤怒', '生气', '不满', '抗议', '谴责'],
            'fear': ['恐惧', '担心', '忧虑', '害怕', '紧张', '风险'],
            'sadness': ['悲伤', '失望', '沮丧', '难过', '遗憾'],
            'surprise': ['惊讶', '意外', '震惊', '突然', '没想到'],
            'trust': ['信任', '相信', '可靠', '稳定', '安全'],
            'anticipation': ['期待', '期望', '希望', '预期', '看好'],
            'disgust': ['厌恶', '反感', '讨厌', '恶心']
        }
        
        # 计算各情绪强度
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    emotions[emotion] += 1.0
        
        # 归一化
        total_emotion = sum(emotions.values())
        if total_emotion > 0:
            emotions = {k: v / total_emotion for k, v in emotions.items()}
        
        # 找到主导情绪
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        intensity = emotions[dominant_emotion]
        
        return EmotionResult(
            emotions=emotions,
            dominant_emotion=dominant_emotion,
            intensity=intensity
        )
    
    async def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """批量情感分析"""
        results = []
        
        for text in texts:
            result = await self.analyze_sentiment(text)
            results.append(result)
        
        logger.info(f"批量情感分析完成，处理{len(texts)}条文本")
        return results
    
    def get_sentiment_distribution(self, results: List[SentimentResult]) -> Dict[str, float]:
        """获取情感分布统计"""
        if not results:
            return {}
        
        label_counts = {label.value: 0 for label in SentimentLabel}
        
        for result in results:
            label_counts[result.label.value] += 1
        
        total = len(results)
        distribution = {k: v / total for k, v in label_counts.items()}
        
        return distribution
    
    def calculate_sentiment_trend(self, results: List[Tuple[datetime, SentimentResult]]) -> Dict[str, Any]:
        """计算情感趋势"""
        if not results:
            return {}
        
        # 按时间排序
        sorted_results = sorted(results, key=lambda x: x[0])
        
        # 计算移动平均
        window_size = min(7, len(sorted_results))  # 7天移动平均
        moving_avg = []
        
        for i in range(len(sorted_results)):
            start_idx = max(0, i - window_size + 1)
            window_scores = [r[1].score for r in sorted_results[start_idx:i+1]]
            avg_score = sum(window_scores) / len(window_scores)
            moving_avg.append(avg_score)
        
        # 计算趋势
        if len(moving_avg) >= 2:
            trend = moving_avg[-1] - moving_avg[0]
            trend_direction = 'positive' if trend > 0.1 else 'negative' if trend < -0.1 else 'stable'
        else:
            trend = 0.0
            trend_direction = 'stable'
        
        return {
            'trend': trend,
            'direction': trend_direction,
            'moving_average': moving_avg,
            'latest_score': sorted_results[-1][1].score if sorted_results else 0.0,
            'score_range': {
                'min': min(r[1].score for r in sorted_results),
                'max': max(r[1].score for r in sorted_results),
                'avg': sum(r[1].score for r in sorted_results) / len(sorted_results)
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取分析器统计信息"""
        return {
            'method': self.method,
            'lexicon_stats': {
                'positive_words': len(self.positive_words),
                'negative_words': len(self.negative_words),
                'neutral_words': len(self.neutral_words),
                'financial_keywords': len(self.financial_keywords)
            },
            'ml_model_available': self.sentiment_pipeline is not None,
            'supported_labels': [label.value for label in SentimentLabel]
        }