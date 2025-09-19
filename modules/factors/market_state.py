# 市场状态分析器模块
# 实现市场状态的综合分析、预测和状态转换检测

import logging
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy import stats
    from scipy.signal import find_peaks
except ImportError:
    stats = None
    find_peaks = None

logger = logging.getLogger(__name__)

class MarketPhase(Enum):
    """市场阶段"""
    ACCUMULATION = "accumulation"      # 积累阶段
    MARKUP = "markup"                  # 上涨阶段
    DISTRIBUTION = "distribution"      # 分配阶段
    MARKDOWN = "markdown"              # 下跌阶段
    CONSOLIDATION = "consolidation"    # 整理阶段

class VolatilityRegime(Enum):
    """波动率状态"""
    LOW = "low"                        # 低波动
    NORMAL = "normal"                  # 正常波动
    HIGH = "high"                      # 高波动
    EXTREME = "extreme"                # 极端波动

class TrendDirection(Enum):
    """趋势方向"""
    STRONG_UPTREND = "strong_uptrend"  # 强上升趋势
    UPTREND = "uptrend"                # 上升趋势
    SIDEWAYS = "sideways"              # 横盘
    DOWNTREND = "downtrend"            # 下降趋势
    STRONG_DOWNTREND = "strong_downtrend"  # 强下降趋势

class LiquidityCondition(Enum):
    """流动性状况"""
    ABUNDANT = "abundant"              # 充裕
    NORMAL = "normal"                  # 正常
    TIGHT = "tight"                    # 紧张
    STRESSED = "stressed"              # 压力

@dataclass
class MarketIndicators:
    """市场指标"""
    # 价格指标
    price_level: float = 0.0
    price_change_1d: float = 0.0
    price_change_5d: float = 0.0
    price_change_20d: float = 0.0
    
    # 波动率指标
    realized_volatility: float = 0.0
    implied_volatility: float = 0.0
    volatility_skew: float = 0.0
    
    # 动量指标
    momentum_short: float = 0.0
    momentum_medium: float = 0.0
    momentum_long: float = 0.0
    
    # 技术指标
    rsi: float = 50.0
    macd: float = 0.0
    bollinger_position: float = 0.5
    
    # 成交量指标
    volume_ratio: float = 1.0
    volume_trend: float = 0.0
    
    # 市场宽度指标
    advance_decline_ratio: float = 0.5
    new_highs_lows_ratio: float = 0.5
    
    # 情绪指标
    vix_level: float = 20.0
    put_call_ratio: float = 1.0
    margin_debt_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'price_level': self.price_level,
            'price_change_1d': self.price_change_1d,
            'price_change_5d': self.price_change_5d,
            'price_change_20d': self.price_change_20d,
            'realized_volatility': self.realized_volatility,
            'implied_volatility': self.implied_volatility,
            'volatility_skew': self.volatility_skew,
            'momentum_short': self.momentum_short,
            'momentum_medium': self.momentum_medium,
            'momentum_long': self.momentum_long,
            'rsi': self.rsi,
            'macd': self.macd,
            'bollinger_position': self.bollinger_position,
            'volume_ratio': self.volume_ratio,
            'volume_trend': self.volume_trend,
            'advance_decline_ratio': self.advance_decline_ratio,
            'new_highs_lows_ratio': self.new_highs_lows_ratio,
            'vix_level': self.vix_level,
            'put_call_ratio': self.put_call_ratio,
            'margin_debt_ratio': self.margin_debt_ratio
        }

@dataclass
class MarketState:
    """市场状态"""
    timestamp: datetime
    phase: MarketPhase
    volatility_regime: VolatilityRegime
    trend_direction: TrendDirection
    liquidity_condition: LiquidityCondition
    
    # 置信度
    phase_confidence: float = 0.0
    volatility_confidence: float = 0.0
    trend_confidence: float = 0.0
    liquidity_confidence: float = 0.0
    
    # 市场指标
    indicators: MarketIndicators = field(default_factory=MarketIndicators)
    
    # 状态持续时间
    phase_duration: int = 0
    volatility_duration: int = 0
    trend_duration: int = 0
    
    # 预测信息
    next_phase_probability: Dict[str, float] = field(default_factory=dict)
    transition_signals: List[str] = field(default_factory=list)
    
    def get_overall_confidence(self) -> float:
        """获取整体置信度"""
        confidences = [
            self.phase_confidence,
            self.volatility_confidence,
            self.trend_confidence,
            self.liquidity_confidence
        ]
        return np.mean([c for c in confidences if c > 0])
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'phase': self.phase.value,
            'volatility_regime': self.volatility_regime.value,
            'trend_direction': self.trend_direction.value,
            'liquidity_condition': self.liquidity_condition.value,
            'phase_confidence': self.phase_confidence,
            'volatility_confidence': self.volatility_confidence,
            'trend_confidence': self.trend_confidence,
            'liquidity_confidence': self.liquidity_confidence,
            'overall_confidence': self.get_overall_confidence(),
            'indicators': self.indicators.to_dict(),
            'phase_duration': self.phase_duration,
            'volatility_duration': self.volatility_duration,
            'trend_duration': self.trend_duration,
            'next_phase_probability': self.next_phase_probability,
            'transition_signals': self.transition_signals
        }

@dataclass
class StateTransition:
    """状态转换记录"""
    timestamp: datetime
    from_state: MarketState
    to_state: MarketState
    transition_type: str  # phase, volatility, trend, liquidity
    trigger_indicators: List[str]
    transition_strength: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'transition_type': self.transition_type,
            'from_phase': self.from_state.phase.value,
            'to_phase': self.to_state.phase.value,
            'from_volatility': self.from_state.volatility_regime.value,
            'to_volatility': self.to_state.volatility_regime.value,
            'from_trend': self.from_state.trend_direction.value,
            'to_trend': self.to_state.trend_direction.value,
            'trigger_indicators': self.trigger_indicators,
            'transition_strength': self.transition_strength
        }

class MarketStateAnalyzer:
    """市场状态分析器
    
    综合分析市场的各个维度状态
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 分析参数
        self.lookback_window = config.get('lookback_window', 252)
        self.volatility_window = config.get('volatility_window', 20)
        self.trend_window = config.get('trend_window', 50)
        
        # 阈值设置
        self._init_thresholds()
        
        # 历史数据
        self.price_history: deque = deque(maxlen=self.lookback_window)
        self.volume_history: deque = deque(maxlen=self.lookback_window)
        self.state_history: List[MarketState] = []
        self.transition_history: List[StateTransition] = []
        
        # 当前状态
        self.current_state: Optional[MarketState] = None
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("市场状态分析器初始化完成")
    
    def _init_thresholds(self):
        """初始化阈值"""
        self.thresholds = {
            # 波动率阈值
            'volatility': {
                'low': 0.10,
                'normal': 0.20,
                'high': 0.30,
                'extreme': 0.50
            },
            
            # 趋势阈值
            'trend': {
                'strong_threshold': 0.05,
                'weak_threshold': 0.02
            },
            
            # RSI阈值
            'rsi': {
                'oversold': 30,
                'overbought': 70
            },
            
            # VIX阈值
            'vix': {
                'low': 15,
                'normal': 25,
                'high': 35
            },
            
            # 成交量阈值
            'volume': {
                'low': 0.7,
                'high': 1.5
            }
        }
    
    async def analyze_market_state(self, market_data: pd.DataFrame) -> MarketState:
        """分析市场状态
        
        Args:
            market_data: 市场数据
            
        Returns:
            市场状态
        """
        try:
            with self.lock:
                # 更新历史数据
                self._update_history(market_data)
                
                # 计算市场指标
                indicators = self._calculate_indicators(market_data)
                
                # 分析各个维度的状态
                phase = self._analyze_market_phase(indicators)
                volatility_regime = self._analyze_volatility_regime(indicators)
                trend_direction = self._analyze_trend_direction(indicators)
                liquidity_condition = self._analyze_liquidity_condition(indicators)
                
                # 计算置信度
                phase_confidence = self._calculate_phase_confidence(indicators, phase)
                volatility_confidence = self._calculate_volatility_confidence(indicators, volatility_regime)
                trend_confidence = self._calculate_trend_confidence(indicators, trend_direction)
                liquidity_confidence = self._calculate_liquidity_confidence(indicators, liquidity_condition)
                
                # 计算状态持续时间
                phase_duration = self._calculate_phase_duration(phase)
                volatility_duration = self._calculate_volatility_duration(volatility_regime)
                trend_duration = self._calculate_trend_duration(trend_direction)
                
                # 预测下一阶段
                next_phase_probability = self._predict_next_phase(phase, indicators)
                
                # 检测转换信号
                transition_signals = self._detect_transition_signals(indicators)
                
                # 创建市场状态
                market_state = MarketState(
                    timestamp=datetime.now(),
                    phase=phase,
                    volatility_regime=volatility_regime,
                    trend_direction=trend_direction,
                    liquidity_condition=liquidity_condition,
                    phase_confidence=phase_confidence,
                    volatility_confidence=volatility_confidence,
                    trend_confidence=trend_confidence,
                    liquidity_confidence=liquidity_confidence,
                    indicators=indicators,
                    phase_duration=phase_duration,
                    volatility_duration=volatility_duration,
                    trend_duration=trend_duration,
                    next_phase_probability=next_phase_probability,
                    transition_signals=transition_signals
                )
                
                # 检测状态转换
                if self.current_state:
                    self._detect_state_transitions(self.current_state, market_state)
                
                # 更新当前状态
                self.current_state = market_state
                self.state_history.append(market_state)
                
                logger.info(f"市场状态分析完成: {phase.value}, 波动率: {volatility_regime.value}, 趋势: {trend_direction.value}")
                return market_state
                
        except Exception as e:
            logger.error(f"市场状态分析失败: {e}")
            return self._get_default_state()
    
    def _update_history(self, market_data: pd.DataFrame):
        """更新历史数据"""
        if market_data.empty:
            return
        
        # 更新价格历史
        if 'close' in market_data.columns:
            latest_prices = market_data['close'].tail(10).tolist()
            self.price_history.extend(latest_prices)
        
        # 更新成交量历史
        if 'volume' in market_data.columns:
            latest_volumes = market_data['volume'].tail(10).tolist()
            self.volume_history.extend(latest_volumes)
    
    def _calculate_indicators(self, market_data: pd.DataFrame) -> MarketIndicators:
        """计算市场指标"""
        indicators = MarketIndicators()
        
        if market_data.empty or 'close' not in market_data.columns:
            return indicators
        
        try:
            # 价格指标
            close_prices = market_data['close']
            indicators.price_level = close_prices.iloc[-1]
            
            if len(close_prices) > 1:
                indicators.price_change_1d = close_prices.pct_change().iloc[-1]
            if len(close_prices) > 5:
                indicators.price_change_5d = close_prices.pct_change(5).iloc[-1]
            if len(close_prices) > 20:
                indicators.price_change_20d = close_prices.pct_change(20).iloc[-1]
            
            # 波动率指标
            if len(close_prices) > self.volatility_window:
                returns = close_prices.pct_change().dropna()
                indicators.realized_volatility = returns.rolling(self.volatility_window).std().iloc[-1] * np.sqrt(252)
            
            # 动量指标
            if len(close_prices) > 10:
                indicators.momentum_short = close_prices.pct_change(10).iloc[-1]
            if len(close_prices) > 50:
                indicators.momentum_medium = close_prices.pct_change(50).iloc[-1]
            if len(close_prices) > 200:
                indicators.momentum_long = close_prices.pct_change(200).iloc[-1]
            
            # RSI指标
            if len(close_prices) > 14:
                indicators.rsi = self._calculate_rsi(close_prices, 14)
            
            # MACD指标
            if len(close_prices) > 26:
                indicators.macd = self._calculate_macd(close_prices)
            
            # 布林带位置
            if len(close_prices) > 20:
                indicators.bollinger_position = self._calculate_bollinger_position(close_prices, 20)
            
            # 成交量指标
            if 'volume' in market_data.columns and len(market_data) > 20:
                volume = market_data['volume']
                avg_volume = volume.rolling(20).mean().iloc[-1]
                indicators.volume_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1.0
                
                # 成交量趋势
                if len(volume) > 10:
                    volume_ma_short = volume.rolling(5).mean().iloc[-1]
                    volume_ma_long = volume.rolling(20).mean().iloc[-1]
                    indicators.volume_trend = (volume_ma_short - volume_ma_long) / volume_ma_long if volume_ma_long > 0 else 0.0
            
            # VIX水平（模拟）
            if indicators.realized_volatility > 0:
                indicators.vix_level = indicators.realized_volatility * 100
            
        except Exception as e:
            logger.error(f"计算市场指标失败: {e}")
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """计算RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50.0
        except:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series) -> float:
        """计算MACD"""
        try:
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd = ema12 - ema26
            return macd.iloc[-1] if not macd.empty else 0.0
        except:
            return 0.0
    
    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20) -> float:
        """计算布林带位置"""
        try:
            ma = prices.rolling(window).mean()
            std = prices.rolling(window).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            
            current_price = prices.iloc[-1]
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            
            if current_upper > current_lower:
                position = (current_price - current_lower) / (current_upper - current_lower)
                return np.clip(position, 0, 1)
            else:
                return 0.5
        except:
            return 0.5
    
    def _analyze_market_phase(self, indicators: MarketIndicators) -> MarketPhase:
        """分析市场阶段"""
        try:
            # 基于多个指标综合判断
            score = 0
            
            # 价格动量
            if indicators.momentum_medium > 0.05:
                score += 2
            elif indicators.momentum_medium > 0.02:
                score += 1
            elif indicators.momentum_medium < -0.05:
                score -= 2
            elif indicators.momentum_medium < -0.02:
                score -= 1
            
            # RSI
            if indicators.rsi > 70:
                score += 1
            elif indicators.rsi < 30:
                score -= 1
            
            # 成交量
            if indicators.volume_ratio > 1.2 and indicators.momentum_short > 0:
                score += 1
            elif indicators.volume_ratio > 1.2 and indicators.momentum_short < 0:
                score -= 1
            
            # 波动率
            if indicators.realized_volatility > 0.25:
                if score > 0:
                    return MarketPhase.MARKUP
                else:
                    return MarketPhase.MARKDOWN
            
            # 根据得分判断阶段
            if score >= 3:
                return MarketPhase.MARKUP
            elif score >= 1:
                return MarketPhase.ACCUMULATION
            elif score <= -3:
                return MarketPhase.MARKDOWN
            elif score <= -1:
                return MarketPhase.DISTRIBUTION
            else:
                return MarketPhase.CONSOLIDATION
                
        except Exception as e:
            logger.error(f"分析市场阶段失败: {e}")
            return MarketPhase.CONSOLIDATION
    
    def _analyze_volatility_regime(self, indicators: MarketIndicators) -> VolatilityRegime:
        """分析波动率状态"""
        vol = indicators.realized_volatility
        
        if vol >= self.thresholds['volatility']['extreme']:
            return VolatilityRegime.EXTREME
        elif vol >= self.thresholds['volatility']['high']:
            return VolatilityRegime.HIGH
        elif vol >= self.thresholds['volatility']['normal']:
            return VolatilityRegime.NORMAL
        else:
            return VolatilityRegime.LOW
    
    def _analyze_trend_direction(self, indicators: MarketIndicators) -> TrendDirection:
        """分析趋势方向"""
        momentum = indicators.momentum_medium
        strong_threshold = self.thresholds['trend']['strong_threshold']
        weak_threshold = self.thresholds['trend']['weak_threshold']
        
        if momentum >= strong_threshold:
            return TrendDirection.STRONG_UPTREND
        elif momentum >= weak_threshold:
            return TrendDirection.UPTREND
        elif momentum <= -strong_threshold:
            return TrendDirection.STRONG_DOWNTREND
        elif momentum <= -weak_threshold:
            return TrendDirection.DOWNTREND
        else:
            return TrendDirection.SIDEWAYS
    
    def _analyze_liquidity_condition(self, indicators: MarketIndicators) -> LiquidityCondition:
        """分析流动性状况"""
        # 基于成交量和波动率判断
        volume_ratio = indicators.volume_ratio
        volatility = indicators.realized_volatility
        
        if volume_ratio > 1.5 and volatility < 0.15:
            return LiquidityCondition.ABUNDANT
        elif volume_ratio < 0.7 or volatility > 0.30:
            if volatility > 0.40:
                return LiquidityCondition.STRESSED
            else:
                return LiquidityCondition.TIGHT
        else:
            return LiquidityCondition.NORMAL
    
    def _calculate_phase_confidence(self, indicators: MarketIndicators, phase: MarketPhase) -> float:
        """计算阶段置信度"""
        # 基于指标一致性计算置信度
        confidence_factors = []
        
        # 动量一致性
        if phase in [MarketPhase.MARKUP, MarketPhase.ACCUMULATION]:
            if indicators.momentum_medium > 0:
                confidence_factors.append(min(abs(indicators.momentum_medium) * 10, 1.0))
        elif phase in [MarketPhase.MARKDOWN, MarketPhase.DISTRIBUTION]:
            if indicators.momentum_medium < 0:
                confidence_factors.append(min(abs(indicators.momentum_medium) * 10, 1.0))
        
        # RSI一致性
        if phase == MarketPhase.MARKUP and indicators.rsi > 50:
            confidence_factors.append((indicators.rsi - 50) / 50)
        elif phase == MarketPhase.MARKDOWN and indicators.rsi < 50:
            confidence_factors.append((50 - indicators.rsi) / 50)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_volatility_confidence(self, indicators: MarketIndicators, regime: VolatilityRegime) -> float:
        """计算波动率置信度"""
        vol = indicators.realized_volatility
        thresholds = self.thresholds['volatility']
        
        if regime == VolatilityRegime.LOW:
            return max(0, 1 - vol / thresholds['low'])
        elif regime == VolatilityRegime.NORMAL:
            if vol < thresholds['normal']:
                return 1 - abs(vol - (thresholds['low'] + thresholds['normal']) / 2) / thresholds['normal']
            else:
                return 0.5
        elif regime == VolatilityRegime.HIGH:
            if thresholds['normal'] <= vol < thresholds['high']:
                return (vol - thresholds['normal']) / (thresholds['high'] - thresholds['normal'])
            else:
                return 0.5
        else:  # EXTREME
            return min(1, vol / thresholds['extreme'])
    
    def _calculate_trend_confidence(self, indicators: MarketIndicators, trend: TrendDirection) -> float:
        """计算趋势置信度"""
        momentum = abs(indicators.momentum_medium)
        strong_threshold = self.thresholds['trend']['strong_threshold']
        
        if trend in [TrendDirection.STRONG_UPTREND, TrendDirection.STRONG_DOWNTREND]:
            return min(1.0, momentum / strong_threshold)
        elif trend in [TrendDirection.UPTREND, TrendDirection.DOWNTREND]:
            return min(1.0, momentum / (strong_threshold / 2))
        else:  # SIDEWAYS
            return max(0, 1 - momentum * 10)
    
    def _calculate_liquidity_confidence(self, indicators: MarketIndicators, condition: LiquidityCondition) -> float:
        """计算流动性置信度"""
        volume_ratio = indicators.volume_ratio
        volatility = indicators.realized_volatility
        
        # 基于成交量和波动率的一致性
        if condition == LiquidityCondition.ABUNDANT:
            return min(1.0, volume_ratio / 1.5) * max(0, 1 - volatility / 0.15)
        elif condition == LiquidityCondition.STRESSED:
            return min(1.0, volatility / 0.40)
        else:
            return 0.7  # 默认置信度
    
    def _calculate_phase_duration(self, current_phase: MarketPhase) -> int:
        """计算阶段持续时间"""
        if not self.state_history:
            return 1
        
        duration = 1
        for state in reversed(self.state_history):
            if state.phase == current_phase:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_volatility_duration(self, current_regime: VolatilityRegime) -> int:
        """计算波动率状态持续时间"""
        if not self.state_history:
            return 1
        
        duration = 1
        for state in reversed(self.state_history):
            if state.volatility_regime == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_trend_duration(self, current_trend: TrendDirection) -> int:
        """计算趋势持续时间"""
        if not self.state_history:
            return 1
        
        duration = 1
        for state in reversed(self.state_history):
            if state.trend_direction == current_trend:
                duration += 1
            else:
                break
        
        return duration
    
    def _predict_next_phase(self, current_phase: MarketPhase, indicators: MarketIndicators) -> Dict[str, float]:
        """预测下一阶段概率"""
        # 简化的转换概率模型
        transitions = {
            MarketPhase.ACCUMULATION: {
                MarketPhase.MARKUP.value: 0.4,
                MarketPhase.CONSOLIDATION.value: 0.3,
                MarketPhase.DISTRIBUTION.value: 0.2,
                MarketPhase.ACCUMULATION.value: 0.1
            },
            MarketPhase.MARKUP: {
                MarketPhase.DISTRIBUTION.value: 0.5,
                MarketPhase.MARKUP.value: 0.3,
                MarketPhase.CONSOLIDATION.value: 0.2
            },
            MarketPhase.DISTRIBUTION: {
                MarketPhase.MARKDOWN.value: 0.4,
                MarketPhase.CONSOLIDATION.value: 0.3,
                MarketPhase.ACCUMULATION.value: 0.2,
                MarketPhase.DISTRIBUTION.value: 0.1
            },
            MarketPhase.MARKDOWN: {
                MarketPhase.ACCUMULATION.value: 0.5,
                MarketPhase.MARKDOWN.value: 0.3,
                MarketPhase.CONSOLIDATION.value: 0.2
            },
            MarketPhase.CONSOLIDATION: {
                MarketPhase.ACCUMULATION.value: 0.3,
                MarketPhase.DISTRIBUTION.value: 0.3,
                MarketPhase.CONSOLIDATION.value: 0.2,
                MarketPhase.MARKUP.value: 0.1,
                MarketPhase.MARKDOWN.value: 0.1
            }
        }
        
        return transitions.get(current_phase, {})
    
    def _detect_transition_signals(self, indicators: MarketIndicators) -> List[str]:
        """检测转换信号"""
        signals = []
        
        # RSI极值信号
        if indicators.rsi > 80:
            signals.append("RSI超买")
        elif indicators.rsi < 20:
            signals.append("RSI超卖")
        
        # 波动率突变信号
        if indicators.realized_volatility > 0.35:
            signals.append("波动率飙升")
        elif indicators.realized_volatility < 0.08:
            signals.append("波动率极低")
        
        # 成交量异常信号
        if indicators.volume_ratio > 2.0:
            signals.append("成交量放大")
        elif indicators.volume_ratio < 0.5:
            signals.append("成交量萎缩")
        
        # 动量背离信号
        if indicators.momentum_short * indicators.momentum_medium < 0:
            signals.append("动量背离")
        
        return signals
    
    def _detect_state_transitions(self, old_state: MarketState, new_state: MarketState):
        """检测状态转换"""
        transitions = []
        
        # 检测阶段转换
        if old_state.phase != new_state.phase:
            transition = StateTransition(
                timestamp=datetime.now(),
                from_state=old_state,
                to_state=new_state,
                transition_type="phase",
                trigger_indicators=new_state.transition_signals
            )
            transitions.append(transition)
        
        # 检测波动率状态转换
        if old_state.volatility_regime != new_state.volatility_regime:
            transition = StateTransition(
                timestamp=datetime.now(),
                from_state=old_state,
                to_state=new_state,
                transition_type="volatility",
                trigger_indicators=["波动率变化"]
            )
            transitions.append(transition)
        
        # 检测趋势转换
        if old_state.trend_direction != new_state.trend_direction:
            transition = StateTransition(
                timestamp=datetime.now(),
                from_state=old_state,
                to_state=new_state,
                transition_type="trend",
                trigger_indicators=["趋势变化"]
            )
            transitions.append(transition)
        
        self.transition_history.extend(transitions)
        
        for transition in transitions:
            logger.info(f"检测到状态转换: {transition.transition_type} - {transition.from_state.phase.value} -> {transition.to_state.phase.value}")
    
    def _get_default_state(self) -> MarketState:
        """获取默认状态"""
        return MarketState(
            timestamp=datetime.now(),
            phase=MarketPhase.CONSOLIDATION,
            volatility_regime=VolatilityRegime.NORMAL,
            trend_direction=TrendDirection.SIDEWAYS,
            liquidity_condition=LiquidityCondition.NORMAL
        )
    
    def get_current_state(self) -> Optional[MarketState]:
        """获取当前市场状态"""
        with self.lock:
            return self.current_state
    
    def get_state_history(self, limit: int = None) -> List[MarketState]:
        """获取状态历史"""
        with self.lock:
            if limit:
                return self.state_history[-limit:]
            return self.state_history.copy()
    
    def get_transition_history(self, limit: int = None) -> List[StateTransition]:
        """获取转换历史"""
        with self.lock:
            if limit:
                return self.transition_history[-limit:]
            return self.transition_history.copy()
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        with self.lock:
            if not self.state_history:
                return {}
            
            # 统计各状态的分布
            phase_counts = defaultdict(int)
            volatility_counts = defaultdict(int)
            trend_counts = defaultdict(int)
            
            for state in self.state_history:
                phase_counts[state.phase.value] += 1
                volatility_counts[state.volatility_regime.value] += 1
                trend_counts[state.trend_direction.value] += 1
            
            total_states = len(self.state_history)
            
            return {
                'total_analyses': total_states,
                'total_transitions': len(self.transition_history),
                'phase_distribution': {phase: count/total_states for phase, count in phase_counts.items()},
                'volatility_distribution': {vol: count/total_states for vol, count in volatility_counts.items()},
                'trend_distribution': {trend: count/total_states for trend, count in trend_counts.items()},
                'current_state': self.current_state.to_dict() if self.current_state else None,
                'avg_confidence': np.mean([state.get_overall_confidence() for state in self.state_history[-50:]]) if self.state_history else 0.0
            }

# 便捷函数
def create_market_state_analyzer(config: Dict[str, Any] = None) -> MarketStateAnalyzer:
    """创建市场状态分析器
    
    Args:
        config: 配置参数
        
    Returns:
        市场状态分析器实例
    """
    if config is None:
        config = {
            'lookback_window': 252,
            'volatility_window': 20,
            'trend_window': 50
        }
    
    return MarketStateAnalyzer(config)