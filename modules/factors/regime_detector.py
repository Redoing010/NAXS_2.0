# 市场状态识别器 - 使用HMM和贝叶斯切换模型识别市场周期

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError:
    GaussianMixture = None
    KMeans = None
    StandardScaler = None

try:
    from hmmlearn import hmm
except ImportError:
    hmm = None

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """市场状态枚举"""
    BULL_MARKET = "bull_market"      # 牛市
    BEAR_MARKET = "bear_market"      # 熊市
    SIDEWAYS = "sideways"            # 震荡市
    HIGH_VOLATILITY = "high_vol"     # 高波动
    LOW_VOLATILITY = "low_vol"       # 低波动
    CRISIS = "crisis"                # 危机状态
    RECOVERY = "recovery"            # 复苏状态

@dataclass
class RegimeState:
    """市场状态"""
    regime: MarketRegime
    probability: float
    confidence: float
    duration: int  # 持续天数
    characteristics: Dict[str, float]
    detected_at: datetime

@dataclass
class RegimeTransition:
    """状态转换"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_date: datetime
    transition_probability: float
    trigger_factors: List[str]

class RegimeDetector:
    """市场状态识别器
    
    使用多种方法识别市场状态：
    1. 隐马尔可夫模型(HMM)
    2. 高斯混合模型(GMM)
    3. 贝叶斯切换模型
    4. 基于规则的识别
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get('method', 'hybrid')  # hmm, gmm, bayesian, rule, hybrid
        self.n_regimes = config.get('n_regimes', 4)
        self.lookback_window = config.get('lookback_window', 252)  # 一年
        
        # 模型组件
        self.hmm_model = None
        self.gmm_model = None
        self.scaler = StandardScaler() if StandardScaler else None
        
        # 历史状态
        self.regime_history: List[RegimeState] = []
        self.transition_history: List[RegimeTransition] = []
        
        # 特征定义
        self._init_features()
        
        # 规则阈值
        self._init_rule_thresholds()
        
        logger.info(f"市场状态识别器初始化完成，方法: {self.method}")
    
    def _init_features(self):
        """初始化特征定义"""
        self.feature_names = [
            'return_1d',      # 1日收益率
            'return_5d',      # 5日收益率
            'return_20d',     # 20日收益率
            'volatility_20d', # 20日波动率
            'volatility_60d', # 60日波动率
            'rsi',            # RSI指标
            'macd',           # MACD指标
            'volume_ratio',   # 成交量比率
            'vix',            # 波动率指数
            'yield_spread',   # 收益率利差
            'momentum_12m',   # 12个月动量
            'drawdown'        # 最大回撤
        ]
    
    def _init_rule_thresholds(self):
        """初始化规则阈值"""
        self.rule_thresholds = {
            'bull_market': {
                'return_20d_min': 0.02,
                'volatility_20d_max': 0.15,
                'rsi_min': 50,
                'drawdown_max': -0.05
            },
            'bear_market': {
                'return_20d_max': -0.02,
                'volatility_20d_min': 0.15,
                'rsi_max': 50,
                'drawdown_min': -0.15
            },
            'high_volatility': {
                'volatility_20d_min': 0.25,
                'vix_min': 25
            },
            'crisis': {
                'return_5d_max': -0.10,
                'volatility_20d_min': 0.30,
                'vix_min': 35,
                'drawdown_min': -0.20
            }
        }
    
    async def detect_regime(self, market_data: pd.DataFrame) -> RegimeState:
        """检测当前市场状态
        
        Args:
            market_data: 市场数据，包含价格、成交量等信息
            
        Returns:
            当前市场状态
        """
        try:
            # 计算特征
            features = self._calculate_features(market_data)
            
            if features is None or len(features) == 0:
                return self._get_default_regime()
            
            # 根据方法选择检测算法
            if self.method == 'hmm':
                regime_state = await self._detect_by_hmm(features)
            elif self.method == 'gmm':
                regime_state = await self._detect_by_gmm(features)
            elif self.method == 'bayesian':
                regime_state = await self._detect_by_bayesian(features)
            elif self.method == 'rule':
                regime_state = await self._detect_by_rules(features)
            else:  # hybrid
                regime_state = await self._detect_by_hybrid(features)
            
            # 更新历史记录
            self._update_regime_history(regime_state)
            
            logger.info(f"市场状态检测完成: {regime_state.regime.value}, 置信度: {regime_state.confidence:.3f}")
            return regime_state
            
        except Exception as e:
            logger.error(f"市场状态检测失败: {str(e)}")
            return self._get_default_regime()
    
    def _calculate_features(self, market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """计算市场特征"""
        try:
            if pd is None or market_data.empty:
                return None
            
            # 确保数据按日期排序
            if 'date' in market_data.columns:
                market_data = market_data.sort_values('date')
            
            # 计算收益率
            if 'close' in market_data.columns:
                market_data['return_1d'] = market_data['close'].pct_change()
                market_data['return_5d'] = market_data['close'].pct_change(5)
                market_data['return_20d'] = market_data['close'].pct_change(20)
                market_data['momentum_12m'] = market_data['close'].pct_change(252)
            
            # 计算波动率
            if 'return_1d' in market_data.columns:
                market_data['volatility_20d'] = market_data['return_1d'].rolling(20).std() * np.sqrt(252)
                market_data['volatility_60d'] = market_data['return_1d'].rolling(60).std() * np.sqrt(252)
            
            # 计算RSI
            if 'close' in market_data.columns:
                market_data['rsi'] = self._calculate_rsi(market_data['close'])
            
            # 计算MACD
            if 'close' in market_data.columns:
                market_data['macd'] = self._calculate_macd(market_data['close'])
            
            # 计算成交量比率
            if 'volume' in market_data.columns:
                market_data['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(20).mean()
            
            # 计算最大回撤
            if 'close' in market_data.columns:
                market_data['drawdown'] = self._calculate_drawdown(market_data['close'])
            
            # 添加VIX和收益率利差（如果有的话）
            if 'vix' not in market_data.columns:
                market_data['vix'] = market_data.get('volatility_20d', 0.15) * 100
            
            if 'yield_spread' not in market_data.columns:
                market_data['yield_spread'] = 0.02  # 默认值
            
            # 提取特征
            feature_data = []
            for feature_name in self.feature_names:
                if feature_name in market_data.columns:
                    values = market_data[feature_name].fillna(0).values
                    if len(values) > 0:
                        feature_data.append(values[-1])  # 取最新值
                    else:
                        feature_data.append(0.0)
                else:
                    feature_data.append(0.0)
            
            return np.array(feature_data).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"特征计算失败: {str(e)}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_drawdown(self, prices: pd.Series) -> pd.Series:
        """计算最大回撤"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown
    
    async def _detect_by_hmm(self, features: np.ndarray) -> RegimeState:
        """使用HMM检测市场状态"""
        try:
            if hmm is None:
                logger.warning("hmmlearn未安装，回退到规则方法")
                return await self._detect_by_rules(features)
            
            # 初始化或训练HMM模型
            if self.hmm_model is None:
                self.hmm_model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_iter=100
                )
                
                # 需要历史数据来训练模型
                # 这里使用当前特征作为示例
                if self.scaler:
                    scaled_features = self.scaler.fit_transform(features)
                else:
                    scaled_features = features
                
                self.hmm_model.fit(scaled_features)
            
            # 预测当前状态
            if self.scaler:
                scaled_features = self.scaler.transform(features)
            else:
                scaled_features = features
            
            state_probs = self.hmm_model.predict_proba(scaled_features)[0]
            predicted_state = np.argmax(state_probs)
            confidence = state_probs[predicted_state]
            
            # 映射到市场状态
            regime = self._map_state_to_regime(predicted_state, features[0])
            
            return RegimeState(
                regime=regime,
                probability=confidence,
                confidence=confidence,
                duration=1,
                characteristics=dict(zip(self.feature_names, features[0])),
                detected_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"HMM检测失败: {str(e)}")
            return await self._detect_by_rules(features)
    
    async def _detect_by_gmm(self, features: np.ndarray) -> RegimeState:
        """使用高斯混合模型检测市场状态"""
        try:
            if GaussianMixture is None:
                logger.warning("sklearn未安装，回退到规则方法")
                return await self._detect_by_rules(features)
            
            # 初始化或训练GMM模型
            if self.gmm_model is None:
                self.gmm_model = GaussianMixture(
                    n_components=self.n_regimes,
                    covariance_type='full',
                    max_iter=100
                )
                
                # 标准化特征
                if self.scaler:
                    scaled_features = self.scaler.fit_transform(features)
                else:
                    scaled_features = features
                
                self.gmm_model.fit(scaled_features)
            
            # 预测当前状态
            if self.scaler:
                scaled_features = self.scaler.transform(features)
            else:
                scaled_features = features
            
            state_probs = self.gmm_model.predict_proba(scaled_features)[0]
            predicted_state = np.argmax(state_probs)
            confidence = state_probs[predicted_state]
            
            # 映射到市场状态
            regime = self._map_state_to_regime(predicted_state, features[0])
            
            return RegimeState(
                regime=regime,
                probability=confidence,
                confidence=confidence,
                duration=1,
                characteristics=dict(zip(self.feature_names, features[0])),
                detected_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"GMM检测失败: {str(e)}")
            return await self._detect_by_rules(features)
    
    async def _detect_by_bayesian(self, features: np.ndarray) -> RegimeState:
        """使用贝叶斯切换模型检测市场状态"""
        # 简化的贝叶斯方法实现
        try:
            feature_dict = dict(zip(self.feature_names, features[0]))
            
            # 计算每个状态的后验概率
            regime_probs = {}
            
            for regime in MarketRegime:
                # 计算似然概率（基于特征的正态分布假设）
                likelihood = self._calculate_likelihood(feature_dict, regime)
                
                # 使用均匀先验
                prior = 1.0 / len(MarketRegime)
                
                # 后验概率
                regime_probs[regime] = likelihood * prior
            
            # 归一化
            total_prob = sum(regime_probs.values())
            if total_prob > 0:
                regime_probs = {k: v / total_prob for k, v in regime_probs.items()}
            
            # 选择概率最高的状态
            best_regime = max(regime_probs.items(), key=lambda x: x[1])
            
            return RegimeState(
                regime=best_regime[0],
                probability=best_regime[1],
                confidence=best_regime[1],
                duration=1,
                characteristics=feature_dict,
                detected_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"贝叶斯检测失败: {str(e)}")
            return await self._detect_by_rules(features)
    
    def _calculate_likelihood(self, features: Dict[str, float], regime: MarketRegime) -> float:
        """计算给定状态下特征的似然概率"""
        # 简化的似然计算
        likelihood = 1.0
        
        if regime == MarketRegime.BULL_MARKET:
            if features.get('return_20d', 0) > 0.01:
                likelihood *= 2.0
            if features.get('volatility_20d', 0.2) < 0.15:
                likelihood *= 1.5
            if features.get('rsi', 50) > 50:
                likelihood *= 1.3
        
        elif regime == MarketRegime.BEAR_MARKET:
            if features.get('return_20d', 0) < -0.01:
                likelihood *= 2.0
            if features.get('volatility_20d', 0.2) > 0.15:
                likelihood *= 1.5
            if features.get('rsi', 50) < 50:
                likelihood *= 1.3
        
        elif regime == MarketRegime.HIGH_VOLATILITY:
            if features.get('volatility_20d', 0.2) > 0.25:
                likelihood *= 3.0
            if features.get('vix', 20) > 25:
                likelihood *= 2.0
        
        elif regime == MarketRegime.CRISIS:
            if features.get('return_5d', 0) < -0.05:
                likelihood *= 2.5
            if features.get('volatility_20d', 0.2) > 0.30:
                likelihood *= 2.0
            if features.get('drawdown', 0) < -0.15:
                likelihood *= 2.0
        
        return likelihood
    
    async def _detect_by_rules(self, features: np.ndarray) -> RegimeState:
        """使用规则方法检测市场状态"""
        try:
            feature_dict = dict(zip(self.feature_names, features[0]))
            
            # 检查危机状态（优先级最高）
            if self._check_regime_conditions(feature_dict, 'crisis'):
                regime = MarketRegime.CRISIS
                confidence = 0.9
            
            # 检查高波动状态
            elif self._check_regime_conditions(feature_dict, 'high_volatility'):
                regime = MarketRegime.HIGH_VOLATILITY
                confidence = 0.8
            
            # 检查牛市状态
            elif self._check_regime_conditions(feature_dict, 'bull_market'):
                regime = MarketRegime.BULL_MARKET
                confidence = 0.8
            
            # 检查熊市状态
            elif self._check_regime_conditions(feature_dict, 'bear_market'):
                regime = MarketRegime.BEAR_MARKET
                confidence = 0.8
            
            # 默认为震荡市
            else:
                regime = MarketRegime.SIDEWAYS
                confidence = 0.6
            
            return RegimeState(
                regime=regime,
                probability=confidence,
                confidence=confidence,
                duration=1,
                characteristics=feature_dict,
                detected_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"规则检测失败: {str(e)}")
            return self._get_default_regime()
    
    def _check_regime_conditions(self, features: Dict[str, float], regime_key: str) -> bool:
        """检查状态条件"""
        if regime_key not in self.rule_thresholds:
            return False
        
        conditions = self.rule_thresholds[regime_key]
        
        for condition, threshold in conditions.items():
            feature_name, operator = condition.rsplit('_', 1)
            feature_value = features.get(feature_name, 0)
            
            if operator == 'min' and feature_value < threshold:
                return False
            elif operator == 'max' and feature_value > threshold:
                return False
        
        return True
    
    async def _detect_by_hybrid(self, features: np.ndarray) -> RegimeState:
        """使用混合方法检测市场状态"""
        try:
            # 获取多种方法的结果
            results = []
            
            # 规则方法（权重0.4）
            rule_result = await self._detect_by_rules(features)
            results.append((rule_result, 0.4))
            
            # GMM方法（权重0.3）
            if GaussianMixture is not None:
                gmm_result = await self._detect_by_gmm(features)
                results.append((gmm_result, 0.3))
            
            # 贝叶斯方法（权重0.3）
            bayesian_result = await self._detect_by_bayesian(features)
            results.append((bayesian_result, 0.3))
            
            # 加权投票
            regime_votes = {}
            total_weight = 0
            
            for result, weight in results:
                regime = result.regime
                confidence = result.confidence
                
                if regime not in regime_votes:
                    regime_votes[regime] = 0
                
                regime_votes[regime] += weight * confidence
                total_weight += weight
            
            # 选择得票最高的状态
            if regime_votes:
                best_regime = max(regime_votes.items(), key=lambda x: x[1])
                final_confidence = best_regime[1] / total_weight if total_weight > 0 else 0.5
                
                return RegimeState(
                    regime=best_regime[0],
                    probability=final_confidence,
                    confidence=final_confidence,
                    duration=1,
                    characteristics=dict(zip(self.feature_names, features[0])),
                    detected_at=datetime.now()
                )
            
            return self._get_default_regime()
            
        except Exception as e:
            logger.error(f"混合检测失败: {str(e)}")
            return await self._detect_by_rules(features)
    
    def _map_state_to_regime(self, state_id: int, features: np.ndarray) -> MarketRegime:
        """将模型状态映射到市场状态"""
        # 基于特征值进行映射
        feature_dict = dict(zip(self.feature_names, features))
        
        return_20d = feature_dict.get('return_20d', 0)
        volatility = feature_dict.get('volatility_20d', 0.2)
        drawdown = feature_dict.get('drawdown', 0)
        
        # 简单的映射逻辑
        if drawdown < -0.20 or volatility > 0.30:
            return MarketRegime.CRISIS
        elif volatility > 0.25:
            return MarketRegime.HIGH_VOLATILITY
        elif return_20d > 0.02:
            return MarketRegime.BULL_MARKET
        elif return_20d < -0.02:
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.SIDEWAYS
    
    def _get_default_regime(self) -> RegimeState:
        """获取默认市场状态"""
        return RegimeState(
            regime=MarketRegime.SIDEWAYS,
            probability=0.5,
            confidence=0.5,
            duration=1,
            characteristics={},
            detected_at=datetime.now()
        )
    
    def _update_regime_history(self, new_state: RegimeState):
        """更新状态历史"""
        # 检查是否发生状态转换
        if self.regime_history:
            last_state = self.regime_history[-1]
            
            if last_state.regime != new_state.regime:
                # 记录状态转换
                transition = RegimeTransition(
                    from_regime=last_state.regime,
                    to_regime=new_state.regime,
                    transition_date=datetime.now(),
                    transition_probability=new_state.probability,
                    trigger_factors=self._identify_trigger_factors(last_state, new_state)
                )
                self.transition_history.append(transition)
                
                logger.info(f"市场状态转换: {last_state.regime.value} -> {new_state.regime.value}")
            else:
                # 更新持续时间
                new_state.duration = last_state.duration + 1
        
        # 添加到历史记录
        self.regime_history.append(new_state)
        
        # 保持历史记录在合理范围内
        max_history = self.config.get('max_history', 1000)
        if len(self.regime_history) > max_history:
            self.regime_history = self.regime_history[-max_history:]
    
    def _identify_trigger_factors(self, old_state: RegimeState, new_state: RegimeState) -> List[str]:
        """识别状态转换的触发因子"""
        triggers = []
        
        old_chars = old_state.characteristics
        new_chars = new_state.characteristics
        
        # 检查显著变化的特征
        for feature_name in self.feature_names:
            old_val = old_chars.get(feature_name, 0)
            new_val = new_chars.get(feature_name, 0)
            
            if abs(new_val - old_val) > 0.1:  # 阈值可配置
                triggers.append(feature_name)
        
        return triggers
    
    def get_regime_probabilities(self, market_data: pd.DataFrame) -> Dict[MarketRegime, float]:
        """获取所有状态的概率分布"""
        try:
            features = self._calculate_features(market_data)
            if features is None:
                return {regime: 1.0/len(MarketRegime) for regime in MarketRegime}
            
            # 使用贝叶斯方法计算所有状态概率
            feature_dict = dict(zip(self.feature_names, features[0]))
            regime_probs = {}
            
            for regime in MarketRegime:
                likelihood = self._calculate_likelihood(feature_dict, regime)
                prior = 1.0 / len(MarketRegime)
                regime_probs[regime] = likelihood * prior
            
            # 归一化
            total_prob = sum(regime_probs.values())
            if total_prob > 0:
                regime_probs = {k: v / total_prob for k, v in regime_probs.items()}
            
            return regime_probs
            
        except Exception as e:
            logger.error(f"概率计算失败: {str(e)}")
            return {regime: 1.0/len(MarketRegime) for regime in MarketRegime}
    
    def get_regime_history(self, days: int = 30) -> List[RegimeState]:
        """获取历史状态记录"""
        if days <= 0:
            return self.regime_history.copy()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            state for state in self.regime_history 
            if state.detected_at >= cutoff_date
        ]
    
    def get_transition_matrix(self) -> Dict[Tuple[MarketRegime, MarketRegime], float]:
        """获取状态转换矩阵"""
        if not self.transition_history:
            return {}
        
        # 统计转换次数
        transition_counts = {}
        regime_counts = {}
        
        for transition in self.transition_history:
            from_regime = transition.from_regime
            to_regime = transition.to_regime
            
            key = (from_regime, to_regime)
            transition_counts[key] = transition_counts.get(key, 0) + 1
            regime_counts[from_regime] = regime_counts.get(from_regime, 0) + 1
        
        # 计算转换概率
        transition_matrix = {}
        for (from_regime, to_regime), count in transition_counts.items():
            total_from = regime_counts.get(from_regime, 1)
            transition_matrix[(from_regime, to_regime)] = count / total_from
        
        return transition_matrix
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检测器统计信息"""
        return {
            'method': self.method,
            'n_regimes': self.n_regimes,
            'lookback_window': self.lookback_window,
            'total_detections': len(self.regime_history),
            'total_transitions': len(self.transition_history),
            'current_regime': self.regime_history[-1].regime.value if self.regime_history else None,
            'regime_distribution': {
                regime.value: sum(1 for s in self.regime_history if s.regime == regime)
                for regime in MarketRegime
            },
            'avg_regime_duration': {
                regime.value: np.mean([s.duration for s in self.regime_history if s.regime == regime])
                for regime in MarketRegime
            } if self.regime_history else {}
        }