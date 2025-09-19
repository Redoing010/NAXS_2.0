# 特征计算引擎 - 计算和生成各种金融特征

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

logger = logging.getLogger(__name__)

class FeatureCategory(Enum):
    """特征类别"""
    PRICE = "price"                    # 价格特征
    VOLUME = "volume"                  # 成交量特征
    TECHNICAL = "technical"            # 技术指标
    FUNDAMENTAL = "fundamental"        # 基本面特征
    SENTIMENT = "sentiment"            # 情绪特征
    MACRO = "macro"                    # 宏观特征
    CROSS_SECTIONAL = "cross_sectional" # 横截面特征
    TIME_SERIES = "time_series"        # 时间序列特征
    INTERACTION = "interaction"        # 交互特征

class ComputeMode(Enum):
    """计算模式"""
    BATCH = "batch"        # 批量计算
    STREAMING = "streaming" # 流式计算
    INCREMENTAL = "incremental" # 增量计算

@dataclass
class FeatureConfig:
    """特征配置"""
    name: str
    category: FeatureCategory
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    window_size: Optional[int] = None
    min_periods: Optional[int] = None
    compute_mode: ComputeMode = ComputeMode.BATCH
    cache_enabled: bool = True
    parallel_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'category': self.category.value,
            'description': self.description,
            'params': self.params,
            'dependencies': self.dependencies,
            'window_size': self.window_size,
            'min_periods': self.min_periods,
            'compute_mode': self.compute_mode.value,
            'cache_enabled': self.cache_enabled,
            'parallel_enabled': self.parallel_enabled
        }

@dataclass
class ComputeResult:
    """计算结果"""
    feature_name: str
    data: pd.Series
    metadata: Dict[str, Any] = field(default_factory=dict)
    compute_time: float = 0.0
    success: bool = True
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'feature_name': self.feature_name,
            'data_shape': self.data.shape if self.data is not None else None,
            'data_type': str(self.data.dtype) if self.data is not None else None,
            'metadata': self.metadata,
            'compute_time': self.compute_time,
            'success': self.success,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat()
        }

class FeatureComputer(ABC):
    """特征计算器基类"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.name = config.name
        self.category = config.category
        self.params = config.params
        self.dependencies = config.dependencies
    
    @abstractmethod
    def compute(self, data: Dict[str, pd.Series], **kwargs) -> pd.Series:
        """计算特征
        
        Args:
            data: 输入数据字典
            **kwargs: 额外参数
            
        Returns:
            计算结果
        """
        pass
    
    def validate_inputs(self, data: Dict[str, pd.Series]) -> bool:
        """验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            是否有效
        """
        # 检查依赖项
        for dep in self.dependencies:
            if dep not in data:
                logger.error(f"缺少依赖项: {dep}")
                return False
            if data[dep] is None or len(data[dep]) == 0:
                logger.error(f"依赖项数据为空: {dep}")
                return False
        
        return True
    
    def preprocess_data(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """预处理数据
        
        Args:
            data: 原始数据
            
        Returns:
            预处理后的数据
        """
        processed_data = {}
        
        for key, series in data.items():
            if key in self.dependencies:
                # 基本清理
                processed_series = series.copy()
                
                # 处理无穷值
                processed_series = processed_series.replace([np.inf, -np.inf], np.nan)
                
                # 如果有窗口大小限制，截取数据
                if self.config.window_size and len(processed_series) > self.config.window_size:
                    processed_series = processed_series.tail(self.config.window_size)
                
                processed_data[key] = processed_series
            else:
                processed_data[key] = series
        
        return processed_data

class PriceFeatureComputer(FeatureComputer):
    """价格特征计算器"""
    
    def compute(self, data: Dict[str, pd.Series], **kwargs) -> pd.Series:
        """计算价格特征"""
        if not self.validate_inputs(data):
            raise ValueError("输入数据验证失败")
        
        processed_data = self.preprocess_data(data)
        
        if self.name == 'returns':
            return self._compute_returns(processed_data)
        elif self.name == 'log_returns':
            return self._compute_log_returns(processed_data)
        elif self.name == 'price_change':
            return self._compute_price_change(processed_data)
        elif self.name == 'price_ratio':
            return self._compute_price_ratio(processed_data)
        elif self.name == 'high_low_ratio':
            return self._compute_high_low_ratio(processed_data)
        elif self.name == 'close_to_high':
            return self._compute_close_to_high(processed_data)
        elif self.name == 'close_to_low':
            return self._compute_close_to_low(processed_data)
        else:
            raise ValueError(f"不支持的价格特征: {self.name}")
    
    def _compute_returns(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算收益率"""
        close = data['close']
        periods = self.params.get('periods', 1)
        return close.pct_change(periods=periods)
    
    def _compute_log_returns(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算对数收益率"""
        close = data['close']
        periods = self.params.get('periods', 1)
        return np.log(close / close.shift(periods))
    
    def _compute_price_change(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算价格变化"""
        close = data['close']
        periods = self.params.get('periods', 1)
        return close.diff(periods=periods)
    
    def _compute_price_ratio(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算价格比率"""
        close = data['close']
        periods = self.params.get('periods', 1)
        return close / close.shift(periods)
    
    def _compute_high_low_ratio(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算最高最低价比率"""
        high = data['high']
        low = data['low']
        return high / low
    
    def _compute_close_to_high(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算收盘价相对最高价位置"""
        close = data['close']
        high = data['high']
        return close / high
    
    def _compute_close_to_low(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算收盘价相对最低价位置"""
        close = data['close']
        low = data['low']
        return close / low

class VolumeFeatureComputer(FeatureComputer):
    """成交量特征计算器"""
    
    def compute(self, data: Dict[str, pd.Series], **kwargs) -> pd.Series:
        """计算成交量特征"""
        if not self.validate_inputs(data):
            raise ValueError("输入数据验证失败")
        
        processed_data = self.preprocess_data(data)
        
        if self.name == 'volume_change':
            return self._compute_volume_change(processed_data)
        elif self.name == 'volume_ratio':
            return self._compute_volume_ratio(processed_data)
        elif self.name == 'volume_ma':
            return self._compute_volume_ma(processed_data)
        elif self.name == 'volume_std':
            return self._compute_volume_std(processed_data)
        elif self.name == 'vwap':
            return self._compute_vwap(processed_data)
        elif self.name == 'volume_price_trend':
            return self._compute_volume_price_trend(processed_data)
        else:
            raise ValueError(f"不支持的成交量特征: {self.name}")
    
    def _compute_volume_change(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算成交量变化"""
        volume = data['volume']
        periods = self.params.get('periods', 1)
        return volume.pct_change(periods=periods)
    
    def _compute_volume_ratio(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算成交量比率"""
        volume = data['volume']
        periods = self.params.get('periods', 1)
        return volume / volume.shift(periods)
    
    def _compute_volume_ma(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算成交量移动平均"""
        volume = data['volume']
        window = self.params.get('window', 20)
        return volume.rolling(window=window, min_periods=self.config.min_periods).mean()
    
    def _compute_volume_std(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算成交量标准差"""
        volume = data['volume']
        window = self.params.get('window', 20)
        return volume.rolling(window=window, min_periods=self.config.min_periods).std()
    
    def _compute_vwap(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算成交量加权平均价格"""
        close = data['close']
        volume = data['volume']
        window = self.params.get('window', 20)
        
        # 计算价格*成交量的滚动和
        pv_sum = (close * volume).rolling(window=window, min_periods=self.config.min_periods).sum()
        # 计算成交量的滚动和
        v_sum = volume.rolling(window=window, min_periods=self.config.min_periods).sum()
        
        return pv_sum / v_sum
    
    def _compute_volume_price_trend(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算成交量价格趋势"""
        close = data['close']
        volume = data['volume']
        
        # 价格变化方向
        price_direction = np.sign(close.diff())
        # 成交量加权的价格趋势
        return (price_direction * volume).rolling(
            window=self.params.get('window', 20),
            min_periods=self.config.min_periods
        ).sum()

class TechnicalFeatureComputer(FeatureComputer):
    """技术指标特征计算器"""
    
    def compute(self, data: Dict[str, pd.Series], **kwargs) -> pd.Series:
        """计算技术指标特征"""
        if not self.validate_inputs(data):
            raise ValueError("输入数据验证失败")
        
        processed_data = self.preprocess_data(data)
        
        if self.name == 'sma':
            return self._compute_sma(processed_data)
        elif self.name == 'ema':
            return self._compute_ema(processed_data)
        elif self.name == 'rsi':
            return self._compute_rsi(processed_data)
        elif self.name == 'macd':
            return self._compute_macd(processed_data)
        elif self.name == 'bollinger_bands':
            return self._compute_bollinger_bands(processed_data)
        elif self.name == 'stochastic':
            return self._compute_stochastic(processed_data)
        elif self.name == 'atr':
            return self._compute_atr(processed_data)
        elif self.name == 'momentum':
            return self._compute_momentum(processed_data)
        else:
            raise ValueError(f"不支持的技术指标: {self.name}")
    
    def _compute_sma(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算简单移动平均"""
        close = data['close']
        window = self.params.get('window', 20)
        return close.rolling(window=window, min_periods=self.config.min_periods).mean()
    
    def _compute_ema(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算指数移动平均"""
        close = data['close']
        span = self.params.get('span', 20)
        return close.ewm(span=span, min_periods=self.config.min_periods).mean()
    
    def _compute_rsi(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算相对强弱指数"""
        close = data['close']
        window = self.params.get('window', 14)
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=self.config.min_periods).mean()
        avg_loss = loss.rolling(window=window, min_periods=self.config.min_periods).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _compute_macd(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算MACD"""
        close = data['close']
        fast_period = self.params.get('fast_period', 12)
        slow_period = self.params.get('slow_period', 26)
        signal_period = self.params.get('signal_period', 9)
        
        ema_fast = close.ewm(span=fast_period).mean()
        ema_slow = close.ewm(span=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        if self.params.get('return_type', 'macd') == 'macd':
            return macd_line
        elif self.params.get('return_type') == 'signal':
            return signal_line
        else:  # histogram
            return macd_line - signal_line
    
    def _compute_bollinger_bands(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算布林带"""
        close = data['close']
        window = self.params.get('window', 20)
        num_std = self.params.get('num_std', 2)
        
        sma = close.rolling(window=window, min_periods=self.config.min_periods).mean()
        std = close.rolling(window=window, min_periods=self.config.min_periods).std()
        
        if self.params.get('return_type', 'middle') == 'middle':
            return sma
        elif self.params.get('return_type') == 'upper':
            return sma + (std * num_std)
        elif self.params.get('return_type') == 'lower':
            return sma - (std * num_std)
        else:  # bandwidth
            upper = sma + (std * num_std)
            lower = sma - (std * num_std)
            return (upper - lower) / sma
    
    def _compute_stochastic(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算随机指标"""
        high = data['high']
        low = data['low']
        close = data['close']
        window = self.params.get('window', 14)
        
        lowest_low = low.rolling(window=window, min_periods=self.config.min_periods).min()
        highest_high = high.rolling(window=window, min_periods=self.config.min_periods).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        if self.params.get('return_type', 'k') == 'k':
            return k_percent
        else:  # d
            d_window = self.params.get('d_window', 3)
            return k_percent.rolling(window=d_window, min_periods=1).mean()
    
    def _compute_atr(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算平均真实波幅"""
        high = data['high']
        low = data['low']
        close = data['close']
        window = self.params.get('window', 14)
        
        # 计算真实波幅
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return true_range.rolling(window=window, min_periods=self.config.min_periods).mean()
    
    def _compute_momentum(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算动量指标"""
        close = data['close']
        window = self.params.get('window', 10)
        
        return close / close.shift(window) - 1

class CrossSectionalFeatureComputer(FeatureComputer):
    """横截面特征计算器"""
    
    def compute(self, data: Dict[str, pd.Series], **kwargs) -> pd.Series:
        """计算横截面特征"""
        if not self.validate_inputs(data):
            raise ValueError("输入数据验证失败")
        
        processed_data = self.preprocess_data(data)
        
        if self.name == 'rank':
            return self._compute_rank(processed_data)
        elif self.name == 'zscore':
            return self._compute_zscore(processed_data)
        elif self.name == 'percentile':
            return self._compute_percentile(processed_data)
        elif self.name == 'industry_neutral':
            return self._compute_industry_neutral(processed_data, kwargs)
        else:
            raise ValueError(f"不支持的横截面特征: {self.name}")
    
    def _compute_rank(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算排名"""
        target = data[self.dependencies[0]]
        ascending = self.params.get('ascending', True)
        pct = self.params.get('pct', False)
        
        return target.rank(ascending=ascending, pct=pct)
    
    def _compute_zscore(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算Z分数"""
        target = data[self.dependencies[0]]
        
        return (target - target.mean()) / target.std()
    
    def _compute_percentile(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算百分位数"""
        target = data[self.dependencies[0]]
        
        return target.rank(pct=True)
    
    def _compute_industry_neutral(self, data: Dict[str, pd.Series], kwargs: Dict[str, Any]) -> pd.Series:
        """计算行业中性化特征"""
        target = data[self.dependencies[0]]
        industry_data = kwargs.get('industry_data')
        
        if industry_data is None:
            logger.warning("缺少行业数据，返回原始特征")
            return target
        
        # 按行业分组计算中性化
        result = target.copy()
        for industry in industry_data.unique():
            mask = industry_data == industry
            industry_values = target[mask]
            if len(industry_values) > 1:
                industry_mean = industry_values.mean()
                industry_std = industry_values.std()
                if industry_std > 0:
                    result[mask] = (industry_values - industry_mean) / industry_std
        
        return result

class InteractionFeatureComputer(FeatureComputer):
    """交互特征计算器"""
    
    def compute(self, data: Dict[str, pd.Series], **kwargs) -> pd.Series:
        """计算交互特征"""
        if not self.validate_inputs(data):
            raise ValueError("输入数据验证失败")
        
        processed_data = self.preprocess_data(data)
        
        if self.name == 'multiply':
            return self._compute_multiply(processed_data)
        elif self.name == 'divide':
            return self._compute_divide(processed_data)
        elif self.name == 'add':
            return self._compute_add(processed_data)
        elif self.name == 'subtract':
            return self._compute_subtract(processed_data)
        elif self.name == 'correlation':
            return self._compute_correlation(processed_data)
        else:
            raise ValueError(f"不支持的交互特征: {self.name}")
    
    def _compute_multiply(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算乘积"""
        feature1 = data[self.dependencies[0]]
        feature2 = data[self.dependencies[1]]
        return feature1 * feature2
    
    def _compute_divide(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算除法"""
        feature1 = data[self.dependencies[0]]
        feature2 = data[self.dependencies[1]]
        return feature1 / feature2
    
    def _compute_add(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算加法"""
        feature1 = data[self.dependencies[0]]
        feature2 = data[self.dependencies[1]]
        return feature1 + feature2
    
    def _compute_subtract(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算减法"""
        feature1 = data[self.dependencies[0]]
        feature2 = data[self.dependencies[1]]
        return feature1 - feature2
    
    def _compute_correlation(self, data: Dict[str, pd.Series]) -> pd.Series:
        """计算滚动相关性"""
        feature1 = data[self.dependencies[0]]
        feature2 = data[self.dependencies[1]]
        window = self.params.get('window', 20)
        
        return feature1.rolling(window=window, min_periods=self.config.min_periods).corr(feature2)

class FeatureEngine:
    """特征计算引擎
    
    提供特征计算和管理功能，包括：
    1. 特征计算器注册和管理
    2. 批量特征计算
    3. 流式特征计算
    4. 特征缓存管理
    5. 并行计算支持
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.computers: Dict[str, FeatureComputer] = {}
        self.feature_configs: Dict[str, FeatureConfig] = {}
        self.cache: Dict[str, pd.Series] = {}
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.max_workers = self.config.get('max_workers', 4)
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 注册默认计算器
        self._register_default_computers()
        
        logger.info(f"特征计算引擎初始化完成，计算器数: {len(self.computers)}")
    
    def _register_default_computers(self):
        """注册默认计算器"""
        # 价格特征
        price_features = [
            ('returns', ['close'], {}),
            ('log_returns', ['close'], {}),
            ('price_change', ['close'], {}),
            ('price_ratio', ['close'], {}),
            ('high_low_ratio', ['high', 'low'], {}),
            ('close_to_high', ['close', 'high'], {}),
            ('close_to_low', ['close', 'low'], {})
        ]
        
        for name, deps, params in price_features:
            config = FeatureConfig(
                name=name,
                category=FeatureCategory.PRICE,
                dependencies=deps,
                params=params
            )
            self.register_computer(config, PriceFeatureComputer(config))
        
        # 成交量特征
        volume_features = [
            ('volume_change', ['volume'], {}),
            ('volume_ratio', ['volume'], {}),
            ('volume_ma', ['volume'], {'window': 20}),
            ('volume_std', ['volume'], {'window': 20}),
            ('vwap', ['close', 'volume'], {'window': 20}),
            ('volume_price_trend', ['close', 'volume'], {'window': 20})
        ]
        
        for name, deps, params in volume_features:
            config = FeatureConfig(
                name=name,
                category=FeatureCategory.VOLUME,
                dependencies=deps,
                params=params
            )
            self.register_computer(config, VolumeFeatureComputer(config))
        
        # 技术指标特征
        technical_features = [
            ('sma', ['close'], {'window': 20}),
            ('ema', ['close'], {'span': 20}),
            ('rsi', ['close'], {'window': 14}),
            ('macd', ['close'], {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
            ('bollinger_bands', ['close'], {'window': 20, 'num_std': 2}),
            ('stochastic', ['high', 'low', 'close'], {'window': 14}),
            ('atr', ['high', 'low', 'close'], {'window': 14}),
            ('momentum', ['close'], {'window': 10})
        ]
        
        for name, deps, params in technical_features:
            config = FeatureConfig(
                name=name,
                category=FeatureCategory.TECHNICAL,
                dependencies=deps,
                params=params
            )
            self.register_computer(config, TechnicalFeatureComputer(config))
    
    def register_computer(self, config: FeatureConfig, computer: FeatureComputer):
        """注册特征计算器
        
        Args:
            config: 特征配置
            computer: 特征计算器
        """
        with self.lock:
            self.feature_configs[config.name] = config
            self.computers[config.name] = computer
            logger.debug(f"注册特征计算器: {config.name}")
    
    def unregister_computer(self, feature_name: str) -> bool:
        """注销特征计算器
        
        Args:
            feature_name: 特征名称
            
        Returns:
            是否成功注销
        """
        with self.lock:
            if feature_name in self.computers:
                del self.computers[feature_name]
                del self.feature_configs[feature_name]
                # 清除缓存
                if feature_name in self.cache:
                    del self.cache[feature_name]
                logger.debug(f"注销特征计算器: {feature_name}")
                return True
            return False
    
    def get_feature_config(self, feature_name: str) -> Optional[FeatureConfig]:
        """获取特征配置
        
        Args:
            feature_name: 特征名称
            
        Returns:
            特征配置
        """
        return self.feature_configs.get(feature_name)
    
    def list_features(self, category: Optional[FeatureCategory] = None) -> List[str]:
        """列出特征
        
        Args:
            category: 特征类别过滤
            
        Returns:
            特征名称列表
        """
        if category:
            return [name for name, config in self.feature_configs.items() 
                   if config.category == category]
        return list(self.feature_configs.keys())
    
    def compute_feature(self, feature_name: str, data: Dict[str, pd.Series], 
                       use_cache: bool = None, **kwargs) -> ComputeResult:
        """计算单个特征
        
        Args:
            feature_name: 特征名称
            data: 输入数据
            use_cache: 是否使用缓存
            **kwargs: 额外参数
            
        Returns:
            计算结果
        """
        start_time = datetime.now()
        
        try:
            # 检查是否有计算器
            if feature_name not in self.computers:
                return ComputeResult(
                    feature_name=feature_name,
                    data=pd.Series(dtype=float),
                    success=False,
                    error_message=f"未找到特征计算器: {feature_name}"
                )
            
            # 检查缓存
            use_cache = use_cache if use_cache is not None else self.cache_enabled
            if use_cache and feature_name in self.cache:
                compute_time = (datetime.now() - start_time).total_seconds()
                return ComputeResult(
                    feature_name=feature_name,
                    data=self.cache[feature_name].copy(),
                    metadata={'from_cache': True},
                    compute_time=compute_time
                )
            
            # 执行计算
            computer = self.computers[feature_name]
            result_data = computer.compute(data, **kwargs)
            
            # 缓存结果
            if use_cache:
                with self.lock:
                    self.cache[feature_name] = result_data.copy()
            
            compute_time = (datetime.now() - start_time).total_seconds()
            
            return ComputeResult(
                feature_name=feature_name,
                data=result_data,
                metadata={'from_cache': False},
                compute_time=compute_time
            )
            
        except Exception as e:
            compute_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"计算特征失败: {feature_name}, 错误: {str(e)}")
            
            return ComputeResult(
                feature_name=feature_name,
                data=pd.Series(dtype=float),
                compute_time=compute_time,
                success=False,
                error_message=str(e)
            )
    
    def compute_features(self, feature_names: List[str], data: Dict[str, pd.Series],
                        parallel: bool = None, use_cache: bool = None, 
                        **kwargs) -> Dict[str, ComputeResult]:
        """批量计算特征
        
        Args:
            feature_names: 特征名称列表
            data: 输入数据
            parallel: 是否并行计算
            use_cache: 是否使用缓存
            **kwargs: 额外参数
            
        Returns:
            计算结果字典
        """
        logger.info(f"开始批量计算特征，特征数: {len(feature_names)}")
        
        # 确定是否并行计算
        parallel = parallel if parallel is not None else (len(feature_names) > 1)
        
        if parallel and len(feature_names) > 1:
            return self._compute_features_parallel(feature_names, data, use_cache, **kwargs)
        else:
            return self._compute_features_sequential(feature_names, data, use_cache, **kwargs)
    
    def _compute_features_sequential(self, feature_names: List[str], data: Dict[str, pd.Series],
                                   use_cache: bool = None, **kwargs) -> Dict[str, ComputeResult]:
        """顺序计算特征"""
        results = {}
        
        for feature_name in feature_names:
            result = self.compute_feature(feature_name, data, use_cache, **kwargs)
            results[feature_name] = result
        
        return results
    
    def _compute_features_parallel(self, feature_names: List[str], data: Dict[str, pd.Series],
                                 use_cache: bool = None, **kwargs) -> Dict[str, ComputeResult]:
        """并行计算特征"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_feature = {
                executor.submit(self.compute_feature, feature_name, data, use_cache, **kwargs): feature_name
                for feature_name in feature_names
            }
            
            # 收集结果
            for future in as_completed(future_to_feature):
                feature_name = future_to_feature[future]
                try:
                    result = future.result()
                    results[feature_name] = result
                except Exception as e:
                    logger.error(f"并行计算特征失败: {feature_name}, 错误: {str(e)}")
                    results[feature_name] = ComputeResult(
                        feature_name=feature_name,
                        data=pd.Series(dtype=float),
                        success=False,
                        error_message=str(e)
                    )
        
        return results
    
    def clear_cache(self, feature_names: Optional[List[str]] = None):
        """清除缓存
        
        Args:
            feature_names: 要清除的特征名称列表，如果为None则清除所有缓存
        """
        with self.lock:
            if feature_names:
                for feature_name in feature_names:
                    if feature_name in self.cache:
                        del self.cache[feature_name]
                logger.info(f"清除特征缓存: {feature_names}")
            else:
                self.cache.clear()
                logger.info("清除所有特征缓存")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息
        
        Returns:
            缓存信息
        """
        with self.lock:
            cache_size = sum(series.memory_usage(deep=True) for series in self.cache.values())
            
            return {
                'cache_enabled': self.cache_enabled,
                'cached_features': list(self.cache.keys()),
                'cache_count': len(self.cache),
                'cache_size_bytes': cache_size,
                'cache_size_mb': cache_size / (1024 * 1024)
            }
    
    def create_custom_computer(self, config: FeatureConfig, 
                             compute_func: Callable[[Dict[str, pd.Series]], pd.Series]) -> FeatureComputer:
        """创建自定义计算器
        
        Args:
            config: 特征配置
            compute_func: 计算函数
            
        Returns:
            自定义计算器
        """
        class CustomComputer(FeatureComputer):
            def __init__(self, config: FeatureConfig, func: Callable):
                super().__init__(config)
                self.compute_func = func
            
            def compute(self, data: Dict[str, pd.Series], **kwargs) -> pd.Series:
                if not self.validate_inputs(data):
                    raise ValueError("输入数据验证失败")
                processed_data = self.preprocess_data(data)
                return self.compute_func(processed_data)
        
        return CustomComputer(config, compute_func)
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息
        
        Returns:
            统计信息
        """
        with self.lock:
            category_counts = {}
            for config in self.feature_configs.values():
                category = config.category.value
                category_counts[category] = category_counts.get(category, 0) + 1
            
            return {
                'total_computers': len(self.computers),
                'category_distribution': category_counts,
                'cache_info': self.get_cache_info(),
                'max_workers': self.max_workers
            }

# 便捷函数
def create_feature_engine(config: Dict[str, Any] = None) -> FeatureEngine:
    """创建特征计算引擎实例
    
    Args:
        config: 配置字典
        
    Returns:
        特征计算引擎实例
    """
    return FeatureEngine(config)

def compute_basic_features(data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """计算基础特征的便捷函数
    
    Args:
        data: 输入数据，应包含 'open', 'high', 'low', 'close', 'volume'
        
    Returns:
        计算结果字典
    """
    engine = create_feature_engine()
    
    # 基础特征列表
    basic_features = [
        'returns', 'log_returns', 'volume_change', 'high_low_ratio',
        'sma', 'ema', 'rsi', 'atr'
    ]
    
    results = engine.compute_features(basic_features, data)
    
    # 提取成功的结果
    feature_data = {}
    for name, result in results.items():
        if result.success:
            feature_data[name] = result.data
        else:
            logger.warning(f"特征计算失败: {name} - {result.error_message}")
    
    return feature_data