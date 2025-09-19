# 因子管理器 - 统一管理因子计算、评估和权重分配

import logging
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy import stats
except ImportError:
    stats = None

logger = logging.getLogger(__name__)

class FactorCategory(Enum):
    """因子分类枚举"""
    TECHNICAL = "technical"          # 技术因子
    FUNDAMENTAL = "fundamental"      # 基本面因子
    SENTIMENT = "sentiment"          # 情绪因子
    MACRO = "macro"                  # 宏观因子
    ALTERNATIVE = "alternative"      # 另类因子
    RISK = "risk"                    # 风险因子
    MOMENTUM = "momentum"            # 动量因子
    MEAN_REVERSION = "mean_reversion" # 均值回归因子

class FactorStatus(Enum):
    """因子状态枚举"""
    ACTIVE = "active"        # 活跃
    INACTIVE = "inactive"    # 非活跃
    DEPRECATED = "deprecated" # 已弃用
    TESTING = "testing"      # 测试中
    FAILED = "failed"        # 失效

@dataclass
class FactorMetadata:
    """因子元数据"""
    factor_id: str
    name: str
    description: str
    category: FactorCategory
    status: FactorStatus
    created_at: datetime
    updated_at: datetime
    version: str = "1.0"
    author: str = "system"
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class FactorValue:
    """因子值"""
    factor_id: str
    timestamp: datetime
    value: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FactorPerformance:
    """因子性能指标"""
    factor_id: str
    ic: float                    # 信息系数
    ic_ir: float                 # IC信息比率
    rank_ic: float               # 排序IC
    turnover: float              # 换手率
    max_drawdown: float          # 最大回撤
    sharpe_ratio: float          # 夏普比率
    calmar_ratio: float          # 卡玛比率
    win_rate: float              # 胜率
    avg_holding_period: float    # 平均持有期
    factor_loading: float        # 因子载荷
    t_stat: float                # t统计量
    p_value: float               # p值
    evaluation_period: Tuple[datetime, datetime]

class BaseFactor(ABC):
    """因子基类"""
    
    def __init__(self, metadata: FactorMetadata):
        self.metadata = metadata
        self.last_calculated = None
        self.cache = {}
        self.cache_ttl = 3600  # 1小时缓存
    
    @abstractmethod
    async def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算因子值"""
        pass
    
    def get_dependencies(self) -> List[str]:
        """获取依赖的数据字段"""
        return self.metadata.dependencies
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        required_columns = self.get_dependencies()
        return all(col in data.columns for col in required_columns)
    
    def is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key].get('timestamp')
        if not cached_time:
            return False
        
        return (datetime.now() - cached_time).total_seconds() < self.cache_ttl

class FactorManager:
    """因子管理器
    
    统一管理所有因子的计算、评估、权重分配和生命周期
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 因子注册表
        self.factors: Dict[str, BaseFactor] = {}
        self.factor_metadata: Dict[str, FactorMetadata] = {}
        
        # 因子值存储
        self.factor_values: Dict[str, List[FactorValue]] = {}
        self.max_history = config.get('max_history', 10000)
        
        # 因子性能记录
        self.factor_performance: Dict[str, FactorPerformance] = {}
        
        # 权重管理
        self.factor_weights: Dict[str, float] = {}
        self.weight_history: List[Tuple[datetime, Dict[str, float]]] = []
        
        # 计算调度
        self.calculation_schedule: Dict[str, datetime] = {}
        self.calculation_intervals: Dict[str, timedelta] = {}
        
        # 初始化内置因子
        self._init_builtin_factors()
        
        logger.info(f"因子管理器初始化完成，注册因子数: {len(self.factors)}")
    
    def _init_builtin_factors(self):
        """初始化内置因子"""
        # 注册一些基础技术因子
        builtin_factors = [
            self._create_momentum_factor(),
            self._create_mean_reversion_factor(),
            self._create_volatility_factor(),
            self._create_volume_factor(),
            self._create_rsi_factor()
        ]
        
        for factor in builtin_factors:
            self.register_factor(factor)
    
    def _create_momentum_factor(self) -> BaseFactor:
        """创建动量因子"""
        metadata = FactorMetadata(
            factor_id="momentum_20d",
            name="20日动量因子",
            description="过去20个交易日的价格动量",
            category=FactorCategory.MOMENTUM,
            status=FactorStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            dependencies=["close"],
            parameters={"window": 20}
        )
        
        class MomentumFactor(BaseFactor):
            async def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                window = self.metadata.parameters.get('window', 20)
                return data['close'].pct_change(window)
        
        return MomentumFactor(metadata)
    
    def _create_mean_reversion_factor(self) -> BaseFactor:
        """创建均值回归因子"""
        metadata = FactorMetadata(
            factor_id="mean_reversion_10d",
            name="10日均值回归因子",
            description="价格相对于10日均线的偏离度",
            category=FactorCategory.MEAN_REVERSION,
            status=FactorStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            dependencies=["close"],
            parameters={"window": 10}
        )
        
        class MeanReversionFactor(BaseFactor):
            async def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                window = self.metadata.parameters.get('window', 10)
                ma = data['close'].rolling(window).mean()
                return (data['close'] - ma) / ma
        
        return MeanReversionFactor(metadata)
    
    def _create_volatility_factor(self) -> BaseFactor:
        """创建波动率因子"""
        metadata = FactorMetadata(
            factor_id="volatility_20d",
            name="20日波动率因子",
            description="过去20个交易日的收益率波动率",
            category=FactorCategory.RISK,
            status=FactorStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            dependencies=["close"],
            parameters={"window": 20}
        )
        
        class VolatilityFactor(BaseFactor):
            async def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                window = self.metadata.parameters.get('window', 20)
                returns = data['close'].pct_change()
                return returns.rolling(window).std() * np.sqrt(252)
        
        return VolatilityFactor(metadata)
    
    def _create_volume_factor(self) -> BaseFactor:
        """创建成交量因子"""
        metadata = FactorMetadata(
            factor_id="volume_ratio_20d",
            name="20日成交量比率因子",
            description="当前成交量相对于20日平均成交量的比率",
            category=FactorCategory.TECHNICAL,
            status=FactorStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            dependencies=["volume"],
            parameters={"window": 20}
        )
        
        class VolumeFactor(BaseFactor):
            async def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                window = self.metadata.parameters.get('window', 20)
                avg_volume = data['volume'].rolling(window).mean()
                return data['volume'] / avg_volume
        
        return VolumeFactor(metadata)
    
    def _create_rsi_factor(self) -> BaseFactor:
        """创建RSI因子"""
        metadata = FactorMetadata(
            factor_id="rsi_14d",
            name="14日RSI因子",
            description="14日相对强弱指数",
            category=FactorCategory.TECHNICAL,
            status=FactorStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            dependencies=["close"],
            parameters={"window": 14}
        )
        
        class RSIFactor(BaseFactor):
            async def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                window = self.metadata.parameters.get('window', 14)
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return (rsi - 50) / 50  # 标准化到[-1, 1]
        
        return RSIFactor(metadata)
    
    def register_factor(self, factor: BaseFactor) -> bool:
        """注册因子
        
        Args:
            factor: 因子实例
            
        Returns:
            注册是否成功
        """
        try:
            factor_id = factor.metadata.factor_id
            
            if factor_id in self.factors:
                logger.warning(f"因子已存在，将被覆盖: {factor_id}")
            
            self.factors[factor_id] = factor
            self.factor_metadata[factor_id] = factor.metadata
            
            # 初始化因子值存储
            if factor_id not in self.factor_values:
                self.factor_values[factor_id] = []
            
            # 设置默认权重
            if factor_id not in self.factor_weights:
                self.factor_weights[factor_id] = 1.0 / len(self.factors)
            
            # 设置计算间隔
            default_interval = timedelta(hours=1)  # 默认1小时计算一次
            self.calculation_intervals[factor_id] = default_interval
            
            logger.info(f"因子注册成功: {factor_id}")
            return True
            
        except Exception as e:
            logger.error(f"因子注册失败: {str(e)}")
            return False
    
    def unregister_factor(self, factor_id: str) -> bool:
        """注销因子"""
        try:
            if factor_id not in self.factors:
                logger.warning(f"因子不存在: {factor_id}")
                return False
            
            # 移除因子
            del self.factors[factor_id]
            del self.factor_metadata[factor_id]
            
            # 清理相关数据
            if factor_id in self.factor_values:
                del self.factor_values[factor_id]
            
            if factor_id in self.factor_weights:
                del self.factor_weights[factor_id]
            
            if factor_id in self.factor_performance:
                del self.factor_performance[factor_id]
            
            if factor_id in self.calculation_schedule:
                del self.calculation_schedule[factor_id]
            
            if factor_id in self.calculation_intervals:
                del self.calculation_intervals[factor_id]
            
            logger.info(f"因子注销成功: {factor_id}")
            return True
            
        except Exception as e:
            logger.error(f"因子注销失败: {str(e)}")
            return False
    
    async def calculate_factor(self, factor_id: str, data: pd.DataFrame, 
                             force_recalculate: bool = False) -> Optional[pd.Series]:
        """计算单个因子
        
        Args:
            factor_id: 因子ID
            data: 输入数据
            force_recalculate: 是否强制重新计算
            
        Returns:
            因子值序列
        """
        try:
            if factor_id not in self.factors:
                logger.error(f"因子不存在: {factor_id}")
                return None
            
            factor = self.factors[factor_id]
            
            # 检查因子状态
            if factor.metadata.status not in [FactorStatus.ACTIVE, FactorStatus.TESTING]:
                logger.warning(f"因子状态不活跃: {factor_id}")
                return None
            
            # 验证数据
            if not factor.validate_data(data):
                logger.error(f"因子数据验证失败: {factor_id}")
                return None
            
            # 检查是否需要计算
            if not force_recalculate and self._should_skip_calculation(factor_id):
                logger.debug(f"跳过因子计算: {factor_id}")
                return self._get_cached_factor_values(factor_id)
            
            # 计算因子
            start_time = datetime.now()
            factor_values = await factor.calculate(data)
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            # 验证计算结果
            if factor_values is None or factor_values.empty:
                logger.warning(f"因子计算结果为空: {factor_id}")
                return None
            
            # 存储因子值
            self._store_factor_values(factor_id, factor_values, calculation_time)
            
            # 更新计算时间
            self.calculation_schedule[factor_id] = datetime.now()
            
            logger.debug(f"因子计算完成: {factor_id}, 耗时: {calculation_time:.3f}秒")
            return factor_values
            
        except Exception as e:
            logger.error(f"因子计算失败: {factor_id}, 错误: {str(e)}")
            return None
    
    async def calculate_all_factors(self, data: pd.DataFrame, 
                                  factor_ids: List[str] = None) -> Dict[str, pd.Series]:
        """计算所有因子或指定因子
        
        Args:
            data: 输入数据
            factor_ids: 指定的因子ID列表，None表示计算所有因子
            
        Returns:
            因子值字典
        """
        if factor_ids is None:
            factor_ids = list(self.factors.keys())
        
        results = {}
        
        # 并发计算因子
        tasks = []
        for factor_id in factor_ids:
            if factor_id in self.factors:
                task = self.calculate_factor(factor_id, data)
                tasks.append((factor_id, task))
        
        # 等待所有计算完成
        for factor_id, task in tasks:
            try:
                factor_values = await task
                if factor_values is not None:
                    results[factor_id] = factor_values
            except Exception as e:
                logger.error(f"因子计算任务失败: {factor_id}, 错误: {str(e)}")
        
        logger.info(f"批量因子计算完成，成功: {len(results)}/{len(factor_ids)}")
        return results
    
    def _should_skip_calculation(self, factor_id: str) -> bool:
        """判断是否应该跳过计算"""
        if factor_id not in self.calculation_schedule:
            return False
        
        last_calculation = self.calculation_schedule[factor_id]
        interval = self.calculation_intervals.get(factor_id, timedelta(hours=1))
        
        return datetime.now() - last_calculation < interval
    
    def _get_cached_factor_values(self, factor_id: str) -> Optional[pd.Series]:
        """获取缓存的因子值"""
        if factor_id not in self.factor_values or not self.factor_values[factor_id]:
            return None
        
        # 返回最新的因子值
        latest_values = self.factor_values[factor_id][-1]
        return latest_values.value if hasattr(latest_values, 'value') else None
    
    def _store_factor_values(self, factor_id: str, values: pd.Series, calculation_time: float):
        """存储因子值"""
        try:
            timestamp = datetime.now()
            
            # 为每个值创建FactorValue对象
            for idx, value in values.items():
                if pd.notna(value):
                    factor_value = FactorValue(
                        factor_id=factor_id,
                        timestamp=timestamp,
                        value=float(value),
                        confidence=1.0,  # 默认置信度
                        metadata={
                            'calculation_time': calculation_time,
                            'data_point': str(idx)
                        }
                    )
                    
                    self.factor_values[factor_id].append(factor_value)
            
            # 限制历史记录长度
            if len(self.factor_values[factor_id]) > self.max_history:
                self.factor_values[factor_id] = self.factor_values[factor_id][-self.max_history:]
                
        except Exception as e:
            logger.error(f"因子值存储失败: {factor_id}, 错误: {str(e)}")
    
    async def evaluate_factor_performance(self, factor_id: str, 
                                        returns: pd.Series,
                                        benchmark_returns: pd.Series = None) -> Optional[FactorPerformance]:
        """评估因子性能
        
        Args:
            factor_id: 因子ID
            returns: 收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            因子性能指标
        """
        try:
            if factor_id not in self.factor_values or not self.factor_values[factor_id]:
                logger.warning(f"因子无历史数据: {factor_id}")
                return None
            
            # 获取因子值序列
            factor_values = [fv.value for fv in self.factor_values[factor_id]]
            if len(factor_values) != len(returns):
                logger.warning(f"因子值与收益率长度不匹配: {factor_id}")
                return None
            
            factor_series = pd.Series(factor_values, index=returns.index)
            
            # 计算信息系数(IC)
            ic = factor_series.corr(returns)
            
            # 计算排序IC
            rank_ic = factor_series.rank().corr(returns.rank())
            
            # 计算IC信息比率
            ic_series = factor_series.rolling(20).corr(returns.rolling(20))
            ic_ir = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0
            
            # 计算换手率（简化）
            factor_changes = factor_series.diff().abs()
            turnover = factor_changes.mean()
            
            # 计算最大回撤
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            # 计算夏普比率
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # 计算卡玛比率
            calmar_ratio = returns.mean() * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # 计算胜率
            win_rate = (returns > 0).mean()
            
            # 计算t统计量和p值
            if stats is not None:
                t_stat, p_value = stats.ttest_1samp(factor_values, 0)
            else:
                t_stat, p_value = 0.0, 1.0
            
            # 创建性能对象
            performance = FactorPerformance(
                factor_id=factor_id,
                ic=ic,
                ic_ir=ic_ir,
                rank_ic=rank_ic,
                turnover=turnover,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                avg_holding_period=len(returns),  # 简化
                factor_loading=1.0,  # 简化
                t_stat=t_stat,
                p_value=p_value,
                evaluation_period=(returns.index[0], returns.index[-1])
            )
            
            # 存储性能指标
            self.factor_performance[factor_id] = performance
            
            logger.info(f"因子性能评估完成: {factor_id}, IC: {ic:.3f}, 夏普: {sharpe_ratio:.3f}")
            return performance
            
        except Exception as e:
            logger.error(f"因子性能评估失败: {factor_id}, 错误: {str(e)}")
            return None
    
    def update_factor_weights(self, new_weights: Dict[str, float]):
        """更新因子权重"""
        try:
            # 验证权重
            total_weight = sum(new_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                logger.warning(f"权重总和不为1: {total_weight}，将进行归一化")
                new_weights = {k: v / total_weight for k, v in new_weights.items()}
            
            # 更新权重
            for factor_id, weight in new_weights.items():
                if factor_id in self.factors:
                    self.factor_weights[factor_id] = weight
                else:
                    logger.warning(f"未知因子ID: {factor_id}")
            
            # 记录权重历史
            self.weight_history.append((datetime.now(), self.factor_weights.copy()))
            
            # 限制历史记录长度
            max_weight_history = self.config.get('max_weight_history', 1000)
            if len(self.weight_history) > max_weight_history:
                self.weight_history = self.weight_history[-max_weight_history:]
            
            logger.info(f"因子权重更新完成，活跃因子数: {len(new_weights)}")
            
        except Exception as e:
            logger.error(f"因子权重更新失败: {str(e)}")
    
    def get_factor_composite_score(self, factor_values: Dict[str, pd.Series]) -> pd.Series:
        """计算因子综合得分
        
        Args:
            factor_values: 因子值字典
            
        Returns:
            综合得分序列
        """
        try:
            if not factor_values:
                return pd.Series()
            
            # 标准化因子值
            normalized_factors = {}
            for factor_id, values in factor_values.items():
                if values is not None and not values.empty:
                    # Z-score标准化
                    mean_val = values.mean()
                    std_val = values.std()
                    if std_val > 0:
                        normalized_factors[factor_id] = (values - mean_val) / std_val
                    else:
                        normalized_factors[factor_id] = values - mean_val
            
            if not normalized_factors:
                return pd.Series()
            
            # 加权合成
            composite_score = None
            total_weight = 0
            
            for factor_id, values in normalized_factors.items():
                weight = self.factor_weights.get(factor_id, 0.0)
                if weight > 0:
                    weighted_values = values * weight
                    if composite_score is None:
                        composite_score = weighted_values
                    else:
                        composite_score = composite_score.add(weighted_values, fill_value=0)
                    total_weight += weight
            
            # 归一化
            if composite_score is not None and total_weight > 0:
                composite_score = composite_score / total_weight
            
            return composite_score if composite_score is not None else pd.Series()
            
        except Exception as e:
            logger.error(f"因子综合得分计算失败: {str(e)}")
            return pd.Series()
    
    def get_factor_ranking(self, metric: str = 'ic') -> List[Tuple[str, float]]:
        """获取因子排名
        
        Args:
            metric: 排名指标 ('ic', 'sharpe_ratio', 'calmar_ratio'等)
            
        Returns:
            因子排名列表 [(factor_id, metric_value)]
        """
        try:
            rankings = []
            
            for factor_id, performance in self.factor_performance.items():
                if hasattr(performance, metric):
                    metric_value = getattr(performance, metric)
                    rankings.append((factor_id, metric_value))
            
            # 按指标值降序排序
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            return rankings
            
        except Exception as e:
            logger.error(f"因子排名计算失败: {str(e)}")
            return []
    
    def get_active_factors(self) -> List[str]:
        """获取活跃因子列表"""
        return [
            factor_id for factor_id, metadata in self.factor_metadata.items()
            if metadata.status == FactorStatus.ACTIVE
        ]
    
    def get_factor_by_category(self, category: FactorCategory) -> List[str]:
        """按类别获取因子"""
        return [
            factor_id for factor_id, metadata in self.factor_metadata.items()
            if metadata.category == category
        ]
    
    def set_factor_status(self, factor_id: str, status: FactorStatus):
        """设置因子状态"""
        if factor_id in self.factor_metadata:
            self.factor_metadata[factor_id].status = status
            self.factor_metadata[factor_id].updated_at = datetime.now()
            logger.info(f"因子状态更新: {factor_id} -> {status.value}")
        else:
            logger.warning(f"因子不存在: {factor_id}")
    
    def get_factor_statistics(self) -> Dict[str, Any]:
        """获取因子统计信息"""
        try:
            stats = {
                'total_factors': len(self.factors),
                'active_factors': len(self.get_active_factors()),
                'factor_categories': {},
                'factor_status': {},
                'avg_ic': 0.0,
                'avg_sharpe': 0.0,
                'total_factor_values': sum(len(values) for values in self.factor_values.values()),
                'weight_updates': len(self.weight_history)
            }
            
            # 统计分类分布
            for metadata in self.factor_metadata.values():
                category = metadata.category.value
                status = metadata.status.value
                
                stats['factor_categories'][category] = stats['factor_categories'].get(category, 0) + 1
                stats['factor_status'][status] = stats['factor_status'].get(status, 0) + 1
            
            # 计算平均性能指标
            if self.factor_performance:
                ic_values = [p.ic for p in self.factor_performance.values() if not np.isnan(p.ic)]
                sharpe_values = [p.sharpe_ratio for p in self.factor_performance.values() if not np.isnan(p.sharpe_ratio)]
                
                stats['avg_ic'] = np.mean(ic_values) if ic_values else 0.0
                stats['avg_sharpe'] = np.mean(sharpe_values) if sharpe_values else 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"因子统计信息计算失败: {str(e)}")
            return {}
    
    def export_factor_data(self, factor_ids: List[str] = None, 
                          start_date: datetime = None, 
                          end_date: datetime = None) -> Dict[str, Any]:
        """导出因子数据"""
        try:
            if factor_ids is None:
                factor_ids = list(self.factors.keys())
            
            export_data = {
                'metadata': {},
                'values': {},
                'performance': {},
                'weights': self.factor_weights,
                'export_time': datetime.now().isoformat()
            }
            
            for factor_id in factor_ids:
                if factor_id in self.factor_metadata:
                    # 导出元数据
                    metadata = self.factor_metadata[factor_id]
                    export_data['metadata'][factor_id] = {
                        'name': metadata.name,
                        'description': metadata.description,
                        'category': metadata.category.value,
                        'status': metadata.status.value,
                        'version': metadata.version,
                        'parameters': metadata.parameters
                    }
                    
                    # 导出因子值
                    if factor_id in self.factor_values:
                        values = self.factor_values[factor_id]
                        
                        # 时间过滤
                        if start_date or end_date:
                            filtered_values = []
                            for value in values:
                                if start_date and value.timestamp < start_date:
                                    continue
                                if end_date and value.timestamp > end_date:
                                    continue
                                filtered_values.append(value)
                            values = filtered_values
                        
                        export_data['values'][factor_id] = [
                            {
                                'timestamp': v.timestamp.isoformat(),
                                'value': v.value,
                                'confidence': v.confidence
                            }
                            for v in values
                        ]
                    
                    # 导出性能指标
                    if factor_id in self.factor_performance:
                        perf = self.factor_performance[factor_id]
                        export_data['performance'][factor_id] = {
                            'ic': perf.ic,
                            'ic_ir': perf.ic_ir,
                            'rank_ic': perf.rank_ic,
                            'sharpe_ratio': perf.sharpe_ratio,
                            'max_drawdown': perf.max_drawdown,
                            'win_rate': perf.win_rate
                        }
            
            return export_data
            
        except Exception as e:
            logger.error(f"因子数据导出失败: {str(e)}")
            return {}
    
    def cleanup_old_data(self, days: int = 30):
        """清理旧数据"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cleaned_count = 0
            
            for factor_id in self.factor_values:
                original_count = len(self.factor_values[factor_id])
                self.factor_values[factor_id] = [
                    fv for fv in self.factor_values[factor_id]
                    if fv.timestamp >= cutoff_date
                ]
                cleaned_count += original_count - len(self.factor_values[factor_id])
            
            # 清理权重历史
            original_weight_count = len(self.weight_history)
            self.weight_history = [
                (timestamp, weights) for timestamp, weights in self.weight_history
                if timestamp >= cutoff_date
            ]
            cleaned_count += original_weight_count - len(self.weight_history)
            
            logger.info(f"数据清理完成，清理了{cleaned_count}条记录")
            
        except Exception as e:
            logger.error(f"数据清理失败: {str(e)}")