# 增强因子评估系统
# 集成实时IC监控、动态权重调整和因子有效性监控

import logging
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import json

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy import stats
    from scipy.optimize import minimize
except ImportError:
    stats = None
    minimize = None

from .factor_evaluator import FactorEvaluator, EvaluationConfig, FactorMetrics, FactorQuality
from .factor_manager import FactorManager, FactorStatus, FactorCategory
from .weight_optimizer import BaseOptimizer, OptimizationConfig, OptimizationMethod

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MarketRegime(Enum):
    """市场状态"""
    BULL = "bull"          # 牛市
    BEAR = "bear"          # 熊市
    SIDEWAYS = "sideways"  # 震荡市
    VOLATILE = "volatile"  # 高波动
    CRISIS = "crisis"      # 危机

@dataclass
class FactorAlert:
    """因子告警"""
    factor_id: str
    alert_type: str
    level: AlertLevel
    message: str
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'factor_id': self.factor_id,
            'alert_type': self.alert_type,
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics
        }

@dataclass
class SystemMetrics:
    """系统指标"""
    total_factors: int = 0
    active_factors: int = 0
    deprecated_factors: int = 0
    avg_ic: float = 0.0
    avg_ic_ir: float = 0.0
    portfolio_return: float = 0.0
    portfolio_volatility: float = 0.0
    portfolio_sharpe: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    last_rebalance: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_factors': self.total_factors,
            'active_factors': self.active_factors,
            'deprecated_factors': self.deprecated_factors,
            'avg_ic': self.avg_ic,
            'avg_ic_ir': self.avg_ic_ir,
            'portfolio_return': self.portfolio_return,
            'portfolio_volatility': self.portfolio_volatility,
            'portfolio_sharpe': self.portfolio_sharpe,
            'max_drawdown': self.max_drawdown,
            'turnover': self.turnover,
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None
        }

class EnhancedFactorSystem:
    """增强因子评估系统
    
    集成因子评估、权重优化、实时监控和自动淘汰功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 核心组件
        self.evaluator = FactorEvaluator(EvaluationConfig(**config.get('evaluation', {})))
        self.factor_manager = FactorManager(config.get('factor_manager', {}))
        
        # 权重优化器
        opt_config = OptimizationConfig(**config.get('optimization', {}))
        self.optimizer = self._create_optimizer(opt_config)
        
        # 实时监控
        self.monitoring_enabled = config.get('monitoring_enabled', True)
        self.monitoring_interval = config.get('monitoring_interval', 60)  # 秒
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # 告警系统
        self.alerts: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # 因子淘汰机制
        self.auto_deprecation = config.get('auto_deprecation', True)
        self.deprecation_thresholds = config.get('deprecation_thresholds', {
            'min_ic': 0.02,
            'min_ic_ir': 0.3,
            'max_turnover': 2.0,
            'min_coverage': 0.8,
            'evaluation_periods': 10
        })
        
        # 市场状态识别
        self.market_regime_detector = MarketRegimeDetector(config.get('regime_detection', {}))
        self.current_market_regime = MarketRegime.SIDEWAYS
        
        # 性能跟踪
        self.system_metrics = SystemMetrics()
        self.performance_history: deque = deque(maxlen=1000)
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("增强因子评估系统初始化完成")
    
    def _create_optimizer(self, config: OptimizationConfig) -> BaseOptimizer:
        """创建权重优化器"""
        if config.method == OptimizationMethod.MEAN_VARIANCE:
            return MeanVarianceOptimizer(config)
        elif config.method == OptimizationMethod.ONLINE_GRADIENT:
            return OnlineGradientOptimizer(config)
        elif config.method == OptimizationMethod.THOMPSON_SAMPLING:
            return ThompsonSamplingOptimizer(config)
        else:
            return MeanVarianceOptimizer(config)  # 默认使用均值方差优化
    
    async def start_monitoring(self):
        """启动实时监控"""
        if not self.monitoring_enabled or self.monitoring_task:
            return
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("因子实时监控已启动")
    
    async def stop_monitoring(self):
        """停止实时监控"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        logger.info("因子实时监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        try:
            while True:
                await self._perform_monitoring_check()
                await asyncio.sleep(self.monitoring_interval)
        except asyncio.CancelledError:
            logger.info("监控循环被取消")
        except Exception as e:
            logger.error(f"监控循环异常: {e}")
    
    async def _perform_monitoring_check(self):
        """执行监控检查"""
        try:
            with self.lock:
                # 检查因子性能
                await self._check_factor_performance()
                
                # 检查系统健康状态
                await self._check_system_health()
                
                # 更新市场状态
                await self._update_market_regime()
                
                # 执行自动淘汰
                if self.auto_deprecation:
                    await self._auto_deprecate_factors()
                
                # 更新系统指标
                self._update_system_metrics()
                
        except Exception as e:
            logger.error(f"监控检查失败: {e}")
    
    async def _check_factor_performance(self):
        """检查因子性能"""
        for factor_id, metrics in self.evaluator.metrics_history.items():
            if not metrics:
                continue
            
            latest_metrics = metrics[-1]
            
            # IC异常检查
            if abs(latest_metrics.ic_mean) < 0.01:
                await self._create_alert(
                    factor_id, "low_ic", AlertLevel.WARNING,
                    f"因子IC过低: {latest_metrics.ic_mean:.4f}",
                    {'ic': latest_metrics.ic_mean}
                )
            
            # IC IR检查
            if latest_metrics.ic_ir < 0.3:
                await self._create_alert(
                    factor_id, "low_ic_ir", AlertLevel.WARNING,
                    f"因子IC IR过低: {latest_metrics.ic_ir:.4f}",
                    {'ic_ir': latest_metrics.ic_ir}
                )
            
            # 换手率检查
            if latest_metrics.turnover > 1.5:
                await self._create_alert(
                    factor_id, "high_turnover", AlertLevel.WARNING,
                    f"因子换手率过高: {latest_metrics.turnover:.4f}",
                    {'turnover': latest_metrics.turnover}
                )
            
            # 覆盖率检查
            if latest_metrics.coverage < 0.8:
                await self._create_alert(
                    factor_id, "low_coverage", AlertLevel.ERROR,
                    f"因子覆盖率过低: {latest_metrics.coverage:.4f}",
                    {'coverage': latest_metrics.coverage}
                )
    
    async def _check_system_health(self):
        """检查系统健康状态"""
        # 检查活跃因子数量
        active_factors = sum(1 for metadata in self.factor_manager.factor_metadata.values() 
                           if metadata.status == FactorStatus.ACTIVE)
        
        if active_factors < 5:
            await self._create_alert(
                "system", "low_active_factors", AlertLevel.ERROR,
                f"活跃因子数量过少: {active_factors}",
                {'active_factors': active_factors}
            )
        
        # 检查系统性能
        if self.system_metrics.portfolio_sharpe < 0.5:
            await self._create_alert(
                "system", "low_sharpe", AlertLevel.WARNING,
                f"组合夏普比率过低: {self.system_metrics.portfolio_sharpe:.4f}",
                {'sharpe_ratio': self.system_metrics.portfolio_sharpe}
            )
    
    async def _update_market_regime(self):
        """更新市场状态"""
        try:
            # 获取市场数据（这里需要实际的市场数据接口）
            market_data = await self._get_market_data()
            if market_data:
                new_regime = await self.market_regime_detector.detect_regime(market_data)
                
                if new_regime != self.current_market_regime:
                    logger.info(f"市场状态变化: {self.current_market_regime.value} -> {new_regime.value}")
                    self.current_market_regime = new_regime
                    
                    await self._create_alert(
                        "market", "regime_change", AlertLevel.INFO,
                        f"市场状态变化为: {new_regime.value}",
                        {'new_regime': new_regime.value}
                    )
        except Exception as e:
            logger.error(f"市场状态更新失败: {e}")
    
    async def _get_market_data(self) -> Optional[Dict[str, Any]]:
        """获取市场数据（占位符）"""
        # 这里应该连接到实际的市场数据源
        return None
    
    async def _auto_deprecate_factors(self):
        """自动淘汰因子"""
        thresholds = self.deprecation_thresholds
        
        for factor_id, metrics_list in self.evaluator.metrics_history.items():
            if len(metrics_list) < thresholds['evaluation_periods']:
                continue
            
            # 获取最近的评估结果
            recent_metrics = metrics_list[-thresholds['evaluation_periods']:]
            
            # 计算平均指标
            avg_ic = np.mean([m.ic_mean for m in recent_metrics])
            avg_ic_ir = np.mean([m.ic_ir for m in recent_metrics])
            avg_turnover = np.mean([m.turnover for m in recent_metrics])
            avg_coverage = np.mean([m.coverage for m in recent_metrics])
            
            # 检查淘汰条件
            should_deprecate = (
                abs(avg_ic) < thresholds['min_ic'] or
                avg_ic_ir < thresholds['min_ic_ir'] or
                avg_turnover > thresholds['max_turnover'] or
                avg_coverage < thresholds['min_coverage']
            )
            
            if should_deprecate:
                # 标记因子为已弃用
                if factor_id in self.factor_manager.factor_metadata:
                    metadata = self.factor_manager.factor_metadata[factor_id]
                    if metadata.status == FactorStatus.ACTIVE:
                        metadata.status = FactorStatus.DEPRECATED
                        metadata.updated_at = datetime.now()
                        
                        await self._create_alert(
                            factor_id, "auto_deprecated", AlertLevel.INFO,
                            f"因子已自动淘汰 - IC:{avg_ic:.4f}, IC_IR:{avg_ic_ir:.4f}, 换手率:{avg_turnover:.4f}",
                            {
                                'avg_ic': avg_ic,
                                'avg_ic_ir': avg_ic_ir,
                                'avg_turnover': avg_turnover,
                                'avg_coverage': avg_coverage
                            }
                        )
                        
                        logger.info(f"因子 {factor_id} 已自动淘汰")
    
    async def _create_alert(self, factor_id: str, alert_type: str, level: AlertLevel, 
                          message: str, metrics: Dict[str, float] = None):
        """创建告警"""
        alert = FactorAlert(
            factor_id=factor_id,
            alert_type=alert_type,
            level=level,
            message=message,
            timestamp=datetime.now(),
            metrics=metrics or {}
        )
        
        self.alerts.append(alert)
        
        # 触发告警回调
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
        
        logger.log(
            getattr(logging, level.value.upper()),
            f"[{factor_id}] {alert_type}: {message}"
        )
    
    def _update_system_metrics(self):
        """更新系统指标"""
        # 统计因子数量
        total_factors = len(self.factor_manager.factor_metadata)
        active_factors = sum(1 for m in self.factor_manager.factor_metadata.values() 
                           if m.status == FactorStatus.ACTIVE)
        deprecated_factors = sum(1 for m in self.factor_manager.factor_metadata.values() 
                               if m.status == FactorStatus.DEPRECATED)
        
        # 计算平均IC指标
        all_metrics = []
        for metrics_list in self.evaluator.metrics_history.values():
            if metrics_list:
                all_metrics.append(metrics_list[-1])
        
        if all_metrics:
            avg_ic = np.mean([m.ic_mean for m in all_metrics])
            avg_ic_ir = np.mean([m.ic_ir for m in all_metrics])
        else:
            avg_ic = 0.0
            avg_ic_ir = 0.0
        
        # 更新系统指标
        self.system_metrics.total_factors = total_factors
        self.system_metrics.active_factors = active_factors
        self.system_metrics.deprecated_factors = deprecated_factors
        self.system_metrics.avg_ic = avg_ic
        self.system_metrics.avg_ic_ir = avg_ic_ir
        
        # 记录性能历史
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': self.system_metrics.to_dict()
        })
    
    async def optimize_factor_weights(self, factor_returns: Dict[str, List[float]], 
                                    current_weights: Dict[str, float] = None) -> Dict[str, float]:
        """优化因子权重"""
        try:
            # 过滤活跃因子
            active_factor_returns = {}
            for factor_id, returns in factor_returns.items():
                if factor_id in self.factor_manager.factor_metadata:
                    metadata = self.factor_manager.factor_metadata[factor_id]
                    if metadata.status == FactorStatus.ACTIVE:
                        active_factor_returns[factor_id] = returns
            
            if not active_factor_returns:
                logger.warning("没有活跃因子可用于权重优化")
                return current_weights or {}
            
            # 执行权重优化
            new_weights = await self.optimizer.optimize_weights(
                active_factor_returns, 
                current_weights or {},
                self.current_market_regime.value
            )
            
            # 记录权重更新
            self.factor_manager.factor_weights.update(new_weights)
            self.factor_manager.weight_history.append(
                (datetime.now(), new_weights.copy())
            )
            
            logger.info(f"因子权重优化完成，活跃因子数: {len(new_weights)}")
            return new_weights
            
        except Exception as e:
            logger.error(f"因子权重优化失败: {e}")
            return current_weights or {}
    
    def add_alert_callback(self, callback: Callable):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def get_recent_alerts(self, hours: int = 24) -> List[FactorAlert]:
        """获取最近的告警"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'system_metrics': self.system_metrics.to_dict(),
            'market_regime': self.current_market_regime.value,
            'monitoring_enabled': self.monitoring_enabled,
            'recent_alerts': len(self.get_recent_alerts()),
            'factor_weights': dict(self.factor_manager.factor_weights),
            'last_update': datetime.now().isoformat()
        }
    
    async def evaluate_and_optimize(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行完整的评估和优化流程"""
        try:
            # 提取因子数据和收益率数据
            factor_data = market_data.get('factors', {})
            return_data = market_data.get('returns', [])
            price_data = market_data.get('prices', [])
            
            # 执行因子评估
            evaluation_result = await self.evaluator.evaluate_factors(
                factor_data, return_data, price_data
            )
            
            # 优化因子权重
            factor_returns = {fid: [0.01] * 20 for fid in factor_data.keys()}  # 模拟收益率
            new_weights = await self.optimize_factor_weights(factor_returns)
            
            return {
                'evaluation_result': evaluation_result.to_dict(),
                'new_weights': new_weights,
                'system_status': self.get_system_status()
            }
            
        except Exception as e:
            logger.error(f"评估和优化流程失败: {e}")
            return {'error': str(e)}

# 辅助类
class MarketRegimeDetector:
    """市场状态检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback_window = config.get('lookback_window', 60)
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        self.trend_threshold = config.get('trend_threshold', 0.05)
    
    async def detect_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """检测市场状态"""
        try:
            # 这里应该实现实际的市场状态检测逻辑
            # 基于价格、波动率、成交量等指标
            return MarketRegime.SIDEWAYS  # 默认返回震荡市
        except Exception as e:
            logger.error(f"市场状态检测失败: {e}")
            return MarketRegime.SIDEWAYS

class MeanVarianceOptimizer(BaseOptimizer):
    """均值方差优化器实现"""
    
    async def optimize_weights(self, factor_returns: Dict[str, List[float]], 
                             current_weights: Dict[str, float],
                             market_regime: str = None) -> Dict[str, float]:
        """均值方差优化"""
        try:
            if not factor_returns:
                return current_weights
            
            # 简化实现：等权重分配
            n_factors = len(factor_returns)
            equal_weight = 1.0 / n_factors
            
            return {factor_id: equal_weight for factor_id in factor_returns.keys()}
            
        except Exception as e:
            logger.error(f"均值方差优化失败: {e}")
            return current_weights

class OnlineGradientOptimizer(BaseOptimizer):
    """在线梯度优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.learning_rate = config.learning_rate
        self.momentum = 0.9
        self.velocity = {}
    
    async def optimize_weights(self, factor_returns: Dict[str, List[float]], 
                             current_weights: Dict[str, float],
                             market_regime: str = None) -> Dict[str, float]:
        """在线梯度优化"""
        try:
            if not factor_returns or not current_weights:
                return current_weights
            
            # 简化实现：基于最近收益率调整权重
            new_weights = current_weights.copy()
            
            for factor_id, returns in factor_returns.items():
                if factor_id in new_weights and returns:
                    recent_return = returns[-1] if returns else 0.0
                    
                    # 梯度更新
                    gradient = recent_return
                    
                    if factor_id not in self.velocity:
                        self.velocity[factor_id] = 0.0
                    
                    self.velocity[factor_id] = (self.momentum * self.velocity[factor_id] + 
                                              self.learning_rate * gradient)
                    
                    new_weights[factor_id] += self.velocity[factor_id]
            
            # 归一化权重
            total_weight = sum(abs(w) for w in new_weights.values())
            if total_weight > 0:
                new_weights = {k: v / total_weight for k, v in new_weights.items()}
            
            return new_weights
            
        except Exception as e:
            logger.error(f"在线梯度优化失败: {e}")
            return current_weights

class ThompsonSamplingOptimizer(BaseOptimizer):
    """汤普森采样优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.alpha = defaultdict(lambda: 1.0)  # 成功次数
        self.beta = defaultdict(lambda: 1.0)   # 失败次数
    
    async def optimize_weights(self, factor_returns: Dict[str, List[float]], 
                             current_weights: Dict[str, float],
                             market_regime: str = None) -> Dict[str, float]:
        """汤普森采样优化"""
        try:
            if not factor_returns:
                return current_weights
            
            # 更新Beta分布参数
            for factor_id, returns in factor_returns.items():
                if returns:
                    recent_return = returns[-1]
                    if recent_return > 0:
                        self.alpha[factor_id] += 1
                    else:
                        self.beta[factor_id] += 1
            
            # 采样权重
            sampled_weights = {}
            for factor_id in factor_returns.keys():
                # 从Beta分布采样
                sample = np.random.beta(self.alpha[factor_id], self.beta[factor_id])
                sampled_weights[factor_id] = sample
            
            # 归一化
            total_weight = sum(sampled_weights.values())
            if total_weight > 0:
                sampled_weights = {k: v / total_weight for k, v in sampled_weights.items()}
            
            return sampled_weights
            
        except Exception as e:
            logger.error(f"汤普森采样优化失败: {e}")
            return current_weights

# 便捷函数
def create_enhanced_factor_system(config_file: str = None) -> EnhancedFactorSystem:
    """创建增强因子评估系统
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        增强因子评估系统实例
    """
    default_config = {
        'evaluation': {
            'evaluation_window': 252,
            'rolling_window': 60,
            'ic_method': 'pearson',
            'min_periods': 20,
            'n_quantiles': 5,
            'ic_threshold': 0.05,
            'ic_ir_threshold': 0.5
        },
        'optimization': {
            'method': 'mean_variance',
            'rebalance_frequency': 'daily',
            'lookback_window': 252,
            'min_weight': 0.0,
            'max_weight': 0.3,
            'target_volatility': 0.15,
            'learning_rate': 0.01
        },
        'monitoring_enabled': True,
        'monitoring_interval': 60,
        'auto_deprecation': True,
        'deprecation_thresholds': {
            'min_ic': 0.02,
            'min_ic_ir': 0.3,
            'max_turnover': 2.0,
            'min_coverage': 0.8,
            'evaluation_periods': 10
        }
    }
    
    if config_file:
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
    
    return EnhancedFactorSystem(default_config)