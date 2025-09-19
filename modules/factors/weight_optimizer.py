# 权重优化器模块
# 实现动态因子权重优化，包括在线学习、强化学习和多臂老虎机算法

import logging
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from collections import defaultdict, deque

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy import optimize
    from scipy.stats import norm
except ImportError:
    optimize = None
    norm = None

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """优化方法枚举"""
    MEAN_VARIANCE = "mean_variance"        # 均值方差优化
    RISK_PARITY = "risk_parity"            # 风险平价
    BLACK_LITTERMAN = "black_litterman"    # Black-Litterman模型
    ONLINE_GRADIENT = "online_gradient"    # 在线梯度下降
    THOMPSON_SAMPLING = "thompson_sampling" # 汤普森采样
    UCB = "ucb"                           # 置信上界
    EXP3 = "exp3"                         # EXP3算法
    ADAPTIVE_HEDGE = "adaptive_hedge"      # 自适应对冲

class RebalanceFrequency(Enum):
    """再平衡频率"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ADAPTIVE = "adaptive"  # 自适应频率

@dataclass
class OptimizationConfig:
    """优化配置"""
    method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    lookback_window: int = 252  # 回望窗口
    min_weight: float = 0.0     # 最小权重
    max_weight: float = 1.0     # 最大权重
    target_volatility: float = 0.15  # 目标波动率
    risk_aversion: float = 3.0  # 风险厌恶系数
    transaction_cost: float = 0.001  # 交易成本
    regularization: float = 0.01     # 正则化参数
    confidence_level: float = 0.95   # 置信水平
    learning_rate: float = 0.01      # 学习率
    exploration_rate: float = 0.1    # 探索率
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'method': self.method.value,
            'rebalance_frequency': self.rebalance_frequency.value,
            'lookback_window': self.lookback_window,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'target_volatility': self.target_volatility,
            'risk_aversion': self.risk_aversion,
            'transaction_cost': self.transaction_cost,
            'regularization': self.regularization,
            'confidence_level': self.confidence_level,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate
        }

@dataclass
class WeightUpdate:
    """权重更新记录"""
    timestamp: datetime
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    method: str
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    rebalance_cost: float = 0.0
    
    def get_weight_change(self) -> Dict[str, float]:
        """获取权重变化"""
        changes = {}
        all_factors = set(self.old_weights.keys()) | set(self.new_weights.keys())
        
        for factor in all_factors:
            old_w = self.old_weights.get(factor, 0.0)
            new_w = self.new_weights.get(factor, 0.0)
            changes[factor] = new_w - old_w
        
        return changes
    
    def get_turnover(self) -> float:
        """计算换手率"""
        changes = self.get_weight_change()
        return sum(abs(change) for change in changes.values()) / 2

@dataclass
class PerformanceMetrics:
    """性能指标"""
    returns: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    hit_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'returns': self.returns,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'information_ratio': self.information_ratio,
            'tracking_error': self.tracking_error,
            'hit_rate': self.hit_rate
        }

class BaseOptimizer(ABC):
    """优化器基类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.factor_returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.lookback_window))
        self.factor_covariance: Optional[np.ndarray] = None
        self.factor_names: List[str] = []
        
    @abstractmethod
    async def optimize_weights(self, factor_returns: Dict[str, List[float]], 
                             current_weights: Dict[str, float],
                             market_regime: str = None) -> Dict[str, float]:
        """优化权重"""
        pass
    
    def update_factor_returns(self, factor_id: str, returns: List[float]):
        """更新因子收益率"""
        self.factor_returns[factor_id].extend(returns)
    
    def calculate_covariance_matrix(self) -> Optional[np.ndarray]:
        """计算协方差矩阵"""
        if not self.factor_returns:
            return None
        
        # 构建收益率矩阵
        min_length = min(len(returns) for returns in self.factor_returns.values())
        if min_length < 10:  # 至少需要10个观测值
            return None
        
        self.factor_names = list(self.factor_returns.keys())
        returns_matrix = np.array([
            list(self.factor_returns[factor])[-min_length:] 
            for factor in self.factor_names
        ]).T
        
        # 计算协方差矩阵
        self.factor_covariance = np.cov(returns_matrix.T)
        return self.factor_covariance

class MeanVarianceOptimizer(BaseOptimizer):
    """均值方差优化器"""
    
    async def optimize_weights(self, factor_returns: Dict[str, List[float]], 
                             current_weights: Dict[str, float],
                             market_regime: str = None) -> Dict[str, float]:
        """均值方差优化"""
        try:
            # 更新因子收益率
            for factor_id, returns in factor_returns.items():
                self.update_factor_returns(factor_id, returns)
            
            # 计算协方差矩阵
            cov_matrix = self.calculate_covariance_matrix()
            if cov_matrix is None:
                return current_weights
            
            # 计算期望收益率
            expected_returns = np.array([
                np.mean(list(self.factor_returns[factor]))
                for factor in self.factor_names
            ])
            
            # 优化目标函数
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                return -portfolio_return + self.config.risk_aversion * portfolio_variance
            
            # 约束条件
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
            ]
            
            bounds = [(self.config.min_weight, self.config.max_weight) 
                     for _ in range(len(self.factor_names))]
            
            # 初始权重
            x0 = np.array([current_weights.get(factor, 1.0/len(self.factor_names)) 
                          for factor in self.factor_names])
            
            # 执行优化
            if optimize:
                result = optimize.minimize(
                    objective, x0, method='SLSQP',
                    bounds=bounds, constraints=constraints
                )
                
                if result.success:
                    optimal_weights = dict(zip(self.factor_names, result.x))
                else:
                    logger.warning("均值方差优化失败，使用等权重")
                    optimal_weights = {factor: 1.0/len(self.factor_names) 
                                     for factor in self.factor_names}
            else:
                # 简化的等权重分配
                optimal_weights = {factor: 1.0/len(self.factor_names) 
                                 for factor in self.factor_names}
            
            return optimal_weights
            
        except Exception as e:
            logger.error(f"均值方差优化失败: {e}")
            return current_weights

class RiskParityOptimizer(BaseOptimizer):
    """风险平价优化器"""
    
    async def optimize_weights(self, factor_returns: Dict[str, List[float]], 
                             current_weights: Dict[str, float],
                             market_regime: str = None) -> Dict[str, float]:
        """风险平价优化"""
        try:
            # 更新因子收益率
            for factor_id, returns in factor_returns.items():
                self.update_factor_returns(factor_id, returns)
            
            # 计算协方差矩阵
            cov_matrix = self.calculate_covariance_matrix()
            if cov_matrix is None:
                return current_weights
            
            n_factors = len(self.factor_names)
            
            # 风险平价目标函数
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                target_contrib = portfolio_vol / n_factors
                return np.sum((contrib - target_contrib) ** 2)
            
            # 约束条件
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            ]
            
            bounds = [(self.config.min_weight, self.config.max_weight) 
                     for _ in range(n_factors)]
            
            # 初始权重
            x0 = np.ones(n_factors) / n_factors
            
            # 执行优化
            if optimize:
                result = optimize.minimize(
                    risk_parity_objective, x0, method='SLSQP',
                    bounds=bounds, constraints=constraints
                )
                
                if result.success:
                    optimal_weights = dict(zip(self.factor_names, result.x))
                else:
                    logger.warning("风险平价优化失败，使用等权重")
                    optimal_weights = {factor: 1.0/n_factors for factor in self.factor_names}
            else:
                # 简化实现：逆波动率权重
                volatilities = np.sqrt(np.diag(cov_matrix))
                inv_vol_weights = 1.0 / volatilities
                inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)
                optimal_weights = dict(zip(self.factor_names, inv_vol_weights))
            
            return optimal_weights
            
        except Exception as e:
            logger.error(f"风险平价优化失败: {e}")
            return current_weights

class OnlineGradientOptimizer(BaseOptimizer):
    """在线梯度下降优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.gradient_history: deque = deque(maxlen=100)
        self.learning_rate_schedule = self._create_learning_rate_schedule()
        
    def _create_learning_rate_schedule(self) -> Callable[[int], float]:
        """创建学习率调度"""
        base_lr = self.config.learning_rate
        
        def schedule(step: int) -> float:
            # 使用1/sqrt(t)衰减
            return base_lr / np.sqrt(max(1, step))
        
        return schedule
    
    async def optimize_weights(self, factor_returns: Dict[str, List[float]], 
                             current_weights: Dict[str, float],
                             market_regime: str = None) -> Dict[str, float]:
        """在线梯度下降优化"""
        try:
            if not factor_returns:
                return current_weights
            
            # 获取最新收益率
            latest_returns = {factor: returns[-1] if returns else 0.0 
                            for factor, returns in factor_returns.items()}
            
            # 计算组合收益率
            portfolio_return = sum(current_weights.get(factor, 0.0) * ret 
                                 for factor, ret in latest_returns.items())
            
            # 计算梯度（简化实现）
            gradients = {}
            for factor in latest_returns:
                # 梯度 = 因子收益率 - 组合收益率
                gradients[factor] = latest_returns[factor] - portfolio_return
            
            # 更新权重
            step = len(self.gradient_history) + 1
            lr = self.learning_rate_schedule(step)
            
            new_weights = {}
            for factor in current_weights:
                gradient = gradients.get(factor, 0.0)
                new_weight = current_weights[factor] + lr * gradient
                new_weights[factor] = np.clip(new_weight, 
                                            self.config.min_weight, 
                                            self.config.max_weight)
            
            # 归一化权重
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                new_weights = {factor: weight / total_weight 
                             for factor, weight in new_weights.items()}
            
            # 记录梯度历史
            self.gradient_history.append(gradients)
            
            return new_weights
            
        except Exception as e:
            logger.error(f"在线梯度优化失败: {e}")
            return current_weights

class ThompsonSamplingOptimizer(BaseOptimizer):
    """汤普森采样优化器（多臂老虎机）"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        # 贝叶斯参数：每个因子的收益率分布参数
        self.alpha: Dict[str, float] = defaultdict(lambda: 1.0)  # 成功次数
        self.beta: Dict[str, float] = defaultdict(lambda: 1.0)   # 失败次数
        self.reward_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    async def optimize_weights(self, factor_returns: Dict[str, List[float]], 
                             current_weights: Dict[str, float],
                             market_regime: str = None) -> Dict[str, float]:
        """汤普森采样优化"""
        try:
            # 更新贝叶斯参数
            for factor, returns in factor_returns.items():
                if returns:
                    latest_return = returns[-1]
                    self.reward_history[factor].append(latest_return)
                    
                    # 更新Beta分布参数
                    if latest_return > 0:
                        self.alpha[factor] += 1
                    else:
                        self.beta[factor] += 1
            
            # 汤普森采样
            sampled_rewards = {}
            for factor in current_weights:
                # 从Beta分布采样
                sampled_reward = np.random.beta(self.alpha[factor], self.beta[factor])
                sampled_rewards[factor] = sampled_reward
            
            # 基于采样结果分配权重
            total_reward = sum(sampled_rewards.values())
            if total_reward > 0:
                new_weights = {factor: reward / total_reward 
                             for factor, reward in sampled_rewards.items()}
            else:
                # 等权重分配
                n_factors = len(current_weights)
                new_weights = {factor: 1.0 / n_factors for factor in current_weights}
            
            # 应用权重约束
            for factor in new_weights:
                new_weights[factor] = np.clip(new_weights[factor],
                                            self.config.min_weight,
                                            self.config.max_weight)
            
            # 重新归一化
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                new_weights = {factor: weight / total_weight 
                             for factor, weight in new_weights.items()}
            
            return new_weights
            
        except Exception as e:
            logger.error(f"汤普森采样优化失败: {e}")
            return current_weights

class UCBOptimizer(BaseOptimizer):
    """置信上界(UCB)优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.factor_counts: Dict[str, int] = defaultdict(int)
        self.factor_rewards: Dict[str, float] = defaultdict(float)
        self.total_rounds = 0
        
    async def optimize_weights(self, factor_returns: Dict[str, List[float]], 
                             current_weights: Dict[str, float],
                             market_regime: str = None) -> Dict[str, float]:
        """UCB优化"""
        try:
            self.total_rounds += 1
            
            # 更新因子统计
            for factor, returns in factor_returns.items():
                if returns:
                    latest_return = returns[-1]
                    self.factor_counts[factor] += 1
                    self.factor_rewards[factor] += latest_return
            
            # 计算UCB值
            ucb_values = {}
            for factor in current_weights:
                if self.factor_counts[factor] == 0:
                    ucb_values[factor] = float('inf')  # 未探索的因子
                else:
                    avg_reward = self.factor_rewards[factor] / self.factor_counts[factor]
                    confidence_bound = np.sqrt(
                        2 * np.log(self.total_rounds) / self.factor_counts[factor]
                    )
                    ucb_values[factor] = avg_reward + confidence_bound
            
            # 基于UCB值分配权重
            if all(v == float('inf') for v in ucb_values.values()):
                # 所有因子都未探索，等权重分配
                n_factors = len(current_weights)
                new_weights = {factor: 1.0 / n_factors for factor in current_weights}
            else:
                # 将无穷大值替换为最大有限值的两倍
                finite_values = [v for v in ucb_values.values() if v != float('inf')]
                if finite_values:
                    max_finite = max(finite_values)
                    for factor in ucb_values:
                        if ucb_values[factor] == float('inf'):
                            ucb_values[factor] = max_finite * 2
                
                # Softmax权重分配
                exp_values = {factor: np.exp(value / self.config.exploration_rate) 
                            for factor, value in ucb_values.items()}
                total_exp = sum(exp_values.values())
                new_weights = {factor: exp_val / total_exp 
                             for factor, exp_val in exp_values.items()}
            
            # 应用权重约束
            for factor in new_weights:
                new_weights[factor] = np.clip(new_weights[factor],
                                            self.config.min_weight,
                                            self.config.max_weight)
            
            # 重新归一化
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                new_weights = {factor: weight / total_weight 
                             for factor, weight in new_weights.items()}
            
            return new_weights
            
        except Exception as e:
            logger.error(f"UCB优化失败: {e}")
            return current_weights

class WeightOptimizer:
    """权重优化器管理器
    
    统一管理不同的权重优化算法
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimizer = self._create_optimizer()
        
        # 权重历史
        self.weight_history: List[WeightUpdate] = []
        self.performance_history: List[PerformanceMetrics] = []
        
        # 当前权重
        self.current_weights: Dict[str, float] = {}
        
        # 再平衡控制
        self.last_rebalance: Optional[datetime] = None
        self.rebalance_threshold = 0.05  # 权重变化阈值
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info(f"权重优化器初始化完成，方法: {config.method.value}")
    
    def _create_optimizer(self) -> BaseOptimizer:
        """创建优化器实例"""
        if self.config.method == OptimizationMethod.MEAN_VARIANCE:
            return MeanVarianceOptimizer(self.config)
        elif self.config.method == OptimizationMethod.RISK_PARITY:
            return RiskParityOptimizer(self.config)
        elif self.config.method == OptimizationMethod.ONLINE_GRADIENT:
            return OnlineGradientOptimizer(self.config)
        elif self.config.method == OptimizationMethod.THOMPSON_SAMPLING:
            return ThompsonSamplingOptimizer(self.config)
        elif self.config.method == OptimizationMethod.UCB:
            return UCBOptimizer(self.config)
        else:
            logger.warning(f"未知优化方法: {self.config.method}, 使用均值方差优化")
            return MeanVarianceOptimizer(self.config)
    
    async def optimize_weights(self, factor_returns: Dict[str, List[float]], 
                             market_regime: str = None,
                             force_rebalance: bool = False) -> Dict[str, float]:
        """优化权重
        
        Args:
            factor_returns: 因子收益率历史
            market_regime: 当前市场状态
            force_rebalance: 是否强制再平衡
            
        Returns:
            优化后的权重
        """
        try:
            with self.lock:
                # 检查是否需要再平衡
                if not force_rebalance and not self._should_rebalance():
                    return self.current_weights.copy()
                
                # 初始化权重（如果为空）
                if not self.current_weights:
                    n_factors = len(factor_returns)
                    self.current_weights = {factor: 1.0 / n_factors 
                                          for factor in factor_returns.keys()}
                
                # 执行优化
                start_time = datetime.now()
                new_weights = await self.optimizer.optimize_weights(
                    factor_returns, self.current_weights, market_regime
                )
                optimization_time = (datetime.now() - start_time).total_seconds()
                
                # 计算再平衡成本
                rebalance_cost = self._calculate_rebalance_cost(
                    self.current_weights, new_weights
                )
                
                # 记录权重更新
                weight_update = WeightUpdate(
                    timestamp=datetime.now(),
                    old_weights=self.current_weights.copy(),
                    new_weights=new_weights.copy(),
                    method=self.config.method.value,
                    performance_metrics={
                        'optimization_time': optimization_time,
                        'turnover': self._calculate_turnover(self.current_weights, new_weights)
                    },
                    rebalance_cost=rebalance_cost
                )
                
                self.weight_history.append(weight_update)
                
                # 更新当前权重
                self.current_weights = new_weights
                self.last_rebalance = datetime.now()
                
                logger.info(f"权重优化完成，换手率: {weight_update.get_turnover():.3f}")
                return new_weights
                
        except Exception as e:
            logger.error(f"权重优化失败: {e}")
            return self.current_weights.copy()
    
    def _should_rebalance(self) -> bool:
        """判断是否应该再平衡"""
        if self.last_rebalance is None:
            return True
        
        now = datetime.now()
        time_since_last = now - self.last_rebalance
        
        # 基于频率的再平衡
        if self.config.rebalance_frequency == RebalanceFrequency.DAILY:
            return time_since_last >= timedelta(days=1)
        elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY:
            return time_since_last >= timedelta(weeks=1)
        elif self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
            return time_since_last >= timedelta(days=30)
        elif self.config.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            return time_since_last >= timedelta(days=90)
        else:  # ADAPTIVE
            # 自适应再平衡：基于权重漂移
            return self._check_weight_drift()
    
    def _check_weight_drift(self) -> bool:
        """检查权重漂移"""
        # 简化实现：总是返回False，实际应该基于市场数据计算权重漂移
        return False
    
    def _calculate_rebalance_cost(self, old_weights: Dict[str, float], 
                                new_weights: Dict[str, float]) -> float:
        """计算再平衡成本"""
        turnover = self._calculate_turnover(old_weights, new_weights)
        return turnover * self.config.transaction_cost
    
    def _calculate_turnover(self, old_weights: Dict[str, float], 
                          new_weights: Dict[str, float]) -> float:
        """计算换手率"""
        all_factors = set(old_weights.keys()) | set(new_weights.keys())
        total_change = 0.0
        
        for factor in all_factors:
            old_w = old_weights.get(factor, 0.0)
            new_w = new_weights.get(factor, 0.0)
            total_change += abs(new_w - old_w)
        
        return total_change / 2
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        with self.lock:
            return self.current_weights.copy()
    
    def get_weight_history(self, limit: int = None) -> List[WeightUpdate]:
        """获取权重历史"""
        with self.lock:
            if limit:
                return self.weight_history[-limit:]
            return self.weight_history.copy()
    
    def calculate_performance_metrics(self, portfolio_returns: List[float], 
                                    benchmark_returns: List[float] = None) -> PerformanceMetrics:
        """计算性能指标"""
        try:
            if not portfolio_returns:
                return PerformanceMetrics()
            
            returns_array = np.array(portfolio_returns)
            
            # 基本指标
            total_return = np.prod(1 + returns_array) - 1
            volatility = np.std(returns_array) * np.sqrt(252)
            
            # 夏普比率
            if volatility > 0:
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # 最大回撤
            cumulative = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # 卡玛比率
            if max_drawdown < 0:
                calmar_ratio = total_return / abs(max_drawdown)
            else:
                calmar_ratio = 0.0
            
            # 信息比率和跟踪误差
            information_ratio = 0.0
            tracking_error = 0.0
            if benchmark_returns and len(benchmark_returns) == len(portfolio_returns):
                excess_returns = returns_array - np.array(benchmark_returns)
                tracking_error = np.std(excess_returns) * np.sqrt(252)
                if tracking_error > 0:
                    information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
            # 胜率
            hit_rate = np.sum(returns_array > 0) / len(returns_array)
            
            metrics = PerformanceMetrics(
                returns=total_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                hit_rate=hit_rate
            )
            
            self.performance_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"计算性能指标失败: {e}")
            return PerformanceMetrics()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        with self.lock:
            if not self.weight_history:
                return {}
            
            # 计算统计指标
            turnovers = [update.get_turnover() for update in self.weight_history]
            rebalance_costs = [update.rebalance_cost for update in self.weight_history]
            
            return {
                'total_rebalances': len(self.weight_history),
                'avg_turnover': np.mean(turnovers) if turnovers else 0.0,
                'total_rebalance_cost': sum(rebalance_costs),
                'avg_rebalance_cost': np.mean(rebalance_costs) if rebalance_costs else 0.0,
                'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
                'current_weights': self.current_weights,
                'optimization_method': self.config.method.value
            }

# 便捷函数
def create_optimization_config(method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
                             rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY,
                             **kwargs) -> OptimizationConfig:
    """创建优化配置
    
    Args:
        method: 优化方法
        rebalance_frequency: 再平衡频率
        **kwargs: 其他配置参数
        
    Returns:
        优化配置
    """
    config = OptimizationConfig(method=method, rebalance_frequency=rebalance_frequency)
    
    # 更新配置参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

def create_weight_optimizer(config: OptimizationConfig = None) -> WeightOptimizer:
    """创建权重优化器
    
    Args:
        config: 优化配置
        
    Returns:
        权重优化器实例
    """
    if config is None:
        config = create_optimization_config()
    
    return WeightOptimizer(config)