# 在线学习器 - 实现权重动态更新和Multi-armed Bandit强化学习

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import deque

try:
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    SGDRegressor = None
    StandardScaler = None

logger = logging.getLogger(__name__)

class LearningMethod(Enum):
    """学习方法枚举"""
    RLS = "recursive_least_squares"    # 递归最小二乘
    EWMA = "exponential_weighted"       # 指数加权移动平均
    SGD = "stochastic_gradient"         # 随机梯度下降
    BANDIT = "multi_armed_bandit"       # 多臂老虎机
    THOMPSON = "thompson_sampling"      # 汤普森采样
    UCB = "upper_confidence_bound"      # 置信上界

@dataclass
class LearningState:
    """学习状态"""
    weights: Dict[str, float]
    performance: float
    confidence: float
    update_count: int
    last_updated: datetime
    learning_rate: float
    convergence_score: float

@dataclass
class BanditArm:
    """老虎机臂"""
    factor_id: str
    weight: float
    reward_sum: float
    pull_count: int
    confidence_interval: Tuple[float, float]
    last_reward: float
    alpha: float = 1.0  # Beta分布参数
    beta: float = 1.0   # Beta分布参数

@dataclass
class PerformanceMetric:
    """性能指标"""
    timestamp: datetime
    factor_weights: Dict[str, float]
    return_value: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    information_ratio: float

class OnlineLearner:
    """在线学习器
    
    实现多种在线学习算法用于动态调整因子权重：
    1. 递归最小二乘法(RLS)
    2. 指数加权移动平均(EWMA)
    3. 随机梯度下降(SGD)
    4. Multi-armed Bandit算法
    5. Thompson采样
    6. 置信上界(UCB)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get('method', 'bandit')
        self.learning_rate = config.get('learning_rate', 0.01)
        self.decay_factor = config.get('decay_factor', 0.95)
        self.exploration_rate = config.get('exploration_rate', 0.1)
        
        # 因子列表
        self.factor_names = config.get('factor_names', [])
        self.n_factors = len(self.factor_names)
        
        # 学习状态
        self.learning_state = self._init_learning_state()
        
        # 性能历史
        self.performance_history: deque = deque(maxlen=config.get('max_history', 1000))
        
        # 方法特定的组件
        self._init_method_components()
        
        logger.info(f"在线学习器初始化完成，方法: {self.method}, 因子数: {self.n_factors}")
    
    def _init_learning_state(self) -> LearningState:
        """初始化学习状态"""
        # 均匀初始化权重
        if self.n_factors > 0:
            initial_weight = 1.0 / self.n_factors
            weights = {factor: initial_weight for factor in self.factor_names}
        else:
            weights = {}
        
        return LearningState(
            weights=weights,
            performance=0.0,
            confidence=0.5,
            update_count=0,
            last_updated=datetime.now(),
            learning_rate=self.learning_rate,
            convergence_score=0.0
        )
    
    def _init_method_components(self):
        """初始化方法特定组件"""
        if self.method == LearningMethod.RLS.value:
            # 递归最小二乘参数
            self.P = np.eye(self.n_factors) * 1000  # 协方差矩阵
            self.theta = np.ones(self.n_factors) / self.n_factors  # 参数向量
            self.lambda_rls = self.config.get('lambda_rls', 0.99)  # 遗忘因子
        
        elif self.method == LearningMethod.SGD.value:
            # SGD回归器
            if SGDRegressor is not None:
                self.sgd_model = SGDRegressor(
                    learning_rate='adaptive',
                    eta0=self.learning_rate,
                    random_state=42
                )
                self.scaler = StandardScaler() if StandardScaler else None
                self.is_fitted = False
            else:
                logger.warning("sklearn未安装，SGD方法不可用")
        
        elif self.method in [LearningMethod.BANDIT.value, LearningMethod.THOMPSON.value, LearningMethod.UCB.value]:
            # 多臂老虎机
            self.bandit_arms = {
                factor: BanditArm(
                    factor_id=factor,
                    weight=1.0 / self.n_factors if self.n_factors > 0 else 0.0,
                    reward_sum=0.0,
                    pull_count=0,
                    confidence_interval=(0.0, 1.0),
                    last_reward=0.0
                )
                for factor in self.factor_names
            }
            self.total_pulls = 0
            self.ucb_confidence = self.config.get('ucb_confidence', 2.0)
    
    async def update_weights(self, factor_returns: Dict[str, float], 
                           portfolio_return: float, 
                           market_regime: str = None) -> LearningState:
        """更新因子权重
        
        Args:
            factor_returns: 各因子的收益率
            portfolio_return: 组合收益率
            market_regime: 当前市场状态
            
        Returns:
            更新后的学习状态
        """
        try:
            # 记录性能指标
            self._record_performance(factor_returns, portfolio_return)
            
            # 根据方法更新权重
            if self.method == LearningMethod.RLS.value:
                new_weights = await self._update_weights_rls(factor_returns, portfolio_return)
            elif self.method == LearningMethod.EWMA.value:
                new_weights = await self._update_weights_ewma(factor_returns, portfolio_return)
            elif self.method == LearningMethod.SGD.value:
                new_weights = await self._update_weights_sgd(factor_returns, portfolio_return)
            elif self.method == LearningMethod.BANDIT.value:
                new_weights = await self._update_weights_bandit(factor_returns, portfolio_return)
            elif self.method == LearningMethod.THOMPSON.value:
                new_weights = await self._update_weights_thompson(factor_returns, portfolio_return)
            elif self.method == LearningMethod.UCB.value:
                new_weights = await self._update_weights_ucb(factor_returns, portfolio_return)
            else:
                new_weights = self.learning_state.weights
            
            # 更新学习状态
            self.learning_state.weights = new_weights
            self.learning_state.performance = portfolio_return
            self.learning_state.update_count += 1
            self.learning_state.last_updated = datetime.now()
            
            # 计算收敛分数
            self.learning_state.convergence_score = self._calculate_convergence_score()
            
            # 自适应学习率
            self._adapt_learning_rate()
            
            logger.debug(f"权重更新完成，方法: {self.method}, 性能: {portfolio_return:.4f}")
            return self.learning_state
            
        except Exception as e:
            logger.error(f"权重更新失败: {str(e)}")
            return self.learning_state
    
    async def _update_weights_rls(self, factor_returns: Dict[str, float], 
                                portfolio_return: float) -> Dict[str, float]:
        """使用递归最小二乘更新权重"""
        try:
            # 构建特征向量
            x = np.array([factor_returns.get(factor, 0.0) for factor in self.factor_names])
            y = portfolio_return
            
            # RLS更新
            k = self.P @ x / (self.lambda_rls + x.T @ self.P @ x)
            self.theta = self.theta + k * (y - x.T @ self.theta)
            self.P = (self.P - np.outer(k, x.T @ self.P)) / self.lambda_rls
            
            # 归一化权重
            weights = np.abs(self.theta)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
            return dict(zip(self.factor_names, weights))
            
        except Exception as e:
            logger.error(f"RLS更新失败: {str(e)}")
            return self.learning_state.weights
    
    async def _update_weights_ewma(self, factor_returns: Dict[str, float], 
                                 portfolio_return: float) -> Dict[str, float]:
        """使用指数加权移动平均更新权重"""
        try:
            current_weights = self.learning_state.weights.copy()
            
            # 计算因子贡献度
            total_return = sum(factor_returns.values())
            if total_return != 0:
                for factor in self.factor_names:
                    factor_contribution = factor_returns.get(factor, 0.0) / total_return
                    
                    # EWMA更新
                    current_weight = current_weights.get(factor, 0.0)
                    new_weight = (1 - self.learning_rate) * current_weight + \
                                self.learning_rate * abs(factor_contribution)
                    current_weights[factor] = new_weight
            
            # 归一化权重
            total_weight = sum(current_weights.values())
            if total_weight > 0:
                current_weights = {k: v / total_weight for k, v in current_weights.items()}
            
            return current_weights
            
        except Exception as e:
            logger.error(f"EWMA更新失败: {str(e)}")
            return self.learning_state.weights
    
    async def _update_weights_sgd(self, factor_returns: Dict[str, float], 
                                portfolio_return: float) -> Dict[str, float]:
        """使用随机梯度下降更新权重"""
        try:
            if SGDRegressor is None or not hasattr(self, 'sgd_model'):
                return self.learning_state.weights
            
            # 准备数据
            X = np.array([list(factor_returns.values())])
            y = np.array([portfolio_return])
            
            # 标准化特征
            if self.scaler is not None:
                if not self.is_fitted:
                    X = self.scaler.fit_transform(X)
                    self.is_fitted = True
                else:
                    X = self.scaler.transform(X)
            
            # 部分拟合
            if hasattr(self.sgd_model, 'partial_fit'):
                self.sgd_model.partial_fit(X, y)
            else:
                self.sgd_model.fit(X, y)
            
            # 获取权重
            if hasattr(self.sgd_model, 'coef_'):
                weights = np.abs(self.sgd_model.coef_)
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
                return dict(zip(self.factor_names, weights))
            
            return self.learning_state.weights
            
        except Exception as e:
            logger.error(f"SGD更新失败: {str(e)}")
            return self.learning_state.weights
    
    async def _update_weights_bandit(self, factor_returns: Dict[str, float], 
                                   portfolio_return: float) -> Dict[str, float]:
        """使用多臂老虎机更新权重"""
        try:
            # 更新每个臂的奖励
            for factor in self.factor_names:
                if factor in self.bandit_arms:
                    arm = self.bandit_arms[factor]
                    factor_return = factor_returns.get(factor, 0.0)
                    
                    # 计算奖励（基于因子收益和权重）
                    reward = factor_return * arm.weight
                    
                    # 更新臂的统计信息
                    arm.reward_sum += reward
                    arm.pull_count += 1
                    arm.last_reward = reward
                    
                    # 更新置信区间
                    if arm.pull_count > 0:
                        mean_reward = arm.reward_sum / arm.pull_count
                        std_error = np.sqrt(2 * np.log(self.total_pulls + 1) / arm.pull_count)
                        arm.confidence_interval = (
                            mean_reward - std_error,
                            mean_reward + std_error
                        )
            
            self.total_pulls += 1
            
            # 使用ε-贪心策略选择权重
            if np.random.random() < self.exploration_rate:
                # 探索：随机权重
                weights = np.random.dirichlet(np.ones(self.n_factors))
            else:
                # 利用：基于平均奖励的权重
                rewards = []
                for factor in self.factor_names:
                    arm = self.bandit_arms[factor]
                    if arm.pull_count > 0:
                        rewards.append(arm.reward_sum / arm.pull_count)
                    else:
                        rewards.append(0.0)
                
                # Softmax权重分配
                rewards = np.array(rewards)
                if np.max(rewards) > np.min(rewards):
                    exp_rewards = np.exp(rewards - np.max(rewards))
                    weights = exp_rewards / np.sum(exp_rewards)
                else:
                    weights = np.ones(self.n_factors) / self.n_factors
            
            # 更新臂的权重
            for i, factor in enumerate(self.factor_names):
                self.bandit_arms[factor].weight = weights[i]
            
            return dict(zip(self.factor_names, weights))
            
        except Exception as e:
            logger.error(f"Bandit更新失败: {str(e)}")
            return self.learning_state.weights
    
    async def _update_weights_thompson(self, factor_returns: Dict[str, float], 
                                     portfolio_return: float) -> Dict[str, float]:
        """使用Thompson采样更新权重"""
        try:
            # 更新Beta分布参数
            for factor in self.factor_names:
                if factor in self.bandit_arms:
                    arm = self.bandit_arms[factor]
                    factor_return = factor_returns.get(factor, 0.0)
                    
                    # 将收益转换为成功/失败
                    if factor_return > 0:
                        arm.alpha += 1
                    else:
                        arm.beta += 1
            
            # Thompson采样
            sampled_values = []
            for factor in self.factor_names:
                arm = self.bandit_arms[factor]
                # 从Beta分布采样
                sampled_value = np.random.beta(arm.alpha, arm.beta)
                sampled_values.append(sampled_value)
            
            # 归一化为权重
            sampled_values = np.array(sampled_values)
            weights = sampled_values / np.sum(sampled_values) if np.sum(sampled_values) > 0 else sampled_values
            
            return dict(zip(self.factor_names, weights))
            
        except Exception as e:
            logger.error(f"Thompson采样更新失败: {str(e)}")
            return self.learning_state.weights
    
    async def _update_weights_ucb(self, factor_returns: Dict[str, float], 
                                portfolio_return: float) -> Dict[str, float]:
        """使用置信上界更新权重"""
        try:
            # 更新每个臂的统计信息
            for factor in self.factor_names:
                if factor in self.bandit_arms:
                    arm = self.bandit_arms[factor]
                    factor_return = factor_returns.get(factor, 0.0)
                    
                    arm.reward_sum += factor_return * arm.weight
                    arm.pull_count += 1
            
            self.total_pulls += 1
            
            # 计算UCB值
            ucb_values = []
            for factor in self.factor_names:
                arm = self.bandit_arms[factor]
                if arm.pull_count > 0:
                    mean_reward = arm.reward_sum / arm.pull_count
                    confidence_bonus = self.ucb_confidence * np.sqrt(
                        np.log(self.total_pulls) / arm.pull_count
                    )
                    ucb_value = mean_reward + confidence_bonus
                else:
                    ucb_value = float('inf')  # 未探索的臂
                
                ucb_values.append(ucb_value)
            
            # 基于UCB值分配权重
            ucb_values = np.array(ucb_values)
            
            # 处理无穷大值
            if np.any(np.isinf(ucb_values)):
                # 如果有未探索的臂，给它们更高权重
                weights = np.where(np.isinf(ucb_values), 1.0, 0.1)
            else:
                # Softmax转换
                exp_ucb = np.exp(ucb_values - np.max(ucb_values))
                weights = exp_ucb / np.sum(exp_ucb)
            
            # 归一化
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
            return dict(zip(self.factor_names, weights))
            
        except Exception as e:
            logger.error(f"UCB更新失败: {str(e)}")
            return self.learning_state.weights
    
    def _record_performance(self, factor_returns: Dict[str, float], portfolio_return: float):
        """记录性能指标"""
        try:
            # 计算夏普比率（简化版）
            if len(self.performance_history) > 0:
                returns = [p.return_value for p in self.performance_history]
                returns.append(portfolio_return)
                
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
                
                # 计算最大回撤
                cumulative_returns = np.cumprod(1 + np.array(returns))
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - peak) / peak
                max_drawdown = np.min(drawdown)
                
                # 计算命中率
                positive_returns = sum(1 for r in returns if r > 0)
                hit_rate = positive_returns / len(returns)
                
                # 信息比率（简化）
                information_ratio = mean_return / std_return if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
                max_drawdown = 0.0
                hit_rate = 1.0 if portfolio_return > 0 else 0.0
                information_ratio = 0.0
            
            # 记录性能指标
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                factor_weights=self.learning_state.weights.copy(),
                return_value=portfolio_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                hit_rate=hit_rate,
                information_ratio=information_ratio
            )
            
            self.performance_history.append(metric)
            
        except Exception as e:
            logger.error(f"性能记录失败: {str(e)}")
    
    def _calculate_convergence_score(self) -> float:
        """计算收敛分数"""
        try:
            if len(self.performance_history) < 10:
                return 0.0
            
            # 计算权重变化的方差
            recent_weights = []
            for metric in list(self.performance_history)[-10:]:
                weights_vector = [metric.factor_weights.get(factor, 0.0) for factor in self.factor_names]
                recent_weights.append(weights_vector)
            
            recent_weights = np.array(recent_weights)
            weight_variance = np.mean(np.var(recent_weights, axis=0))
            
            # 收敛分数：方差越小，收敛分数越高
            convergence_score = 1.0 / (1.0 + weight_variance * 10)
            
            return convergence_score
            
        except Exception as e:
            logger.error(f"收敛分数计算失败: {str(e)}")
            return 0.0
    
    def _adapt_learning_rate(self):
        """自适应调整学习率"""
        try:
            # 基于收敛分数调整学习率
            if self.learning_state.convergence_score > 0.8:
                # 已收敛，降低学习率
                self.learning_state.learning_rate *= 0.95
            elif self.learning_state.convergence_score < 0.3:
                # 未收敛，提高学习率
                self.learning_state.learning_rate *= 1.05
            
            # 限制学习率范围
            self.learning_state.learning_rate = np.clip(
                self.learning_state.learning_rate, 0.001, 0.1
            )
            
        except Exception as e:
            logger.error(f"学习率调整失败: {str(e)}")
    
    def get_factor_importance(self) -> Dict[str, float]:
        """获取因子重要性排序"""
        try:
            if self.method in [LearningMethod.BANDIT.value, LearningMethod.THOMPSON.value, LearningMethod.UCB.value]:
                # 基于累积奖励计算重要性
                importance = {}
                for factor, arm in self.bandit_arms.items():
                    if arm.pull_count > 0:
                        importance[factor] = arm.reward_sum / arm.pull_count
                    else:
                        importance[factor] = 0.0
                
                return importance
            else:
                # 基于当前权重
                return self.learning_state.weights.copy()
                
        except Exception as e:
            logger.error(f"因子重要性计算失败: {str(e)}")
            return {}
    
    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """获取性能指标"""
        try:
            if not self.performance_history:
                return {}
            
            # 获取指定天数的数据
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_metrics = [
                m for m in self.performance_history 
                if m.timestamp >= cutoff_date
            ]
            
            if not recent_metrics:
                return {}
            
            # 计算统计指标
            returns = [m.return_value for m in recent_metrics]
            sharpe_ratios = [m.sharpe_ratio for m in recent_metrics]
            
            return {
                'total_return': np.sum(returns),
                'avg_return': np.mean(returns),
                'volatility': np.std(returns),
                'sharpe_ratio': np.mean(sharpe_ratios),
                'max_drawdown': min(m.max_drawdown for m in recent_metrics),
                'hit_rate': np.mean([m.hit_rate for m in recent_metrics]),
                'information_ratio': np.mean([m.information_ratio for m in recent_metrics]),
                'total_trades': len(recent_metrics),
                'convergence_score': self.learning_state.convergence_score
            }
            
        except Exception as e:
            logger.error(f"性能指标计算失败: {str(e)}")
            return {}
    
    def reset_learning(self):
        """重置学习状态"""
        try:
            self.learning_state = self._init_learning_state()
            self.performance_history.clear()
            
            # 重置方法特定组件
            self._init_method_components()
            
            logger.info("学习状态已重置")
            
        except Exception as e:
            logger.error(f"学习状态重置失败: {str(e)}")
    
    def save_state(self) -> Dict[str, Any]:
        """保存学习状态"""
        try:
            state = {
                'learning_state': {
                    'weights': self.learning_state.weights,
                    'performance': self.learning_state.performance,
                    'confidence': self.learning_state.confidence,
                    'update_count': self.learning_state.update_count,
                    'learning_rate': self.learning_state.learning_rate,
                    'convergence_score': self.learning_state.convergence_score
                },
                'method': self.method,
                'config': self.config,
                'performance_history_length': len(self.performance_history)
            }
            
            # 保存方法特定状态
            if self.method in [LearningMethod.BANDIT.value, LearningMethod.THOMPSON.value, LearningMethod.UCB.value]:
                state['bandit_arms'] = {
                    factor: {
                        'weight': arm.weight,
                        'reward_sum': arm.reward_sum,
                        'pull_count': arm.pull_count,
                        'alpha': arm.alpha,
                        'beta': arm.beta
                    }
                    for factor, arm in self.bandit_arms.items()
                }
                state['total_pulls'] = self.total_pulls
            
            return state
            
        except Exception as e:
            logger.error(f"状态保存失败: {str(e)}")
            return {}
    
    def load_state(self, state: Dict[str, Any]):
        """加载学习状态"""
        try:
            if 'learning_state' in state:
                ls = state['learning_state']
                self.learning_state.weights = ls.get('weights', {})
                self.learning_state.performance = ls.get('performance', 0.0)
                self.learning_state.confidence = ls.get('confidence', 0.5)
                self.learning_state.update_count = ls.get('update_count', 0)
                self.learning_state.learning_rate = ls.get('learning_rate', self.learning_rate)
                self.learning_state.convergence_score = ls.get('convergence_score', 0.0)
            
            # 加载方法特定状态
            if 'bandit_arms' in state and hasattr(self, 'bandit_arms'):
                for factor, arm_data in state['bandit_arms'].items():
                    if factor in self.bandit_arms:
                        arm = self.bandit_arms[factor]
                        arm.weight = arm_data.get('weight', arm.weight)
                        arm.reward_sum = arm_data.get('reward_sum', 0.0)
                        arm.pull_count = arm_data.get('pull_count', 0)
                        arm.alpha = arm_data.get('alpha', 1.0)
                        arm.beta = arm_data.get('beta', 1.0)
                
                self.total_pulls = state.get('total_pulls', 0)
            
            logger.info("学习状态加载成功")
            
        except Exception as e:
            logger.error(f"状态加载失败: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取学习器统计信息"""
        return {
            'method': self.method,
            'n_factors': self.n_factors,
            'update_count': self.learning_state.update_count,
            'convergence_score': self.learning_state.convergence_score,
            'current_learning_rate': self.learning_state.learning_rate,
            'performance_history_length': len(self.performance_history),
            'current_weights': self.learning_state.weights,
            'total_pulls': getattr(self, 'total_pulls', 0),
            'exploration_rate': self.exploration_rate
        }