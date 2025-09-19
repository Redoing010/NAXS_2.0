# 因子评估器模块
# 实现因子性能评估、筛选和排名功能

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
    from scipy.stats import spearmanr, pearsonr
except ImportError:
    stats = None
    spearmanr = None
    pearsonr = None

logger = logging.getLogger(__name__)

class EvaluationMetric(Enum):
    """评估指标枚举"""
    IC = "ic"                          # 信息系数
    RANK_IC = "rank_ic"                # 排序信息系数
    IC_IR = "ic_ir"                    # IC信息比率
    TURNOVER = "turnover"              # 换手率
    DECAY = "decay"                    # 衰减性
    STABILITY = "stability"            # 稳定性
    MONOTONICITY = "monotonicity"      # 单调性
    COVERAGE = "coverage"              # 覆盖率
    FACTOR_LOADING = "factor_loading"  # 因子载荷
    T_STAT = "t_stat"                  # t统计量
    P_VALUE = "p_value"                # p值
    SHARPE_RATIO = "sharpe_ratio"      # 夏普比率
    MAX_DRAWDOWN = "max_drawdown"      # 最大回撤
    WIN_RATE = "win_rate"              # 胜率

class FactorQuality(Enum):
    """因子质量等级"""
    EXCELLENT = "excellent"    # 优秀
    GOOD = "good"              # 良好
    AVERAGE = "average"        # 一般
    POOR = "poor"              # 较差
    INVALID = "invalid"        # 无效

@dataclass
class EvaluationConfig:
    """评估配置"""
    # 评估窗口
    evaluation_window: int = 252  # 评估窗口长度
    rolling_window: int = 60      # 滚动窗口长度
    
    # IC计算参数
    ic_method: str = "pearson"    # pearson, spearman
    min_periods: int = 20         # 最小计算周期
    
    # 分层回测参数
    n_quantiles: int = 5          # 分层数量
    holding_period: int = 1       # 持有期
    
    # 阈值设置
    ic_threshold: float = 0.05    # IC阈值
    ic_ir_threshold: float = 0.5  # IC IR阈值
    turnover_threshold: float = 0.5  # 换手率阈值
    
    # 其他参数
    risk_free_rate: float = 0.03  # 无风险利率
    benchmark_return: float = 0.08  # 基准收益率
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'evaluation_window': self.evaluation_window,
            'rolling_window': self.rolling_window,
            'ic_method': self.ic_method,
            'min_periods': self.min_periods,
            'n_quantiles': self.n_quantiles,
            'holding_period': self.holding_period,
            'ic_threshold': self.ic_threshold,
            'ic_ir_threshold': self.ic_ir_threshold,
            'turnover_threshold': self.turnover_threshold,
            'risk_free_rate': self.risk_free_rate,
            'benchmark_return': self.benchmark_return
        }

@dataclass
class FactorMetrics:
    """因子指标"""
    factor_id: str
    
    # 基础统计
    mean: float = 0.0
    std: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # IC相关指标
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    rank_ic_mean: float = 0.0
    rank_ic_std: float = 0.0
    rank_ic_ir: float = 0.0
    
    # 回归指标
    factor_loading: float = 0.0
    t_stat: float = 0.0
    p_value: float = 1.0
    r_squared: float = 0.0
    
    # 分层回测指标
    long_short_return: float = 0.0
    long_short_sharpe: float = 0.0
    long_short_volatility: float = 0.0
    max_drawdown: float = 0.0
    
    # 换手率和衰减
    turnover: float = 0.0
    auto_correlation: float = 0.0
    
    # 稳定性指标
    ic_win_rate: float = 0.0
    monotonicity: float = 0.0
    
    # 覆盖率
    coverage: float = 0.0
    
    # 质量评级
    quality: FactorQuality = FactorQuality.AVERAGE
    quality_score: float = 0.0
    
    # 评估时间
    evaluation_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_id': self.factor_id,
            'mean': self.mean,
            'std': self.std,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'ic_mean': self.ic_mean,
            'ic_std': self.ic_std,
            'ic_ir': self.ic_ir,
            'rank_ic_mean': self.rank_ic_mean,
            'rank_ic_std': self.rank_ic_std,
            'rank_ic_ir': self.rank_ic_ir,
            'factor_loading': self.factor_loading,
            't_stat': self.t_stat,
            'p_value': self.p_value,
            'r_squared': self.r_squared,
            'long_short_return': self.long_short_return,
            'long_short_sharpe': self.long_short_sharpe,
            'long_short_volatility': self.long_short_volatility,
            'max_drawdown': self.max_drawdown,
            'turnover': self.turnover,
            'auto_correlation': self.auto_correlation,
            'ic_win_rate': self.ic_win_rate,
            'monotonicity': self.monotonicity,
            'coverage': self.coverage,
            'quality': self.quality.value,
            'quality_score': self.quality_score,
            'evaluation_date': self.evaluation_date.isoformat()
        }

@dataclass
class FactorRanking:
    """因子排名"""
    factor_id: str
    rank: int
    score: float
    metrics: FactorMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_id': self.factor_id,
            'rank': self.rank,
            'score': self.score,
            'metrics': self.metrics.to_dict()
        }

@dataclass
class EvaluationResult:
    """评估结果"""
    factor_metrics: Dict[str, FactorMetrics] = field(default_factory=dict)
    factor_rankings: List[FactorRanking] = field(default_factory=list)
    correlation_matrix: Optional[np.ndarray] = None
    factor_names: List[str] = field(default_factory=list)
    evaluation_summary: Dict[str, Any] = field(default_factory=dict)
    
    def get_top_factors(self, n: int = 10) -> List[FactorRanking]:
        """获取排名前N的因子"""
        return self.factor_rankings[:n]
    
    def get_factor_by_quality(self, quality: FactorQuality) -> List[str]:
        """按质量等级获取因子"""
        return [factor_id for factor_id, metrics in self.factor_metrics.items() 
                if metrics.quality == quality]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_metrics': {fid: metrics.to_dict() for fid, metrics in self.factor_metrics.items()},
            'factor_rankings': [ranking.to_dict() for ranking in self.factor_rankings],
            'correlation_matrix': self.correlation_matrix.tolist() if self.correlation_matrix is not None else None,
            'factor_names': self.factor_names,
            'evaluation_summary': self.evaluation_summary
        }

class FactorEvaluator:
    """因子评估器
    
    提供全面的因子性能评估功能
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # 历史数据存储
        self.factor_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.evaluation_window))
        self.return_data: deque = deque(maxlen=config.evaluation_window)
        self.price_data: deque = deque(maxlen=config.evaluation_window)
        
        # 评估结果缓存
        self.evaluation_cache: Dict[str, EvaluationResult] = {}
        self.metrics_history: Dict[str, List[FactorMetrics]] = defaultdict(list)
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("因子评估器初始化完成")
    
    async def evaluate_factors(self, factor_data: Dict[str, List[float]], 
                             return_data: List[float],
                             price_data: List[float] = None) -> EvaluationResult:
        """评估因子性能
        
        Args:
            factor_data: 因子数据 {factor_id: [values]}
            return_data: 收益率数据
            price_data: 价格数据（可选）
            
        Returns:
            评估结果
        """
        try:
            with self.lock:
                # 更新历史数据
                self._update_data(factor_data, return_data, price_data)
                
                # 计算各因子指标
                factor_metrics = {}
                for factor_id in factor_data.keys():
                    metrics = await self._evaluate_single_factor(factor_id)
                    if metrics:
                        factor_metrics[factor_id] = metrics
                
                # 计算因子相关性
                correlation_matrix, factor_names = self._calculate_factor_correlation(factor_data.keys())
                
                # 因子排名
                factor_rankings = self._rank_factors(factor_metrics)
                
                # 生成评估摘要
                evaluation_summary = self._generate_evaluation_summary(factor_metrics)
                
                # 创建评估结果
                result = EvaluationResult(
                    factor_metrics=factor_metrics,
                    factor_rankings=factor_rankings,
                    correlation_matrix=correlation_matrix,
                    factor_names=factor_names,
                    evaluation_summary=evaluation_summary
                )
                
                # 缓存结果
                cache_key = f"evaluation_{datetime.now().strftime('%Y%m%d')}"
                self.evaluation_cache[cache_key] = result
                
                logger.info(f"因子评估完成，评估因子数: {len(factor_metrics)}")
                return result
                
        except Exception as e:
            logger.error(f"因子评估失败: {e}")
            return EvaluationResult()
    
    def _update_data(self, factor_data: Dict[str, List[float]], 
                    return_data: List[float], price_data: List[float] = None):
        """更新历史数据"""
        # 更新因子数据
        for factor_id, values in factor_data.items():
            self.factor_data[factor_id].extend(values)
        
        # 更新收益率数据
        self.return_data.extend(return_data)
        
        # 更新价格数据
        if price_data:
            self.price_data.extend(price_data)
    
    async def _evaluate_single_factor(self, factor_id: str) -> Optional[FactorMetrics]:
        """评估单个因子"""
        try:
            factor_values = list(self.factor_data[factor_id])
            return_values = list(self.return_data)
            
            if len(factor_values) < self.config.min_periods or len(return_values) < self.config.min_periods:
                return None
            
            # 确保数据长度一致
            min_length = min(len(factor_values), len(return_values))
            factor_values = factor_values[-min_length:]
            return_values = return_values[-min_length:]
            
            # 创建因子指标对象
            metrics = FactorMetrics(factor_id=factor_id)
            
            # 基础统计
            metrics.mean = np.mean(factor_values)
            metrics.std = np.std(factor_values)
            if stats:
                metrics.skewness = stats.skew(factor_values)
                metrics.kurtosis = stats.kurtosis(factor_values)
            
            # 计算IC指标
            ic_metrics = self._calculate_ic_metrics(factor_values, return_values)
            metrics.ic_mean = ic_metrics['ic_mean']
            metrics.ic_std = ic_metrics['ic_std']
            metrics.ic_ir = ic_metrics['ic_ir']
            metrics.rank_ic_mean = ic_metrics['rank_ic_mean']
            metrics.rank_ic_std = ic_metrics['rank_ic_std']
            metrics.rank_ic_ir = ic_metrics['rank_ic_ir']
            metrics.ic_win_rate = ic_metrics['ic_win_rate']
            
            # 计算回归指标
            regression_metrics = self._calculate_regression_metrics(factor_values, return_values)
            metrics.factor_loading = regression_metrics['factor_loading']
            metrics.t_stat = regression_metrics['t_stat']
            metrics.p_value = regression_metrics['p_value']
            metrics.r_squared = regression_metrics['r_squared']
            
            # 计算分层回测指标
            backtest_metrics = self._calculate_backtest_metrics(factor_values, return_values)
            metrics.long_short_return = backtest_metrics['long_short_return']
            metrics.long_short_sharpe = backtest_metrics['long_short_sharpe']
            metrics.long_short_volatility = backtest_metrics['long_short_volatility']
            metrics.max_drawdown = backtest_metrics['max_drawdown']
            
            # 计算换手率和自相关
            metrics.turnover = self._calculate_turnover(factor_values)
            metrics.auto_correlation = self._calculate_auto_correlation(factor_values)
            
            # 计算单调性
            metrics.monotonicity = self._calculate_monotonicity(factor_values, return_values)
            
            # 计算覆盖率
            metrics.coverage = self._calculate_coverage(factor_values)
            
            # 评估因子质量
            metrics.quality, metrics.quality_score = self._evaluate_factor_quality(metrics)
            
            # 记录评估历史
            self.metrics_history[factor_id].append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"评估因子 {factor_id} 失败: {e}")
            return None
    
    def _calculate_ic_metrics(self, factor_values: List[float], 
                            return_values: List[float]) -> Dict[str, float]:
        """计算IC相关指标"""
        try:
            # 滚动计算IC
            window = self.config.rolling_window
            ics = []
            rank_ics = []
            
            for i in range(window, len(factor_values)):
                factor_window = factor_values[i-window:i]
                return_window = return_values[i-window:i]
                
                # 计算IC
                if self.config.ic_method == "pearson" and pearsonr:
                    ic, _ = pearsonr(factor_window, return_window)
                elif spearmanr:
                    ic, _ = spearmanr(factor_window, return_window)
                else:
                    # 简化计算
                    ic = np.corrcoef(factor_window, return_window)[0, 1]
                
                if not np.isnan(ic):
                    ics.append(ic)
                
                # 计算Rank IC
                if spearmanr:
                    rank_ic, _ = spearmanr(factor_window, return_window)
                    if not np.isnan(rank_ic):
                        rank_ics.append(rank_ic)
            
            # 计算统计指标
            ic_mean = np.mean(ics) if ics else 0.0
            ic_std = np.std(ics) if ics else 0.0
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
            
            rank_ic_mean = np.mean(rank_ics) if rank_ics else 0.0
            rank_ic_std = np.std(rank_ics) if rank_ics else 0.0
            rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0.0
            
            # IC胜率
            ic_win_rate = np.sum(np.array(ics) > 0) / len(ics) if ics else 0.0
            
            return {
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_ir': ic_ir,
                'rank_ic_mean': rank_ic_mean,
                'rank_ic_std': rank_ic_std,
                'rank_ic_ir': rank_ic_ir,
                'ic_win_rate': ic_win_rate
            }
            
        except Exception as e:
            logger.error(f"计算IC指标失败: {e}")
            return {key: 0.0 for key in ['ic_mean', 'ic_std', 'ic_ir', 'rank_ic_mean', 'rank_ic_std', 'rank_ic_ir', 'ic_win_rate']}
    
    def _calculate_regression_metrics(self, factor_values: List[float], 
                                    return_values: List[float]) -> Dict[str, float]:
        """计算回归指标"""
        try:
            # 线性回归: return = alpha + beta * factor + error
            X = np.array(factor_values).reshape(-1, 1)
            y = np.array(return_values)
            
            # 添加截距项
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            
            # 最小二乘法
            coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            alpha, beta = coeffs[0], coeffs[1]
            
            # 预测值和残差
            y_pred = X_with_intercept @ coeffs
            residuals = y - y_pred
            
            # R平方
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # t统计量和p值
            n = len(y)
            mse = ss_res / (n - 2) if n > 2 else 0.0
            
            if mse > 0:
                # 计算标准误
                x_centered = X.flatten() - np.mean(X.flatten())
                se_beta = np.sqrt(mse / np.sum(x_centered ** 2)) if np.sum(x_centered ** 2) > 0 else 0.0
                
                # t统计量
                t_stat = beta / se_beta if se_beta > 0 else 0.0
                
                # p值（双尾检验）
                if stats:
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                else:
                    p_value = 0.05 if abs(t_stat) > 2 else 0.5  # 简化估计
            else:
                t_stat = 0.0
                p_value = 1.0
            
            return {
                'factor_loading': beta,
                't_stat': t_stat,
                'p_value': p_value,
                'r_squared': r_squared
            }
            
        except Exception as e:
            logger.error(f"计算回归指标失败: {e}")
            return {'factor_loading': 0.0, 't_stat': 0.0, 'p_value': 1.0, 'r_squared': 0.0}
    
    def _calculate_backtest_metrics(self, factor_values: List[float], 
                                  return_values: List[float]) -> Dict[str, float]:
        """计算分层回测指标"""
        try:
            if len(factor_values) != len(return_values):
                return {'long_short_return': 0.0, 'long_short_sharpe': 0.0, 
                       'long_short_volatility': 0.0, 'max_drawdown': 0.0}
            
            # 分层
            n_quantiles = self.config.n_quantiles
            data = list(zip(factor_values, return_values))
            
            # 按因子值排序
            data.sort(key=lambda x: x[0])
            
            # 分组
            group_size = len(data) // n_quantiles
            groups = []
            
            for i in range(n_quantiles):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < n_quantiles - 1 else len(data)
                group_returns = [item[1] for item in data[start_idx:end_idx]]
                groups.append(group_returns)
            
            # 计算各组平均收益
            group_mean_returns = [np.mean(group) for group in groups if group]
            
            if len(group_mean_returns) < 2:
                return {'long_short_return': 0.0, 'long_short_sharpe': 0.0, 
                       'long_short_volatility': 0.0, 'max_drawdown': 0.0}
            
            # 多空组合收益（最高分位 - 最低分位）
            long_short_return = group_mean_returns[-1] - group_mean_returns[0]
            
            # 计算多空组合的时间序列收益
            long_short_series = []
            window = min(len(groups[0]), len(groups[-1]))
            
            for i in range(window):
                if i < len(groups[0]) and i < len(groups[-1]):
                    ls_return = groups[-1][i] - groups[0][i]
                    long_short_series.append(ls_return)
            
            # 计算夏普比率和波动率
            if long_short_series:
                long_short_volatility = np.std(long_short_series) * np.sqrt(252)
                excess_return = np.mean(long_short_series) * 252 - self.config.risk_free_rate
                long_short_sharpe = excess_return / long_short_volatility if long_short_volatility > 0 else 0.0
                
                # 计算最大回撤
                cumulative = np.cumprod(1 + np.array(long_short_series))
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown)
            else:
                long_short_volatility = 0.0
                long_short_sharpe = 0.0
                max_drawdown = 0.0
            
            return {
                'long_short_return': long_short_return,
                'long_short_sharpe': long_short_sharpe,
                'long_short_volatility': long_short_volatility,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            logger.error(f"计算分层回测指标失败: {e}")
            return {'long_short_return': 0.0, 'long_short_sharpe': 0.0, 
                   'long_short_volatility': 0.0, 'max_drawdown': 0.0}
    
    def _calculate_turnover(self, factor_values: List[float]) -> float:
        """计算换手率"""
        try:
            if len(factor_values) < 2:
                return 0.0
            
            # 计算因子值的变化
            changes = []
            for i in range(1, len(factor_values)):
                if factor_values[i-1] != 0:
                    change = abs(factor_values[i] - factor_values[i-1]) / abs(factor_values[i-1])
                    changes.append(change)
            
            return np.mean(changes) if changes else 0.0
            
        except Exception as e:
            logger.error(f"计算换手率失败: {e}")
            return 0.0
    
    def _calculate_auto_correlation(self, factor_values: List[float], lag: int = 1) -> float:
        """计算自相关系数"""
        try:
            if len(factor_values) <= lag:
                return 0.0
            
            x = np.array(factor_values[:-lag])
            y = np.array(factor_values[lag:])
            
            correlation = np.corrcoef(x, y)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"计算自相关失败: {e}")
            return 0.0
    
    def _calculate_monotonicity(self, factor_values: List[float], 
                              return_values: List[float]) -> float:
        """计算单调性"""
        try:
            if len(factor_values) != len(return_values) or len(factor_values) < self.config.n_quantiles:
                return 0.0
            
            # 分层
            data = list(zip(factor_values, return_values))
            data.sort(key=lambda x: x[0])
            
            group_size = len(data) // self.config.n_quantiles
            group_returns = []
            
            for i in range(self.config.n_quantiles):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < self.config.n_quantiles - 1 else len(data)
                group_return = np.mean([item[1] for item in data[start_idx:end_idx]])
                group_returns.append(group_return)
            
            # 计算单调性（相邻组收益率差的符号一致性）
            if len(group_returns) < 2:
                return 0.0
            
            diffs = [group_returns[i+1] - group_returns[i] for i in range(len(group_returns)-1)]
            
            # 计算符号一致性
            if not diffs:
                return 0.0
            
            positive_diffs = sum(1 for d in diffs if d > 0)
            negative_diffs = sum(1 for d in diffs if d < 0)
            
            monotonicity = max(positive_diffs, negative_diffs) / len(diffs)
            return monotonicity
            
        except Exception as e:
            logger.error(f"计算单调性失败: {e}")
            return 0.0
    
    def _calculate_coverage(self, factor_values: List[float]) -> float:
        """计算覆盖率"""
        try:
            # 非空值比例
            non_null_count = sum(1 for v in factor_values if not (np.isnan(v) or np.isinf(v)))
            coverage = non_null_count / len(factor_values) if factor_values else 0.0
            return coverage
            
        except Exception as e:
            logger.error(f"计算覆盖率失败: {e}")
            return 0.0
    
    def _evaluate_factor_quality(self, metrics: FactorMetrics) -> Tuple[FactorQuality, float]:
        """评估因子质量"""
        try:
            # 质量评分权重
            weights = {
                'ic_ir': 0.3,
                'rank_ic_ir': 0.2,
                'long_short_sharpe': 0.2,
                'monotonicity': 0.1,
                'coverage': 0.1,
                'ic_win_rate': 0.1
            }
            
            # 标准化各指标到[0,1]区间
            normalized_scores = {}
            
            # IC IR
            normalized_scores['ic_ir'] = min(1.0, max(0.0, abs(metrics.ic_ir) / 2.0))
            
            # Rank IC IR
            normalized_scores['rank_ic_ir'] = min(1.0, max(0.0, abs(metrics.rank_ic_ir) / 2.0))
            
            # 多空夏普比率
            normalized_scores['long_short_sharpe'] = min(1.0, max(0.0, abs(metrics.long_short_sharpe) / 3.0))
            
            # 单调性
            normalized_scores['monotonicity'] = metrics.monotonicity
            
            # 覆盖率
            normalized_scores['coverage'] = metrics.coverage
            
            # IC胜率
            normalized_scores['ic_win_rate'] = abs(metrics.ic_win_rate - 0.5) * 2  # 转换为偏离随机的程度
            
            # 计算加权得分
            quality_score = sum(weights[key] * normalized_scores[key] for key in weights)
            
            # 确定质量等级
            if quality_score >= 0.8:
                quality = FactorQuality.EXCELLENT
            elif quality_score >= 0.6:
                quality = FactorQuality.GOOD
            elif quality_score >= 0.4:
                quality = FactorQuality.AVERAGE
            elif quality_score >= 0.2:
                quality = FactorQuality.POOR
            else:
                quality = FactorQuality.INVALID
            
            return quality, quality_score
            
        except Exception as e:
            logger.error(f"评估因子质量失败: {e}")
            return FactorQuality.AVERAGE, 0.5
    
    def _calculate_factor_correlation(self, factor_ids: List[str]) -> Tuple[Optional[np.ndarray], List[str]]:
        """计算因子相关性矩阵"""
        try:
            factor_names = list(factor_ids)
            if len(factor_names) < 2:
                return None, factor_names
            
            # 构建因子数据矩阵
            min_length = min(len(self.factor_data[fid]) for fid in factor_names)
            if min_length < self.config.min_periods:
                return None, factor_names
            
            factor_matrix = np.array([
                list(self.factor_data[fid])[-min_length:] for fid in factor_names
            ])
            
            # 计算相关性矩阵
            correlation_matrix = np.corrcoef(factor_matrix)
            
            return correlation_matrix, factor_names
            
        except Exception as e:
            logger.error(f"计算因子相关性失败: {e}")
            return None, list(factor_ids)
    
    def _rank_factors(self, factor_metrics: Dict[str, FactorMetrics]) -> List[FactorRanking]:
        """因子排名"""
        try:
            # 按质量得分排序
            sorted_factors = sorted(
                factor_metrics.items(),
                key=lambda x: x[1].quality_score,
                reverse=True
            )
            
            rankings = []
            for rank, (factor_id, metrics) in enumerate(sorted_factors, 1):
                ranking = FactorRanking(
                    factor_id=factor_id,
                    rank=rank,
                    score=metrics.quality_score,
                    metrics=metrics
                )
                rankings.append(ranking)
            
            return rankings
            
        except Exception as e:
            logger.error(f"因子排名失败: {e}")
            return []
    
    def _generate_evaluation_summary(self, factor_metrics: Dict[str, FactorMetrics]) -> Dict[str, Any]:
        """生成评估摘要"""
        try:
            if not factor_metrics:
                return {}
            
            # 统计各质量等级的因子数量
            quality_counts = defaultdict(int)
            for metrics in factor_metrics.values():
                quality_counts[metrics.quality.value] += 1
            
            # 计算平均指标
            avg_metrics = {
                'avg_ic_ir': np.mean([m.ic_ir for m in factor_metrics.values()]),
                'avg_rank_ic_ir': np.mean([m.rank_ic_ir for m in factor_metrics.values()]),
                'avg_long_short_sharpe': np.mean([m.long_short_sharpe for m in factor_metrics.values()]),
                'avg_turnover': np.mean([m.turnover for m in factor_metrics.values()]),
                'avg_coverage': np.mean([m.coverage for m in factor_metrics.values()]),
                'avg_quality_score': np.mean([m.quality_score for m in factor_metrics.values()])
            }
            
            # 找出最佳因子
            best_factor = max(factor_metrics.items(), key=lambda x: x[1].quality_score)
            
            return {
                'total_factors': len(factor_metrics),
                'quality_distribution': dict(quality_counts),
                'average_metrics': avg_metrics,
                'best_factor': {
                    'factor_id': best_factor[0],
                    'quality_score': best_factor[1].quality_score,
                    'quality': best_factor[1].quality.value
                },
                'evaluation_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"生成评估摘要失败: {e}")
            return {}
    
    def get_factor_metrics(self, factor_id: str) -> Optional[FactorMetrics]:
        """获取因子指标"""
        with self.lock:
            if factor_id in self.metrics_history and self.metrics_history[factor_id]:
                return self.metrics_history[factor_id][-1]
            return None
    
    def get_metrics_history(self, factor_id: str, limit: int = None) -> List[FactorMetrics]:
        """获取指标历史"""
        with self.lock:
            history = self.metrics_history.get(factor_id, [])
            if limit:
                return history[-limit:]
            return history.copy()
    
    def get_evaluation_cache(self, cache_key: str = None) -> Optional[EvaluationResult]:
        """获取评估缓存"""
        with self.lock:
            if cache_key:
                return self.evaluation_cache.get(cache_key)
            
            # 返回最新的评估结果
            if self.evaluation_cache:
                latest_key = max(self.evaluation_cache.keys())
                return self.evaluation_cache[latest_key]
            
            return None
    
    def clear_cache(self):
        """清空缓存"""
        with self.lock:
            self.evaluation_cache.clear()
            logger.info("评估缓存已清空")
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        with self.lock:
            return {
                'total_factors_tracked': len(self.factor_data),
                'total_evaluations': len(self.evaluation_cache),
                'data_points_per_factor': {fid: len(data) for fid, data in self.factor_data.items()},
                'return_data_points': len(self.return_data),
                'evaluation_window': self.config.evaluation_window,
                'rolling_window': self.config.rolling_window
            }

# 便捷函数
def create_evaluation_config(**kwargs) -> EvaluationConfig:
    """创建评估配置
    
    Args:
        **kwargs: 配置参数
        
    Returns:
        评估配置
    """
    config = EvaluationConfig()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

def create_factor_evaluator(config: EvaluationConfig = None) -> FactorEvaluator:
    """创建因子评估器
    
    Args:
        config: 评估配置
        
    Returns:
        因子评估器实例
    """
    if config is None:
        config = create_evaluation_config()
    
    return FactorEvaluator(config)