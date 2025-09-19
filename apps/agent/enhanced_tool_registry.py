# 增强Agent工具注册表
# 扩展量化分析工具、工具链式调用和执行监控功能

import logging
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import json
import time

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import talib
except ImportError:
    talib = None

try:
    from scipy import stats, optimize
except ImportError:
    stats = None
    optimize = None

from .tool_registry import ToolRegistry, Tool, ToolMetadata, ToolParameter, ToolCategory, ParameterType, ToolResult

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ChainExecutionMode(Enum):
    """链式执行模式"""
    SEQUENTIAL = "sequential"    # 顺序执行
    PARALLEL = "parallel"       # 并行执行
    CONDITIONAL = "conditional" # 条件执行
    PIPELINE = "pipeline"       # 管道执行

@dataclass
class ToolExecution:
    """工具执行记录"""
    execution_id: str
    tool_name: str
    parameters: Dict[str, Any]
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[ToolResult] = None
    parent_chain_id: Optional[str] = None
    
    def get_duration(self) -> float:
        """获取执行时长"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_id': self.execution_id,
            'tool_name': self.tool_name,
            'parameters': self.parameters,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.get_duration(),
            'result': self.result.to_dict() if self.result else None,
            'parent_chain_id': self.parent_chain_id
        }

@dataclass
class ToolChain:
    """工具链"""
    chain_id: str
    name: str
    description: str
    tools: List[Dict[str, Any]]  # 工具配置列表
    execution_mode: ChainExecutionMode
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chain_id': self.chain_id,
            'name': self.name,
            'description': self.description,
            'tools': self.tools,
            'execution_mode': self.execution_mode.value,
            'created_at': self.created_at.isoformat()
        }

class EnhancedToolRegistry(ToolRegistry):
    """增强工具注册表
    
    扩展量化分析工具、工具链式调用和执行监控功能
    """
    
    def __init__(self):
        super().__init__()
        
        # 执行监控
        self.executions: Dict[str, ToolExecution] = {}
        self.execution_history: deque = deque(maxlen=10000)
        
        # 工具链管理
        self.tool_chains: Dict[str, ToolChain] = {}
        
        # 性能统计
        self.performance_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 错误统计
        self.error_stats: Dict[str, int] = defaultdict(int)
        
        # 注册量化分析工具
        self._register_quantitative_tools()
        
        # 注册复合工具
        self._register_composite_tools()
        
        logger.info("增强工具注册表初始化完成")
    
    def _register_quantitative_tools(self):
        """注册量化分析工具"""
        
        # 技术指标工具
        self._register_technical_indicators()
        
        # 风险指标工具
        self._register_risk_indicators()
        
        # 组合优化工具
        self._register_portfolio_optimization()
        
        # 统计分析工具
        self._register_statistical_analysis()
        
        # 因子分析工具
        self._register_factor_analysis()
    
    def _register_technical_indicators(self):
        """注册技术指标工具"""
        
        # RSI指标
        def calculate_rsi(prices: List[float], period: int = 14) -> Dict[str, Any]:
            """计算RSI指标"""
            if talib:
                rsi = talib.RSI(np.array(prices), timeperiod=period)
                return {
                    'rsi': rsi.tolist(),
                    'current_rsi': float(rsi[-1]) if not np.isnan(rsi[-1]) else None,
                    'overbought': float(rsi[-1]) > 70 if not np.isnan(rsi[-1]) else False,
                    'oversold': float(rsi[-1]) < 30 if not np.isnan(rsi[-1]) else False
                }
            else:
                # 简化RSI计算
                if len(prices) < period + 1:
                    return {'error': '数据不足'}
                
                deltas = np.diff(prices)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                avg_gain = np.mean(gains[-period:])
                avg_loss = np.mean(losses[-period:])
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                return {
                    'current_rsi': rsi,
                    'overbought': rsi > 70,
                    'oversold': rsi < 30
                }
        
        rsi_metadata = ToolMetadata(
            name="calculate_rsi",
            description="计算相对强弱指数(RSI)",
            category=ToolCategory.ANALYSIS,
            parameters=[
                ToolParameter("prices", ParameterType.LIST, "价格序列", required=True),
                ToolParameter("period", ParameterType.INTEGER, "计算周期", required=False, default=14, min_value=2, max_value=100)
            ],
            tags=["技术指标", "RSI", "超买超卖"]
        )
        self.register_tool(rsi_metadata, calculate_rsi)
        
        # MACD指标
        def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, Any]:
            """计算MACD指标"""
            if talib:
                macd, signal, histogram = talib.MACD(np.array(prices), fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
                return {
                    'macd': macd.tolist(),
                    'signal': signal.tolist(),
                    'histogram': histogram.tolist(),
                    'current_macd': float(macd[-1]) if not np.isnan(macd[-1]) else None,
                    'current_signal': float(signal[-1]) if not np.isnan(signal[-1]) else None,
                    'bullish_crossover': macd[-1] > signal[-1] and macd[-2] <= signal[-2] if len(macd) > 1 else False
                }
            else:
                # 简化MACD计算
                if len(prices) < slow_period:
                    return {'error': '数据不足'}
                
                prices_array = np.array(prices)
                ema_fast = pd.Series(prices).ewm(span=fast_period).mean() if pd else None
                ema_slow = pd.Series(prices).ewm(span=slow_period).mean() if pd else None
                
                if ema_fast is not None and ema_slow is not None:
                    macd = ema_fast - ema_slow
                    signal = macd.ewm(span=signal_period).mean()
                    histogram = macd - signal
                    
                    return {
                        'current_macd': float(macd.iloc[-1]),
                        'current_signal': float(signal.iloc[-1]),
                        'current_histogram': float(histogram.iloc[-1]),
                        'bullish_crossover': macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2] if len(macd) > 1 else False
                    }
                
                return {'error': 'pandas未安装，无法计算MACD'}
        
        macd_metadata = ToolMetadata(
            name="calculate_macd",
            description="计算MACD指标",
            category=ToolCategory.ANALYSIS,
            parameters=[
                ToolParameter("prices", ParameterType.LIST, "价格序列", required=True),
                ToolParameter("fast_period", ParameterType.INTEGER, "快线周期", required=False, default=12),
                ToolParameter("slow_period", ParameterType.INTEGER, "慢线周期", required=False, default=26),
                ToolParameter("signal_period", ParameterType.INTEGER, "信号线周期", required=False, default=9)
            ],
            tags=["技术指标", "MACD", "趋势"]
        )
        self.register_tool(macd_metadata, calculate_macd)
        
        # 布林带指标
        def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, Any]:
            """计算布林带指标"""
            if talib:
                upper, middle, lower = talib.BBANDS(np.array(prices), timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
                current_price = prices[-1]
                return {
                    'upper_band': upper.tolist(),
                    'middle_band': middle.tolist(),
                    'lower_band': lower.tolist(),
                    'current_upper': float(upper[-1]) if not np.isnan(upper[-1]) else None,
                    'current_middle': float(middle[-1]) if not np.isnan(middle[-1]) else None,
                    'current_lower': float(lower[-1]) if not np.isnan(lower[-1]) else None,
                    'position': 'above_upper' if current_price > upper[-1] else 'below_lower' if current_price < lower[-1] else 'within_bands',
                    'bandwidth': (upper[-1] - lower[-1]) / middle[-1] * 100 if not np.isnan(upper[-1]) else None
                }
            else:
                # 简化布林带计算
                if len(prices) < period:
                    return {'error': '数据不足'}
                
                prices_array = np.array(prices)
                sma = np.mean(prices_array[-period:])
                std = np.std(prices_array[-period:])
                
                upper = sma + (std_dev * std)
                lower = sma - (std_dev * std)
                current_price = prices[-1]
                
                return {
                    'current_upper': upper,
                    'current_middle': sma,
                    'current_lower': lower,
                    'position': 'above_upper' if current_price > upper else 'below_lower' if current_price < lower else 'within_bands',
                    'bandwidth': (upper - lower) / sma * 100
                }
        
        bb_metadata = ToolMetadata(
            name="calculate_bollinger_bands",
            description="计算布林带指标",
            category=ToolCategory.ANALYSIS,
            parameters=[
                ToolParameter("prices", ParameterType.LIST, "价格序列", required=True),
                ToolParameter("period", ParameterType.INTEGER, "计算周期", required=False, default=20),
                ToolParameter("std_dev", ParameterType.FLOAT, "标准差倍数", required=False, default=2.0)
            ],
            tags=["技术指标", "布林带", "波动率"]
        )
        self.register_tool(bb_metadata, calculate_bollinger_bands)
    
    def _register_risk_indicators(self):
        """注册风险指标工具"""
        
        # VaR计算
        def calculate_var(returns: List[float], confidence_level: float = 0.95, method: str = "historical") -> Dict[str, Any]:
            """计算风险价值(VaR)"""
            if not returns:
                return {'error': '收益率数据为空'}
            
            returns_array = np.array(returns)
            
            if method == "historical":
                # 历史模拟法
                var = np.percentile(returns_array, (1 - confidence_level) * 100)
            elif method == "parametric":
                # 参数法（假设正态分布）
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                if stats:
                    var = stats.norm.ppf(1 - confidence_level, mean_return, std_return)
                else:
                    # 简化计算
                    z_score = 1.645 if confidence_level == 0.95 else 2.326 if confidence_level == 0.99 else 1.282
                    var = mean_return - z_score * std_return
            else:
                return {'error': f'不支持的方法: {method}'}
            
            # 计算CVaR (条件VaR)
            cvar = np.mean(returns_array[returns_array <= var]) if np.any(returns_array <= var) else var
            
            return {
                'var': float(var),
                'cvar': float(cvar),
                'confidence_level': confidence_level,
                'method': method,
                'worst_return': float(np.min(returns_array)),
                'best_return': float(np.max(returns_array))
            }
        
        var_metadata = ToolMetadata(
            name="calculate_var",
            description="计算风险价值(VaR)和条件VaR",
            category=ToolCategory.ANALYSIS,
            parameters=[
                ToolParameter("returns", ParameterType.LIST, "收益率序列", required=True),
                ToolParameter("confidence_level", ParameterType.FLOAT, "置信水平", required=False, default=0.95, min_value=0.01, max_value=0.99),
                ToolParameter("method", ParameterType.STRING, "计算方法", required=False, default="historical", choices=["historical", "parametric"])
            ],
            tags=["风险指标", "VaR", "风险管理"]
        )
        self.register_tool(var_metadata, calculate_var)
        
        # 最大回撤计算
        def calculate_max_drawdown(prices: List[float]) -> Dict[str, Any]:
            """计算最大回撤"""
            if len(prices) < 2:
                return {'error': '价格数据不足'}
            
            prices_array = np.array(prices)
            cumulative = np.cumprod(1 + np.diff(prices_array) / prices_array[:-1])
            
            # 计算回撤
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            
            max_dd = np.min(drawdown)
            max_dd_idx = np.argmin(drawdown)
            
            # 找到回撤开始和结束点
            peak_idx = np.argmax(running_max[:max_dd_idx+1])
            
            return {
                'max_drawdown': float(max_dd),
                'max_drawdown_pct': float(max_dd * 100),
                'peak_date_idx': int(peak_idx),
                'trough_date_idx': int(max_dd_idx),
                'recovery_date_idx': None,  # 需要更复杂的逻辑来计算恢复点
                'current_drawdown': float(drawdown[-1]),
                'drawdown_series': drawdown.tolist()
            }
        
        mdd_metadata = ToolMetadata(
            name="calculate_max_drawdown",
            description="计算最大回撤",
            category=ToolCategory.ANALYSIS,
            parameters=[
                ToolParameter("prices", ParameterType.LIST, "价格序列", required=True)
            ],
            tags=["风险指标", "回撤", "风险管理"]
        )
        self.register_tool(mdd_metadata, calculate_max_drawdown)
    
    def _register_portfolio_optimization(self):
        """注册组合优化工具"""
        
        # 均值方差优化
        def optimize_portfolio(expected_returns: List[float], covariance_matrix: List[List[float]], 
                             risk_aversion: float = 1.0, constraints: Dict[str, Any] = None) -> Dict[str, Any]:
            """组合优化"""
            try:
                expected_returns = np.array(expected_returns)
                cov_matrix = np.array(covariance_matrix)
                n_assets = len(expected_returns)
                
                if cov_matrix.shape != (n_assets, n_assets):
                    return {'error': '协方差矩阵维度不匹配'}
                
                # 目标函数：最大化 expected_return - 0.5 * risk_aversion * variance
                def objective(weights):
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                    return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
                
                # 约束条件
                constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # 权重和为1
                
                # 权重边界
                bounds = [(0, 1) for _ in range(n_assets)]  # 默认多头约束
                
                if constraints:
                    if 'min_weights' in constraints:
                        min_weights = constraints['min_weights']
                        bounds = [(min_weights[i] if i < len(min_weights) else 0, 1) for i in range(n_assets)]
                    
                    if 'max_weights' in constraints:
                        max_weights = constraints['max_weights']
                        bounds = [(bounds[i][0], min(max_weights[i] if i < len(max_weights) else 1, bounds[i][1])) 
                                for i in range(n_assets)]
                
                # 初始权重（等权重）
                x0 = np.ones(n_assets) / n_assets
                
                if optimize:
                    # 使用scipy优化
                    result = optimize.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
                    
                    if result.success:
                        optimal_weights = result.x
                        portfolio_return = np.dot(optimal_weights, expected_returns)
                        portfolio_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
                        portfolio_volatility = np.sqrt(portfolio_variance)
                        
                        return {
                            'optimal_weights': optimal_weights.tolist(),
                            'expected_return': float(portfolio_return),
                            'expected_volatility': float(portfolio_volatility),
                            'sharpe_ratio': float(portfolio_return / portfolio_volatility) if portfolio_volatility > 0 else 0,
                            'optimization_success': True
                        }
                    else:
                        return {'error': f'优化失败: {result.message}'}
                else:
                    # 简化：返回等权重组合
                    equal_weights = np.ones(n_assets) / n_assets
                    portfolio_return = np.dot(equal_weights, expected_returns)
                    portfolio_variance = np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights))
                    portfolio_volatility = np.sqrt(portfolio_variance)
                    
                    return {
                        'optimal_weights': equal_weights.tolist(),
                        'expected_return': float(portfolio_return),
                        'expected_volatility': float(portfolio_volatility),
                        'sharpe_ratio': float(portfolio_return / portfolio_volatility) if portfolio_volatility > 0 else 0,
                        'optimization_success': False,
                        'note': 'scipy未安装，返回等权重组合'
                    }
                
            except Exception as e:
                return {'error': f'组合优化失败: {str(e)}'}
        
        portfolio_metadata = ToolMetadata(
            name="optimize_portfolio",
            description="均值方差组合优化",
            category=ToolCategory.MODELING,
            parameters=[
                ToolParameter("expected_returns", ParameterType.LIST, "预期收益率", required=True),
                ToolParameter("covariance_matrix", ParameterType.LIST, "协方差矩阵", required=True),
                ToolParameter("risk_aversion", ParameterType.FLOAT, "风险厌恶系数", required=False, default=1.0),
                ToolParameter("constraints", ParameterType.DICT, "约束条件", required=False)
            ],
            tags=["组合优化", "均值方差", "资产配置"]
        )
        self.register_tool(portfolio_metadata, optimize_portfolio)
    
    def _register_statistical_analysis(self):
        """注册统计分析工具"""
        
        # 相关性分析
        def calculate_correlation(data1: List[float], data2: List[float], method: str = "pearson") -> Dict[str, Any]:
            """计算相关性"""
            if len(data1) != len(data2):
                return {'error': '数据长度不匹配'}
            
            if len(data1) < 2:
                return {'error': '数据点不足'}
            
            try:
                if stats:
                    if method == "pearson":
                        corr, p_value = stats.pearsonr(data1, data2)
                    elif method == "spearman":
                        corr, p_value = stats.spearmanr(data1, data2)
                    else:
                        return {'error': f'不支持的方法: {method}'}
                else:
                    # 简化计算
                    corr = np.corrcoef(data1, data2)[0, 1]
                    p_value = None
                
                return {
                    'correlation': float(corr) if not np.isnan(corr) else 0.0,
                    'p_value': float(p_value) if p_value is not None else None,
                    'method': method,
                    'sample_size': len(data1),
                    'significant': p_value < 0.05 if p_value is not None else None
                }
                
            except Exception as e:
                return {'error': f'相关性计算失败: {str(e)}'}
        
        corr_metadata = ToolMetadata(
            name="calculate_correlation",
            description="计算两个序列的相关性",
            category=ToolCategory.ANALYSIS,
            parameters=[
                ToolParameter("data1", ParameterType.LIST, "数据序列1", required=True),
                ToolParameter("data2", ParameterType.LIST, "数据序列2", required=True),
                ToolParameter("method", ParameterType.STRING, "计算方法", required=False, default="pearson", choices=["pearson", "spearman"])
            ],
            tags=["统计分析", "相关性", "皮尔逊", "斯皮尔曼"]
        )
        self.register_tool(corr_metadata, calculate_correlation)
    
    def _register_factor_analysis(self):
        """注册因子分析工具"""
        
        # IC分析
        def calculate_ic(factor_values: List[float], returns: List[float], method: str = "pearson") -> Dict[str, Any]:
            """计算信息系数(IC)"""
            if len(factor_values) != len(returns):
                return {'error': '因子值和收益率长度不匹配'}
            
            if len(factor_values) < 10:
                return {'error': '数据点不足，至少需要10个观测值'}
            
            try:
                if stats:
                    if method == "pearson":
                        ic, p_value = stats.pearsonr(factor_values, returns)
                    elif method == "spearman":
                        ic, p_value = stats.spearmanr(factor_values, returns)
                    else:
                        return {'error': f'不支持的方法: {method}'}
                else:
                    ic = np.corrcoef(factor_values, returns)[0, 1]
                    p_value = None
                
                # 计算IC的t统计量
                n = len(factor_values)
                if not np.isnan(ic) and abs(ic) < 1:
                    t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                else:
                    t_stat = None
                
                return {
                    'ic': float(ic) if not np.isnan(ic) else 0.0,
                    'abs_ic': float(abs(ic)) if not np.isnan(ic) else 0.0,
                    'p_value': float(p_value) if p_value is not None else None,
                    't_statistic': float(t_stat) if t_stat is not None else None,
                    'method': method,
                    'sample_size': n,
                    'significant': p_value < 0.05 if p_value is not None else None,
                    'ic_quality': 'excellent' if abs(ic) > 0.1 else 'good' if abs(ic) > 0.05 else 'average' if abs(ic) > 0.02 else 'poor'
                }
                
            except Exception as e:
                return {'error': f'IC计算失败: {str(e)}'}
        
        ic_metadata = ToolMetadata(
            name="calculate_ic",
            description="计算因子信息系数(IC)",
            category=ToolCategory.ANALYSIS,
            parameters=[
                ToolParameter("factor_values", ParameterType.LIST, "因子值序列", required=True),
                ToolParameter("returns", ParameterType.LIST, "收益率序列", required=True),
                ToolParameter("method", ParameterType.STRING, "计算方法", required=False, default="pearson", choices=["pearson", "spearman"])
            ],
            tags=["因子分析", "IC", "信息系数"]
        )
        self.register_tool(ic_metadata, calculate_ic)
    
    def _register_composite_tools(self):
        """注册复合工具"""
        
        # 技术分析综合工具
        async def technical_analysis_suite(prices: List[float], volumes: List[float] = None) -> Dict[str, Any]:
            """技术分析套件"""
            results = {}
            
            try:
                # RSI
                rsi_tool = self.get_tool("calculate_rsi")
                if rsi_tool:
                    rsi_result = await rsi_tool.execute({'prices': prices})
                    results['rsi'] = rsi_result.data if rsi_result.success else {'error': rsi_result.error}
                
                # MACD
                macd_tool = self.get_tool("calculate_macd")
                if macd_tool:
                    macd_result = await macd_tool.execute({'prices': prices})
                    results['macd'] = macd_result.data if macd_result.success else {'error': macd_result.error}
                
                # 布林带
                bb_tool = self.get_tool("calculate_bollinger_bands")
                if bb_tool:
                    bb_result = await bb_tool.execute({'prices': prices})
                    results['bollinger_bands'] = bb_result.data if bb_result.success else {'error': bb_result.error}
                
                # 综合信号
                signals = []
                if 'rsi' in results and 'current_rsi' in results['rsi']:
                    rsi_val = results['rsi']['current_rsi']
                    if rsi_val > 70:
                        signals.append('RSI超买')
                    elif rsi_val < 30:
                        signals.append('RSI超卖')
                
                if 'macd' in results and 'bullish_crossover' in results['macd']:
                    if results['macd']['bullish_crossover']:
                        signals.append('MACD金叉')
                
                if 'bollinger_bands' in results and 'position' in results['bollinger_bands']:
                    position = results['bollinger_bands']['position']
                    if position == 'above_upper':
                        signals.append('突破布林上轨')
                    elif position == 'below_lower':
                        signals.append('跌破布林下轨')
                
                results['综合信号'] = signals
                results['信号数量'] = len(signals)
                
                return results
                
            except Exception as e:
                return {'error': f'技术分析套件执行失败: {str(e)}'}
        
        tech_suite_metadata = ToolMetadata(
            name="technical_analysis_suite",
            description="技术分析综合套件",
            category=ToolCategory.ANALYSIS,
            parameters=[
                ToolParameter("prices", ParameterType.LIST, "价格序列", required=True),
                ToolParameter("volumes", ParameterType.LIST, "成交量序列", required=False)
            ],
            tags=["技术分析", "综合工具", "套件"],
            async_execution=True
        )
        self.register_tool(tech_suite_metadata, technical_analysis_suite)
    
    async def execute_tool_chain(self, chain_id: str, initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行工具链"""
        if chain_id not in self.tool_chains:
            return {'error': f'工具链不存在: {chain_id}'}
        
        chain = self.tool_chains[chain_id]
        execution_id = f"chain_{chain_id}_{int(time.time())}"
        
        try:
            if chain.execution_mode == ChainExecutionMode.SEQUENTIAL:
                return await self._execute_sequential_chain(chain, execution_id, initial_data)
            elif chain.execution_mode == ChainExecutionMode.PARALLEL:
                return await self._execute_parallel_chain(chain, execution_id, initial_data)
            elif chain.execution_mode == ChainExecutionMode.PIPELINE:
                return await self._execute_pipeline_chain(chain, execution_id, initial_data)
            else:
                return {'error': f'不支持的执行模式: {chain.execution_mode}'}
                
        except Exception as e:
            logger.error(f"工具链执行失败: {chain_id} - {e}")
            return {'error': f'工具链执行失败: {str(e)}'}
    
    async def _execute_sequential_chain(self, chain: ToolChain, execution_id: str, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """顺序执行工具链"""
        results = {'chain_id': chain.chain_id, 'execution_id': execution_id, 'results': []}
        current_data = initial_data or {}
        
        for i, tool_config in enumerate(chain.tools):
            tool_name = tool_config['name']
            tool_params = tool_config.get('parameters', {})
            
            # 合并当前数据和工具参数
            merged_params = {**current_data, **tool_params}
            
            # 执行工具
            tool_result = await self.execute_tool_with_monitoring(tool_name, merged_params, execution_id)
            results['results'].append({
                'step': i + 1,
                'tool_name': tool_name,
                'result': tool_result
            })
            
            # 如果工具执行失败，停止链式执行
            if not tool_result.get('success', False):
                results['status'] = 'failed'
                results['failed_at_step'] = i + 1
                break
            
            # 更新当前数据（用于下一个工具）
            if tool_result.get('data'):
                current_data.update(tool_result['data'])
        
        if 'status' not in results:
            results['status'] = 'completed'
        
        return results
    
    async def _execute_parallel_chain(self, chain: ToolChain, execution_id: str, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """并行执行工具链"""
        results = {'chain_id': chain.chain_id, 'execution_id': execution_id, 'results': []}
        
        # 创建并行任务
        tasks = []
        for i, tool_config in enumerate(chain.tools):
            tool_name = tool_config['name']
            tool_params = tool_config.get('parameters', {})
            merged_params = {**(initial_data or {}), **tool_params}
            
            task = asyncio.create_task(
                self.execute_tool_with_monitoring(tool_name, merged_params, execution_id)
            )
            tasks.append((i + 1, tool_name, task))
        
        # 等待所有任务完成
        for step, tool_name, task in tasks:
            try:
                tool_result = await task
                results['results'].append({
                    'step': step,
                    'tool_name': tool_name,
                    'result': tool_result
                })
            except Exception as e:
                results['results'].append({
                    'step': step,
                    'tool_name': tool_name,
                    'result': {'success': False, 'error': str(e)}
                })
        
        # 检查是否所有工具都成功执行
        all_success = all(r['result'].get('success', False) for r in results['results'])
        results['status'] = 'completed' if all_success else 'partial_failure'
        
        return results
    
    async def _execute_pipeline_chain(self, chain: ToolChain, execution_id: str, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """管道执行工具链（数据流式传递）"""
        results = {'chain_id': chain.chain_id, 'execution_id': execution_id, 'results': []}
        pipeline_data = initial_data or {}
        
        for i, tool_config in enumerate(chain.tools):
            tool_name = tool_config['name']
            tool_params = tool_config.get('parameters', {})
            
            # 从管道数据中提取输入
            input_mapping = tool_config.get('input_mapping', {})
            actual_params = {}
            
            for param_name, source_key in input_mapping.items():
                if source_key in pipeline_data:
                    actual_params[param_name] = pipeline_data[source_key]
            
            # 合并参数
            merged_params = {**actual_params, **tool_params}
            
            # 执行工具
            tool_result = await self.execute_tool_with_monitoring(tool_name, merged_params, execution_id)
            results['results'].append({
                'step': i + 1,
                'tool_name': tool_name,
                'result': tool_result
            })
            
            # 如果工具执行失败，停止管道
            if not tool_result.get('success', False):
                results['status'] = 'failed'
                results['failed_at_step'] = i + 1
                break
            
            # 更新管道数据
            output_mapping = tool_config.get('output_mapping', {})
            if tool_result.get('data'):
                for output_key, target_key in output_mapping.items():
                    if output_key in tool_result['data']:
                        pipeline_data[target_key] = tool_result['data'][output_key]
                
                # 默认将所有输出添加到管道数据
                pipeline_data.update(tool_result['data'])
        
        if 'status' not in results:
            results['status'] = 'completed'
            results['final_data'] = pipeline_data
        
        return results
    
    async def execute_tool_with_monitoring(self, tool_name: str, parameters: Dict[str, Any], 
                                         parent_chain_id: str = None) -> Dict[str, Any]:
        """带监控的工具执行"""
        execution_id = f"{tool_name}_{int(time.time() * 1000)}"
        
        # 创建执行记录
        execution = ToolExecution(
            execution_id=execution_id,
            tool_name=tool_name,
            parameters=parameters,
            status=ExecutionStatus.PENDING,
            start_time=datetime.now(),
            parent_chain_id=parent_chain_id
        )
        
        self.executions[execution_id] = execution
        
        try:
            # 更新状态为运行中
            execution.status = ExecutionStatus.RUNNING
            
            # 获取工具
            tool = self.get_tool(tool_name)
            if not tool:
                raise ValueError(f"工具不存在: {tool_name}")
            
            # 执行工具
            result = await tool.execute(parameters)
            
            # 更新执行记录
            execution.status = ExecutionStatus.COMPLETED if result.success else ExecutionStatus.FAILED
            execution.end_time = datetime.now()
            execution.result = result
            
            # 更新性能统计
            self._update_performance_stats(tool_name, result)
            
            # 记录到历史
            self.execution_history.append(execution)
            
            return result.to_dict()
            
        except Exception as e:
            # 更新执行记录
            execution.status = ExecutionStatus.FAILED
            execution.end_time = datetime.now()
            execution.result = ToolResult(success=False, error=str(e), error_type=type(e).__name__)
            
            # 更新错误统计
            self.error_stats[tool_name] += 1
            
            # 记录到历史
            self.execution_history.append(execution)
            
            logger.error(f"工具执行失败: {tool_name} - {e}")
            return {'success': False, 'error': str(e), 'error_type': type(e).__name__}
    
    def _update_performance_stats(self, tool_name: str, result: ToolResult):
        """更新性能统计"""
        if tool_name not in self.performance_stats:
            self.performance_stats[tool_name] = {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'total_execution_time': 0.0,
                'avg_execution_time': 0.0,
                'min_execution_time': float('inf'),
                'max_execution_time': 0.0
            }
        
        stats = self.performance_stats[tool_name]
        stats['total_executions'] += 1
        
        if result.success:
            stats['successful_executions'] += 1
        else:
            stats['failed_executions'] += 1
        
        stats['total_execution_time'] += result.execution_time
        stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_executions']
        stats['min_execution_time'] = min(stats['min_execution_time'], result.execution_time)
        stats['max_execution_time'] = max(stats['max_execution_time'], result.execution_time)
    
    def create_tool_chain(self, name: str, description: str, tools: List[Dict[str, Any]], 
                         execution_mode: ChainExecutionMode = ChainExecutionMode.SEQUENTIAL) -> str:
        """创建工具链"""
        chain_id = f"chain_{len(self.tool_chains)}_{int(time.time())}"
        
        chain = ToolChain(
            chain_id=chain_id,
            name=name,
            description=description,
            tools=tools,
            execution_mode=execution_mode,
            created_at=datetime.now()
        )
        
        self.tool_chains[chain_id] = chain
        logger.info(f"工具链创建成功: {name} ({chain_id})")
        
        return chain_id
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """获取执行状态"""
        if execution_id in self.executions:
            return self.executions[execution_id].to_dict()
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'tool_performance': dict(self.performance_stats),
            'error_statistics': dict(self.error_stats),
            'total_executions': len(self.execution_history),
            'active_executions': len([e for e in self.executions.values() if e.status == ExecutionStatus.RUNNING]),
            'tool_chains': len(self.tool_chains)
        }

# 便捷函数
def create_enhanced_tool_registry() -> EnhancedToolRegistry:
    """创建增强工具注册表"""
    return EnhancedToolRegistry()