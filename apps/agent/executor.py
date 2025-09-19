# 任务执行器模块
# 实现任务的异步执行、状态管理和结果处理功能

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import traceback
import json

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """执行状态"""
    PENDING = "pending"        # 等待执行
    RUNNING = "running"        # 正在执行
    COMPLETED = "completed"    # 执行完成
    FAILED = "failed"          # 执行失败
    CANCELLED = "cancelled"    # 已取消
    TIMEOUT = "timeout"        # 超时
    RETRYING = "retrying"      # 重试中

@dataclass
class ExecutionContext:
    """执行上下文"""
    task_id: str
    task_graph_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # 执行环境
    environment: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # 配置
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 300.0
    
    # 回调函数
    on_start: Optional[Callable] = None
    on_complete: Optional[Callable] = None
    on_error: Optional[Callable] = None
    on_progress: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'task_graph_id': self.task_graph_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'environment': self.environment,
            'variables': self.variables,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'timeout': self.timeout
        }

@dataclass
class ExecutionResult:
    """执行结果"""
    task_id: str
    status: ExecutionStatus
    
    # 结果数据
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None
    
    # 执行统计
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    retry_count: int = 0
    
    # 资源使用
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    
    @property
    def is_success(self) -> bool:
        """是否执行成功"""
        return self.status == ExecutionStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """是否执行失败"""
        return self.status in [ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT, ExecutionStatus.CANCELLED]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'data': self.data,
            'error': self.error,
            'error_type': self.error_type,
            'traceback': self.traceback,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'retry_count': self.retry_count,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'is_success': self.is_success,
            'is_failed': self.is_failed
        }

class TaskExecutor:
    """任务执行器基类
    
    定义任务执行的标准接口
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def execute(self, task_node, context: ExecutionContext) -> ExecutionResult:
        """执行任务
        
        Args:
            task_node: 任务节点
            context: 执行上下文
            
        Returns:
            执行结果
        """
        raise NotImplementedError("子类必须实现execute方法")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数
        
        Args:
            parameters: 任务参数
            
        Returns:
            是否有效
        """
        return True
    
    def estimate_duration(self, parameters: Dict[str, Any]) -> float:
        """估算执行时间
        
        Args:
            parameters: 任务参数
            
        Returns:
            预估时间（秒）
        """
        return 60.0  # 默认1分钟

class DataFetchExecutor(TaskExecutor):
    """数据获取执行器"""
    
    def __init__(self):
        super().__init__("DataFetchExecutor")
    
    async def execute(self, task_node, context: ExecutionContext) -> ExecutionResult:
        """执行数据获取任务"""
        result = ExecutionResult(
            task_id=task_node.id,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            parameters = task_node.parameters
            
            # 模拟数据获取
            await asyncio.sleep(1)  # 模拟网络请求
            
            # 根据参数类型获取不同数据
            if 'symbols' in parameters:
                # 获取股票数据
                symbols = parameters['symbols']
                data = {
                    'symbols': symbols,
                    'data': {
                        symbol: {
                            'price': 100.0 + hash(symbol) % 50,
                            'change': (hash(symbol) % 10 - 5) / 10,
                            'volume': hash(symbol) % 1000000
                        } for symbol in symbols
                    },
                    'timestamp': datetime.now().isoformat()
                }
            elif 'market' in parameters:
                # 获取市场数据
                data = {
                    'market': parameters['market'],
                    'stocks': [f"stock_{i}" for i in range(100)],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # 通用数据获取
                data = {
                    'message': '数据获取完成',
                    'parameters': parameters,
                    'timestamp': datetime.now().isoformat()
                }
            
            result.status = ExecutionStatus.COMPLETED
            result.data = data
            
            self.logger.info(f"数据获取完成: {task_node.id}")
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            result.error_type = type(e).__name__
            result.traceback = traceback.format_exc()
            
            self.logger.error(f"数据获取失败: {task_node.id} - {e}")
        
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
        
        return result

class DataAnalysisExecutor(TaskExecutor):
    """数据分析执行器"""
    
    def __init__(self):
        super().__init__("DataAnalysisExecutor")
    
    async def execute(self, task_node, context: ExecutionContext) -> ExecutionResult:
        """执行数据分析任务"""
        result = ExecutionResult(
            task_id=task_node.id,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            parameters = task_node.parameters
            
            # 模拟数据分析
            await asyncio.sleep(2)  # 模拟计算时间
            
            # 根据分析类型生成不同结果
            analysis_type = parameters.get('analysis_type', 'general')
            
            if analysis_type == 'trend_analysis':
                data = {
                    'trend': 'upward',
                    'strength': 0.75,
                    'support_level': 3000,
                    'resistance_level': 3200,
                    'recommendation': 'hold'
                }
            elif analysis_type == 'technical_analysis':
                data = {
                    'indicators': {
                        'MA5': 100.5,
                        'MA20': 98.2,
                        'RSI': 65.3,
                        'MACD': 1.2
                    },
                    'signals': ['golden_cross', 'volume_breakout'],
                    'score': 7.5
                }
            elif analysis_type == 'fundamental_analysis':
                data = {
                    'metrics': {
                        'PE': 15.2,
                        'PB': 1.8,
                        'ROE': 0.12,
                        'debt_ratio': 0.35
                    },
                    'industry_rank': 15,
                    'valuation': 'reasonable'
                }
            else:
                data = {
                    'analysis_result': '分析完成',
                    'score': 8.0,
                    'confidence': 0.85,
                    'parameters': parameters
                }
            
            result.status = ExecutionStatus.COMPLETED
            result.data = data
            
            self.logger.info(f"数据分析完成: {task_node.id}")
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            result.error_type = type(e).__name__
            result.traceback = traceback.format_exc()
            
            self.logger.error(f"数据分析失败: {task_node.id} - {e}")
        
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
        
        return result

class StrategyExecutionExecutor(TaskExecutor):
    """策略执行执行器"""
    
    def __init__(self):
        super().__init__("StrategyExecutionExecutor")
    
    async def execute(self, task_node, context: ExecutionContext) -> ExecutionResult:
        """执行策略执行任务"""
        result = ExecutionResult(
            task_id=task_node.id,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            parameters = task_node.parameters
            
            # 模拟策略执行
            await asyncio.sleep(3)  # 模拟回测时间
            
            strategy_type = parameters.get('strategy_type', 'multi_factor')
            
            data = {
                'strategy_type': strategy_type,
                'backtest_period': parameters.get('start_date', '2020-01-01') + ' to ' + parameters.get('end_date', '2023-12-31'),
                'performance': {
                    'total_return': 0.25,
                    'annual_return': 0.08,
                    'volatility': 0.15,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.12,
                    'calmar_ratio': 0.67
                },
                'trades': {
                    'total_trades': 156,
                    'win_rate': 0.58,
                    'avg_holding_period': 15.2
                },
                'positions': [
                    {'symbol': '000001.SZ', 'weight': 0.05, 'return': 0.12},
                    {'symbol': '000002.SZ', 'weight': 0.04, 'return': 0.08},
                    {'symbol': '600000.SH', 'weight': 0.06, 'return': 0.15}
                ]
            }
            
            result.status = ExecutionStatus.COMPLETED
            result.data = data
            
            self.logger.info(f"策略执行完成: {task_node.id}")
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            result.error_type = type(e).__name__
            result.traceback = traceback.format_exc()
            
            self.logger.error(f"策略执行失败: {task_node.id} - {e}")
        
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
        
        return result

class ReportGenerationExecutor(TaskExecutor):
    """报告生成执行器"""
    
    def __init__(self):
        super().__init__("ReportGenerationExecutor")
    
    async def execute(self, task_node, context: ExecutionContext) -> ExecutionResult:
        """执行报告生成任务"""
        result = ExecutionResult(
            task_id=task_node.id,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            parameters = task_node.parameters
            
            # 模拟报告生成
            await asyncio.sleep(1.5)  # 模拟生成时间
            
            report_type = parameters.get('report_type', 'general')
            
            data = {
                'report_type': report_type,
                'title': f"{report_type.replace('_', ' ').title()} 报告",
                'generated_at': datetime.now().isoformat(),
                'sections': [
                    {
                        'title': '执行摘要',
                        'content': '本报告基于最新的市场数据和分析结果生成。'
                    },
                    {
                        'title': '主要发现',
                        'content': '通过深入分析，我们发现了以下关键趋势和机会。'
                    },
                    {
                        'title': '投资建议',
                        'content': '基于分析结果，我们提出以下投资建议。'
                    }
                ],
                'charts': parameters.get('include_charts', False),
                'format': 'html',
                'file_path': f'/reports/{report_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            }
            
            result.status = ExecutionStatus.COMPLETED
            result.data = data
            
            self.logger.info(f"报告生成完成: {task_node.id}")
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            result.error_type = type(e).__name__
            result.traceback = traceback.format_exc()
            
            self.logger.error(f"报告生成失败: {task_node.id} - {e}")
        
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
        
        return result

class Executor:
    """主执行器
    
    负责任务的调度、执行和状态管理
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # 执行器注册表
        self.executors = {
            'DATA_FETCH': DataFetchExecutor(),
            'DATA_ANALYSIS': DataAnalysisExecutor(),
            'STRATEGY_EXECUTION': StrategyExecutionExecutor(),
            'REPORT_GENERATION': ReportGenerationExecutor()
        }
        
        # 执行状态跟踪
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, ExecutionResult] = {}
        self.task_contexts: Dict[str, ExecutionContext] = {}
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # 线程池（用于CPU密集型任务）
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # 事件回调
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info(f"执行器初始化完成: 最大并发任务数={max_concurrent_tasks}")
    
    def register_executor(self, task_type: str, executor: TaskExecutor):
        """注册任务执行器
        
        Args:
            task_type: 任务类型
            executor: 执行器实例
        """
        self.executors[task_type] = executor
        logger.debug(f"注册执行器: {task_type} -> {executor.name}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """添加事件处理器
        
        Args:
            event_type: 事件类型 (task_start, task_complete, task_error)
            handler: 处理函数
        """
        self.event_handlers[event_type].append(handler)
    
    async def execute_task(self, task_node, context: ExecutionContext = None) -> ExecutionResult:
        """执行单个任务
        
        Args:
            task_node: 任务节点
            context: 执行上下文
            
        Returns:
            执行结果
        """
        if context is None:
            context = ExecutionContext(
                task_id=task_node.id,
                task_graph_id="standalone"
            )
        
        # 获取执行器
        task_type = task_node.type.name if hasattr(task_node.type, 'name') else str(task_node.type)
        executor = self.executors.get(task_type)
        
        if not executor:
            result = ExecutionResult(
                task_id=task_node.id,
                status=ExecutionStatus.FAILED,
                error=f"未找到任务类型 {task_type} 的执行器",
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            return result
        
        # 验证参数
        if not executor.validate_parameters(task_node.parameters):
            result = ExecutionResult(
                task_id=task_node.id,
                status=ExecutionStatus.FAILED,
                error="任务参数验证失败",
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            return result
        
        # 执行任务
        async with self.semaphore:
            try:
                # 触发开始事件
                await self._trigger_event('task_start', task_node, context)
                
                # 执行任务（带重试机制）
                result = await self._execute_with_retry(executor, task_node, context)
                
                # 保存结果
                self.task_results[task_node.id] = result
                
                # 触发完成事件
                if result.is_success:
                    await self._trigger_event('task_complete', task_node, context, result)
                else:
                    await self._trigger_event('task_error', task_node, context, result)
                
                return result
                
            except Exception as e:
                result = ExecutionResult(
                    task_id=task_node.id,
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=traceback.format_exc(),
                    start_time=datetime.now(),
                    end_time=datetime.now()
                )
                
                self.task_results[task_node.id] = result
                await self._trigger_event('task_error', task_node, context, result)
                
                return result
    
    async def execute_task_graph(self, task_graph, context: ExecutionContext = None) -> Dict[str, ExecutionResult]:
        """执行任务图
        
        Args:
            task_graph: 任务图
            context: 执行上下文
            
        Returns:
            所有任务的执行结果
        """
        if context is None:
            context = ExecutionContext(
                task_id="graph_execution",
                task_graph_id=task_graph.id
            )
        
        logger.info(f"开始执行任务图: {task_graph.name} ({len(task_graph.nodes)} 个任务)")
        
        # 获取执行顺序
        execution_order = task_graph.get_execution_order()
        
        results = {}
        
        # 按层级执行任务
        for level, task_ids in enumerate(execution_order):
            logger.info(f"执行第 {level + 1} 层任务: {task_ids}")
            
            # 并行执行同一层级的任务
            level_tasks = []
            for task_id in task_ids:
                if task_id in task_graph.nodes:
                    task_node = task_graph.nodes[task_id]
                    task_context = ExecutionContext(
                        task_id=task_id,
                        task_graph_id=task_graph.id,
                        user_id=context.user_id,
                        session_id=context.session_id,
                        environment=context.environment.copy(),
                        variables=context.variables.copy()
                    )
                    
                    # 将前置任务的结果添加到上下文
                    for dep_id in task_node.dependencies:
                        if dep_id in results:
                            task_context.variables[f"result_{dep_id}"] = results[dep_id].data
                    
                    level_tasks.append(self.execute_task(task_node, task_context))
            
            # 等待当前层级所有任务完成
            if level_tasks:
                level_results = await asyncio.gather(*level_tasks, return_exceptions=True)
                
                for i, result in enumerate(level_results):
                    task_id = task_ids[i]
                    if isinstance(result, Exception):
                        # 处理异常
                        error_result = ExecutionResult(
                            task_id=task_id,
                            status=ExecutionStatus.FAILED,
                            error=str(result),
                            error_type=type(result).__name__,
                            traceback=traceback.format_exc(),
                            start_time=datetime.now(),
                            end_time=datetime.now()
                        )
                        results[task_id] = error_result
                    else:
                        results[task_id] = result
                    
                    # 更新任务图中的任务状态
                    if task_id in task_graph.nodes:
                        task_node = task_graph.nodes[task_id]
                        task_node.status = results[task_id].status.value
                        task_node.result = results[task_id].data
                        task_node.error = results[task_id].error
                        task_node.start_time = results[task_id].start_time
                        task_node.end_time = results[task_id].end_time
        
        # 统计执行结果
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results.values() if r.is_success)
        failed_tasks = sum(1 for r in results.values() if r.is_failed)
        
        logger.info(f"任务图执行完成: {task_graph.name} - 成功: {successful_tasks}, 失败: {failed_tasks}, 总计: {total_tasks}")
        
        return results
    
    async def _execute_with_retry(self, executor: TaskExecutor, task_node, context: ExecutionContext) -> ExecutionResult:
        """带重试机制的任务执行
        
        Args:
            executor: 执行器
            task_node: 任务节点
            context: 执行上下文
            
        Returns:
            执行结果
        """
        max_retries = context.max_retries
        retry_delay = context.retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                # 设置超时
                timeout = context.timeout or task_node.timeout
                result = await asyncio.wait_for(
                    executor.execute(task_node, context),
                    timeout=timeout
                )
                
                result.retry_count = attempt
                
                if result.is_success:
                    return result
                elif attempt < max_retries:
                    # 重试
                    logger.warning(f"任务执行失败，准备重试: {task_node.id} (尝试 {attempt + 1}/{max_retries + 1})")
                    result.status = ExecutionStatus.RETRYING
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # 指数退避
                else:
                    # 最后一次尝试失败
                    return result
                    
            except asyncio.TimeoutError:
                result = ExecutionResult(
                    task_id=task_node.id,
                    status=ExecutionStatus.TIMEOUT,
                    error=f"任务执行超时 ({timeout}s)",
                    retry_count=attempt,
                    start_time=datetime.now(),
                    end_time=datetime.now()
                )
                
                if attempt < max_retries:
                    logger.warning(f"任务执行超时，准备重试: {task_node.id} (尝试 {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    return result
            
            except Exception as e:
                result = ExecutionResult(
                    task_id=task_node.id,
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=traceback.format_exc(),
                    retry_count=attempt,
                    start_time=datetime.now(),
                    end_time=datetime.now()
                )
                
                if attempt < max_retries:
                    logger.warning(f"任务执行异常，准备重试: {task_node.id} - {e} (尝试 {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    return result
        
        # 不应该到达这里
        return ExecutionResult(
            task_id=task_node.id,
            status=ExecutionStatus.FAILED,
            error="未知错误",
            start_time=datetime.now(),
            end_time=datetime.now()
        )
    
    async def _trigger_event(self, event_type: str, task_node, context: ExecutionContext, result: ExecutionResult = None):
        """触发事件
        
        Args:
            event_type: 事件类型
            task_node: 任务节点
            context: 执行上下文
            result: 执行结果（可选）
        """
        handlers = self.event_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(task_node, context, result)
                else:
                    handler(task_node, context, result)
            except Exception as e:
                logger.error(f"事件处理器执行失败: {event_type} - {e}")
    
    def get_task_result(self, task_id: str) -> Optional[ExecutionResult]:
        """获取任务执行结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            执行结果
        """
        return self.task_results.get(task_id)
    
    def get_running_tasks(self) -> List[str]:
        """获取正在运行的任务列表
        
        Returns:
            任务ID列表
        """
        return list(self.running_tasks.keys())
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            del self.running_tasks[task_id]
            
            # 更新结果
            self.task_results[task_id] = ExecutionResult(
                task_id=task_id,
                status=ExecutionStatus.CANCELLED,
                error="任务被用户取消",
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            
            logger.info(f"任务已取消: {task_id}")
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息
        
        Returns:
            统计信息
        """
        total_tasks = len(self.task_results)
        if total_tasks == 0:
            return {
                'total_tasks': 0,
                'successful_tasks': 0,
                'failed_tasks': 0,
                'running_tasks': len(self.running_tasks),
                'success_rate': 0.0,
                'avg_duration': 0.0
            }
        
        successful_tasks = sum(1 for r in self.task_results.values() if r.is_success)
        failed_tasks = sum(1 for r in self.task_results.values() if r.is_failed)
        total_duration = sum(r.duration for r in self.task_results.values())
        
        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'running_tasks': len(self.running_tasks),
            'success_rate': successful_tasks / total_tasks,
            'avg_duration': total_duration / total_tasks,
            'total_duration': total_duration
        }
    
    async def shutdown(self):
        """关闭执行器"""
        # 取消所有运行中的任务
        for task_id in list(self.running_tasks.keys()):
            self.cancel_task(task_id)
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        logger.info("执行器已关闭")

# 便捷函数
def create_executor(max_concurrent_tasks: int = 10) -> Executor:
    """创建执行器
    
    Args:
        max_concurrent_tasks: 最大并发任务数
        
    Returns:
        执行器实例
    """
    return Executor(max_concurrent_tasks)

def create_execution_context(task_id: str, task_graph_id: str, **kwargs) -> ExecutionContext:
    """创建执行上下文
    
    Args:
        task_id: 任务ID
        task_graph_id: 任务图ID
        **kwargs: 其他参数
        
    Returns:
        执行上下文实例
    """
    return ExecutionContext(task_id=task_id, task_graph_id=task_graph_id, **kwargs)