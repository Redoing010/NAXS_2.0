# 管道管理器 - 实现数据处理管道的编排和执行

import logging
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import uuid
import json
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time

logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    """管道状态"""
    DRAFT = "draft"            # 草稿
    READY = "ready"            # 就绪
    RUNNING = "running"        # 运行中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"    # 已取消
    PAUSED = "paused"          # 暂停

class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"        # 等待中
    RUNNING = "running"        # 运行中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败
    SKIPPED = "skipped"        # 跳过
    CANCELLED = "cancelled"    # 已取消

class StepType(Enum):
    """步骤类型"""
    EXTRACT = "extract"        # 数据提取
    TRANSFORM = "transform"    # 数据转换
    LOAD = "load"              # 数据加载
    VALIDATE = "validate"      # 数据验证
    AGGREGATE = "aggregate"    # 数据聚合
    FILTER = "filter"          # 数据过滤
    CUSTOM = "custom"          # 自定义

class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = "sequential"  # 顺序执行
    PARALLEL = "parallel"      # 并行执行
    STREAMING = "streaming"    # 流式执行

@dataclass
class StepResult:
    """步骤执行结果"""
    step_id: str
    status: StepStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    input_data: Any = None
    output_data: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'step_id': self.step_id,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'error': self.error,
            'metrics': self.metrics,
            'metadata': self.metadata
        }

@dataclass
class PipelineConfig:
    """管道配置"""
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_parallel_steps: int = 4
    timeout: Optional[float] = None
    retry_failed_steps: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 5  # 每5个步骤保存一次检查点
    output_validation: bool = True
    cache_intermediate_results: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'execution_mode': self.execution_mode.value,
            'max_parallel_steps': self.max_parallel_steps,
            'timeout': self.timeout,
            'retry_failed_steps': self.retry_failed_steps,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'checkpoint_enabled': self.checkpoint_enabled,
            'checkpoint_interval': self.checkpoint_interval,
            'output_validation': self.output_validation,
            'cache_intermediate_results': self.cache_intermediate_results
        }

@dataclass
class PipelineStep:
    """管道步骤"""
    id: str
    name: str
    step_type: StepType
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[Callable[[Any], bool]] = None
    condition_expression: str = ""
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0
    retry_delay: float = 1.0
    cache_enabled: bool = True
    validation_func: Optional[Callable[[Any], bool]] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 运行时状态
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    
    def should_execute(self, input_data: Any, context: Dict[str, Any]) -> bool:
        """判断是否应该执行此步骤
        
        Args:
            input_data: 输入数据
            context: 执行上下文
            
        Returns:
            是否应该执行
        """
        if self.condition:
            return self.condition(input_data)
        elif self.condition_expression:
            return self._evaluate_expression(input_data, context)
        else:
            return True
    
    def _evaluate_expression(self, input_data: Any, context: Dict[str, Any]) -> bool:
        """评估条件表达式"""
        try:
            local_vars = {
                'input_data': input_data,
                'context': context,
                'pd': pd,
                'np': np
            }
            return eval(self.condition_expression, {"__builtins__": {}}, local_vars)
        except Exception as e:
            logger.error(f"步骤条件表达式评估失败: {self.condition_expression}, 错误: {str(e)}")
            return False
    
    def validate_output(self, output_data: Any) -> bool:
        """验证输出数据
        
        Args:
            output_data: 输出数据
            
        Returns:
            是否有效
        """
        if self.validation_func:
            try:
                return self.validation_func(output_data)
            except Exception as e:
                logger.error(f"输出验证失败: {self.id}, 错误: {str(e)}")
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'step_type': self.step_type.value,
            'args': self.args,
            'kwargs': self.kwargs,
            'dependencies': self.dependencies,
            'condition_expression': self.condition_expression,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'cache_enabled': self.cache_enabled,
            'description': self.description,
            'metadata': self.metadata,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error': self.error
        }

@dataclass
class Pipeline:
    """数据处理管道"""
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    steps: Dict[str, PipelineStep] = field(default_factory=dict)
    config: PipelineConfig = field(default_factory=PipelineConfig)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 运行时状态
    status: PipelineStatus = PipelineStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_step(self, step: PipelineStep):
        """添加步骤
        
        Args:
            step: 管道步骤
        """
        self.steps[step.id] = step
        self.updated_at = datetime.now()
    
    def remove_step(self, step_id: str):
        """移除步骤
        
        Args:
            step_id: 步骤ID
        """
        if step_id in self.steps:
            del self.steps[step_id]
            
            # 移除其他步骤对此步骤的依赖
            for step in self.steps.values():
                if step_id in step.dependencies:
                    step.dependencies.remove(step_id)
            
            self.updated_at = datetime.now()
    
    def get_execution_order(self) -> List[str]:
        """获取执行顺序
        
        Returns:
            步骤ID列表（按执行顺序）
        """
        # 使用拓扑排序确定执行顺序
        visited = set()
        temp_visited = set()
        result = []
        
        def dfs(step_id: str):
            if step_id in temp_visited:
                raise ValueError(f"检测到循环依赖: {step_id}")
            if step_id in visited:
                return
            
            temp_visited.add(step_id)
            
            step = self.steps.get(step_id)
            if step:
                for dep_id in step.dependencies:
                    if dep_id in self.steps:
                        dfs(dep_id)
            
            temp_visited.remove(step_id)
            visited.add(step_id)
            result.append(step_id)
        
        for step_id in self.steps.keys():
            if step_id not in visited:
                dfs(step_id)
        
        return result
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证管道
        
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查是否有步骤
        if not self.steps:
            errors.append("管道没有步骤")
            return False, errors
        
        # 检查依赖关系
        for step in self.steps.values():
            for dep_id in step.dependencies:
                if dep_id not in self.steps:
                    errors.append(f"步骤 {step.id} 的依赖项不存在: {dep_id}")
        
        # 检查循环依赖
        try:
            self.get_execution_order()
        except ValueError as e:
            errors.append(str(e))
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'steps': {step_id: step.to_dict() for step_id, step in self.steps.items()},
            'config': self.config.to_dict(),
            'tags': self.tags,
            'metadata': self.metadata,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class StepExecutor(ABC):
    """步骤执行器基类"""
    
    @abstractmethod
    async def execute(self, step: PipelineStep, input_data: Any, 
                     context: Dict[str, Any]) -> StepResult:
        """执行步骤
        
        Args:
            step: 管道步骤
            input_data: 输入数据
            context: 执行上下文
            
        Returns:
            步骤执行结果
        """
        pass

class DefaultStepExecutor(StepExecutor):
    """默认步骤执行器"""
    
    async def execute(self, step: PipelineStep, input_data: Any, 
                     context: Dict[str, Any]) -> StepResult:
        """执行步骤"""
        start_time = datetime.now()
        result = StepResult(
            step_id=step.id,
            status=StepStatus.RUNNING,
            start_time=start_time,
            input_data=input_data
        )
        
        try:
            if not step.func:
                raise ValueError("步骤没有指定执行函数")
            
            # 执行步骤函数
            if asyncio.iscoroutinefunction(step.func):
                # 异步函数
                if step.timeout:
                    output_data = await asyncio.wait_for(
                        step.func(input_data, *step.args, **step.kwargs, context=context),
                        timeout=step.timeout
                    )
                else:
                    output_data = await step.func(input_data, *step.args, **step.kwargs, context=context)
            else:
                # 同步函数
                loop = asyncio.get_event_loop()
                if step.timeout:
                    output_data = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, 
                            lambda: step.func(input_data, *step.args, **step.kwargs, context=context)
                        ),
                        timeout=step.timeout
                    )
                else:
                    output_data = await loop.run_in_executor(
                        None, 
                        lambda: step.func(input_data, *step.args, **step.kwargs, context=context)
                    )
            
            # 验证输出
            if not step.validate_output(output_data):
                raise ValueError("输出数据验证失败")
            
            # 步骤成功完成
            end_time = datetime.now()
            result.status = StepStatus.COMPLETED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.output_data = output_data
            
            # 计算指标
            result.metrics = self._calculate_metrics(input_data, output_data)
            
        except asyncio.TimeoutError:
            # 超时
            end_time = datetime.now()
            result.status = StepStatus.FAILED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.error = f"步骤执行超时: {step.timeout}秒"
            
        except Exception as e:
            # 执行失败
            end_time = datetime.now()
            result.status = StepStatus.FAILED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.error = str(e)
        
        return result
    
    def _calculate_metrics(self, input_data: Any, output_data: Any) -> Dict[str, Any]:
        """计算执行指标"""
        metrics = {}
        
        try:
            # 数据大小指标
            if hasattr(input_data, '__len__'):
                metrics['input_size'] = len(input_data)
            if hasattr(output_data, '__len__'):
                metrics['output_size'] = len(output_data)
            
            # DataFrame特定指标
            if isinstance(input_data, pd.DataFrame):
                metrics['input_rows'] = len(input_data)
                metrics['input_columns'] = len(input_data.columns)
                metrics['input_memory_usage'] = input_data.memory_usage(deep=True).sum()
            
            if isinstance(output_data, pd.DataFrame):
                metrics['output_rows'] = len(output_data)
                metrics['output_columns'] = len(output_data.columns)
                metrics['output_memory_usage'] = output_data.memory_usage(deep=True).sum()
                
                # 计算数据变化率
                if isinstance(input_data, pd.DataFrame):
                    if len(input_data) > 0:
                        metrics['row_change_rate'] = (len(output_data) - len(input_data)) / len(input_data)
                    metrics['column_change_rate'] = (len(output_data.columns) - len(input_data.columns)) / len(input_data.columns)
            
        except Exception as e:
            logger.warning(f"计算指标失败: {str(e)}")
        
        return metrics

class PipelineManager:
    """管道管理器
    
    提供数据处理管道的管理和执行功能，包括：
    1. 管道定义管理
    2. 步骤执行器管理
    3. 管道执行控制
    4. 检查点和恢复
    5. 缓存管理
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.pipelines: Dict[str, Pipeline] = {}
        self.executors: Dict[StepType, StepExecutor] = {}
        self.running_pipelines: Dict[str, asyncio.Task] = {}
        self.pipeline_contexts: Dict[str, Dict[str, Any]] = {}
        self.cache: Dict[str, Any] = {}
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        
        # 配置参数
        self.max_concurrent_pipelines = self.config.get('max_concurrent_pipelines', 5)
        self.cache_dir = Path(self.config.get('cache_dir', './pipeline_cache'))
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', './pipeline_checkpoints'))
        
        # 创建目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 注册默认执行器
        self._register_default_executors()
        
        logger.info(f"管道管理器初始化完成")
    
    def _register_default_executors(self):
        """注册默认执行器"""
        default_executor = DefaultStepExecutor()
        for step_type in StepType:
            self.executors[step_type] = default_executor
    
    def register_executor(self, step_type: StepType, executor: StepExecutor):
        """注册步骤执行器
        
        Args:
            step_type: 步骤类型
            executor: 步骤执行器
        """
        self.executors[step_type] = executor
        logger.debug(f"注册步骤执行器: {step_type.value}")
    
    def register_pipeline(self, pipeline: Pipeline) -> bool:
        """注册管道
        
        Args:
            pipeline: 管道对象
            
        Returns:
            是否成功注册
        """
        try:
            # 验证管道
            is_valid, errors = pipeline.validate()
            if not is_valid:
                logger.error(f"管道验证失败: {pipeline.id}, 错误: {errors}")
                return False
            
            with self.lock:
                self.pipelines[pipeline.id] = pipeline
                pipeline.status = PipelineStatus.READY
                
            logger.info(f"注册管道成功: {pipeline.id} ({pipeline.name})")
            return True
            
        except Exception as e:
            logger.error(f"注册管道失败: {pipeline.id}, 错误: {str(e)}")
            return False
    
    def unregister_pipeline(self, pipeline_id: str) -> bool:
        """注销管道
        
        Args:
            pipeline_id: 管道ID
            
        Returns:
            是否成功注销
        """
        try:
            with self.lock:
                if pipeline_id not in self.pipelines:
                    logger.warning(f"管道不存在: {pipeline_id}")
                    return False
                
                # 停止正在运行的管道
                if pipeline_id in self.running_pipelines:
                    task = self.running_pipelines[pipeline_id]
                    task.cancel()
                    del self.running_pipelines[pipeline_id]
                
                # 清理上下文和缓存
                if pipeline_id in self.pipeline_contexts:
                    del self.pipeline_contexts[pipeline_id]
                
                self._clear_pipeline_cache(pipeline_id)
                self._clear_pipeline_checkpoints(pipeline_id)
                
                # 移除管道
                del self.pipelines[pipeline_id]
                
            logger.info(f"注销管道成功: {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"注销管道失败: {pipeline_id}, 错误: {str(e)}")
            return False
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """获取管道
        
        Args:
            pipeline_id: 管道ID
            
        Returns:
            管道对象
        """
        return self.pipelines.get(pipeline_id)
    
    def list_pipelines(self, status: Optional[PipelineStatus] = None,
                      tags: Optional[List[str]] = None) -> List[Pipeline]:
        """列出管道
        
        Args:
            status: 状态过滤
            tags: 标签过滤
            
        Returns:
            管道列表
        """
        with self.lock:
            pipelines = list(self.pipelines.values())
            
            if status:
                pipelines = [p for p in pipelines if p.status == status]
            
            if tags:
                pipelines = [p for p in pipelines if any(tag in p.tags for tag in tags)]
            
            return pipelines
    
    async def execute_pipeline(self, pipeline_id: str, input_data: Any,
                              context: Dict[str, Any] = None,
                              resume_from_checkpoint: bool = False) -> Dict[str, StepResult]:
        """执行管道
        
        Args:
            pipeline_id: 管道ID
            input_data: 输入数据
            context: 执行上下文
            resume_from_checkpoint: 是否从检查点恢复
            
        Returns:
            步骤执行结果字典
        """
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"管道不存在: {pipeline_id}")
        
        if pipeline.status not in [PipelineStatus.READY, PipelineStatus.PAUSED]:
            raise ValueError(f"管道状态不正确: {pipeline.status}")
        
        # 检查并发限制
        if len(self.running_pipelines) >= self.max_concurrent_pipelines:
            raise RuntimeError("达到最大并发管道数量限制")
        
        # 创建执行上下文
        execution_context = context or {}
        execution_context.update({
            'pipeline_id': pipeline_id,
            'start_time': datetime.now().isoformat(),
            'input_data': input_data
        })
        
        with self.lock:
            self.pipeline_contexts[pipeline_id] = execution_context
            pipeline.status = PipelineStatus.RUNNING
        
        try:
            logger.info(f"开始执行管道: {pipeline_id} ({pipeline.name})")
            
            # 创建执行任务
            task = asyncio.create_task(
                self._execute_pipeline_internal(pipeline, input_data, execution_context, resume_from_checkpoint)
            )
            
            with self.lock:
                self.running_pipelines[pipeline_id] = task
            
            # 等待执行完成
            results = await task
            
            # 更新管道状态
            if all(result.status == StepStatus.COMPLETED for result in results.values()):
                pipeline.status = PipelineStatus.COMPLETED
            else:
                pipeline.status = PipelineStatus.FAILED
            
            logger.info(f"管道执行完成: {pipeline_id}, 状态: {pipeline.status.value}")
            return results
            
        except asyncio.CancelledError:
            pipeline.status = PipelineStatus.CANCELLED
            logger.info(f"管道被取消: {pipeline_id}")
            raise
            
        except Exception as e:
            pipeline.status = PipelineStatus.FAILED
            logger.error(f"管道执行失败: {pipeline_id}, 错误: {str(e)}")
            raise
            
        finally:
            # 清理运行状态
            with self.lock:
                if pipeline_id in self.running_pipelines:
                    del self.running_pipelines[pipeline_id]
    
    async def _execute_pipeline_internal(self, pipeline: Pipeline, input_data: Any,
                                       context: Dict[str, Any], resume_from_checkpoint: bool) -> Dict[str, StepResult]:
        """内部管道执行逻辑"""
        results = {}
        
        # 尝试从检查点恢复
        if resume_from_checkpoint:
            checkpoint_data = self._load_checkpoint(pipeline.id)
            if checkpoint_data:
                results = checkpoint_data.get('results', {})
                logger.info(f"从检查点恢复管道执行: {pipeline.id}")
        
        # 获取执行顺序
        execution_order = pipeline.get_execution_order()
        
        # 初始化步骤状态
        for step in pipeline.steps.values():
            if step.id not in results:
                step.status = StepStatus.PENDING
        
        # 根据执行模式执行步骤
        if pipeline.config.execution_mode == ExecutionMode.SEQUENTIAL:
            results = await self._execute_sequential(pipeline, execution_order, input_data, context, results)
        elif pipeline.config.execution_mode == ExecutionMode.PARALLEL:
            results = await self._execute_parallel(pipeline, execution_order, input_data, context, results)
        elif pipeline.config.execution_mode == ExecutionMode.STREAMING:
            results = await self._execute_streaming(pipeline, execution_order, input_data, context, results)
        
        return results
    
    async def _execute_sequential(self, pipeline: Pipeline, execution_order: List[str],
                                input_data: Any, context: Dict[str, Any], 
                                existing_results: Dict[str, StepResult]) -> Dict[str, StepResult]:
        """顺序执行管道"""
        results = existing_results.copy()
        current_data = input_data
        
        for i, step_id in enumerate(execution_order):
            # 跳过已完成的步骤
            if step_id in results and results[step_id].status == StepStatus.COMPLETED:
                current_data = results[step_id].output_data
                continue
            
            step = pipeline.steps[step_id]
            
            # 检查步骤是否应该执行
            if not step.should_execute(current_data, context):
                result = StepResult(
                    step_id=step_id,
                    status=StepStatus.SKIPPED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    input_data=current_data,
                    output_data=current_data
                )
                results[step_id] = result
                continue
            
            # 执行步骤
            result = await self._execute_step_with_retry(step, current_data, context)
            results[step_id] = result
            
            # 更新步骤状态
            step.status = result.status
            step.start_time = result.start_time
            step.end_time = result.end_time
            step.result = result.output_data
            step.error = result.error
            
            if result.status == StepStatus.COMPLETED:
                current_data = result.output_data
            else:
                # 步骤失败，停止执行
                logger.error(f"步骤执行失败，停止管道: {step_id}")
                break
            
            # 保存检查点
            if pipeline.config.checkpoint_enabled and (i + 1) % pipeline.config.checkpoint_interval == 0:
                self._save_checkpoint(pipeline.id, {'results': results, 'current_step': i + 1})
        
        return results
    
    async def _execute_parallel(self, pipeline: Pipeline, execution_order: List[str],
                              input_data: Any, context: Dict[str, Any],
                              existing_results: Dict[str, StepResult]) -> Dict[str, StepResult]:
        """并行执行管道"""
        results = existing_results.copy()
        
        # 构建依赖图
        dependency_graph = self._build_dependency_graph(pipeline)
        
        # 获取可并行执行的步骤组
        execution_groups = self._get_parallel_execution_groups(pipeline, dependency_graph)
        
        current_data = input_data
        
        for group in execution_groups:
            # 并行执行当前组的步骤
            tasks = []
            for step_id in group:
                # 跳过已完成的步骤
                if step_id in results and results[step_id].status == StepStatus.COMPLETED:
                    continue
                
                step = pipeline.steps[step_id]
                
                # 获取步骤的输入数据
                step_input_data = self._get_step_input_data(step, current_data, results)
                
                # 检查步骤是否应该执行
                if step.should_execute(step_input_data, context):
                    task = asyncio.create_task(
                        self._execute_step_with_retry(step, step_input_data, context)
                    )
                    tasks.append((step_id, task))
                else:
                    # 跳过步骤
                    result = StepResult(
                        step_id=step_id,
                        status=StepStatus.SKIPPED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        input_data=step_input_data,
                        output_data=step_input_data
                    )
                    results[step_id] = result
            
            # 等待所有任务完成
            for step_id, task in tasks:
                try:
                    result = await task
                    results[step_id] = result
                    
                    # 更新步骤状态
                    step = pipeline.steps[step_id]
                    step.status = result.status
                    step.start_time = result.start_time
                    step.end_time = result.end_time
                    step.result = result.output_data
                    step.error = result.error
                    
                except Exception as e:
                    logger.error(f"并行步骤执行异常: {step_id}, 错误: {str(e)}")
                    result = StepResult(
                        step_id=step_id,
                        status=StepStatus.FAILED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error=str(e)
                    )
                    results[step_id] = result
        
        return results
    
    async def _execute_streaming(self, pipeline: Pipeline, execution_order: List[str],
                               input_data: Any, context: Dict[str, Any],
                               existing_results: Dict[str, StepResult]) -> Dict[str, StepResult]:
        """流式执行管道"""
        # 流式执行的简化实现
        # 实际应用中可能需要更复杂的流处理逻辑
        return await self._execute_sequential(pipeline, execution_order, input_data, context, existing_results)
    
    async def _execute_step_with_retry(self, step: PipelineStep, input_data: Any,
                                     context: Dict[str, Any]) -> StepResult:
        """执行步骤（包含重试逻辑）"""
        executor = self.executors.get(step.step_type)
        if not executor:
            raise ValueError(f"没有找到步骤类型的执行器: {step.step_type}")
        
        # 检查缓存
        if step.cache_enabled:
            cache_key = self._get_cache_key(step, input_data)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.debug(f"使用缓存结果: {step.id}")
                return cached_result
        
        # 执行步骤（包含重试逻辑）
        for attempt in range(step.max_retries + 1):
            try:
                result = await executor.execute(step, input_data, context)
                
                if result.status == StepStatus.COMPLETED:
                    # 缓存结果
                    if step.cache_enabled:
                        self._cache_result(cache_key, result)
                    return result
                elif attempt < step.max_retries:
                    # 重试
                    logger.info(f"步骤执行失败，准备重试: {step.id}, 尝试次数: {attempt + 1}/{step.max_retries + 1}")
                    await asyncio.sleep(step.retry_delay)
                else:
                    # 重试次数用尽
                    return result
                    
            except Exception as e:
                if attempt < step.max_retries:
                    logger.info(f"步骤执行异常，准备重试: {step.id}, 错误: {str(e)}, 尝试次数: {attempt + 1}/{step.max_retries + 1}")
                    await asyncio.sleep(step.retry_delay)
                else:
                    # 重试次数用尽
                    result = StepResult(
                        step_id=step.id,
                        status=StepStatus.FAILED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        input_data=input_data,
                        error=str(e)
                    )
                    return result
        
        # 不应该到达这里
        return StepResult(
            step_id=step.id,
            status=StepStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            input_data=input_data,
            error="未知错误"
        )
    
    def _build_dependency_graph(self, pipeline: Pipeline) -> Dict[str, List[str]]:
        """构建依赖图"""
        graph = {}
        for step_id, step in pipeline.steps.items():
            graph[step_id] = step.dependencies.copy()
        return graph
    
    def _get_parallel_execution_groups(self, pipeline: Pipeline, 
                                     dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """获取并行执行组"""
        groups = []
        remaining_steps = set(pipeline.steps.keys())
        completed_steps = set()
        
        while remaining_steps:
            # 找到当前可执行的步骤（没有未完成的依赖）
            ready_steps = []
            for step_id in remaining_steps:
                dependencies = dependency_graph[step_id]
                if all(dep in completed_steps for dep in dependencies):
                    ready_steps.append(step_id)
            
            if not ready_steps:
                # 可能存在循环依赖
                raise ValueError("检测到循环依赖或无法解析的依赖关系")
            
            groups.append(ready_steps)
            
            # 更新状态
            for step_id in ready_steps:
                remaining_steps.remove(step_id)
                completed_steps.add(step_id)
        
        return groups
    
    def _get_step_input_data(self, step: PipelineStep, default_data: Any,
                           results: Dict[str, StepResult]) -> Any:
        """获取步骤的输入数据"""
        if not step.dependencies:
            return default_data
        
        # 如果有依赖，使用最后一个依赖的输出作为输入
        last_dependency = step.dependencies[-1]
        if last_dependency in results:
            return results[last_dependency].output_data
        
        return default_data
    
    def _get_cache_key(self, step: PipelineStep, input_data: Any) -> str:
        """生成缓存键"""
        import hashlib
        
        # 简化的缓存键生成
        key_data = {
            'step_id': step.id,
            'step_type': step.step_type.value,
            'args': step.args,
            'kwargs': step.kwargs
        }
        
        # 添加输入数据的哈希
        try:
            if isinstance(input_data, pd.DataFrame):
                input_hash = hashlib.md5(pd.util.hash_pandas_object(input_data).values).hexdigest()
            else:
                input_hash = hashlib.md5(str(input_data).encode()).hexdigest()
            key_data['input_hash'] = input_hash
        except Exception:
            key_data['input_hash'] = 'unknown'
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[StepResult]:
        """获取缓存结果"""
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 尝试从文件加载
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                self.cache[cache_key] = result
                return result
            except Exception as e:
                logger.warning(f"加载缓存文件失败: {cache_file}, 错误: {str(e)}")
        
        return None
    
    def _cache_result(self, cache_key: str, result: StepResult):
        """缓存结果"""
        self.cache[cache_key] = result
        
        # 保存到文件
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"保存缓存文件失败: {cache_file}, 错误: {str(e)}")
    
    def _save_checkpoint(self, pipeline_id: str, checkpoint_data: Dict[str, Any]):
        """保存检查点"""
        self.checkpoints[pipeline_id] = checkpoint_data
        
        # 保存到文件
        checkpoint_file = self.checkpoint_dir / f"{pipeline_id}.json"
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                # 序列化检查点数据
                serializable_data = self._serialize_checkpoint_data(checkpoint_data)
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"保存检查点: {pipeline_id}")
        except Exception as e:
            logger.warning(f"保存检查点文件失败: {checkpoint_file}, 错误: {str(e)}")
    
    def _load_checkpoint(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        if pipeline_id in self.checkpoints:
            return self.checkpoints[pipeline_id]
        
        # 尝试从文件加载
        checkpoint_file = self.checkpoint_dir / f"{pipeline_id}.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    serializable_data = json.load(f)
                
                # 反序列化检查点数据
                checkpoint_data = self._deserialize_checkpoint_data(serializable_data)
                self.checkpoints[pipeline_id] = checkpoint_data
                return checkpoint_data
            except Exception as e:
                logger.warning(f"加载检查点文件失败: {checkpoint_file}, 错误: {str(e)}")
        
        return None
    
    def _serialize_checkpoint_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """序列化检查点数据"""
        serializable_data = {}
        
        for key, value in data.items():
            if key == 'results':
                # 序列化步骤结果
                serializable_data[key] = {
                    step_id: result.to_dict() for step_id, result in value.items()
                }
            else:
                serializable_data[key] = value
        
        return serializable_data
    
    def _deserialize_checkpoint_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """反序列化检查点数据"""
        checkpoint_data = {}
        
        for key, value in data.items():
            if key == 'results':
                # 反序列化步骤结果
                results = {}
                for step_id, result_dict in value.items():
                    result = StepResult(
                        step_id=result_dict['step_id'],
                        status=StepStatus(result_dict['status']),
                        start_time=datetime.fromisoformat(result_dict['start_time']),
                        end_time=datetime.fromisoformat(result_dict['end_time']) if result_dict['end_time'] else None,
                        duration=result_dict['duration'],
                        error=result_dict['error'],
                        metrics=result_dict['metrics'],
                        metadata=result_dict['metadata']
                    )
                    results[step_id] = result
                checkpoint_data[key] = results
            else:
                checkpoint_data[key] = value
        
        return checkpoint_data
    
    def _clear_pipeline_cache(self, pipeline_id: str):
        """清理管道缓存"""
        # 清理内存缓存
        keys_to_remove = []
        for key in self.cache.keys():
            if pipeline_id in key:  # 简化的匹配逻辑
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        # 清理文件缓存（这里简化处理，实际可能需要更精确的匹配）
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"清理缓存文件失败: {str(e)}")
    
    def _clear_pipeline_checkpoints(self, pipeline_id: str):
        """清理管道检查点"""
        if pipeline_id in self.checkpoints:
            del self.checkpoints[pipeline_id]
        
        checkpoint_file = self.checkpoint_dir / f"{pipeline_id}.json"
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
            except Exception as e:
                logger.warning(f"删除检查点文件失败: {checkpoint_file}, 错误: {str(e)}")
    
    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """取消管道执行
        
        Args:
            pipeline_id: 管道ID
            
        Returns:
            是否成功取消
        """
        try:
            with self.lock:
                if pipeline_id not in self.running_pipelines:
                    logger.warning(f"管道未在运行: {pipeline_id}")
                    return False
                
                task = self.running_pipelines[pipeline_id]
                task.cancel()
                
                # 更新管道状态
                pipeline = self.pipelines.get(pipeline_id)
                if pipeline:
                    pipeline.status = PipelineStatus.CANCELLED
                
            logger.info(f"取消管道成功: {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消管道失败: {pipeline_id}, 错误: {str(e)}")
            return False
    
    def get_pipeline_context(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """获取管道上下文
        
        Args:
            pipeline_id: 管道ID
            
        Returns:
            管道上下文
        """
        return self.pipeline_contexts.get(pipeline_id)
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息
        
        Returns:
            统计信息
        """
        with self.lock:
            status_counts = {}
            for pipeline in self.pipelines.values():
                status = pipeline.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'total_pipelines': len(self.pipelines),
                'running_pipelines': len(self.running_pipelines),
                'status_distribution': status_counts,
                'cache_size': len(self.cache),
                'checkpoint_count': len(self.checkpoints),
                'max_concurrent_pipelines': self.max_concurrent_pipelines
            }

# 便捷函数
def create_pipeline_manager(config: Dict[str, Any] = None) -> PipelineManager:
    """创建管道管理器实例
    
    Args:
        config: 配置字典
        
    Returns:
        管道管理器实例
    """
    return PipelineManager(config)

def create_extract_step(step_id: str, name: str, func: Callable, 
                       *args, **kwargs) -> PipelineStep:
    """创建数据提取步骤
    
    Args:
        step_id: 步骤ID
        name: 步骤名称
        func: 执行函数
        *args: 函数参数
        **kwargs: 函数关键字参数
        
    Returns:
        管道步骤
    """
    return PipelineStep(
        id=step_id,
        name=name,
        step_type=StepType.EXTRACT,
        func=func,
        args=args,
        kwargs=kwargs
    )

def create_transform_step(step_id: str, name: str, func: Callable,
                         dependencies: List[str] = None, *args, **kwargs) -> PipelineStep:
    """创建数据转换步骤
    
    Args:
        step_id: 步骤ID
        name: 步骤名称
        func: 执行函数
        dependencies: 依赖步骤列表
        *args: 函数参数
        **kwargs: 函数关键字参数
        
    Returns:
        管道步骤
    """
    return PipelineStep(
        id=step_id,
        name=name,
        step_type=StepType.TRANSFORM,
        func=func,
        dependencies=dependencies or [],
        args=args,
        kwargs=kwargs
    )

def create_load_step(step_id: str, name: str, func: Callable,
                    dependencies: List[str] = None, *args, **kwargs) -> PipelineStep:
    """创建数据加载步骤
    
    Args:
        step_id: 步骤ID
        name: 步骤名称
        func: 执行函数
        dependencies: 依赖步骤列表
        *args: 函数参数
        **kwargs: 函数关键字参数
        
    Returns:
        管道步骤
    """
    return PipelineStep(
        id=step_id,
        name=name,
        step_type=StepType.LOAD,
        func=func,
        dependencies=dependencies or [],
        args=args,
        kwargs=kwargs
    )