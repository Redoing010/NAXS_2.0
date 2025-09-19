# 任务调度器 - 实现基于时间和事件的任务调度

import logging
import asyncio
import threading
import time
from typing import Dict, Any, List, Optional, Callable, Union, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import uuid
import json
from concurrent.futures import ThreadPoolExecutor, Future
import schedule
from croniter import croniter

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"        # 等待中
    RUNNING = "running"        # 运行中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"    # 已取消
    RETRYING = "retrying"      # 重试中
    PAUSED = "paused"          # 暂停

class TriggerType(Enum):
    """触发器类型"""
    CRON = "cron"              # Cron表达式
    INTERVAL = "interval"      # 时间间隔
    ONCE = "once"              # 一次性
    EVENT = "event"            # 事件触发
    DEPENDENCY = "dependency"  # 依赖触发

class Priority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'result': self.result,
            'error': self.error,
            'retry_count': self.retry_count,
            'metadata': self.metadata
        }

@dataclass
class ScheduleConfig:
    """调度配置"""
    trigger_type: TriggerType
    trigger_config: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    retry_delay: float = 60.0  # 重试延迟（秒）
    timeout: Optional[float] = None  # 超时时间（秒）
    priority: Priority = Priority.NORMAL
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'trigger_type': self.trigger_type.value,
            'trigger_config': self.trigger_config,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'timeout': self.timeout,
            'priority': self.priority.value,
            'enabled': self.enabled
        }

@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    schedule_config: ScheduleConfig = field(default_factory=lambda: ScheduleConfig(TriggerType.ONCE))
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 运行时状态
    status: TaskStatus = TaskStatus.PENDING
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'args': self.args,
            'kwargs': self.kwargs,
            'schedule_config': self.schedule_config.to_dict(),
            'dependencies': self.dependencies,
            'tags': self.tags,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status.value,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'run_count': self.run_count
        }

class TaskTrigger(ABC):
    """任务触发器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def should_trigger(self, task: Task, current_time: datetime) -> bool:
        """判断是否应该触发任务
        
        Args:
            task: 任务对象
            current_time: 当前时间
            
        Returns:
            是否应该触发
        """
        pass
    
    @abstractmethod
    def get_next_run_time(self, task: Task, current_time: datetime) -> Optional[datetime]:
        """获取下次运行时间
        
        Args:
            task: 任务对象
            current_time: 当前时间
            
        Returns:
            下次运行时间
        """
        pass

class CronTrigger(TaskTrigger):
    """Cron触发器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cron_expression = config.get('cron_expression', '0 0 * * *')
        self.timezone = config.get('timezone', 'UTC')
    
    def should_trigger(self, task: Task, current_time: datetime) -> bool:
        """判断是否应该触发"""
        if task.next_run is None:
            task.next_run = self.get_next_run_time(task, current_time)
        
        return task.next_run is not None and current_time >= task.next_run
    
    def get_next_run_time(self, task: Task, current_time: datetime) -> Optional[datetime]:
        """获取下次运行时间"""
        try:
            cron = croniter(self.cron_expression, current_time)
            return cron.get_next(datetime)
        except Exception as e:
            logger.error(f"解析Cron表达式失败: {self.cron_expression}, 错误: {str(e)}")
            return None

class IntervalTrigger(TaskTrigger):
    """间隔触发器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.interval = config.get('interval', 3600)  # 默认1小时
        self.start_time = config.get('start_time')
    
    def should_trigger(self, task: Task, current_time: datetime) -> bool:
        """判断是否应该触发"""
        if task.next_run is None:
            task.next_run = self.get_next_run_time(task, current_time)
        
        return task.next_run is not None and current_time >= task.next_run
    
    def get_next_run_time(self, task: Task, current_time: datetime) -> Optional[datetime]:
        """获取下次运行时间"""
        if task.last_run is None:
            # 首次运行
            if self.start_time:
                if isinstance(self.start_time, str):
                    return datetime.fromisoformat(self.start_time)
                return self.start_time
            else:
                return current_time
        else:
            # 基于上次运行时间计算
            return task.last_run + timedelta(seconds=self.interval)

class OnceTrigger(TaskTrigger):
    """一次性触发器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.run_time = config.get('run_time')
    
    def should_trigger(self, task: Task, current_time: datetime) -> bool:
        """判断是否应该触发"""
        if task.run_count > 0:
            return False  # 已经运行过
        
        if self.run_time:
            if isinstance(self.run_time, str):
                run_time = datetime.fromisoformat(self.run_time)
            else:
                run_time = self.run_time
            return current_time >= run_time
        else:
            return True  # 立即运行
    
    def get_next_run_time(self, task: Task, current_time: datetime) -> Optional[datetime]:
        """获取下次运行时间"""
        if task.run_count > 0:
            return None  # 已经运行过，不再运行
        
        if self.run_time:
            if isinstance(self.run_time, str):
                return datetime.fromisoformat(self.run_time)
            return self.run_time
        else:
            return current_time

class EventTrigger(TaskTrigger):
    """事件触发器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.event_type = config.get('event_type')
        self.event_filter = config.get('event_filter', {})
        self.triggered_events = []
    
    def should_trigger(self, task: Task, current_time: datetime) -> bool:
        """判断是否应该触发"""
        return len(self.triggered_events) > 0
    
    def get_next_run_time(self, task: Task, current_time: datetime) -> Optional[datetime]:
        """获取下次运行时间"""
        if self.triggered_events:
            return current_time
        return None
    
    def trigger_event(self, event_data: Dict[str, Any]):
        """触发事件
        
        Args:
            event_data: 事件数据
        """
        # 检查事件过滤条件
        if self._match_event_filter(event_data):
            self.triggered_events.append({
                'timestamp': datetime.now(),
                'data': event_data
            })
    
    def _match_event_filter(self, event_data: Dict[str, Any]) -> bool:
        """检查事件是否匹配过滤条件"""
        if not self.event_filter:
            return True
        
        for key, expected_value in self.event_filter.items():
            if key not in event_data or event_data[key] != expected_value:
                return False
        
        return True
    
    def consume_events(self) -> List[Dict[str, Any]]:
        """消费事件"""
        events = self.triggered_events.copy()
        self.triggered_events.clear()
        return events

class TaskScheduler:
    """任务调度器
    
    提供任务调度和执行功能，包括：
    1. 多种触发器支持（Cron、间隔、一次性、事件）
    2. 任务依赖管理
    3. 重试机制
    4. 并发控制
    5. 任务监控
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tasks: Dict[str, Task] = {}
        self.triggers: Dict[str, TaskTrigger] = {}
        self.results: Dict[str, List[TaskResult]] = {}
        self.running_tasks: Dict[str, Future] = {}
        
        # 配置参数
        self.max_workers = self.config.get('max_workers', 10)
        self.check_interval = self.config.get('check_interval', 10)  # 检查间隔（秒）
        self.max_result_history = self.config.get('max_result_history', 100)
        
        # 线程池和控制
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.running = False
        self.scheduler_thread = None
        self.lock = threading.RLock()
        
        logger.info(f"任务调度器初始化完成，最大工作线程: {self.max_workers}")
    
    def add_task(self, task: Task) -> bool:
        """添加任务
        
        Args:
            task: 任务对象
            
        Returns:
            是否成功添加
        """
        try:
            with self.lock:
                if task.id in self.tasks:
                    logger.warning(f"任务已存在: {task.id}")
                    return False
                
                # 创建触发器
                trigger = self._create_trigger(task.schedule_config)
                if trigger is None:
                    logger.error(f"创建触发器失败: {task.id}")
                    return False
                
                self.tasks[task.id] = task
                self.triggers[task.id] = trigger
                self.results[task.id] = []
                
                # 计算下次运行时间
                task.next_run = trigger.get_next_run_time(task, datetime.now())
                
                logger.info(f"添加任务成功: {task.id} ({task.name})")
                return True
                
        except Exception as e:
            logger.error(f"添加任务失败: {task.id}, 错误: {str(e)}")
            return False
    
    def remove_task(self, task_id: str) -> bool:
        """移除任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功移除
        """
        try:
            with self.lock:
                if task_id not in self.tasks:
                    logger.warning(f"任务不存在: {task_id}")
                    return False
                
                # 取消正在运行的任务
                if task_id in self.running_tasks:
                    future = self.running_tasks[task_id]
                    future.cancel()
                    del self.running_tasks[task_id]
                
                # 移除任务相关数据
                del self.tasks[task_id]
                del self.triggers[task_id]
                if task_id in self.results:
                    del self.results[task_id]
                
                logger.info(f"移除任务成功: {task_id}")
                return True
                
        except Exception as e:
            logger.error(f"移除任务失败: {task_id}, 错误: {str(e)}")
            return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务对象
        """
        return self.tasks.get(task_id)
    
    def list_tasks(self, status: Optional[TaskStatus] = None, 
                  tags: Optional[List[str]] = None) -> List[Task]:
        """列出任务
        
        Args:
            status: 状态过滤
            tags: 标签过滤
            
        Returns:
            任务列表
        """
        with self.lock:
            tasks = list(self.tasks.values())
            
            if status:
                tasks = [t for t in tasks if t.status == status]
            
            if tags:
                tasks = [t for t in tasks if any(tag in t.tags for tag in tags)]
            
            return tasks
    
    def start(self):
        """启动调度器"""
        if self.running:
            logger.warning("调度器已在运行")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("任务调度器启动")
    
    def stop(self):
        """停止调度器"""
        if not self.running:
            logger.warning("调度器未在运行")
            return
        
        self.running = False
        
        # 等待调度线程结束
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        # 取消所有运行中的任务
        with self.lock:
            for future in self.running_tasks.values():
                future.cancel()
            self.running_tasks.clear()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("任务调度器停止")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        logger.info("调度器主循环启动")
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # 检查需要运行的任务
                tasks_to_run = self._get_tasks_to_run(current_time)
                
                # 执行任务
                for task in tasks_to_run:
                    if self._can_run_task(task):
                        self._submit_task(task)
                
                # 清理完成的任务
                self._cleanup_completed_tasks()
                
                # 等待下次检查
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"调度器循环异常: {str(e)}")
                time.sleep(self.check_interval)
        
        logger.info("调度器主循环结束")
    
    def _get_tasks_to_run(self, current_time: datetime) -> List[Task]:
        """获取需要运行的任务"""
        tasks_to_run = []
        
        with self.lock:
            for task_id, task in self.tasks.items():
                if not task.schedule_config.enabled:
                    continue
                
                if task.status in [TaskStatus.RUNNING, TaskStatus.CANCELLED]:
                    continue
                
                trigger = self.triggers.get(task_id)
                if trigger and trigger.should_trigger(task, current_time):
                    tasks_to_run.append(task)
        
        return tasks_to_run
    
    def _can_run_task(self, task: Task) -> bool:
        """检查任务是否可以运行"""
        # 检查依赖任务
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task:
                logger.warning(f"依赖任务不存在: {dep_id}")
                return False
            
            if dep_task.status != TaskStatus.COMPLETED:
                logger.debug(f"依赖任务未完成: {dep_id}")
                return False
        
        # 检查是否已在运行
        if task.id in self.running_tasks:
            return False
        
        return True
    
    def _submit_task(self, task: Task):
        """提交任务执行"""
        try:
            with self.lock:
                task.status = TaskStatus.RUNNING
                
                # 提交到线程池
                future = self.executor.submit(self._execute_task, task)
                self.running_tasks[task.id] = future
                
                logger.info(f"提交任务执行: {task.id} ({task.name})")
                
        except Exception as e:
            logger.error(f"提交任务失败: {task.id}, 错误: {str(e)}")
            task.status = TaskStatus.FAILED
    
    def _execute_task(self, task: Task) -> TaskResult:
        """执行任务"""
        start_time = datetime.now()
        result = TaskResult(
            task_id=task.id,
            status=TaskStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            logger.info(f"开始执行任务: {task.id} ({task.name})")
            
            # 执行任务函数
            if asyncio.iscoroutinefunction(task.func):
                # 异步函数
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    task_result = loop.run_until_complete(
                        asyncio.wait_for(
                            task.func(*task.args, **task.kwargs),
                            timeout=task.schedule_config.timeout
                        )
                    )
                finally:
                    loop.close()
            else:
                # 同步函数
                task_result = task.func(*task.args, **task.kwargs)
            
            # 任务成功完成
            end_time = datetime.now()
            result.status = TaskStatus.COMPLETED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.result = task_result
            
            # 更新任务状态
            with self.lock:
                task.status = TaskStatus.COMPLETED
                task.last_run = start_time
                task.run_count += 1
                
                # 计算下次运行时间
                trigger = self.triggers.get(task.id)
                if trigger:
                    task.next_run = trigger.get_next_run_time(task, end_time)
            
            logger.info(f"任务执行成功: {task.id}, 耗时: {result.duration:.2f}秒")
            
        except Exception as e:
            # 任务执行失败
            end_time = datetime.now()
            result.status = TaskStatus.FAILED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.error = str(e)
            
            logger.error(f"任务执行失败: {task.id}, 错误: {str(e)}")
            
            # 检查是否需要重试
            if result.retry_count < task.schedule_config.max_retries:
                result.retry_count += 1
                result.status = TaskStatus.RETRYING
                
                # 安排重试
                retry_time = datetime.now() + timedelta(seconds=task.schedule_config.retry_delay)
                with self.lock:
                    task.status = TaskStatus.RETRYING
                    task.next_run = retry_time
                
                logger.info(f"任务将重试: {task.id}, 重试次数: {result.retry_count}/{task.schedule_config.max_retries}")
            else:
                # 重试次数用尽
                with self.lock:
                    task.status = TaskStatus.FAILED
        
        finally:
            # 保存执行结果
            with self.lock:
                if task.id not in self.results:
                    self.results[task.id] = []
                
                self.results[task.id].append(result)
                
                # 限制历史记录数量
                if len(self.results[task.id]) > self.max_result_history:
                    self.results[task.id] = self.results[task.id][-self.max_result_history:]
                
                # 从运行任务列表中移除
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
        
        return result
    
    def _cleanup_completed_tasks(self):
        """清理已完成的任务"""
        with self.lock:
            completed_tasks = []
            for task_id, future in self.running_tasks.items():
                if future.done():
                    completed_tasks.append(task_id)
            
            for task_id in completed_tasks:
                del self.running_tasks[task_id]
    
    def _create_trigger(self, schedule_config: ScheduleConfig) -> Optional[TaskTrigger]:
        """创建触发器"""
        try:
            if schedule_config.trigger_type == TriggerType.CRON:
                return CronTrigger(schedule_config.trigger_config)
            elif schedule_config.trigger_type == TriggerType.INTERVAL:
                return IntervalTrigger(schedule_config.trigger_config)
            elif schedule_config.trigger_type == TriggerType.ONCE:
                return OnceTrigger(schedule_config.trigger_config)
            elif schedule_config.trigger_type == TriggerType.EVENT:
                return EventTrigger(schedule_config.trigger_config)
            else:
                logger.error(f"不支持的触发器类型: {schedule_config.trigger_type}")
                return None
                
        except Exception as e:
            logger.error(f"创建触发器失败: {str(e)}")
            return None
    
    def trigger_event(self, event_type: str, event_data: Dict[str, Any]):
        """触发事件
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        with self.lock:
            for task_id, trigger in self.triggers.items():
                if isinstance(trigger, EventTrigger) and trigger.event_type == event_type:
                    trigger.trigger_event(event_data)
    
    def get_task_results(self, task_id: str, limit: Optional[int] = None) -> List[TaskResult]:
        """获取任务执行结果
        
        Args:
            task_id: 任务ID
            limit: 结果数量限制
            
        Returns:
            执行结果列表
        """
        with self.lock:
            results = self.results.get(task_id, [])
            if limit:
                return results[-limit:]
            return results.copy()
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息
        
        Returns:
            统计信息
        """
        with self.lock:
            status_counts = {}
            for task in self.tasks.values():
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'total_tasks': len(self.tasks),
                'running_tasks': len(self.running_tasks),
                'status_distribution': status_counts,
                'max_workers': self.max_workers,
                'is_running': self.running
            }
    
    def pause_task(self, task_id: str) -> bool:
        """暂停任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功暂停
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if task.status == TaskStatus.RUNNING:
                # 取消正在运行的任务
                if task_id in self.running_tasks:
                    future = self.running_tasks[task_id]
                    future.cancel()
            
            task.status = TaskStatus.PAUSED
            task.schedule_config.enabled = False
            
            logger.info(f"任务已暂停: {task_id}")
            return True
    
    def resume_task(self, task_id: str) -> bool:
        """恢复任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功恢复
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if task.status == TaskStatus.PAUSED:
                task.status = TaskStatus.PENDING
                task.schedule_config.enabled = True
                
                # 重新计算下次运行时间
                trigger = self.triggers.get(task_id)
                if trigger:
                    task.next_run = trigger.get_next_run_time(task, datetime.now())
                
                logger.info(f"任务已恢复: {task_id}")
                return True
            
            return False

# 便捷函数
def create_task_scheduler(config: Dict[str, Any] = None) -> TaskScheduler:
    """创建任务调度器实例
    
    Args:
        config: 配置字典
        
    Returns:
        任务调度器实例
    """
    return TaskScheduler(config)

def create_cron_task(task_id: str, name: str, func: Callable, 
                    cron_expression: str, *args, **kwargs) -> Task:
    """创建Cron任务
    
    Args:
        task_id: 任务ID
        name: 任务名称
        func: 任务函数
        cron_expression: Cron表达式
        *args: 任务参数
        **kwargs: 任务关键字参数
        
    Returns:
        任务对象
    """
    schedule_config = ScheduleConfig(
        trigger_type=TriggerType.CRON,
        trigger_config={'cron_expression': cron_expression}
    )
    
    return Task(
        id=task_id,
        name=name,
        func=func,
        args=args,
        kwargs=kwargs,
        schedule_config=schedule_config
    )

def create_interval_task(task_id: str, name: str, func: Callable, 
                        interval: int, *args, **kwargs) -> Task:
    """创建间隔任务
    
    Args:
        task_id: 任务ID
        name: 任务名称
        func: 任务函数
        interval: 间隔时间（秒）
        *args: 任务参数
        **kwargs: 任务关键字参数
        
    Returns:
        任务对象
    """
    schedule_config = ScheduleConfig(
        trigger_type=TriggerType.INTERVAL,
        trigger_config={'interval': interval}
    )
    
    return Task(
        id=task_id,
        name=name,
        func=func,
        args=args,
        kwargs=kwargs,
        schedule_config=schedule_config
    )

def create_once_task(task_id: str, name: str, func: Callable, 
                    run_time: Optional[datetime] = None, *args, **kwargs) -> Task:
    """创建一次性任务
    
    Args:
        task_id: 任务ID
        name: 任务名称
        func: 任务函数
        run_time: 运行时间，如果为None则立即运行
        *args: 任务参数
        **kwargs: 任务关键字参数
        
    Returns:
        任务对象
    """
    trigger_config = {}
    if run_time:
        trigger_config['run_time'] = run_time
    
    schedule_config = ScheduleConfig(
        trigger_type=TriggerType.ONCE,
        trigger_config=trigger_config
    )
    
    return Task(
        id=task_id,
        name=name,
        func=func,
        args=args,
        kwargs=kwargs,
        schedule_config=schedule_config
    )