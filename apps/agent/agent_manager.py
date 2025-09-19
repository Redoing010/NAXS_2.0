# Agent管理器
# 管理和调度智能投研Agent实例

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

class AgentStatus(Enum):
    """Agent状态"""
    IDLE = "idle"                    # 空闲
    BUSY = "busy"                    # 忙碌
    THINKING = "thinking"            # 思考中
    EXECUTING = "executing"          # 执行中
    WAITING = "waiting"              # 等待中
    ERROR = "error"                  # 错误
    OFFLINE = "offline"              # 离线

class AgentType(Enum):
    """Agent类型"""
    GENERAL = "general"              # 通用Agent
    MARKET_ANALYST = "market_analyst" # 市场分析师
    STOCK_PICKER = "stock_picker"    # 选股专家
    RISK_MANAGER = "risk_manager"    # 风险管理师
    STRATEGY_ADVISOR = "strategy_advisor" # 策略顾问
    DATA_SCIENTIST = "data_scientist" # 数据科学家
    REPORT_WRITER = "report_writer"  # 报告撰写员
    CUSTOM = "custom"                # 自定义

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class AgentConfig:
    """Agent配置"""
    agent_id: str
    name: str
    agent_type: AgentType = AgentType.GENERAL
    description: str = ""
    
    # 能力配置
    capabilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    
    # LLM配置
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # 行为配置
    auto_execute: bool = False
    confidence_threshold: float = 0.8
    max_iterations: int = 10
    timeout: int = 300  # 秒
    
    # 记忆配置
    memory_size: int = 1000
    context_window: int = 10
    
    # 其他配置
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Task:
    """任务定义"""
    task_id: str
    user_id: str
    content: str
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # 任务配置
    required_capabilities: List[str] = field(default_factory=list)
    preferred_agent_type: Optional[AgentType] = None
    timeout: int = 300
    
    # 状态信息
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 结果信息
    result: Optional[Any] = None
    error: Optional[str] = None
    agent_id: Optional[str] = None
    
    # 上下文
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """任务执行时长"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.created_at:
            return datetime.now() - self.created_at > timedelta(seconds=self.timeout)
        return False

@dataclass
class AgentStats:
    """Agent统计信息"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_response_time: float = 0.0
    success_rate: float = 0.0
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    last_activity: Optional[datetime] = None
    
    def update_completion(self, duration: timedelta, success: bool):
        """更新完成统计"""
        self.total_tasks += 1
        if success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1
        
        # 更新平均响应时间
        if self.avg_response_time == 0:
            self.avg_response_time = duration.total_seconds()
        else:
            self.avg_response_time = (self.avg_response_time + duration.total_seconds()) / 2
        
        # 更新成功率
        self.success_rate = self.completed_tasks / self.total_tasks
        self.last_activity = datetime.now()

class Agent:
    """智能Agent实例"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(f"Agent.{config.name}")
        
        # 状态管理
        self.status = AgentStatus.IDLE
        self.current_tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
        
        # 统计信息
        self.stats = AgentStats()
        self.created_at = datetime.now()
        
        # 组件引用
        self.planner = None
        self.executor = None
        self.llm_interface = None
        self.context_manager = None
        self.memory_manager = None
        
        # 事件回调
        self.on_task_start: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None
        self.on_task_error: Optional[Callable] = None
        self.on_status_change: Optional[Callable] = None
    
    async def initialize(self):
        """初始化Agent"""
        try:
            # 这里会注入各种组件
            self.logger.info(f"Agent {self.config.name} initialized")
            self.status = AgentStatus.IDLE
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def shutdown(self):
        """关闭Agent"""
        try:
            # 取消所有正在执行的任务
            for task in self.current_tasks.values():
                task.status = "cancelled"
            
            self.current_tasks.clear()
            self.task_queue.clear()
            self.status = AgentStatus.OFFLINE
            
            self.logger.info(f"Agent {self.config.name} shutdown")
            return True
        except Exception as e:
            self.logger.error(f"Failed to shutdown agent: {e}")
            return False
    
    async def assign_task(self, task: Task) -> bool:
        """分配任务"""
        try:
            # 检查Agent状态
            if self.status == AgentStatus.OFFLINE:
                return False
            
            # 检查并发限制
            if len(self.current_tasks) >= self.config.max_concurrent_tasks:
                # 添加到队列
                self.task_queue.append(task)
                self.logger.info(f"Task {task.task_id} queued")
                return True
            
            # 立即执行任务
            await self._execute_task(task)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to assign task {task.task_id}: {e}")
            return False
    
    async def _execute_task(self, task: Task):
        """执行任务"""
        task.agent_id = self.config.agent_id
        task.started_at = datetime.now()
        task.status = "running"
        
        self.current_tasks[task.task_id] = task
        self._update_status()
        
        # 触发任务开始回调
        if self.on_task_start:
            await self.on_task_start(task)
        
        try:
            # 这里会调用实际的任务执行逻辑
            result = await self._process_task(task)
            
            # 任务完成
            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()
            
            # 更新统计
            if task.duration:
                self.stats.update_completion(task.duration, True)
            
            # 触发完成回调
            if self.on_task_complete:
                await self.on_task_complete(task)
            
            self.logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            # 任务失败
            task.error = str(e)
            task.status = "failed"
            task.completed_at = datetime.now()
            
            # 更新统计
            if task.duration:
                self.stats.update_completion(task.duration, False)
            
            # 触发错误回调
            if self.on_task_error:
                await self.on_task_error(task, e)
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # 清理任务
            if task.task_id in self.current_tasks:
                del self.current_tasks[task.task_id]
            
            # 处理队列中的下一个任务
            await self._process_queue()
            
            # 更新状态
            self._update_status()
    
    async def _process_task(self, task: Task) -> Any:
        """处理任务的核心逻辑"""
        # 这里是任务处理的核心逻辑
        # 会调用planner进行任务规划，executor执行具体操作
        
        # 1. 任务理解和规划
        if self.planner:
            plan = await self.planner.create_plan(task.content, task.context)
        else:
            plan = {"steps": [{"action": "respond", "content": task.content}]}
        
        # 2. 执行计划
        if self.executor:
            result = await self.executor.execute_plan(plan, task.context)
        else:
            result = {"response": f"Processed: {task.content}"}
        
        return result
    
    async def _process_queue(self):
        """处理任务队列"""
        if (self.task_queue and 
            len(self.current_tasks) < self.config.max_concurrent_tasks):
            
            # 按优先级排序
            self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
            
            # 取出下一个任务
            next_task = self.task_queue.pop(0)
            await self._execute_task(next_task)
    
    def _update_status(self):
        """更新Agent状态"""
        old_status = self.status
        
        if len(self.current_tasks) == 0:
            self.status = AgentStatus.IDLE
        elif len(self.current_tasks) >= self.config.max_concurrent_tasks:
            self.status = AgentStatus.BUSY
        else:
            self.status = AgentStatus.EXECUTING
        
        # 触发状态变化回调
        if old_status != self.status and self.on_status_change:
            asyncio.create_task(self.on_status_change(old_status, self.status))
    
    def can_handle_task(self, task: Task) -> bool:
        """检查是否能处理任务"""
        # 检查Agent状态
        if self.status == AgentStatus.OFFLINE or not self.config.enabled:
            return False
        
        # 检查能力匹配
        if task.required_capabilities:
            if not all(cap in self.config.capabilities for cap in task.required_capabilities):
                return False
        
        # 检查Agent类型匹配
        if (task.preferred_agent_type and 
            task.preferred_agent_type != self.config.agent_type):
            return False
        
        return True
    
    def get_load_factor(self) -> float:
        """获取负载因子"""
        if self.config.max_concurrent_tasks == 0:
            return 0.0
        
        current_load = len(self.current_tasks) + len(self.task_queue)
        return current_load / self.config.max_concurrent_tasks
    
    def get_status_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            'agent_id': self.config.agent_id,
            'name': self.config.name,
            'type': self.config.agent_type.value,
            'status': self.status.value,
            'current_tasks': len(self.current_tasks),
            'queued_tasks': len(self.task_queue),
            'load_factor': self.get_load_factor(),
            'capabilities': self.config.capabilities,
            'stats': {
                'total_tasks': self.stats.total_tasks,
                'completed_tasks': self.stats.completed_tasks,
                'failed_tasks': self.stats.failed_tasks,
                'success_rate': self.stats.success_rate,
                'avg_response_time': self.stats.avg_response_time,
                'uptime': (datetime.now() - self.created_at).total_seconds(),
                'last_activity': self.stats.last_activity.isoformat() if self.stats.last_activity else None,
            }
        }

class AgentManager:
    """Agent管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger("AgentManager")
        self.agents: Dict[str, Agent] = {}
        self.task_history: Dict[str, Task] = {}
        
        # 调度配置
        self.load_balance_strategy = "least_loaded"  # round_robin, least_loaded, capability_match
        self.max_queue_size = 1000
        
        # 统计信息
        self.total_tasks_processed = 0
        self.total_agents_created = 0
        
        # 事件回调
        self.on_agent_created: Optional[Callable] = None
        self.on_agent_removed: Optional[Callable] = None
        self.on_task_assigned: Optional[Callable] = None
        self.on_task_completed: Optional[Callable] = None
    
    async def create_agent(self, config: AgentConfig) -> bool:
        """创建Agent"""
        try:
            if config.agent_id in self.agents:
                self.logger.warning(f"Agent {config.agent_id} already exists")
                return False
            
            agent = Agent(config)
            
            # 设置事件回调
            agent.on_task_start = self._on_agent_task_start
            agent.on_task_complete = self._on_agent_task_complete
            agent.on_task_error = self._on_agent_task_error
            agent.on_status_change = self._on_agent_status_change
            
            # 初始化Agent
            if await agent.initialize():
                self.agents[config.agent_id] = agent
                self.total_agents_created += 1
                
                # 触发创建回调
                if self.on_agent_created:
                    await self.on_agent_created(agent)
                
                self.logger.info(f"Agent {config.name} created successfully")
                return True
            else:
                self.logger.error(f"Failed to initialize agent {config.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to create agent {config.name}: {e}")
            return False
    
    async def remove_agent(self, agent_id: str) -> bool:
        """移除Agent"""
        try:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            
            # 关闭Agent
            await agent.shutdown()
            
            # 移除Agent
            del self.agents[agent_id]
            
            # 触发移除回调
            if self.on_agent_removed:
                await self.on_agent_removed(agent)
            
            self.logger.info(f"Agent {agent.config.name} removed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove agent {agent_id}: {e}")
            return False
    
    async def submit_task(self, task: Task) -> bool:
        """提交任务"""
        try:
            # 选择合适的Agent
            agent = await self._select_agent(task)
            if not agent:
                self.logger.warning(f"No suitable agent found for task {task.task_id}")
                return False
            
            # 分配任务
            success = await agent.assign_task(task)
            if success:
                self.task_history[task.task_id] = task
                self.total_tasks_processed += 1
                
                # 触发任务分配回调
                if self.on_task_assigned:
                    await self.on_task_assigned(task, agent)
                
                self.logger.info(f"Task {task.task_id} assigned to agent {agent.config.name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    async def _select_agent(self, task: Task) -> Optional[Agent]:
        """选择合适的Agent"""
        # 过滤可用的Agent
        available_agents = [
            agent for agent in self.agents.values()
            if agent.can_handle_task(task)
        ]
        
        if not available_agents:
            return None
        
        # 根据策略选择Agent
        if self.load_balance_strategy == "round_robin":
            # 轮询选择
            return min(available_agents, key=lambda a: a.stats.total_tasks)
        
        elif self.load_balance_strategy == "least_loaded":
            # 选择负载最低的
            return min(available_agents, key=lambda a: a.get_load_factor())
        
        elif self.load_balance_strategy == "capability_match":
            # 选择能力最匹配的
            if task.required_capabilities:
                scored_agents = []
                for agent in available_agents:
                    score = len(set(task.required_capabilities) & set(agent.config.capabilities))
                    scored_agents.append((agent, score))
                
                scored_agents.sort(key=lambda x: x[1], reverse=True)
                return scored_agents[0][0] if scored_agents else available_agents[0]
        
        # 默认选择第一个可用的
        return available_agents[0]
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if task_id in self.task_history:
            task = self.task_history[task_id]
            return {
                'task_id': task.task_id,
                'status': task.status,
                'created_at': task.created_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'duration': task.duration.total_seconds() if task.duration else None,
                'agent_id': task.agent_id,
                'result': task.result,
                'error': task.error,
            }
        return None
    
    def get_agent_status(self, agent_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """获取Agent状态"""
        if agent_id:
            if agent_id in self.agents:
                return self.agents[agent_id].get_status_info()
            return None
        else:
            return [agent.get_status_info() for agent in self.agents.values()]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        total_current_tasks = sum(len(agent.current_tasks) for agent in self.agents.values())
        total_queued_tasks = sum(len(agent.task_queue) for agent in self.agents.values())
        
        agent_status_counts = {}
        for agent in self.agents.values():
            status = agent.status.value
            agent_status_counts[status] = agent_status_counts.get(status, 0) + 1
        
        return {
            'total_agents': len(self.agents),
            'total_tasks_processed': self.total_tasks_processed,
            'current_active_tasks': total_current_tasks,
            'queued_tasks': total_queued_tasks,
            'agent_status_distribution': agent_status_counts,
            'load_balance_strategy': self.load_balance_strategy,
        }
    
    async def _on_agent_task_start(self, task: Task):
        """Agent任务开始回调"""
        self.logger.debug(f"Task {task.task_id} started on agent {task.agent_id}")
    
    async def _on_agent_task_complete(self, task: Task):
        """Agent任务完成回调"""
        self.logger.info(f"Task {task.task_id} completed by agent {task.agent_id}")
        
        # 触发系统级回调
        if self.on_task_completed:
            await self.on_task_completed(task)
    
    async def _on_agent_task_error(self, task: Task, error: Exception):
        """Agent任务错误回调"""
        self.logger.error(f"Task {task.task_id} failed on agent {task.agent_id}: {error}")
    
    async def _on_agent_status_change(self, old_status: AgentStatus, new_status: AgentStatus):
        """Agent状态变化回调"""
        self.logger.debug(f"Agent status changed from {old_status.value} to {new_status.value}")
    
    async def shutdown(self):
        """关闭管理器"""
        self.logger.info("Shutting down AgentManager")
        
        # 关闭所有Agent
        for agent in self.agents.values():
            await agent.shutdown()
        
        self.agents.clear()
        self.logger.info("AgentManager shutdown complete")

# 便捷函数
def create_agent_config(
    name: str,
    agent_type: AgentType = AgentType.GENERAL,
    capabilities: Optional[List[str]] = None,
    **kwargs
) -> AgentConfig:
    """创建Agent配置"""
    return AgentConfig(
        agent_id=str(uuid.uuid4()),
        name=name,
        agent_type=agent_type,
        capabilities=capabilities or [],
        **kwargs
    )

def create_task(
    user_id: str,
    content: str,
    priority: TaskPriority = TaskPriority.MEDIUM,
    **kwargs
) -> Task:
    """创建任务"""
    return Task(
        task_id=str(uuid.uuid4()),
        user_id=user_id,
        content=content,
        priority=priority,
        **kwargs
    )

def create_agent_manager() -> AgentManager:
    """创建Agent管理器"""
    return AgentManager()