# 工作流引擎 - 实现复杂的工作流编排和执行

import logging
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import uuid
import json
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import networkx as nx
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """工作流状态"""
    DRAFT = "draft"            # 草稿
    READY = "ready"            # 就绪
    RUNNING = "running"        # 运行中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"    # 已取消
    PAUSED = "paused"          # 暂停

class NodeStatus(Enum):
    """节点状态"""
    PENDING = "pending"        # 等待中
    READY = "ready"            # 就绪
    RUNNING = "running"        # 运行中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败
    SKIPPED = "skipped"        # 跳过
    CANCELLED = "cancelled"    # 已取消

class NodeType(Enum):
    """节点类型"""
    TASK = "task"              # 任务节点
    CONDITION = "condition"    # 条件节点
    PARALLEL = "parallel"      # 并行节点
    SEQUENTIAL = "sequential"  # 顺序节点
    SUBWORKFLOW = "subworkflow" # 子工作流节点
    HUMAN = "human"            # 人工节点
    TIMER = "timer"            # 定时器节点

class EdgeType(Enum):
    """边类型"""
    NORMAL = "normal"          # 普通边
    CONDITIONAL = "conditional" # 条件边
    ERROR = "error"            # 错误边
    TIMEOUT = "timeout"        # 超时边

@dataclass
class NodeResult:
    """节点执行结果"""
    node_id: str
    status: NodeStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'node_id': self.node_id,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'result': self.result,
            'error': self.error,
            'metadata': self.metadata
        }

@dataclass
class WorkflowContext:
    """工作流上下文"""
    workflow_id: str
    variables: Dict[str, Any] = field(default_factory=dict)
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """获取变量"""
        return self.variables.get(name, default)
    
    def set_variable(self, name: str, value: Any):
        """设置变量"""
        self.variables[name] = value
    
    def get_node_result(self, node_id: str) -> Optional[NodeResult]:
        """获取节点结果"""
        return self.node_results.get(node_id)
    
    def set_node_result(self, node_id: str, result: NodeResult):
        """设置节点结果"""
        self.node_results[node_id] = result

@dataclass
class WorkflowEdge:
    """工作流边"""
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.NORMAL
    condition: Optional[Callable[[WorkflowContext], bool]] = None
    condition_expression: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_traverse(self, context: WorkflowContext) -> bool:
        """判断是否应该遍历此边
        
        Args:
            context: 工作流上下文
            
        Returns:
            是否应该遍历
        """
        if self.edge_type == EdgeType.NORMAL:
            return True
        elif self.edge_type == EdgeType.CONDITIONAL:
            if self.condition:
                return self.condition(context)
            elif self.condition_expression:
                return self._evaluate_expression(context)
            else:
                return True
        else:
            return True
    
    def _evaluate_expression(self, context: WorkflowContext) -> bool:
        """评估条件表达式"""
        try:
            # 简单的表达式评估（生产环境应使用更安全的方法）
            local_vars = {
                'context': context,
                'variables': context.variables,
                'results': context.node_results
            }
            return eval(self.condition_expression, {"__builtins__": {}}, local_vars)
        except Exception as e:
            logger.error(f"条件表达式评估失败: {self.condition_expression}, 错误: {str(e)}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type.value,
            'condition_expression': self.condition_expression,
            'metadata': self.metadata
        }

@dataclass
class WorkflowNode:
    """工作流节点"""
    id: str
    name: str
    node_type: NodeType
    func: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0
    retry_delay: float = 1.0
    condition: Optional[Callable[[WorkflowContext], bool]] = None
    condition_expression: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 运行时状态
    status: NodeStatus = NodeStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    
    def should_execute(self, context: WorkflowContext) -> bool:
        """判断是否应该执行此节点
        
        Args:
            context: 工作流上下文
            
        Returns:
            是否应该执行
        """
        if self.condition:
            return self.condition(context)
        elif self.condition_expression:
            return self._evaluate_expression(context)
        else:
            return True
    
    def _evaluate_expression(self, context: WorkflowContext) -> bool:
        """评估条件表达式"""
        try:
            local_vars = {
                'context': context,
                'variables': context.variables,
                'results': context.node_results
            }
            return eval(self.condition_expression, {"__builtins__": {}}, local_vars)
        except Exception as e:
            logger.error(f"节点条件表达式评估失败: {self.condition_expression}, 错误: {str(e)}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'node_type': self.node_type.value,
            'args': self.args,
            'kwargs': self.kwargs,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'condition_expression': self.condition_expression,
            'metadata': self.metadata,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'result': self.result,
            'error': self.error
        }

@dataclass
class Workflow:
    """工作流定义"""
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    edges: Dict[str, WorkflowEdge] = field(default_factory=dict)
    start_nodes: List[str] = field(default_factory=list)
    end_nodes: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    max_parallel: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 运行时状态
    status: WorkflowStatus = WorkflowStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_node(self, node: WorkflowNode):
        """添加节点
        
        Args:
            node: 工作流节点
        """
        self.nodes[node.id] = node
        self.updated_at = datetime.now()
    
    def add_edge(self, edge: WorkflowEdge):
        """添加边
        
        Args:
            edge: 工作流边
        """
        self.edges[edge.id] = edge
        self.updated_at = datetime.now()
    
    def remove_node(self, node_id: str):
        """移除节点
        
        Args:
            node_id: 节点ID
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            
            # 移除相关的边
            edges_to_remove = []
            for edge_id, edge in self.edges.items():
                if edge.source_id == node_id or edge.target_id == node_id:
                    edges_to_remove.append(edge_id)
            
            for edge_id in edges_to_remove:
                del self.edges[edge_id]
            
            # 从开始和结束节点列表中移除
            if node_id in self.start_nodes:
                self.start_nodes.remove(node_id)
            if node_id in self.end_nodes:
                self.end_nodes.remove(node_id)
            
            self.updated_at = datetime.now()
    
    def remove_edge(self, edge_id: str):
        """移除边
        
        Args:
            edge_id: 边ID
        """
        if edge_id in self.edges:
            del self.edges[edge_id]
            self.updated_at = datetime.now()
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证工作流
        
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查是否有节点
        if not self.nodes:
            errors.append("工作流没有节点")
            return False, errors
        
        # 检查开始节点
        if not self.start_nodes:
            errors.append("工作流没有开始节点")
        
        # 检查节点是否存在
        for node_id in self.start_nodes:
            if node_id not in self.nodes:
                errors.append(f"开始节点不存在: {node_id}")
        
        for node_id in self.end_nodes:
            if node_id not in self.nodes:
                errors.append(f"结束节点不存在: {node_id}")
        
        # 检查边的节点是否存在
        for edge in self.edges.values():
            if edge.source_id not in self.nodes:
                errors.append(f"边的源节点不存在: {edge.source_id}")
            if edge.target_id not in self.nodes:
                errors.append(f"边的目标节点不存在: {edge.target_id}")
        
        # 检查是否有环路（使用NetworkX）
        try:
            graph = nx.DiGraph()
            for node_id in self.nodes.keys():
                graph.add_node(node_id)
            for edge in self.edges.values():
                graph.add_edge(edge.source_id, edge.target_id)
            
            if not nx.is_directed_acyclic_graph(graph):
                errors.append("工作流包含环路")
        except Exception as e:
            errors.append(f"图结构验证失败: {str(e)}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            'start_nodes': self.start_nodes,
            'end_nodes': self.end_nodes,
            'timeout': self.timeout,
            'max_parallel': self.max_parallel,
            'metadata': self.metadata,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class NodeExecutor(ABC):
    """节点执行器基类"""
    
    @abstractmethod
    async def execute(self, node: WorkflowNode, context: WorkflowContext) -> NodeResult:
        """执行节点
        
        Args:
            node: 工作流节点
            context: 工作流上下文
            
        Returns:
            节点执行结果
        """
        pass

class TaskNodeExecutor(NodeExecutor):
    """任务节点执行器"""
    
    async def execute(self, node: WorkflowNode, context: WorkflowContext) -> NodeResult:
        """执行任务节点"""
        start_time = datetime.now()
        result = NodeResult(
            node_id=node.id,
            status=NodeStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            if not node.func:
                raise ValueError("任务节点没有指定执行函数")
            
            # 执行任务函数
            if asyncio.iscoroutinefunction(node.func):
                # 异步函数
                if node.timeout:
                    task_result = await asyncio.wait_for(
                        node.func(*node.args, **node.kwargs, context=context),
                        timeout=node.timeout
                    )
                else:
                    task_result = await node.func(*node.args, **node.kwargs, context=context)
            else:
                # 同步函数
                loop = asyncio.get_event_loop()
                if node.timeout:
                    task_result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, 
                            lambda: node.func(*node.args, **node.kwargs, context=context)
                        ),
                        timeout=node.timeout
                    )
                else:
                    task_result = await loop.run_in_executor(
                        None, 
                        lambda: node.func(*node.args, **node.kwargs, context=context)
                    )
            
            # 任务成功完成
            end_time = datetime.now()
            result.status = NodeStatus.COMPLETED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.result = task_result
            
        except asyncio.TimeoutError:
            # 超时
            end_time = datetime.now()
            result.status = NodeStatus.FAILED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.error = f"任务执行超时: {node.timeout}秒"
            
        except Exception as e:
            # 执行失败
            end_time = datetime.now()
            result.status = NodeStatus.FAILED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.error = str(e)
        
        return result

class ConditionNodeExecutor(NodeExecutor):
    """条件节点执行器"""
    
    async def execute(self, node: WorkflowNode, context: WorkflowContext) -> NodeResult:
        """执行条件节点"""
        start_time = datetime.now()
        result = NodeResult(
            node_id=node.id,
            status=NodeStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # 评估条件
            condition_result = node.should_execute(context)
            
            end_time = datetime.now()
            result.status = NodeStatus.COMPLETED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.result = condition_result
            
        except Exception as e:
            end_time = datetime.now()
            result.status = NodeStatus.FAILED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.error = str(e)
        
        return result

class ParallelNodeExecutor(NodeExecutor):
    """并行节点执行器"""
    
    async def execute(self, node: WorkflowNode, context: WorkflowContext) -> NodeResult:
        """执行并行节点"""
        start_time = datetime.now()
        result = NodeResult(
            node_id=node.id,
            status=NodeStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # 并行节点通常用于控制流程，不执行具体任务
            end_time = datetime.now()
            result.status = NodeStatus.COMPLETED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.result = "parallel_completed"
            
        except Exception as e:
            end_time = datetime.now()
            result.status = NodeStatus.FAILED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.error = str(e)
        
        return result

class TimerNodeExecutor(NodeExecutor):
    """定时器节点执行器"""
    
    async def execute(self, node: WorkflowNode, context: WorkflowContext) -> NodeResult:
        """执行定时器节点"""
        start_time = datetime.now()
        result = NodeResult(
            node_id=node.id,
            status=NodeStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # 获取等待时间
            wait_time = node.kwargs.get('wait_time', 1.0)
            
            # 等待指定时间
            await asyncio.sleep(wait_time)
            
            end_time = datetime.now()
            result.status = NodeStatus.COMPLETED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.result = f"waited_{wait_time}_seconds"
            
        except Exception as e:
            end_time = datetime.now()
            result.status = NodeStatus.FAILED
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.error = str(e)
        
        return result

class WorkflowEngine:
    """工作流引擎
    
    提供工作流编排和执行功能，包括：
    1. 工作流定义管理
    2. 节点执行器管理
    3. 工作流执行控制
    4. 并发执行支持
    5. 错误处理和重试
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.workflows: Dict[str, Workflow] = {}
        self.executors: Dict[NodeType, NodeExecutor] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.workflow_contexts: Dict[str, WorkflowContext] = {}
        
        # 配置参数
        self.max_concurrent_workflows = self.config.get('max_concurrent_workflows', 10)
        self.default_timeout = self.config.get('default_timeout', 3600)  # 1小时
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 注册默认执行器
        self._register_default_executors()
        
        logger.info(f"工作流引擎初始化完成")
    
    def _register_default_executors(self):
        """注册默认执行器"""
        self.executors[NodeType.TASK] = TaskNodeExecutor()
        self.executors[NodeType.CONDITION] = ConditionNodeExecutor()
        self.executors[NodeType.PARALLEL] = ParallelNodeExecutor()
        self.executors[NodeType.TIMER] = TimerNodeExecutor()
    
    def register_executor(self, node_type: NodeType, executor: NodeExecutor):
        """注册节点执行器
        
        Args:
            node_type: 节点类型
            executor: 节点执行器
        """
        self.executors[node_type] = executor
        logger.debug(f"注册节点执行器: {node_type.value}")
    
    def register_workflow(self, workflow: Workflow) -> bool:
        """注册工作流
        
        Args:
            workflow: 工作流对象
            
        Returns:
            是否成功注册
        """
        try:
            # 验证工作流
            is_valid, errors = workflow.validate()
            if not is_valid:
                logger.error(f"工作流验证失败: {workflow.id}, 错误: {errors}")
                return False
            
            with self.lock:
                self.workflows[workflow.id] = workflow
                workflow.status = WorkflowStatus.READY
                
            logger.info(f"注册工作流成功: {workflow.id} ({workflow.name})")
            return True
            
        except Exception as e:
            logger.error(f"注册工作流失败: {workflow.id}, 错误: {str(e)}")
            return False
    
    def unregister_workflow(self, workflow_id: str) -> bool:
        """注销工作流
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            是否成功注销
        """
        try:
            with self.lock:
                if workflow_id not in self.workflows:
                    logger.warning(f"工作流不存在: {workflow_id}")
                    return False
                
                # 停止正在运行的工作流
                if workflow_id in self.running_workflows:
                    task = self.running_workflows[workflow_id]
                    task.cancel()
                    del self.running_workflows[workflow_id]
                
                # 清理上下文
                if workflow_id in self.workflow_contexts:
                    del self.workflow_contexts[workflow_id]
                
                # 移除工作流
                del self.workflows[workflow_id]
                
            logger.info(f"注销工作流成功: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"注销工作流失败: {workflow_id}, 错误: {str(e)}")
            return False
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """获取工作流
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            工作流对象
        """
        return self.workflows.get(workflow_id)
    
    def list_workflows(self, status: Optional[WorkflowStatus] = None) -> List[Workflow]:
        """列出工作流
        
        Args:
            status: 状态过滤
            
        Returns:
            工作流列表
        """
        with self.lock:
            workflows = list(self.workflows.values())
            
            if status:
                workflows = [w for w in workflows if w.status == status]
            
            return workflows
    
    async def execute_workflow(self, workflow_id: str, 
                              initial_variables: Dict[str, Any] = None) -> WorkflowContext:
        """执行工作流
        
        Args:
            workflow_id: 工作流ID
            initial_variables: 初始变量
            
        Returns:
            工作流上下文
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"工作流不存在: {workflow_id}")
        
        if workflow.status != WorkflowStatus.READY:
            raise ValueError(f"工作流状态不正确: {workflow.status}")
        
        # 检查并发限制
        if len(self.running_workflows) >= self.max_concurrent_workflows:
            raise RuntimeError("达到最大并发工作流数量限制")
        
        # 创建工作流上下文
        context = WorkflowContext(
            workflow_id=workflow_id,
            variables=initial_variables or {},
            metadata={'start_time': datetime.now().isoformat()}
        )
        
        with self.lock:
            self.workflow_contexts[workflow_id] = context
            workflow.status = WorkflowStatus.RUNNING
        
        try:
            logger.info(f"开始执行工作流: {workflow_id} ({workflow.name})")
            
            # 创建执行任务
            task = asyncio.create_task(self._execute_workflow_internal(workflow, context))
            
            with self.lock:
                self.running_workflows[workflow_id] = task
            
            # 等待执行完成
            await task
            
            # 更新工作流状态
            if all(node.status == NodeStatus.COMPLETED for node in workflow.nodes.values()):
                workflow.status = WorkflowStatus.COMPLETED
            else:
                workflow.status = WorkflowStatus.FAILED
            
            logger.info(f"工作流执行完成: {workflow_id}, 状态: {workflow.status.value}")
            
        except asyncio.CancelledError:
            workflow.status = WorkflowStatus.CANCELLED
            logger.info(f"工作流被取消: {workflow_id}")
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            logger.error(f"工作流执行失败: {workflow_id}, 错误: {str(e)}")
            
        finally:
            # 清理运行状态
            with self.lock:
                if workflow_id in self.running_workflows:
                    del self.running_workflows[workflow_id]
        
        return context
    
    async def _execute_workflow_internal(self, workflow: Workflow, context: WorkflowContext):
        """内部工作流执行逻辑"""
        # 构建执行图
        execution_graph = self._build_execution_graph(workflow)
        
        # 初始化节点状态
        for node in workflow.nodes.values():
            node.status = NodeStatus.PENDING
        
        # 获取可执行的节点
        ready_nodes = self._get_ready_nodes(workflow, execution_graph)
        
        # 执行节点
        while ready_nodes:
            # 并行执行就绪的节点
            tasks = []
            for node_id in ready_nodes:
                node = workflow.nodes[node_id]
                if node.should_execute(context):
                    task = asyncio.create_task(self._execute_node(node, context))
                    tasks.append((node_id, task))
                else:
                    # 跳过节点
                    node.status = NodeStatus.SKIPPED
                    result = NodeResult(
                        node_id=node_id,
                        status=NodeStatus.SKIPPED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        duration=0.0
                    )
                    context.set_node_result(node_id, result)
            
            # 等待所有任务完成
            if tasks:
                for node_id, task in tasks:
                    try:
                        result = await task
                        context.set_node_result(node_id, result)
                        
                        # 更新节点状态
                        node = workflow.nodes[node_id]
                        node.status = result.status
                        node.start_time = result.start_time
                        node.end_time = result.end_time
                        node.result = result.result
                        node.error = result.error
                        
                    except Exception as e:
                        logger.error(f"节点执行异常: {node_id}, 错误: {str(e)}")
                        node = workflow.nodes[node_id]
                        node.status = NodeStatus.FAILED
                        node.error = str(e)
            
            # 获取下一批可执行的节点
            ready_nodes = self._get_ready_nodes(workflow, execution_graph)
    
    async def _execute_node(self, node: WorkflowNode, context: WorkflowContext) -> NodeResult:
        """执行单个节点"""
        executor = self.executors.get(node.node_type)
        if not executor:
            raise ValueError(f"没有找到节点类型的执行器: {node.node_type}")
        
        # 执行节点（包含重试逻辑）
        for attempt in range(node.max_retries + 1):
            try:
                result = await executor.execute(node, context)
                
                if result.status == NodeStatus.COMPLETED:
                    return result
                elif attempt < node.max_retries:
                    # 重试
                    logger.info(f"节点执行失败，准备重试: {node.id}, 尝试次数: {attempt + 1}/{node.max_retries + 1}")
                    await asyncio.sleep(node.retry_delay)
                else:
                    # 重试次数用尽
                    return result
                    
            except Exception as e:
                if attempt < node.max_retries:
                    logger.info(f"节点执行异常，准备重试: {node.id}, 错误: {str(e)}, 尝试次数: {attempt + 1}/{node.max_retries + 1}")
                    await asyncio.sleep(node.retry_delay)
                else:
                    # 重试次数用尽
                    result = NodeResult(
                        node_id=node.id,
                        status=NodeStatus.FAILED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error=str(e)
                    )
                    return result
        
        # 不应该到达这里
        return NodeResult(
            node_id=node.id,
            status=NodeStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            error="未知错误"
        )
    
    def _build_execution_graph(self, workflow: Workflow) -> nx.DiGraph:
        """构建执行图"""
        graph = nx.DiGraph()
        
        # 添加节点
        for node_id in workflow.nodes.keys():
            graph.add_node(node_id)
        
        # 添加边
        for edge in workflow.edges.values():
            graph.add_edge(edge.source_id, edge.target_id, edge_data=edge)
        
        return graph
    
    def _get_ready_nodes(self, workflow: Workflow, graph: nx.DiGraph) -> List[str]:
        """获取就绪的节点"""
        ready_nodes = []
        
        for node_id, node in workflow.nodes.items():
            if node.status != NodeStatus.PENDING:
                continue
            
            # 检查前置节点是否都已完成
            predecessors = list(graph.predecessors(node_id))
            if not predecessors:
                # 没有前置节点，是开始节点
                if node_id in workflow.start_nodes:
                    ready_nodes.append(node_id)
            else:
                # 检查所有前置节点是否已完成
                all_predecessors_completed = True
                for pred_id in predecessors:
                    pred_node = workflow.nodes[pred_id]
                    if pred_node.status not in [NodeStatus.COMPLETED, NodeStatus.SKIPPED]:
                        all_predecessors_completed = False
                        break
                    
                    # 检查边的条件
                    edge_data = graph.get_edge_data(pred_id, node_id)
                    if edge_data and 'edge_data' in edge_data:
                        edge = edge_data['edge_data']
                        context = self.workflow_contexts.get(workflow.id)
                        if context and not edge.should_traverse(context):
                            all_predecessors_completed = False
                            break
                
                if all_predecessors_completed:
                    ready_nodes.append(node_id)
        
        return ready_nodes
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流执行
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            是否成功取消
        """
        try:
            with self.lock:
                if workflow_id not in self.running_workflows:
                    logger.warning(f"工作流未在运行: {workflow_id}")
                    return False
                
                task = self.running_workflows[workflow_id]
                task.cancel()
                
                # 更新工作流状态
                workflow = self.workflows.get(workflow_id)
                if workflow:
                    workflow.status = WorkflowStatus.CANCELLED
                
            logger.info(f"取消工作流成功: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消工作流失败: {workflow_id}, 错误: {str(e)}")
            return False
    
    def get_workflow_context(self, workflow_id: str) -> Optional[WorkflowContext]:
        """获取工作流上下文
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            工作流上下文
        """
        return self.workflow_contexts.get(workflow_id)
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息
        
        Returns:
            统计信息
        """
        with self.lock:
            status_counts = {}
            for workflow in self.workflows.values():
                status = workflow.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'total_workflows': len(self.workflows),
                'running_workflows': len(self.running_workflows),
                'status_distribution': status_counts,
                'max_concurrent_workflows': self.max_concurrent_workflows,
                'registered_executors': list(self.executors.keys())
            }

# 便捷函数
def create_workflow_engine(config: Dict[str, Any] = None) -> WorkflowEngine:
    """创建工作流引擎实例
    
    Args:
        config: 配置字典
        
    Returns:
        工作流引擎实例
    """
    return WorkflowEngine(config)

def create_task_node(node_id: str, name: str, func: Callable, 
                    *args, **kwargs) -> WorkflowNode:
    """创建任务节点
    
    Args:
        node_id: 节点ID
        name: 节点名称
        func: 执行函数
        *args: 函数参数
        **kwargs: 函数关键字参数
        
    Returns:
        工作流节点
    """
    return WorkflowNode(
        id=node_id,
        name=name,
        node_type=NodeType.TASK,
        func=func,
        args=args,
        kwargs=kwargs
    )

def create_condition_node(node_id: str, name: str, 
                         condition: Optional[Callable] = None,
                         condition_expression: str = "") -> WorkflowNode:
    """创建条件节点
    
    Args:
        node_id: 节点ID
        name: 节点名称
        condition: 条件函数
        condition_expression: 条件表达式
        
    Returns:
        工作流节点
    """
    return WorkflowNode(
        id=node_id,
        name=name,
        node_type=NodeType.CONDITION,
        condition=condition,
        condition_expression=condition_expression
    )

def create_timer_node(node_id: str, name: str, wait_time: float) -> WorkflowNode:
    """创建定时器节点
    
    Args:
        node_id: 节点ID
        name: 节点名称
        wait_time: 等待时间（秒）
        
    Returns:
        工作流节点
    """
    return WorkflowNode(
        id=node_id,
        name=name,
        node_type=NodeType.TIMER,
        kwargs={'wait_time': wait_time}
    )

def create_workflow_edge(edge_id: str, source_id: str, target_id: str,
                        edge_type: EdgeType = EdgeType.NORMAL,
                        condition: Optional[Callable] = None,
                        condition_expression: str = "") -> WorkflowEdge:
    """创建工作流边
    
    Args:
        edge_id: 边ID
        source_id: 源节点ID
        target_id: 目标节点ID
        edge_type: 边类型
        condition: 条件函数
        condition_expression: 条件表达式
        
    Returns:
        工作流边
    """
    return WorkflowEdge(
        id=edge_id,
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        condition=condition,
        condition_expression=condition_expression
    )