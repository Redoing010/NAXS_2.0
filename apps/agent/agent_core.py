# Agent核心模块
# 整合规划器、执行器、工具注册表和LLM适配器，实现完整的Agent功能

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import threading
from collections import deque

from .planner import Planner, Intent, TaskGraph, PlanningResult
from .executor import Executor, ExecutionContext, ExecutionResult
from .tool_registry import ToolRegistry
from .llm_adapter import (
    LLMAdapter, LLMRequest, LLMResponse, ChatMessage, MessageRole,
    create_llm_adapter, LLMConfig, LLMProvider
)

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent状态"""
    IDLE = "idle"              # 空闲
    THINKING = "thinking"      # 思考中
    PLANNING = "planning"      # 规划中
    EXECUTING = "executing"    # 执行中
    RESPONDING = "responding"  # 响应中
    ERROR = "error"            # 错误状态

@dataclass
class ConversationHistory:
    """对话历史"""
    messages: List[ChatMessage] = field(default_factory=list)
    max_history: int = 50
    
    def add_message(self, message: ChatMessage):
        """添加消息"""
        self.messages.append(message)
        
        # 保持历史长度限制
        if len(self.messages) > self.max_history:
            # 保留系统消息和最近的消息
            system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
            recent_messages = [msg for msg in self.messages if msg.role != MessageRole.SYSTEM][-self.max_history + len(system_messages):]
            self.messages = system_messages + recent_messages
    
    def get_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        """获取最近的消息"""
        return self.messages[-count:] if count > 0 else self.messages
    
    def clear(self):
        """清空历史"""
        # 保留系统消息
        system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
        self.messages = system_messages
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'messages': [msg.to_dict() for msg in self.messages],
            'max_history': self.max_history,
            'message_count': len(self.messages)
        }

@dataclass
class AgentMemory:
    """Agent记忆"""
    # 短期记忆（当前会话）
    working_memory: Dict[str, Any] = field(default_factory=dict)
    
    # 长期记忆（持久化）
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    
    # 用户偏好
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # 执行历史
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def remember(self, key: str, value: Any, persistent: bool = False):
        """记住信息
        
        Args:
            key: 键
            value: 值
            persistent: 是否持久化
        """
        if persistent:
            self.knowledge_base[key] = {
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'access_count': self.knowledge_base.get(key, {}).get('access_count', 0) + 1
            }
        else:
            self.working_memory[key] = value
    
    def recall(self, key: str, default: Any = None) -> Any:
        """回忆信息
        
        Args:
            key: 键
            default: 默认值
            
        Returns:
            记忆的值
        """
        # 先查工作记忆
        if key in self.working_memory:
            return self.working_memory[key]
        
        # 再查知识库
        if key in self.knowledge_base:
            entry = self.knowledge_base[key]
            entry['access_count'] = entry.get('access_count', 0) + 1
            entry['last_accessed'] = datetime.now().isoformat()
            return entry['value']
        
        return default
    
    def forget(self, key: str, persistent: bool = False):
        """忘记信息
        
        Args:
            key: 键
            persistent: 是否从持久化记忆中删除
        """
        if key in self.working_memory:
            del self.working_memory[key]
        
        if persistent and key in self.knowledge_base:
            del self.knowledge_base[key]
    
    def add_execution_record(self, task_graph_id: str, results: Dict[str, ExecutionResult]):
        """添加执行记录
        
        Args:
            task_graph_id: 任务图ID
            results: 执行结果
        """
        record = {
            'task_graph_id': task_graph_id,
            'timestamp': datetime.now().isoformat(),
            'success_count': sum(1 for r in results.values() if r.is_success),
            'total_count': len(results),
            'duration': sum(r.duration for r in results.values()),
            'summary': f"执行了{len(results)}个任务，成功{sum(1 for r in results.values() if r.is_success)}个"
        }
        
        self.execution_history.append(record)
        
        # 保持历史长度限制
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """获取用户偏好
        
        Args:
            key: 偏好键
            default: 默认值
            
        Returns:
            偏好值
        """
        return self.user_preferences.get(key, default)
    
    def set_user_preference(self, key: str, value: Any):
        """设置用户偏好
        
        Args:
            key: 偏好键
            value: 偏好值
        """
        self.user_preferences[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'working_memory': self.working_memory,
            'knowledge_base': self.knowledge_base,
            'user_preferences': self.user_preferences,
            'execution_history': self.execution_history[-10:],  # 只返回最近10条
            'stats': {
                'working_memory_size': len(self.working_memory),
                'knowledge_base_size': len(self.knowledge_base),
                'execution_count': len(self.execution_history)
            }
        }

@dataclass
class AgentConfig:
    """Agent配置"""
    # 基本信息
    name: str = "NAXS智能投研助手"
    description: str = "专业的量化投资研究助手"
    version: str = "1.0.0"
    
    # LLM配置
    llm_config: Optional[LLMConfig] = None
    
    # 系统提示词
    system_prompt: str = """
你是NAXS智能投研系统的AI助手，专门帮助用户进行量化投资研究和分析。

你的核心能力包括：
1. 市场数据分析和趋势判断
2. 个股深度研究和估值分析
3. 投资策略设计和回测
4. 风险管理和组合优化
5. 选股筛选和推荐

请始终保持专业、准确、有用的回答风格，并根据用户的具体需求提供个性化的投资建议。
当需要执行复杂分析时，我会调用相应的量化工具来获取准确的数据和结果。
"""
    
    # 行为配置
    max_thinking_time: float = 30.0  # 最大思考时间
    max_execution_time: float = 300.0  # 最大执行时间
    auto_execute: bool = True  # 是否自动执行任务
    
    # 记忆配置
    max_conversation_history: int = 50
    enable_long_term_memory: bool = True
    
    # 工具配置
    enable_tools: bool = True
    tool_timeout: float = 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'llm_config': self.llm_config.to_dict() if self.llm_config else None,
            'system_prompt': self.system_prompt,
            'max_thinking_time': self.max_thinking_time,
            'max_execution_time': self.max_execution_time,
            'auto_execute': self.auto_execute,
            'max_conversation_history': self.max_conversation_history,
            'enable_long_term_memory': self.enable_long_term_memory,
            'enable_tools': self.enable_tools,
            'tool_timeout': self.tool_timeout
        }

class Agent:
    """智能Agent
    
    整合规划、执行、工具调用和对话能力的完整Agent系统
    """
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.agent_id = str(uuid.uuid4())
        
        # 状态管理
        self.state = AgentState.IDLE
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # 核心组件
        self.llm_adapter: Optional[LLMAdapter] = None
        self.planner = Planner()
        self.executor = Executor()
        self.tool_registry = ToolRegistry()
        
        # 记忆和历史
        self.memory = AgentMemory()
        self.conversation_history = ConversationHistory(max_history=self.config.max_conversation_history)
        
        # 当前执行状态
        self.current_task_graph: Optional[TaskGraph] = None
        self.current_execution_results: Dict[str, ExecutionResult] = {}
        
        # 事件回调
        self.event_handlers: Dict[str, List[Callable]] = {
            'state_change': [],
            'message_received': [],
            'task_started': [],
            'task_completed': [],
            'error_occurred': []
        }
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 初始化系统消息
        if self.config.system_prompt:
            system_message = ChatMessage(
                role=MessageRole.SYSTEM,
                content=self.config.system_prompt
            )
            self.conversation_history.add_message(system_message)
        
        logger.info(f"Agent初始化完成: {self.config.name} (ID: {self.agent_id})")
    
    async def start(self):
        """启动Agent"""
        # 初始化LLM适配器
        if self.config.llm_config:
            self.llm_adapter = create_llm_adapter(self.config.llm_config)
            await self.llm_adapter.start()
        
        # 启动执行器
        # executor不需要特殊启动
        
        self._change_state(AgentState.IDLE)
        
        logger.info(f"Agent启动完成: {self.config.name}")
    
    async def stop(self):
        """停止Agent"""
        # 停止LLM适配器
        if self.llm_adapter:
            await self.llm_adapter.stop()
        
        # 停止执行器
        await self.executor.shutdown()
        
        logger.info(f"Agent停止: {self.config.name}")
    
    async def chat(self, message: str, user_id: str = None, session_id: str = None) -> str:
        """与Agent对话
        
        Args:
            message: 用户消息
            user_id: 用户ID
            session_id: 会话ID
            
        Returns:
            Agent回复
        """
        self.last_activity = datetime.now()
        
        # 添加用户消息到历史
        user_message = ChatMessage(role=MessageRole.USER, content=message)
        self.conversation_history.add_message(user_message)
        
        # 触发消息接收事件
        await self._trigger_event('message_received', user_message)
        
        try:
            # 状态：思考中
            self._change_state(AgentState.THINKING)
            
            # 如果启用了工具，尝试解析意图并执行任务
            if self.config.enable_tools:
                response = await self._handle_with_tools(message, user_id, session_id)
            else:
                response = await self._handle_simple_chat(message)
            
            # 添加助手回复到历史
            assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=response)
            self.conversation_history.add_message(assistant_message)
            
            self._change_state(AgentState.IDLE)
            
            return response
            
        except Exception as e:
            self._change_state(AgentState.ERROR)
            error_message = f"处理请求时发生错误: {str(e)}"
            
            # 触发错误事件
            await self._trigger_event('error_occurred', e)
            
            logger.error(f"Agent对话错误: {e}")
            
            # 添加错误回复到历史
            error_reply = ChatMessage(role=MessageRole.ASSISTANT, content=error_message)
            self.conversation_history.add_message(error_reply)
            
            return error_message
    
    async def _handle_with_tools(self, message: str, user_id: str = None, session_id: str = None) -> str:
        """使用工具处理消息
        
        Args:
            message: 用户消息
            user_id: 用户ID
            session_id: 会话ID
            
        Returns:
            处理结果
        """
        # 状态：规划中
        self._change_state(AgentState.PLANNING)
        
        # 解析意图并生成任务图
        context = {
            'user_id': user_id,
            'session_id': session_id,
            'conversation_history': self.conversation_history.get_recent_messages(5),
            'user_preferences': self.memory.user_preferences
        }
        
        planning_result = await self.planner.plan(message, context)
        
        # 检查是否需要执行任务
        if len(planning_result.task_graph.nodes) > 0 and self.config.auto_execute:
            # 状态：执行中
            self._change_state(AgentState.EXECUTING)
            
            # 触发任务开始事件
            await self._trigger_event('task_started', planning_result.task_graph)
            
            # 执行任务图
            execution_context = ExecutionContext(
                task_id="agent_execution",
                task_graph_id=planning_result.task_graph.id,
                user_id=user_id,
                session_id=session_id,
                timeout=self.config.max_execution_time
            )
            
            self.current_task_graph = planning_result.task_graph
            self.current_execution_results = await self.executor.execute_task_graph(
                planning_result.task_graph, execution_context
            )
            
            # 记录执行历史
            self.memory.add_execution_record(
                planning_result.task_graph.id,
                self.current_execution_results
            )
            
            # 触发任务完成事件
            await self._trigger_event('task_completed', self.current_execution_results)
            
            # 生成基于执行结果的回复
            response = await self._generate_response_with_results(
                message, planning_result, self.current_execution_results
            )
        else:
            # 直接使用LLM生成回复
            response = await self._generate_simple_response(message, planning_result)
        
        return response
    
    async def _handle_simple_chat(self, message: str) -> str:
        """简单对话处理
        
        Args:
            message: 用户消息
            
        Returns:
            回复内容
        """
        if not self.llm_adapter:
            return "抱歉，LLM服务未配置，无法进行对话。"
        
        # 状态：响应中
        self._change_state(AgentState.RESPONDING)
        
        # 准备对话请求
        recent_messages = self.conversation_history.get_recent_messages(10)
        
        llm_request = LLMRequest(
            messages=recent_messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        # 调用LLM
        llm_response = await self.llm_adapter.chat(llm_request)
        
        return llm_response.content
    
    async def _generate_response_with_results(
        self, 
        original_message: str, 
        planning_result: PlanningResult, 
        execution_results: Dict[str, ExecutionResult]
    ) -> str:
        """基于执行结果生成回复
        
        Args:
            original_message: 原始用户消息
            planning_result: 规划结果
            execution_results: 执行结果
            
        Returns:
            生成的回复
        """
        if not self.llm_adapter:
            return self._generate_simple_summary(execution_results)
        
        # 状态：响应中
        self._change_state(AgentState.RESPONDING)
        
        # 准备结果摘要
        results_summary = self._prepare_results_summary(execution_results)
        
        # 构建提示词
        prompt = f"""
基于以下执行结果，请为用户的问题提供专业的回答：

用户问题：{original_message}

执行的任务：{planning_result.task_graph.name}
任务描述：{planning_result.task_graph.description}

执行结果摘要：
{results_summary}

请基于这些结果，用专业但易懂的语言回答用户的问题。如果有具体的数据或建议，请明确指出。
"""
        
        # 准备消息
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.config.system_prompt),
            ChatMessage(role=MessageRole.USER, content=prompt)
        ]
        
        llm_request = LLMRequest(
            messages=messages,
            temperature=0.3,  # 降低温度以获得更准确的回答
            max_tokens=1024
        )
        
        # 调用LLM生成回复
        llm_response = await self.llm_adapter.chat(llm_request)
        
        return llm_response.content
    
    async def _generate_simple_response(self, message: str, planning_result: PlanningResult) -> str:
        """生成简单回复
        
        Args:
            message: 用户消息
            planning_result: 规划结果
            
        Returns:
            回复内容
        """
        if not self.llm_adapter:
            return f"我理解您想要{planning_result.intent.type.value}，但当前无法执行具体的分析任务。"
        
        # 状态：响应中
        self._change_state(AgentState.RESPONDING)
        
        # 使用对话历史生成回复
        recent_messages = self.conversation_history.get_recent_messages(10)
        
        llm_request = LLMRequest(
            messages=recent_messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        llm_response = await self.llm_adapter.chat(llm_request)
        
        return llm_response.content
    
    def _prepare_results_summary(self, execution_results: Dict[str, ExecutionResult]) -> str:
        """准备执行结果摘要
        
        Args:
            execution_results: 执行结果
            
        Returns:
            结果摘要文本
        """
        summary_parts = []
        
        successful_tasks = [r for r in execution_results.values() if r.is_success]
        failed_tasks = [r for r in execution_results.values() if r.is_failed]
        
        summary_parts.append(f"总共执行了{len(execution_results)}个任务")
        summary_parts.append(f"成功完成{len(successful_tasks)}个，失败{len(failed_tasks)}个")
        
        # 添加成功任务的结果
        for result in successful_tasks:
            if result.data:
                task_name = result.task_id.replace('_', ' ').title()
                summary_parts.append(f"\n{task_name}:")
                
                # 简化数据展示
                if isinstance(result.data, dict):
                    for key, value in result.data.items():
                        if isinstance(value, (str, int, float, bool)):
                            summary_parts.append(f"  - {key}: {value}")
                        elif isinstance(value, dict) and len(value) <= 3:
                            summary_parts.append(f"  - {key}: {json.dumps(value, ensure_ascii=False)}")
        
        # 添加失败任务的错误信息
        if failed_tasks:
            summary_parts.append("\n执行中遇到的问题:")
            for result in failed_tasks:
                summary_parts.append(f"  - {result.task_id}: {result.error}")
        
        return "\n".join(summary_parts)
    
    def _generate_simple_summary(self, execution_results: Dict[str, ExecutionResult]) -> str:
        """生成简单摘要（无LLM）
        
        Args:
            execution_results: 执行结果
            
        Returns:
            简单摘要
        """
        successful_count = sum(1 for r in execution_results.values() if r.is_success)
        total_count = len(execution_results)
        
        if successful_count == total_count:
            return f"已成功完成您的请求，执行了{total_count}个分析任务。"
        elif successful_count > 0:
            return f"部分完成您的请求，{total_count}个任务中成功执行了{successful_count}个。"
        else:
            return "抱歉，执行过程中遇到了问题，请稍后重试。"
    
    def _change_state(self, new_state: AgentState):
        """改变Agent状态
        
        Args:
            new_state: 新状态
        """
        old_state = self.state
        self.state = new_state
        self.last_activity = datetime.now()
        
        logger.debug(f"Agent状态变更: {old_state.value} -> {new_state.value}")
        
        # 触发状态变更事件
        asyncio.create_task(self._trigger_event('state_change', old_state, new_state))
    
    async def _trigger_event(self, event_type: str, *args):
        """触发事件
        
        Args:
            event_type: 事件类型
            *args: 事件参数
        """
        handlers = self.event_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args)
                else:
                    handler(*args)
            except Exception as e:
                logger.error(f"事件处理器执行失败: {event_type} - {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """添加事件处理器
        
        Args:
            event_type: 事件类型
            handler: 处理函数
        """
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    def get_status(self) -> Dict[str, Any]:
        """获取Agent状态
        
        Returns:
            状态信息
        """
        return {
            'agent_id': self.agent_id,
            'name': self.config.name,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'conversation_messages': len(self.conversation_history.messages),
            'current_task_graph': self.current_task_graph.name if self.current_task_graph else None,
            'executor_stats': self.executor.get_statistics(),
            'tool_stats': self.tool_registry.get_statistics(),
            'memory_stats': {
                'working_memory_size': len(self.memory.working_memory),
                'knowledge_base_size': len(self.memory.knowledge_base),
                'execution_history_count': len(self.memory.execution_history)
            }
        }
    
    def get_conversation_history(self) -> Dict[str, Any]:
        """获取对话历史
        
        Returns:
            对话历史
        """
        return self.conversation_history.to_dict()
    
    def get_memory(self) -> Dict[str, Any]:
        """获取记忆信息
        
        Returns:
            记忆信息
        """
        return self.memory.to_dict()
    
    def clear_conversation(self):
        """清空对话历史"""
        self.conversation_history.clear()
        logger.info(f"Agent对话历史已清空: {self.config.name}")
    
    def reset_memory(self, keep_preferences: bool = True):
        """重置记忆
        
        Args:
            keep_preferences: 是否保留用户偏好
        """
        preferences = self.memory.user_preferences.copy() if keep_preferences else {}
        self.memory = AgentMemory()
        if keep_preferences:
            self.memory.user_preferences = preferences
        
        logger.info(f"Agent记忆已重置: {self.config.name}")

# 便捷函数
def create_agent(config: AgentConfig = None) -> Agent:
    """创建Agent实例
    
    Args:
        config: Agent配置
        
    Returns:
        Agent实例
    """
    return Agent(config)

def create_agent_config(
    name: str = "NAXS智能投研助手",
    llm_provider: LLMProvider = LLMProvider.OPENAI,
    llm_model: str = "gpt-3.5-turbo",
    api_key: str = None,
    **kwargs
) -> AgentConfig:
    """创建Agent配置
    
    Args:
        name: Agent名称
        llm_provider: LLM提供商
        llm_model: LLM模型
        api_key: API密钥
        **kwargs: 其他配置参数
        
    Returns:
        Agent配置实例
    """
    llm_config = None
    if api_key:
        llm_config = LLMConfig(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key
        )
    
    return AgentConfig(
        name=name,
        llm_config=llm_config,
        **kwargs
    )