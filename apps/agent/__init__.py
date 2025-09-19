# Agent中台系统
# 智能投研Agent调度和管理平台

from .planner import Planner, TaskPlanner, IntentParser
from .executor import Executor, TaskExecutor, ToolExecutor
from .tool_registry import ToolRegistry, Tool, ToolCategory
from .enhanced_tool_registry import (
    EnhancedToolRegistry,
    ExecutionStatus,
    ChainExecutionMode,
    ToolExecution,
    ToolChain,
    create_enhanced_tool_registry
)
from .agent_manager import AgentManager, Agent, AgentConfig
from .llm_interface import LLMInterface, LLMProvider, ModelConfig
from .context_manager import ContextManager, ConversationContext
from .workflow_engine import WorkflowEngine, Workflow, WorkflowStep
from .memory_manager import MemoryManager, ShortTermMemory, LongTermMemory
from .reasoning_engine import ReasoningEngine, ReasoningStrategy
from .knowledge_integration import KnowledgeIntegrator

# Agent类型
__all__ = [
    # 核心组件
    'Planner',
    'TaskPlanner',
    'IntentParser',
    'Executor',
    'TaskExecutor', 
    'ToolExecutor',
    'ToolRegistry',
    'Tool',
    'ToolCategory',
    'EnhancedToolRegistry',
    'ExecutionStatus',
    'ChainExecutionMode',
    'ToolExecution',
    'ToolChain',
    'create_enhanced_tool_registry',
    
    # 管理器
    'AgentManager',
    'Agent',
    'AgentConfig',
    
    # LLM接口
    'LLMInterface',
    'LLMProvider',
    'ModelConfig',
    
    # 上下文和工作流
    'ContextManager',
    'ConversationContext',
    'WorkflowEngine',
    'Workflow',
    'WorkflowStep',
    
    # 记忆和推理
    'MemoryManager',
    'ShortTermMemory',
    'LongTermMemory',
    'ReasoningEngine',
    'ReasoningStrategy',
    
    # 知识集成
    'KnowledgeIntegrator',
]

__version__ = '1.0.0'
__description__ = 'Intelligent Agent Platform for Investment Research'