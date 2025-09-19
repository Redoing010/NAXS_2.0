# 工具注册表 - 管理所有可用的量化分析工具

import logging
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    description: str
    version: str
    category: str
    capabilities: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str] = None
    timeout: int = 300
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class BaseTool(ABC):
    """工具基类
    
    所有量化分析工具都应继承此类
    """
    
    def __init__(self):
        self.metadata = self.get_metadata()
        self.is_initialized = False
        self.last_used = None
    
    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """获取工具元数据"""
        pass
    
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具功能
        
        Args:
            params: 执行参数
            
        Returns:
            执行结果
        """
        pass
    
    async def initialize(self) -> bool:
        """初始化工具"""
        try:
            await self._initialize_impl()
            self.is_initialized = True
            logger.info(f"工具初始化成功: {self.metadata.name}")
            return True
        except Exception as e:
            logger.error(f"工具初始化失败: {self.metadata.name}, 错误: {str(e)}")
            return False
    
    async def _initialize_impl(self):
        """子类可重写的初始化实现"""
        pass
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            return await self._health_check_impl()
        except Exception as e:
            logger.error(f"工具健康检查失败: {self.metadata.name}, 错误: {str(e)}")
            return False
    
    async def _health_check_impl(self) -> bool:
        """子类可重写的健康检查实现"""
        return self.is_initialized
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """验证输入参数"""
        # TODO: 根据input_schema验证参数
        return True
    
    def update_usage(self):
        """更新使用时间"""
        self.last_used = datetime.now()

class ToolRegistry:
    """工具注册表
    
    管理所有可用的量化分析工具，提供工具发现、注册、调用等功能
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
        self._capabilities: Dict[str, List[str]] = {}
        self._tool_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info("工具注册表初始化完成")
    
    def register(self, tool: BaseTool) -> bool:
        """注册工具
        
        Args:
            tool: 要注册的工具实例
            
        Returns:
            注册是否成功
        """
        try:
            metadata = tool.metadata
            tool_name = metadata.name
            
            # 检查工具是否已存在
            if tool_name in self._tools:
                logger.warning(f"工具已存在，将被覆盖: {tool_name}")
            
            # 注册工具
            self._tools[tool_name] = tool
            
            # 更新分类索引
            category = metadata.category
            if category not in self._categories:
                self._categories[category] = []
            if tool_name not in self._categories[category]:
                self._categories[category].append(tool_name)
            
            # 更新能力索引
            for capability in metadata.capabilities:
                if capability not in self._capabilities:
                    self._capabilities[capability] = []
                if tool_name not in self._capabilities[capability]:
                    self._capabilities[capability].append(tool_name)
            
            # 初始化统计信息
            self._tool_stats[tool_name] = {
                'registered_at': datetime.now(),
                'usage_count': 0,
                'success_count': 0,
                'error_count': 0,
                'avg_execution_time': 0.0
            }
            
            logger.info(f"工具注册成功: {tool_name} (类别: {category})")
            return True
            
        except Exception as e:
            logger.error(f"工具注册失败: {str(e)}")
            return False
    
    def unregister(self, tool_name: str) -> bool:
        """注销工具"""
        try:
            if tool_name not in self._tools:
                logger.warning(f"工具不存在: {tool_name}")
                return False
            
            tool = self._tools[tool_name]
            metadata = tool.metadata
            
            # 从索引中移除
            category = metadata.category
            if category in self._categories:
                self._categories[category] = [
                    name for name in self._categories[category] if name != tool_name
                ]
            
            for capability in metadata.capabilities:
                if capability in self._capabilities:
                    self._capabilities[capability] = [
                        name for name in self._capabilities[capability] if name != tool_name
                    ]
            
            # 移除工具和统计信息
            del self._tools[tool_name]
            del self._tool_stats[tool_name]
            
            logger.info(f"工具注销成功: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"工具注销失败: {str(e)}")
            return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self._tools.get(tool_name)
    
    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否存在"""
        return tool_name in self._tools
    
    def list_tools(self) -> List[str]:
        """获取所有工具名称列表"""
        return list(self._tools.keys())
    
    def list_tools_by_category(self, category: str) -> List[str]:
        """按类别获取工具列表"""
        return self._categories.get(category, [])
    
    def list_tools_by_capability(self, capability: str) -> List[str]:
        """按能力获取工具列表"""
        return self._capabilities.get(capability, [])
    
    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """获取工具元数据"""
        tool = self.get_tool(tool_name)
        return tool.metadata if tool else None
    
    def search_tools(self, query: str) -> List[str]:
        """搜索工具
        
        Args:
            query: 搜索关键词
            
        Returns:
            匹配的工具名称列表
        """
        query_lower = query.lower()
        matched_tools = []
        
        for tool_name, tool in self._tools.items():
            metadata = tool.metadata
            
            # 搜索工具名称、描述、能力
            if (query_lower in tool_name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in cap.lower() for cap in metadata.capabilities)):
                matched_tools.append(tool_name)
        
        return matched_tools
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具
        
        Args:
            tool_name: 工具名称
            params: 执行参数
            
        Returns:
            执行结果
        """
        start_time = datetime.now()
        
        try:
            # 获取工具
            tool = self.get_tool(tool_name)
            if not tool:
                raise Exception(f"工具不存在: {tool_name}")
            
            # 检查工具是否已初始化
            if not tool.is_initialized:
                success = await tool.initialize()
                if not success:
                    raise Exception(f"工具初始化失败: {tool_name}")
            
            # 验证参数
            if not tool.validate_params(params):
                raise Exception(f"参数验证失败: {tool_name}")
            
            # 执行工具
            result = await tool.execute(params)
            
            # 更新统计信息
            self._update_tool_stats(tool_name, True, start_time)
            tool.update_usage()
            
            return {
                'success': True,
                'tool_name': tool_name,
                'result': result,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            # 更新错误统计
            self._update_tool_stats(tool_name, False, start_time)
            
            logger.error(f"工具执行失败: {tool_name}, 错误: {str(e)}")
            return {
                'success': False,
                'tool_name': tool_name,
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _update_tool_stats(self, tool_name: str, success: bool, start_time: datetime):
        """更新工具统计信息"""
        if tool_name not in self._tool_stats:
            return
        
        stats = self._tool_stats[tool_name]
        execution_time = (datetime.now() - start_time).total_seconds()
        
        stats['usage_count'] += 1
        if success:
            stats['success_count'] += 1
        else:
            stats['error_count'] += 1
        
        # 更新平均执行时间
        current_avg = stats['avg_execution_time']
        usage_count = stats['usage_count']
        stats['avg_execution_time'] = (
            (current_avg * (usage_count - 1) + execution_time) / usage_count
        )
    
    def get_tool_stats(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具统计信息"""
        return self._tool_stats.get(tool_name)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        total_tools = len(self._tools)
        categories = list(self._categories.keys())
        capabilities = list(self._capabilities.keys())
        
        total_usage = sum(stats['usage_count'] for stats in self._tool_stats.values())
        total_success = sum(stats['success_count'] for stats in self._tool_stats.values())
        total_errors = sum(stats['error_count'] for stats in self._tool_stats.values())
        
        return {
            'total_tools': total_tools,
            'categories': categories,
            'capabilities': capabilities,
            'total_usage': total_usage,
            'success_rate': total_success / max(total_usage, 1),
            'error_rate': total_errors / max(total_usage, 1)
        }
    
    async def health_check_all(self) -> Dict[str, bool]:
        """检查所有工具的健康状态"""
        health_status = {}
        
        for tool_name, tool in self._tools.items():
            try:
                health_status[tool_name] = await tool.health_check()
            except Exception as e:
                logger.error(f"工具健康检查异常: {tool_name}, 错误: {str(e)}")
                health_status[tool_name] = False
        
        return health_status
    
    async def initialize_all(self) -> Dict[str, bool]:
        """初始化所有工具"""
        init_results = {}
        
        for tool_name, tool in self._tools.items():
            if not tool.is_initialized:
                init_results[tool_name] = await tool.initialize()
            else:
                init_results[tool_name] = True
        
        return init_results