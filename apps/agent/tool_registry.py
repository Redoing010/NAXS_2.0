# 工具注册表模块
# 实现工具的注册、发现、调用和管理功能

import logging
import inspect
import json
from typing import Dict, Any, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import threading
from functools import wraps
import importlib
import pkgutil

logger = logging.getLogger(__name__)

class ToolCategory(Enum):
    """工具分类"""
    DATA_SOURCE = "data_source"        # 数据源
    ANALYSIS = "analysis"              # 分析工具
    MODELING = "modeling"              # 建模工具
    VISUALIZATION = "visualization"    # 可视化工具
    NOTIFICATION = "notification"      # 通知工具
    UTILITY = "utility"                # 实用工具
    INTEGRATION = "integration"        # 集成工具
    CUSTOM = "custom"                  # 自定义工具

class ParameterType(Enum):
    """参数类型"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    FILE = "file"
    DATE = "date"
    DATETIME = "datetime"

@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: ParameterType
    description: str = ""
    required: bool = True
    default: Any = None
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None  # 正则表达式模式
    
    def validate(self, value: Any) -> bool:
        """验证参数值
        
        Args:
            value: 参数值
            
        Returns:
            是否有效
        """
        if value is None:
            return not self.required
        
        # 类型检查
        if self.type == ParameterType.STRING and not isinstance(value, str):
            return False
        elif self.type == ParameterType.INTEGER and not isinstance(value, int):
            return False
        elif self.type == ParameterType.FLOAT and not isinstance(value, (int, float)):
            return False
        elif self.type == ParameterType.BOOLEAN and not isinstance(value, bool):
            return False
        elif self.type == ParameterType.LIST and not isinstance(value, list):
            return False
        elif self.type == ParameterType.DICT and not isinstance(value, dict):
            return False
        
        # 选择值检查
        if self.choices and value not in self.choices:
            return False
        
        # 数值范围检查
        if self.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
        
        # 正则表达式检查
        if self.pattern and self.type == ParameterType.STRING:
            import re
            if not re.match(self.pattern, value):
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'required': self.required,
            'default': self.default,
            'choices': self.choices,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'pattern': self.pattern
        }

@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: List[ToolParameter] = field(default_factory=list)
    
    # 执行特性
    async_execution: bool = False
    estimated_duration: float = 0.0  # 预估执行时间（秒）
    resource_intensive: bool = False
    
    # 依赖信息
    dependencies: List[str] = field(default_factory=list)
    
    # 使用统计
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'version': self.version,
            'author': self.author,
            'tags': self.tags,
            'parameters': [p.to_dict() for p in self.parameters],
            'async_execution': self.async_execution,
            'estimated_duration': self.estimated_duration,
            'resource_intensive': self.resource_intensive,
            'dependencies': self.dependencies,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }

@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'error_type': self.error_type,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }

class Tool:
    """工具基类
    
    定义工具的标准接口和行为
    """
    
    def __init__(self, metadata: ToolMetadata, func: Callable):
        self.metadata = metadata
        self.func = func
        self.lock = threading.RLock()
        
        # 验证函数签名
        self._validate_function_signature()
        
        logger.debug(f"工具初始化: {metadata.name}")
    
    def _validate_function_signature(self):
        """验证函数签名"""
        sig = inspect.signature(self.func)
        
        # 检查参数是否匹配
        func_params = list(sig.parameters.keys())
        metadata_params = [p.name for p in self.metadata.parameters]
        
        # 允许函数有额外的 **kwargs 参数
        if '**kwargs' in str(sig):
            # 移除 kwargs 参数进行比较
            func_params = [p for p in func_params if p != 'kwargs']
        
        missing_params = set(metadata_params) - set(func_params)
        if missing_params:
            logger.warning(f"工具 {self.metadata.name} 缺少参数: {missing_params}")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """验证参数
        
        Args:
            parameters: 参数字典
            
        Returns:
            错误信息列表
        """
        errors = []
        
        # 检查必需参数
        for param in self.metadata.parameters:
            if param.required and param.name not in parameters:
                errors.append(f"缺少必需参数: {param.name}")
            elif param.name in parameters:
                if not param.validate(parameters[param.name]):
                    errors.append(f"参数 {param.name} 验证失败")
        
        return errors
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """执行工具
        
        Args:
            parameters: 执行参数
            
        Returns:
            执行结果
        """
        start_time = datetime.now()
        
        try:
            # 验证参数
            validation_errors = self.validate_parameters(parameters)
            if validation_errors:
                return ToolResult(
                    success=False,
                    error=f"参数验证失败: {'; '.join(validation_errors)}",
                    error_type="ValidationError"
                )
            
            # 添加默认参数
            final_params = {}
            for param in self.metadata.parameters:
                if param.name in parameters:
                    final_params[param.name] = parameters[param.name]
                elif param.default is not None:
                    final_params[param.name] = param.default
            
            # 执行函数
            if self.metadata.async_execution:
                if asyncio.iscoroutinefunction(self.func):
                    result_data = await self.func(**final_params)
                else:
                    # 在线程池中执行同步函数
                    loop = asyncio.get_event_loop()
                    result_data = await loop.run_in_executor(None, lambda: self.func(**final_params))
            else:
                if asyncio.iscoroutinefunction(self.func):
                    result_data = await self.func(**final_params)
                else:
                    result_data = self.func(**final_params)
            
            # 更新使用统计
            with self.lock:
                self.metadata.usage_count += 1
                self.metadata.last_used = datetime.now()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                data=result_data if isinstance(result_data, dict) else {'result': result_data},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"工具执行失败: {self.metadata.name} - {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                execution_time=execution_time
            )
    
    def get_info(self) -> Dict[str, Any]:
        """获取工具信息
        
        Returns:
            工具信息字典
        """
        return self.metadata.to_dict()

class ToolRegistry:
    """工具注册表
    
    管理所有可用工具的注册、发现和调用
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
        self.lock = threading.RLock()
        
        # 注册内置工具
        self._register_builtin_tools()
        
        logger.info("工具注册表初始化完成")
    
    def register_tool(self, metadata: ToolMetadata, func: Callable) -> bool:
        """注册工具
        
        Args:
            metadata: 工具元数据
            func: 工具函数
            
        Returns:
            是否注册成功
        """
        with self.lock:
            if metadata.name in self.tools:
                logger.warning(f"工具已存在，将被覆盖: {metadata.name}")
            
            try:
                tool = Tool(metadata, func)
                self.tools[metadata.name] = tool
                
                # 更新分类索引
                if metadata.name not in self.categories[metadata.category]:
                    self.categories[metadata.category].append(metadata.name)
                
                logger.info(f"工具注册成功: {metadata.name} ({metadata.category.value})")
                return True
                
            except Exception as e:
                logger.error(f"工具注册失败: {metadata.name} - {e}")
                return False
    
    def unregister_tool(self, name: str) -> bool:
        """注销工具
        
        Args:
            name: 工具名称
            
        Returns:
            是否注销成功
        """
        with self.lock:
            if name not in self.tools:
                return False
            
            tool = self.tools[name]
            category = tool.metadata.category
            
            # 从注册表中移除
            del self.tools[name]
            
            # 从分类索引中移除
            if name in self.categories[category]:
                self.categories[category].remove(name)
            
            logger.info(f"工具注销成功: {name}")
            return True
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具
        
        Args:
            name: 工具名称
            
        Returns:
            工具实例
        """
        return self.tools.get(name)
    
    def list_tools(self, category: Optional[ToolCategory] = None, tags: Optional[List[str]] = None) -> List[str]:
        """列出工具
        
        Args:
            category: 工具分类过滤
            tags: 标签过滤
            
        Returns:
            工具名称列表
        """
        tools = list(self.tools.keys())
        
        # 按分类过滤
        if category:
            tools = [name for name in tools if self.tools[name].metadata.category == category]
        
        # 按标签过滤
        if tags:
            tools = [name for name in tools 
                    if any(tag in self.tools[name].metadata.tags for tag in tags)]
        
        return sorted(tools)
    
    def search_tools(self, query: str) -> List[str]:
        """搜索工具
        
        Args:
            query: 搜索查询
            
        Returns:
            匹配的工具名称列表
        """
        query = query.lower()
        matches = []
        
        for name, tool in self.tools.items():
            # 搜索名称
            if query in name.lower():
                matches.append((name, 3))  # 名称匹配权重最高
                continue
            
            # 搜索描述
            if query in tool.metadata.description.lower():
                matches.append((name, 2))
                continue
            
            # 搜索标签
            if any(query in tag.lower() for tag in tool.metadata.tags):
                matches.append((name, 1))
        
        # 按权重排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in matches]
    
    async def execute_tool(self, name: str, parameters: Dict[str, Any]) -> ToolResult:
        """执行工具
        
        Args:
            name: 工具名称
            parameters: 执行参数
            
        Returns:
            执行结果
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"工具不存在: {name}",
                error_type="ToolNotFoundError"
            )
        
        return await tool.execute(parameters)
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取工具信息
        
        Args:
            name: 工具名称
            
        Returns:
            工具信息字典
        """
        tool = self.get_tool(name)
        return tool.get_info() if tool else None
    
    def get_all_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有工具信息
        
        Returns:
            所有工具信息字典
        """
        return {name: tool.get_info() for name, tool in self.tools.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        total_tools = len(self.tools)
        category_counts = {cat.value: len(tools) for cat, tools in self.categories.items()}
        
        # 使用统计
        total_usage = sum(tool.metadata.usage_count for tool in self.tools.values())
        most_used = max(self.tools.values(), key=lambda t: t.metadata.usage_count, default=None)
        
        return {
            'total_tools': total_tools,
            'category_counts': category_counts,
            'total_usage': total_usage,
            'most_used_tool': most_used.metadata.name if most_used else None,
            'avg_usage_per_tool': total_usage / total_tools if total_tools > 0 else 0
        }
    
    def _register_builtin_tools(self):
        """注册内置工具"""
        # 市场数据获取工具
        def get_market_data(symbols: List[str], fields: List[str] = None) -> Dict[str, Any]:
            """获取市场数据"""
            fields = fields or ['price', 'change', 'volume']
            return {
                'symbols': symbols,
                'fields': fields,
                'data': {
                    symbol: {
                        field: hash(symbol + field) % 1000 / 10.0
                        for field in fields
                    } for symbol in symbols
                },
                'timestamp': datetime.now().isoformat()
            }
        
        market_data_metadata = ToolMetadata(
            name="get_market_data",
            description="获取股票市场实时数据",
            category=ToolCategory.DATA_SOURCE,
            tags=["market", "stock", "realtime"],
            parameters=[
                ToolParameter("symbols", ParameterType.LIST, "股票代码列表", required=True),
                ToolParameter("fields", ParameterType.LIST, "数据字段列表", required=False, 
                            default=['price', 'change', 'volume'])
            ]
        )
        
        self.register_tool(market_data_metadata, get_market_data)
        
        # 技术分析工具
        def technical_analysis(symbol: str, indicators: List[str], period: str = "1Y") -> Dict[str, Any]:
            """技术分析"""
            return {
                'symbol': symbol,
                'period': period,
                'indicators': {
                    indicator: {
                        'value': hash(symbol + indicator) % 100,
                        'signal': 'buy' if hash(symbol + indicator) % 3 == 0 else 'hold'
                    } for indicator in indicators
                },
                'overall_signal': 'buy',
                'confidence': 0.75
            }
        
        technical_metadata = ToolMetadata(
            name="technical_analysis",
            description="股票技术分析",
            category=ToolCategory.ANALYSIS,
            tags=["technical", "analysis", "indicators"],
            parameters=[
                ToolParameter("symbol", ParameterType.STRING, "股票代码", required=True),
                ToolParameter("indicators", ParameterType.LIST, "技术指标列表", required=True),
                ToolParameter("period", ParameterType.STRING, "分析周期", required=False, default="1Y")
            ]
        )
        
        self.register_tool(technical_metadata, technical_analysis)
        
        # 选股筛选工具
        def stock_screening(criteria: Dict[str, Any], top_n: int = 20) -> Dict[str, Any]:
            """股票筛选"""
            return {
                'criteria': criteria,
                'total_candidates': 1000,
                'selected_stocks': [
                    {
                        'symbol': f"00000{i}.SZ",
                        'name': f"股票{i}",
                        'score': 90 - i,
                        'reason': f"符合{len(criteria)}个筛选条件"
                    } for i in range(1, min(top_n + 1, 21))
                ],
                'screening_time': datetime.now().isoformat()
            }
        
        screening_metadata = ToolMetadata(
            name="stock_screening",
            description="基于条件筛选股票",
            category=ToolCategory.ANALYSIS,
            tags=["screening", "selection", "filter"],
            parameters=[
                ToolParameter("criteria", ParameterType.DICT, "筛选条件", required=True),
                ToolParameter("top_n", ParameterType.INTEGER, "返回数量", required=False, 
                            default=20, min_value=1, max_value=100)
            ]
        )
        
        self.register_tool(screening_metadata, stock_screening)
        
        # 策略回测工具
        async def strategy_backtest(strategy_config: Dict[str, Any], start_date: str, end_date: str) -> Dict[str, Any]:
            """策略回测"""
            # 模拟异步回测
            await asyncio.sleep(1)
            
            return {
                'strategy_config': strategy_config,
                'period': f"{start_date} to {end_date}",
                'performance': {
                    'total_return': 0.25,
                    'annual_return': 0.08,
                    'volatility': 0.15,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.12
                },
                'trades': 156,
                'win_rate': 0.58
            }
        
        backtest_metadata = ToolMetadata(
            name="strategy_backtest",
            description="投资策略回测",
            category=ToolCategory.MODELING,
            tags=["backtest", "strategy", "performance"],
            async_execution=True,
            estimated_duration=30.0,
            parameters=[
                ToolParameter("strategy_config", ParameterType.DICT, "策略配置", required=True),
                ToolParameter("start_date", ParameterType.STRING, "开始日期", required=True),
                ToolParameter("end_date", ParameterType.STRING, "结束日期", required=True)
            ]
        )
        
        self.register_tool(backtest_metadata, strategy_backtest)
        
        logger.info("内置工具注册完成")
    
    def load_tools_from_module(self, module_name: str) -> int:
        """从模块加载工具
        
        Args:
            module_name: 模块名称
            
        Returns:
            加载的工具数量
        """
        try:
            module = importlib.import_module(module_name)
            loaded_count = 0
            
            # 查找带有工具装饰器的函数
            for name in dir(module):
                obj = getattr(module, name)
                if hasattr(obj, '_tool_metadata'):
                    metadata = obj._tool_metadata
                    if self.register_tool(metadata, obj):
                        loaded_count += 1
            
            logger.info(f"从模块 {module_name} 加载了 {loaded_count} 个工具")
            return loaded_count
            
        except Exception as e:
            logger.error(f"从模块 {module_name} 加载工具失败: {e}")
            return 0

# 工具装饰器
def tool_decorator(name: str, description: str, category: ToolCategory, **kwargs):
    """工具装饰器
    
    Args:
        name: 工具名称
        description: 工具描述
        category: 工具分类
        **kwargs: 其他元数据参数
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable):
        # 从函数签名自动生成参数定义
        sig = inspect.signature(func)
        parameters = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'kwargs':
                continue
                
            # 推断参数类型
            param_type = ParameterType.STRING  # 默认类型
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = ParameterType.INTEGER
                elif param.annotation == float:
                    param_type = ParameterType.FLOAT
                elif param.annotation == bool:
                    param_type = ParameterType.BOOLEAN
                elif param.annotation == list:
                    param_type = ParameterType.LIST
                elif param.annotation == dict:
                    param_type = ParameterType.DICT
            
            # 判断是否必需
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            
            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=f"{param_name} 参数",
                required=required,
                default=default
            ))
        
        # 创建元数据
        metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            async_execution=asyncio.iscoroutinefunction(func),
            **kwargs
        )
        
        # 将元数据附加到函数
        func._tool_metadata = metadata
        
        return func
    
    return decorator

# 全局工具注册表实例
_global_registry = ToolRegistry()

# 便捷函数
def create_tool_registry() -> ToolRegistry:
    """创建工具注册表
    
    Returns:
        工具注册表实例
    """
    return ToolRegistry()

def register_tool(metadata: ToolMetadata, func: Callable, registry: ToolRegistry = None) -> bool:
    """注册工具到全局注册表
    
    Args:
        metadata: 工具元数据
        func: 工具函数
        registry: 工具注册表（可选，默认使用全局注册表）
        
    Returns:
        是否注册成功
    """
    target_registry = registry or _global_registry
    return target_registry.register_tool(metadata, func)

def get_global_registry() -> ToolRegistry:
    """获取全局工具注册表
    
    Returns:
        全局工具注册表实例
    """
    return _global_registry

# 常用工具装饰器
def data_source_tool(name: str, description: str, **kwargs):
    """数据源工具装饰器"""
    return tool_decorator(name, description, ToolCategory.DATA_SOURCE, **kwargs)

def analysis_tool(name: str, description: str, **kwargs):
    """分析工具装饰器"""
    return tool_decorator(name, description, ToolCategory.ANALYSIS, **kwargs)

def modeling_tool(name: str, description: str, **kwargs):
    """建模工具装饰器"""
    return tool_decorator(name, description, ToolCategory.MODELING, **kwargs)

def visualization_tool(name: str, description: str, **kwargs):
    """可视化工具装饰器"""
    return tool_decorator(name, description, ToolCategory.VISUALIZATION, **kwargs)