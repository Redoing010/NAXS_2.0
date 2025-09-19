# 基础数据连接器
# 定义统一的数据源连接接口和数据结构

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Iterator, Callable
from datetime import datetime, date
from enum import Enum
import pandas as pd
import asyncio
import logging

# 数据类型枚举
class DataType(Enum):
    STOCK_PRICE = "stock_price"          # 股票价格
    STOCK_INFO = "stock_info"            # 股票基本信息
    FINANCIAL_REPORT = "financial_report" # 财务报表
    NEWS = "news"                        # 新闻资讯
    ANNOUNCEMENT = "announcement"        # 公告
    RESEARCH_REPORT = "research_report"  # 研报
    MARKET_INDEX = "market_index"        # 市场指数
    SECTOR_DATA = "sector_data"          # 行业数据
    MACRO_DATA = "macro_data"            # 宏观数据
    ALTERNATIVE_DATA = "alternative_data" # 替代数据

# 数据频率枚举
class DataFrequency(Enum):
    TICK = "tick"          # 逐笔
    MINUTE_1 = "1min"      # 1分钟
    MINUTE_5 = "5min"      # 5分钟
    MINUTE_15 = "15min"    # 15分钟
    MINUTE_30 = "30min"    # 30分钟
    HOUR_1 = "1h"          # 1小时
    DAILY = "daily"        # 日线
    WEEKLY = "weekly"      # 周线
    MONTHLY = "monthly"    # 月线
    QUARTERLY = "quarterly" # 季度
    YEARLY = "yearly"      # 年度

# 数据源状态
class DataSourceStatus(Enum):
    ACTIVE = "active"          # 活跃
    INACTIVE = "inactive"      # 非活跃
    MAINTENANCE = "maintenance" # 维护中
    ERROR = "error"            # 错误
    RATE_LIMITED = "rate_limited" # 限流中

@dataclass
class DataPoint:
    """单个数据点"""
    timestamp: datetime
    symbol: str
    data_type: DataType
    values: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'data_type': self.data_type.value,
            'values': self.values,
            'metadata': self.metadata,
            'source': self.source,
            'quality_score': self.quality_score,
        }

@dataclass
class DataBatch:
    """数据批次"""
    data_points: List[DataPoint]
    batch_id: str
    source: str
    data_type: DataType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.data_points)
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        if not self.data_points:
            return pd.DataFrame()
        
        records = []
        for point in self.data_points:
            record = {
                'timestamp': point.timestamp,
                'symbol': point.symbol,
                'source': point.source,
                'quality_score': point.quality_score,
            }
            record.update(point.values)
            records.append(record)
        
        return pd.DataFrame(records)

@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    source_type: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: int = 100  # 每分钟请求数
    timeout: int = 30      # 超时时间(秒)
    retry_count: int = 3   # 重试次数
    retry_delay: float = 1.0  # 重试延迟(秒)
    cache_ttl: int = 300   # 缓存TTL(秒)
    enabled: bool = True
    priority: int = 1      # 优先级(1-10)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """验证配置"""
        if not self.name or not self.source_type:
            return False
        return True

class BaseConnector(ABC):
    """基础数据连接器抽象类"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.status = DataSourceStatus.INACTIVE
        self._session = None
        self._last_request_time = None
        self._request_count = 0
        self._error_count = 0
        
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def source_type(self) -> str:
        return self.config.source_type
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接数据源"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """断开连接"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """测试连接"""
        pass
    
    @abstractmethod
    async def get_stock_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY,
        fields: Optional[List[str]] = None
    ) -> DataBatch:
        """获取股票数据"""
        pass
    
    @abstractmethod
    async def get_market_data(
        self,
        symbols: List[str],
        data_type: DataType,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        **kwargs
    ) -> DataBatch:
        """获取市场数据"""
        pass
    
    @abstractmethod
    async def get_financial_data(
        self,
        symbol: str,
        report_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> DataBatch:
        """获取财务数据"""
        pass
    
    @abstractmethod
    async def get_news_data(
        self,
        symbols: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> DataBatch:
        """获取新闻数据"""
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """获取支持的股票代码列表"""
        pass
    
    @abstractmethod
    def get_supported_data_types(self) -> List[DataType]:
        """获取支持的数据类型"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            is_connected = await self.test_connection()
            return {
                'status': self.status.value,
                'connected': is_connected,
                'error_count': self._error_count,
                'request_count': self._request_count,
                'last_request': self._last_request_time,
                'config': {
                    'name': self.config.name,
                    'enabled': self.config.enabled,
                    'rate_limit': self.config.rate_limit,
                }
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'status': DataSourceStatus.ERROR.value,
                'connected': False,
                'error': str(e)
            }
    
    def _update_request_stats(self):
        """更新请求统计"""
        self._request_count += 1
        self._last_request_time = datetime.now()
    
    def _handle_error(self, error: Exception):
        """处理错误"""
        self._error_count += 1
        self.logger.error(f"Connector error: {error}")
        
        # 如果错误过多，标记为错误状态
        if self._error_count > 10:
            self.status = DataSourceStatus.ERROR
    
    async def _retry_request(
        self,
        func: Callable,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """重试请求"""
        max_retries = max_retries or self.config.retry_count
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                self._update_request_stats()
                return result
            except Exception as e:
                last_error = e
                self._handle_error(e)
                
                if attempt < max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)  # 指数退避
                    await asyncio.sleep(delay)
                    self.logger.warning(f"Retrying request (attempt {attempt + 1}/{max_retries})")
        
        raise last_error
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.name})"
    
    def __repr__(self) -> str:
        return self.__str__()

# 连接器工厂
class ConnectorFactory:
    """连接器工厂"""
    
    _connectors: Dict[str, type] = {}
    
    @classmethod
    def register(cls, source_type: str, connector_class: type):
        """注册连接器"""
        cls._connectors[source_type] = connector_class
    
    @classmethod
    def create(cls, config: DataSourceConfig) -> BaseConnector:
        """创建连接器实例"""
        connector_class = cls._connectors.get(config.source_type)
        if not connector_class:
            raise ValueError(f"Unknown connector type: {config.source_type}")
        
        return connector_class(config)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """获取可用的连接器类型"""
        return list(cls._connectors.keys())

# 数据验证器
class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_data_point(point: DataPoint) -> bool:
        """验证数据点"""
        if not point.timestamp or not point.symbol:
            return False
        
        if not point.values:
            return False
        
        # 检查数值字段
        for key, value in point.values.items():
            if isinstance(value, (int, float)) and (value < 0 or value > 1e10):
                return False
        
        return True
    
    @staticmethod
    def validate_data_batch(batch: DataBatch) -> Dict[str, Any]:
        """验证数据批次"""
        total_points = len(batch.data_points)
        valid_points = sum(1 for point in batch.data_points 
                          if DataValidator.validate_data_point(point))
        
        return {
            'total_points': total_points,
            'valid_points': valid_points,
            'invalid_points': total_points - valid_points,
            'validity_rate': valid_points / total_points if total_points > 0 else 0,
            'is_valid': valid_points == total_points
        }

# 便捷函数
def create_data_point(
    symbol: str,
    data_type: DataType,
    values: Dict[str, Any],
    timestamp: Optional[datetime] = None,
    source: str = "",
    **metadata
) -> DataPoint:
    """创建数据点"""
    return DataPoint(
        timestamp=timestamp or datetime.now(),
        symbol=symbol,
        data_type=data_type,
        values=values,
        source=source,
        metadata=metadata
    )

def create_data_batch(
    data_points: List[DataPoint],
    source: str,
    data_type: DataType,
    batch_id: Optional[str] = None
) -> DataBatch:
    """创建数据批次"""
    return DataBatch(
        data_points=data_points,
        batch_id=batch_id or f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        source=source,
        data_type=data_type
    )