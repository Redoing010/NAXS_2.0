# 连接器管理器
# 统一管理和调度所有数据源连接器

import asyncio
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from dataclasses import dataclass, field
import json

from .base_connector import (
    BaseConnector, DataSourceConfig, DataPoint, DataBatch,
    DataType, DataFrequency, DataSourceStatus, ConnectorFactory
)
from .rate_limiter import RateLimiter
from .cache_manager import CacheManager
from .data_normalizer import DataNormalizer
from .quality_checker import DataQualityChecker

class DataSourceType(Enum):
    """数据源类型"""
    AKSHARE = "akshare"
    TUSHARE = "tushare"
    EASTMONEY = "eastmoney"
    YAHOO = "yahoo"
    JUKUAN = "jukuan"
    BROKER = "broker"
    CUSTOM = "custom"

class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"      # 轮询
    PRIORITY = "priority"            # 优先级
    RANDOM = "random"                # 随机
    LEAST_LOADED = "least_loaded"    # 最少负载
    FASTEST = "fastest"              # 最快响应

@dataclass
class ConnectorStats:
    """连接器统计信息"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    error_rate: float = 0.0
    uptime: float = 100.0
    
    def update_success(self, response_time: float):
        """更新成功统计"""
        self.total_requests += 1
        self.successful_requests += 1
        self.last_request_time = datetime.now()
        
        # 更新平均响应时间
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time + response_time) / 2
        
        # 更新错误率
        self.error_rate = (self.failed_requests / self.total_requests) * 100
    
    def update_failure(self):
        """更新失败统计"""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_request_time = datetime.now()
        
        # 更新错误率
        self.error_rate = (self.failed_requests / self.total_requests) * 100
        
        # 更新可用性
        self.uptime = (self.successful_requests / self.total_requests) * 100

@dataclass
class DataRequest:
    """数据请求"""
    request_id: str
    data_type: DataType
    symbols: List[str]
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    frequency: DataFrequency = DataFrequency.DAILY
    fields: Optional[List[str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1-10, 10最高
    timeout: int = 30
    retry_count: int = 3
    preferred_sources: Optional[List[str]] = None
    exclude_sources: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConnectorManager:
    """连接器管理器"""
    
    def __init__(
        self,
        load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.PRIORITY,
        enable_cache: bool = True,
        enable_quality_check: bool = True,
        enable_normalization: bool = True
    ):
        self.logger = logging.getLogger("ConnectorManager")
        self.connectors: Dict[str, BaseConnector] = {}
        self.connector_stats: Dict[str, ConnectorStats] = {}
        self.load_balance_strategy = load_balance_strategy
        
        # 组件
        self.rate_limiter = RateLimiter()
        self.cache_manager = CacheManager() if enable_cache else None
        self.data_normalizer = DataNormalizer() if enable_normalization else None
        self.quality_checker = DataQualityChecker() if enable_quality_check else None
        
        # 状态
        self._round_robin_index = 0
        self._active_requests: Set[str] = set()
        self._request_queue: List[DataRequest] = []
        
        # 配置
        self.max_concurrent_requests = 10
        self.default_timeout = 30
        self.health_check_interval = 60  # 秒
        
        # 启动健康检查
        self._health_check_task = None
        self._start_health_check()
    
    async def add_connector(self, config: DataSourceConfig) -> bool:
        """添加连接器"""
        try:
            connector = ConnectorFactory.create(config)
            
            # 连接测试
            if await connector.connect():
                self.connectors[config.name] = connector
                self.connector_stats[config.name] = ConnectorStats()
                self.logger.info(f"Added connector: {config.name}")
                return True
            else:
                self.logger.error(f"Failed to connect: {config.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to add connector {config.name}: {e}")
            return False
    
    async def remove_connector(self, name: str) -> bool:
        """移除连接器"""
        try:
            if name in self.connectors:
                await self.connectors[name].disconnect()
                del self.connectors[name]
                del self.connector_stats[name]
                self.logger.info(f"Removed connector: {name}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove connector {name}: {e}")
            return False
    
    async def get_data(self, request: DataRequest) -> DataBatch:
        """获取数据"""
        start_time = datetime.now()
        
        try:
            # 检查缓存
            if self.cache_manager:
                cached_data = await self.cache_manager.get(self._generate_cache_key(request))
                if cached_data:
                    self.logger.debug(f"Cache hit for request {request.request_id}")
                    return cached_data
            
            # 选择连接器
            connector = await self._select_connector(request)
            if not connector:
                raise ValueError("No available connector for request")
            
            # 频率限制
            await self.rate_limiter.acquire(connector.name)
            
            # 执行请求
            self._active_requests.add(request.request_id)
            
            try:
                data_batch = await self._execute_request(connector, request)
                
                # 数据质量检查
                if self.quality_checker:
                    quality_result = await self.quality_checker.check(data_batch)
                    if not quality_result.is_valid:
                        self.logger.warning(f"Data quality check failed: {quality_result.issues}")
                
                # 数据标准化
                if self.data_normalizer:
                    data_batch = await self.data_normalizer.normalize(data_batch)
                
                # 缓存结果
                if self.cache_manager:
                    await self.cache_manager.set(
                        self._generate_cache_key(request),
                        data_batch,
                        ttl=connector.config.cache_ttl
                    )
                
                # 更新统计
                response_time = (datetime.now() - start_time).total_seconds()
                self.connector_stats[connector.name].update_success(response_time)
                
                return data_batch
                
            finally:
                self._active_requests.discard(request.request_id)
                
        except Exception as e:
            # 更新失败统计
            if 'connector' in locals():
                self.connector_stats[connector.name].update_failure()
            
            self.logger.error(f"Failed to get data for request {request.request_id}: {e}")
            raise
    
    async def get_stock_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY,
        preferred_sources: Optional[List[str]] = None
    ) -> DataBatch:
        """获取股票数据的便捷方法"""
        request = DataRequest(
            request_id=f"stock_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            data_type=DataType.STOCK_PRICE,
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            preferred_sources=preferred_sources
        )
        
        return await self.get_data(request)
    
    async def get_market_data(
        self,
        symbols: List[str],
        data_type: DataType,
        preferred_sources: Optional[List[str]] = None,
        **kwargs
    ) -> DataBatch:
        """获取市场数据的便捷方法"""
        request = DataRequest(
            request_id=f"market_{data_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            data_type=data_type,
            symbols=symbols,
            preferred_sources=preferred_sources,
            metadata=kwargs
        )
        
        return await self.get_data(request)
    
    async def get_financial_data(
        self,
        symbol: str,
        report_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        preferred_sources: Optional[List[str]] = None
    ) -> DataBatch:
        """获取财务数据的便捷方法"""
        request = DataRequest(
            request_id=f"financial_{symbol}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            data_type=DataType.FINANCIAL_REPORT,
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            preferred_sources=preferred_sources,
            metadata={'report_type': report_type}
        )
        
        return await self.get_data(request)
    
    async def _select_connector(self, request: DataRequest) -> Optional[BaseConnector]:
        """选择连接器"""
        # 过滤可用连接器
        available_connectors = []
        
        for name, connector in self.connectors.items():
            # 检查状态
            if connector.status != DataSourceStatus.ACTIVE:
                continue
            
            # 检查数据类型支持
            if request.data_type not in connector.get_supported_data_types():
                continue
            
            # 检查偏好源
            if request.preferred_sources and name not in request.preferred_sources:
                continue
            
            # 检查排除源
            if request.exclude_sources and name in request.exclude_sources:
                continue
            
            available_connectors.append((name, connector))
        
        if not available_connectors:
            return None
        
        # 根据策略选择
        if self.load_balance_strategy == LoadBalanceStrategy.PRIORITY:
            # 按优先级排序
            available_connectors.sort(key=lambda x: x[1].config.priority, reverse=True)
            return available_connectors[0][1]
            
        elif self.load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN:
            # 轮询
            selected = available_connectors[self._round_robin_index % len(available_connectors)]
            self._round_robin_index += 1
            return selected[1]
            
        elif self.load_balance_strategy == LoadBalanceStrategy.FASTEST:
            # 选择响应最快的
            fastest = min(available_connectors, 
                         key=lambda x: self.connector_stats[x[0]].avg_response_time)
            return fastest[1]
            
        elif self.load_balance_strategy == LoadBalanceStrategy.LEAST_LOADED:
            # 选择负载最少的
            least_loaded = min(available_connectors,
                             key=lambda x: len([r for r in self._active_requests 
                                              if r.startswith(x[0])]))
            return least_loaded[1]
            
        else:
            # 随机选择
            import random
            return random.choice(available_connectors)[1]
    
    async def _execute_request(self, connector: BaseConnector, request: DataRequest) -> DataBatch:
        """执行数据请求"""
        if request.data_type == DataType.STOCK_PRICE:
            return await connector.get_stock_data(
                symbol=request.symbols[0],
                start_date=request.start_date,
                end_date=request.end_date,
                frequency=request.frequency,
                fields=request.fields
            )
        elif request.data_type in [DataType.MARKET_INDEX, DataType.STOCK_INFO, DataType.SECTOR_DATA]:
            return await connector.get_market_data(
                symbols=request.symbols,
                data_type=request.data_type,
                start_date=request.start_date,
                end_date=request.end_date,
                **request.metadata
            )
        elif request.data_type == DataType.FINANCIAL_REPORT:
            return await connector.get_financial_data(
                symbol=request.symbols[0],
                report_type=request.metadata.get('report_type', 'income'),
                start_date=request.start_date,
                end_date=request.end_date
            )
        elif request.data_type == DataType.NEWS:
            return await connector.get_news_data(
                symbols=request.symbols,
                keywords=request.metadata.get('keywords'),
                start_date=request.start_date,
                end_date=request.end_date,
                limit=request.metadata.get('limit', 100)
            )
        else:
            raise ValueError(f"Unsupported data type: {request.data_type}")
    
    def _generate_cache_key(self, request: DataRequest) -> str:
        """生成缓存键"""
        key_parts = [
            request.data_type.value,
            '_'.join(request.symbols),
            request.start_date.isoformat() if request.start_date else 'None',
            request.end_date.isoformat() if request.end_date else 'None',
            request.frequency.value,
        ]
        
        if request.fields:
            key_parts.append('_'.join(request.fields))
        
        if request.metadata:
            key_parts.append(json.dumps(request.metadata, sort_keys=True))
        
        return 'data_' + '_'.join(key_parts)
    
    async def get_connector_status(self) -> Dict[str, Any]:
        """获取所有连接器状态"""
        status = {}
        
        for name, connector in self.connectors.items():
            health = await connector.health_check()
            stats = self.connector_stats[name]
            
            status[name] = {
                'health': health,
                'stats': {
                    'total_requests': stats.total_requests,
                    'successful_requests': stats.successful_requests,
                    'failed_requests': stats.failed_requests,
                    'error_rate': stats.error_rate,
                    'avg_response_time': stats.avg_response_time,
                    'uptime': stats.uptime,
                    'last_request_time': stats.last_request_time.isoformat() if stats.last_request_time else None,
                },
                'config': {
                    'source_type': connector.config.source_type,
                    'priority': connector.config.priority,
                    'enabled': connector.config.enabled,
                    'rate_limit': connector.config.rate_limit,
                }
            }
        
        return status
    
    async def get_supported_symbols(self, source: Optional[str] = None) -> List[str]:
        """获取支持的股票代码"""
        all_symbols = set()
        
        for name, connector in self.connectors.items():
            if source and name != source:
                continue
            
            try:
                symbols = connector.get_supported_symbols()
                all_symbols.update(symbols)
            except Exception as e:
                self.logger.warning(f"Failed to get symbols from {name}: {e}")
        
        return list(all_symbols)
    
    def _start_health_check(self):
        """启动健康检查任务"""
        async def health_check_loop():
            while True:
                try:
                    await asyncio.sleep(self.health_check_interval)
                    await self._perform_health_check()
                except Exception as e:
                    self.logger.error(f"Health check error: {e}")
        
        self._health_check_task = asyncio.create_task(health_check_loop())
    
    async def _perform_health_check(self):
        """执行健康检查"""
        for name, connector in self.connectors.items():
            try:
                is_healthy = await connector.test_connection()
                if not is_healthy and connector.status == DataSourceStatus.ACTIVE:
                    connector.status = DataSourceStatus.ERROR
                    self.logger.warning(f"Connector {name} health check failed")
                elif is_healthy and connector.status == DataSourceStatus.ERROR:
                    connector.status = DataSourceStatus.ACTIVE
                    self.logger.info(f"Connector {name} recovered")
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
    
    async def shutdown(self):
        """关闭管理器"""
        # 取消健康检查任务
        if self._health_check_task:
            self._health_check_task.cancel()
        
        # 断开所有连接器
        for connector in self.connectors.values():
            try:
                await connector.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting connector: {e}")
        
        self.connectors.clear()
        self.connector_stats.clear()
        
        self.logger.info("ConnectorManager shutdown complete")

# 便捷函数
def create_connector_manager(
    configs: List[DataSourceConfig],
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.PRIORITY
) -> ConnectorManager:
    """创建连接器管理器"""
    manager = ConnectorManager(load_balance_strategy=load_balance_strategy)
    
    # 异步添加连接器
    async def add_connectors():
        for config in configs:
            await manager.add_connector(config)
    
    # 这里需要在异步环境中调用
    return manager

def create_data_request(
    data_type: DataType,
    symbols: List[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    frequency: DataFrequency = DataFrequency.DAILY,
    **kwargs
) -> DataRequest:
    """创建数据请求"""
    return DataRequest(
        request_id=f"{data_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        data_type=data_type,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        **kwargs
    )