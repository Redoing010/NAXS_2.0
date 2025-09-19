# 数据连接器 - 实现多源数据接入

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import tushare as ts
except ImportError:
    ts = None

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """数据源枚举"""
    AKSHARE = "akshare"              # AkShare
    TUSHARE = "tushare"              # TuShare
    YAHOO_FINANCE = "yahoo_finance"  # Yahoo Finance
    EASTMONEY = "eastmoney"          # 东方财富
    SINA_FINANCE = "sina_finance"    # 新浪财经
    TENCENT_FINANCE = "tencent"      # 腾讯财经
    CUSTOM_API = "custom_api"        # 自定义API
    WEBSOCKET = "websocket"          # WebSocket数据源
    FILE = "file"                    # 文件数据源

class DataType(Enum):
    """数据类型枚举"""
    REAL_TIME_QUOTES = "quotes"      # 实时行情
    HISTORICAL_DATA = "historical"   # 历史数据
    FINANCIAL_DATA = "financial"     # 财务数据
    NEWS_DATA = "news"               # 新闻数据
    ANNOUNCEMENT = "announcement"    # 公告数据
    SENTIMENT_DATA = "sentiment"     # 情绪数据
    MACRO_DATA = "macro"             # 宏观数据
    ALTERNATIVE_DATA = "alternative" # 替代数据
    CUSTOM = "custom"                # 自定义数据

class DataStatus(Enum):
    """数据状态枚举"""
    PENDING = "pending"              # 等待中
    FETCHING = "fetching"            # 获取中
    SUCCESS = "success"              # 成功
    FAILED = "failed"                # 失败
    TIMEOUT = "timeout"              # 超时
    RATE_LIMITED = "rate_limited"    # 限流

@dataclass
class DataRequest:
    """数据请求"""
    request_id: str
    data_source: DataSource
    data_type: DataType
    symbols: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1-10, 10最高
    timeout: int = 30  # 秒
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class DataResponse:
    """数据响应"""
    request_id: str
    data_source: DataSource
    data_type: DataType
    status: DataStatus
    data: Any = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    fetch_time: float = 0.0  # 获取耗时
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseDataConnector(ABC):
    """数据连接器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.enabled = config.get('enabled', True)
        self.rate_limit = config.get('rate_limit', 100)  # 每分钟请求数
        self.timeout = config.get('timeout', 30)
        
        # 限流控制
        self.request_times = []
        self.rate_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'last_request_time': None
        }
        
        logger.info(f"数据连接器初始化: {self.name}")
    
    @abstractmethod
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """获取数据"""
        pass
    
    @abstractmethod
    def get_supported_data_types(self) -> List[DataType]:
        """获取支持的数据类型"""
        pass
    
    def check_rate_limit(self) -> bool:
        """检查限流"""
        with self.rate_lock:
            current_time = time.time()
            # 清理1分钟前的请求记录
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            
            if len(self.request_times) >= self.rate_limit:
                return False
            
            self.request_times.append(current_time)
            return True
    
    def update_stats(self, success: bool, response_time: float):
        """更新统计信息"""
        self.stats['total_requests'] += 1
        self.stats['last_request_time'] = datetime.now()
        
        if success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        # 更新平均响应时间
        total_time = self.stats['avg_response_time'] * (self.stats['total_requests'] - 1)
        self.stats['avg_response_time'] = (total_time + response_time) / self.stats['total_requests']
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'supported_types': [dt.value for dt in self.get_supported_data_types()],
            **self.stats
        }

class AkShareConnector(BaseDataConnector):
    """AkShare数据连接器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if ak is None:
            logger.error("AkShare未安装，请安装: pip install akshare")
            self.enabled = False
        
        logger.info("AkShare连接器初始化完成")
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """获取AkShare数据"""
        start_time = time.time()
        
        try:
            if not self.enabled:
                return DataResponse(
                    request_id=request.request_id,
                    data_source=DataSource.AKSHARE,
                    data_type=request.data_type,
                    status=DataStatus.FAILED,
                    error="AkShare connector is disabled"
                )
            
            if not self.check_rate_limit():
                return DataResponse(
                    request_id=request.request_id,
                    data_source=DataSource.AKSHARE,
                    data_type=request.data_type,
                    status=DataStatus.RATE_LIMITED,
                    error="Rate limit exceeded"
                )
            
            # 在线程池中执行同步操作
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                data = await loop.run_in_executor(
                    executor, 
                    self._fetch_akshare_data, 
                    request
                )
            
            fetch_time = time.time() - start_time
            self.update_stats(True, fetch_time)
            
            return DataResponse(
                request_id=request.request_id,
                data_source=DataSource.AKSHARE,
                data_type=request.data_type,
                status=DataStatus.SUCCESS,
                data=data,
                fetch_time=fetch_time
            )
            
        except Exception as e:
            fetch_time = time.time() - start_time
            self.update_stats(False, fetch_time)
            
            logger.error(f"AkShare数据获取失败: {str(e)}")
            return DataResponse(
                request_id=request.request_id,
                data_source=DataSource.AKSHARE,
                data_type=request.data_type,
                status=DataStatus.FAILED,
                error=str(e),
                fetch_time=fetch_time
            )
    
    def _fetch_akshare_data(self, request: DataRequest) -> Any:
        """获取AkShare数据（同步方法）"""
        try:
            if request.data_type == DataType.REAL_TIME_QUOTES:
                # 实时行情
                if request.symbols:
                    symbol = request.symbols[0].replace('.', '')
                    return ak.stock_zh_a_spot_em().query(f"代码 == '{symbol}'")
                else:
                    return ak.stock_zh_a_spot_em()
            
            elif request.data_type == DataType.HISTORICAL_DATA:
                # 历史数据
                symbol = request.parameters.get('symbol', '000001')
                period = request.parameters.get('period', 'daily')
                start_date = request.parameters.get('start_date')
                end_date = request.parameters.get('end_date')
                
                if period == 'daily':
                    return ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date
                    )
                elif period == 'minute':
                    return ak.stock_zh_a_hist_min_em(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date
                    )
            
            elif request.data_type == DataType.NEWS_DATA:
                # 新闻数据
                return ak.stock_news_em()
            
            elif request.data_type == DataType.FINANCIAL_DATA:
                # 财务数据
                symbol = request.parameters.get('symbol', '000001')
                return ak.stock_financial_abstract(
                    symbol=symbol
                )
            
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
                
        except Exception as e:
            logger.error(f"AkShare数据获取异常: {str(e)}")
            raise
    
    def get_supported_data_types(self) -> List[DataType]:
        """获取支持的数据类型"""
        return [
            DataType.REAL_TIME_QUOTES,
            DataType.HISTORICAL_DATA,
            DataType.FINANCIAL_DATA,
            DataType.NEWS_DATA
        ]

class TuShareConnector(BaseDataConnector):
    """TuShare数据连接器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.token = config.get('token')
        
        if ts is None:
            logger.error("TuShare未安装，请安装: pip install tushare")
            self.enabled = False
        elif not self.token:
            logger.warning("TuShare token未配置")
            self.enabled = False
        else:
            try:
                ts.set_token(self.token)
                self.pro = ts.pro_api()
                logger.info("TuShare连接器初始化完成")
            except Exception as e:
                logger.error(f"TuShare初始化失败: {str(e)}")
                self.enabled = False
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """获取TuShare数据"""
        start_time = time.time()
        
        try:
            if not self.enabled:
                return DataResponse(
                    request_id=request.request_id,
                    data_source=DataSource.TUSHARE,
                    data_type=request.data_type,
                    status=DataStatus.FAILED,
                    error="TuShare connector is disabled"
                )
            
            if not self.check_rate_limit():
                return DataResponse(
                    request_id=request.request_id,
                    data_source=DataSource.TUSHARE,
                    data_type=request.data_type,
                    status=DataStatus.RATE_LIMITED,
                    error="Rate limit exceeded"
                )
            
            # 在线程池中执行同步操作
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                data = await loop.run_in_executor(
                    executor, 
                    self._fetch_tushare_data, 
                    request
                )
            
            fetch_time = time.time() - start_time
            self.update_stats(True, fetch_time)
            
            return DataResponse(
                request_id=request.request_id,
                data_source=DataSource.TUSHARE,
                data_type=request.data_type,
                status=DataStatus.SUCCESS,
                data=data,
                fetch_time=fetch_time
            )
            
        except Exception as e:
            fetch_time = time.time() - start_time
            self.update_stats(False, fetch_time)
            
            logger.error(f"TuShare数据获取失败: {str(e)}")
            return DataResponse(
                request_id=request.request_id,
                data_source=DataSource.TUSHARE,
                data_type=request.data_type,
                status=DataStatus.FAILED,
                error=str(e),
                fetch_time=fetch_time
            )
    
    def _fetch_tushare_data(self, request: DataRequest) -> Any:
        """获取TuShare数据（同步方法）"""
        try:
            if request.data_type == DataType.REAL_TIME_QUOTES:
                # 实时行情
                return self.pro.daily(trade_date=datetime.now().strftime('%Y%m%d'))
            
            elif request.data_type == DataType.HISTORICAL_DATA:
                # 历史数据
                ts_code = request.parameters.get('ts_code')
                start_date = request.parameters.get('start_date')
                end_date = request.parameters.get('end_date')
                
                return self.pro.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
            
            elif request.data_type == DataType.FINANCIAL_DATA:
                # 财务数据
                ts_code = request.parameters.get('ts_code')
                period = request.parameters.get('period')
                
                return self.pro.income(
                    ts_code=ts_code,
                    period=period
                )
            
            elif request.data_type == DataType.NEWS_DATA:
                # 新闻数据
                return self.pro.news(
                    start_date=request.parameters.get('start_date'),
                    end_date=request.parameters.get('end_date')
                )
            
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
                
        except Exception as e:
            logger.error(f"TuShare数据获取异常: {str(e)}")
            raise
    
    def get_supported_data_types(self) -> List[DataType]:
        """获取支持的数据类型"""
        return [
            DataType.REAL_TIME_QUOTES,
            DataType.HISTORICAL_DATA,
            DataType.FINANCIAL_DATA,
            DataType.NEWS_DATA
        ]

class YahooFinanceConnector(BaseDataConnector):
    """Yahoo Finance数据连接器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if yf is None:
            logger.error("yfinance未安装，请安装: pip install yfinance")
            self.enabled = False
        
        logger.info("Yahoo Finance连接器初始化完成")
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """获取Yahoo Finance数据"""
        start_time = time.time()
        
        try:
            if not self.enabled:
                return DataResponse(
                    request_id=request.request_id,
                    data_source=DataSource.YAHOO_FINANCE,
                    data_type=request.data_type,
                    status=DataStatus.FAILED,
                    error="Yahoo Finance connector is disabled"
                )
            
            if not self.check_rate_limit():
                return DataResponse(
                    request_id=request.request_id,
                    data_source=DataSource.YAHOO_FINANCE,
                    data_type=request.data_type,
                    status=DataStatus.RATE_LIMITED,
                    error="Rate limit exceeded"
                )
            
            # 在线程池中执行同步操作
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                data = await loop.run_in_executor(
                    executor, 
                    self._fetch_yahoo_data, 
                    request
                )
            
            fetch_time = time.time() - start_time
            self.update_stats(True, fetch_time)
            
            return DataResponse(
                request_id=request.request_id,
                data_source=DataSource.YAHOO_FINANCE,
                data_type=request.data_type,
                status=DataStatus.SUCCESS,
                data=data,
                fetch_time=fetch_time
            )
            
        except Exception as e:
            fetch_time = time.time() - start_time
            self.update_stats(False, fetch_time)
            
            logger.error(f"Yahoo Finance数据获取失败: {str(e)}")
            return DataResponse(
                request_id=request.request_id,
                data_source=DataSource.YAHOO_FINANCE,
                data_type=request.data_type,
                status=DataStatus.FAILED,
                error=str(e),
                fetch_time=fetch_time
            )
    
    def _fetch_yahoo_data(self, request: DataRequest) -> Any:
        """获取Yahoo Finance数据（同步方法）"""
        try:
            if request.data_type == DataType.HISTORICAL_DATA:
                # 历史数据
                symbols = request.symbols or ['AAPL']
                period = request.parameters.get('period', '1y')
                interval = request.parameters.get('interval', '1d')
                
                if len(symbols) == 1:
                    ticker = yf.Ticker(symbols[0])
                    return ticker.history(period=period, interval=interval)
                else:
                    return yf.download(symbols, period=period, interval=interval)
            
            elif request.data_type == DataType.REAL_TIME_QUOTES:
                # 实时行情
                symbols = request.symbols or ['AAPL']
                tickers = yf.Tickers(' '.join(symbols))
                
                result = {}
                for symbol in symbols:
                    ticker = tickers.tickers[symbol]
                    result[symbol] = ticker.info
                
                return result
            
            elif request.data_type == DataType.FINANCIAL_DATA:
                # 财务数据
                symbol = request.symbols[0] if request.symbols else 'AAPL'
                ticker = yf.Ticker(symbol)
                
                return {
                    'info': ticker.info,
                    'financials': ticker.financials,
                    'balance_sheet': ticker.balance_sheet,
                    'cashflow': ticker.cashflow
                }
            
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
                
        except Exception as e:
            logger.error(f"Yahoo Finance数据获取异常: {str(e)}")
            raise
    
    def get_supported_data_types(self) -> List[DataType]:
        """获取支持的数据类型"""
        return [
            DataType.REAL_TIME_QUOTES,
            DataType.HISTORICAL_DATA,
            DataType.FINANCIAL_DATA
        ]

class DataConnectorManager:
    """数据连接器管理器
    
    统一管理所有数据连接器，提供：
    1. 连接器注册和管理
    2. 数据请求路由
    3. 负载均衡
    4. 故障转移
    5. 缓存管理
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connectors: Dict[DataSource, BaseDataConnector] = {}
        self.request_queue = asyncio.Queue(maxsize=config.get('queue_size', 1000))
        self.response_callbacks: Dict[str, Callable] = {}
        
        # 缓存配置
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_ttl = config.get('cache_ttl', 300)  # 5分钟
        self.cache: Dict[str, tuple] = {}  # (data, timestamp)
        
        # 工作线程
        self.worker_count = config.get('worker_count', 5)
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        
        logger.info("数据连接器管理器初始化完成")
    
    def register_connector(self, data_source: DataSource, connector: BaseDataConnector):
        """注册数据连接器"""
        self.connectors[data_source] = connector
        logger.info(f"数据连接器注册成功: {data_source.value}")
    
    def unregister_connector(self, data_source: DataSource):
        """注销数据连接器"""
        if data_source in self.connectors:
            del self.connectors[data_source]
            logger.info(f"数据连接器注销成功: {data_source.value}")
    
    async def start(self):
        """启动管理器"""
        if self.is_running:
            logger.warning("数据连接器管理器已在运行")
            return
        
        self.is_running = True
        
        # 启动工作线程
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # 启动缓存清理任务
        cache_cleaner = asyncio.create_task(self._cache_cleaner())
        self.workers.append(cache_cleaner)
        
        logger.info(f"数据连接器管理器启动成功，工作线程数: {self.worker_count}")
    
    async def stop(self):
        """停止管理器"""
        self.is_running = False
        
        # 取消所有工作线程
        for worker in self.workers:
            if not worker.done():
                worker.cancel()
        
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        logger.info("数据连接器管理器已停止")
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """获取数据
        
        Args:
            request: 数据请求
            
        Returns:
            数据响应
        """
        try:
            # 检查缓存
            if self.cache_enabled:
                cache_key = self._get_cache_key(request)
                cached_data = self._get_cached_data(cache_key)
                if cached_data is not None:
                    return DataResponse(
                        request_id=request.request_id,
                        data_source=request.data_source,
                        data_type=request.data_type,
                        status=DataStatus.SUCCESS,
                        data=cached_data,
                        metadata={'from_cache': True}
                    )
            
            # 检查连接器是否存在
            if request.data_source not in self.connectors:
                return DataResponse(
                    request_id=request.request_id,
                    data_source=request.data_source,
                    data_type=request.data_type,
                    status=DataStatus.FAILED,
                    error=f"Connector not found: {request.data_source.value}"
                )
            
            connector = self.connectors[request.data_source]
            
            # 检查数据类型支持
            if request.data_type not in connector.get_supported_data_types():
                return DataResponse(
                    request_id=request.request_id,
                    data_source=request.data_source,
                    data_type=request.data_type,
                    status=DataStatus.FAILED,
                    error=f"Data type not supported: {request.data_type.value}"
                )
            
            # 获取数据
            response = await connector.fetch_data(request)
            
            # 缓存成功的响应
            if (self.cache_enabled and 
                response.status == DataStatus.SUCCESS and 
                response.data is not None):
                cache_key = self._get_cache_key(request)
                self._cache_data(cache_key, response.data)
            
            return response
            
        except Exception as e:
            logger.error(f"数据获取失败: {str(e)}")
            return DataResponse(
                request_id=request.request_id,
                data_source=request.data_source,
                data_type=request.data_type,
                status=DataStatus.FAILED,
                error=str(e)
            )
    
    async def fetch_data_async(self, request: DataRequest, callback: Callable = None):
        """异步获取数据
        
        Args:
            request: 数据请求
            callback: 回调函数
        """
        try:
            if callback:
                self.response_callbacks[request.request_id] = callback
            
            await self.request_queue.put(request)
            logger.debug(f"数据请求已加入队列: {request.request_id}")
            
        except asyncio.QueueFull:
            logger.error("请求队列已满")
            if callback:
                error_response = DataResponse(
                    request_id=request.request_id,
                    data_source=request.data_source,
                    data_type=request.data_type,
                    status=DataStatus.FAILED,
                    error="Request queue is full"
                )
                await callback(error_response)
    
    async def _worker(self, worker_name: str):
        """工作线程"""
        logger.info(f"工作线程启动: {worker_name}")
        
        try:
            while self.is_running:
                try:
                    # 获取请求
                    request = await asyncio.wait_for(
                        self.request_queue.get(), 
                        timeout=1.0
                    )
                    
                    # 处理请求
                    response = await self.fetch_data(request)
                    
                    # 调用回调
                    callback = self.response_callbacks.pop(request.request_id, None)
                    if callback:
                        try:
                            await callback(response)
                        except Exception as e:
                            logger.error(f"回调执行失败: {str(e)}")
                    
                    # 标记任务完成
                    self.request_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"工作线程异常: {worker_name}, 错误: {str(e)}")
                    
        except asyncio.CancelledError:
            logger.info(f"工作线程取消: {worker_name}")
        except Exception as e:
            logger.error(f"工作线程失败: {worker_name}, 错误: {str(e)}")
    
    async def _cache_cleaner(self):
        """缓存清理任务"""
        try:
            while self.is_running:
                await asyncio.sleep(60)  # 每分钟清理一次
                
                current_time = time.time()
                expired_keys = []
                
                for key, (data, timestamp) in self.cache.items():
                    if current_time - timestamp > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.cache[key]
                
                if expired_keys:
                    logger.debug(f"清理过期缓存: {len(expired_keys)}个")
                    
        except asyncio.CancelledError:
            logger.info("缓存清理任务取消")
        except Exception as e:
            logger.error(f"缓存清理任务失败: {str(e)}")
    
    def _get_cache_key(self, request: DataRequest) -> str:
        """生成缓存键"""
        key_parts = [
            request.data_source.value,
            request.data_type.value,
            ','.join(sorted(request.symbols)),
            json.dumps(request.parameters, sort_keys=True)
        ]
        return '|'.join(key_parts)
    
    def _get_cached_data(self, cache_key: str) -> Any:
        """获取缓存数据"""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp <= self.cache_ttl:
                return data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, data: Any):
        """缓存数据"""
        self.cache[cache_key] = (data, time.time())
    
    def get_connector_stats(self) -> Dict[str, Any]:
        """获取连接器统计信息"""
        stats = {
            'total_connectors': len(self.connectors),
            'enabled_connectors': sum(1 for c in self.connectors.values() if c.enabled),
            'queue_size': self.request_queue.qsize(),
            'cache_size': len(self.cache),
            'is_running': self.is_running,
            'worker_count': len(self.workers),
            'connectors': {}
        }
        
        for source, connector in self.connectors.items():
            stats['connectors'][source.value] = connector.get_stats()
        
        return stats
    
    def get_supported_sources(self) -> List[Dict[str, Any]]:
        """获取支持的数据源"""
        sources = []
        for source, connector in self.connectors.items():
            sources.append({
                'source': source.value,
                'enabled': connector.enabled,
                'supported_types': [dt.value for dt in connector.get_supported_data_types()]
            })
        return sources