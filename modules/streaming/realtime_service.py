# 实时数据推送服务 - 整合WebSocket和数据连接器

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .websocket_server import WebSocketServer, SubscriptionType, MessageType
from .data_connector import (
    DataConnectorManager, DataSource, DataType, DataRequest, DataResponse, DataStatus,
    AkShareConnector, TuShareConnector, YahooFinanceConnector
)
from .stream_processor import StreamProcessor, StreamType

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """服务状态枚举"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class RealtimeConfig:
    """实时服务配置"""
    # WebSocket配置
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    max_connections: int = 1000
    heartbeat_interval: int = 30
    
    # 数据源配置
    data_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 推送配置
    push_interval: int = 1  # 秒
    batch_size: int = 100
    
    # 缓存配置
    cache_enabled: bool = True
    cache_ttl: int = 300
    
    # 其他配置
    enable_auth: bool = False
    jwt_secret: str = "your-secret-key"
    log_level: str = "INFO"

@dataclass
class DataSubscription:
    """数据订阅配置"""
    subscription_id: str
    data_source: DataSource
    data_type: DataType
    symbols: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    push_interval: int = 1  # 秒
    last_push: Optional[datetime] = None
    is_active: bool = True

class RealtimeDataService:
    """实时数据推送服务
    
    整合WebSocket服务器和数据连接器，提供：
    1. 实时数据推送
    2. 客户端连接管理
    3. 数据源管理
    4. 订阅管理
    5. 数据缓存
    """
    
    def __init__(self, config: RealtimeConfig):
        self.config = config
        self.status = ServiceStatus.STOPPED
        
        # 初始化WebSocket服务器
        websocket_config = {
            'host': config.websocket_host,
            'port': config.websocket_port,
            'max_connections': config.max_connections,
            'heartbeat_interval': config.heartbeat_interval,
            'auth_required': config.enable_auth,
            'jwt_secret': config.jwt_secret
        }
        self.websocket_server = WebSocketServer(websocket_config)
        
        # 初始化数据连接器管理器
        connector_config = {
            'cache_enabled': config.cache_enabled,
            'cache_ttl': config.cache_ttl,
            'queue_size': 1000,
            'worker_count': 5
        }
        self.connector_manager = DataConnectorManager(connector_config)
        
        # 初始化流处理器
        stream_config = {
            'backend': 'memory',  # 可配置为redis或kafka
            'batch_size': config.batch_size
        }
        self.stream_processor = StreamProcessor(stream_config)
        
        # 数据订阅管理
        self.data_subscriptions: Dict[str, DataSubscription] = {}
        self.subscription_mapping: Dict[str, Set[str]] = {}  # websocket_subscription_id -> data_subscription_ids
        
        # 后台任务
        self.background_tasks: Set[asyncio.Task] = set()
        
        # 统计信息
        self.stats = {
            'start_time': None,
            'total_data_requests': 0,
            'successful_pushes': 0,
            'failed_pushes': 0,
            'active_subscriptions': 0,
            'last_activity': None
        }
        
        logger.info("实时数据服务初始化完成")
    
    async def start(self):
        """启动服务"""
        try:
            if self.status != ServiceStatus.STOPPED:
                logger.warning(f"服务已在运行，当前状态: {self.status.value}")
                return
            
            self.status = ServiceStatus.STARTING
            logger.info("启动实时数据服务...")
            
            # 初始化数据连接器
            await self._initialize_connectors()
            
            # 启动数据连接器管理器
            await self.connector_manager.start()
            
            # 启动流处理器
            await self.stream_processor.start()
            
            # 启动WebSocket服务器
            await self.websocket_server.start_server()
            
            # 启动后台任务
            await self._start_background_tasks()
            
            self.status = ServiceStatus.RUNNING
            self.stats['start_time'] = datetime.now()
            
            logger.info(f"实时数据服务启动成功: ws://{self.config.websocket_host}:{self.config.websocket_port}")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            logger.error(f"实时数据服务启动失败: {str(e)}")
            raise
    
    async def stop(self):
        """停止服务"""
        try:
            if self.status == ServiceStatus.STOPPED:
                logger.warning("服务已停止")
                return
            
            self.status = ServiceStatus.STOPPING
            logger.info("停止实时数据服务...")
            
            # 停止后台任务
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.background_tasks.clear()
            
            # 停止WebSocket服务器
            await self.websocket_server.stop_server()
            
            # 停止流处理器
            await self.stream_processor.stop()
            
            # 停止数据连接器管理器
            await self.connector_manager.stop()
            
            # 清理订阅
            self.data_subscriptions.clear()
            self.subscription_mapping.clear()
            
            self.status = ServiceStatus.STOPPED
            logger.info("实时数据服务已停止")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            logger.error(f"实时数据服务停止失败: {str(e)}")
    
    async def _initialize_connectors(self):
        """初始化数据连接器"""
        try:
            # 注册AkShare连接器
            if 'akshare' in self.config.data_sources:
                akshare_config = self.config.data_sources['akshare']
                akshare_connector = AkShareConnector(akshare_config)
                self.connector_manager.register_connector(DataSource.AKSHARE, akshare_connector)
                logger.info("AkShare连接器注册成功")
            
            # 注册TuShare连接器
            if 'tushare' in self.config.data_sources:
                tushare_config = self.config.data_sources['tushare']
                tushare_connector = TuShareConnector(tushare_config)
                self.connector_manager.register_connector(DataSource.TUSHARE, tushare_connector)
                logger.info("TuShare连接器注册成功")
            
            # 注册Yahoo Finance连接器
            if 'yahoo_finance' in self.config.data_sources:
                yahoo_config = self.config.data_sources['yahoo_finance']
                yahoo_connector = YahooFinanceConnector(yahoo_config)
                self.connector_manager.register_connector(DataSource.YAHOO_FINANCE, yahoo_connector)
                logger.info("Yahoo Finance连接器注册成功")
            
            logger.info(f"数据连接器初始化完成，共注册{len(self.connector_manager.connectors)}个连接器")
            
        except Exception as e:
            logger.error(f"数据连接器初始化失败: {str(e)}")
            raise
    
    async def _start_background_tasks(self):
        """启动后台任务"""
        try:
            # 数据推送任务
            push_task = asyncio.create_task(self._data_push_loop())
            self.background_tasks.add(push_task)
            push_task.add_done_callback(self.background_tasks.discard)
            
            # 订阅管理任务
            subscription_task = asyncio.create_task(self._subscription_manager())
            self.background_tasks.add(subscription_task)
            subscription_task.add_done_callback(self.background_tasks.discard)
            
            # 统计更新任务
            stats_task = asyncio.create_task(self._stats_updater())
            self.background_tasks.add(stats_task)
            stats_task.add_done_callback(self.background_tasks.discard)
            
            logger.info(f"后台任务启动成功，共{len(self.background_tasks)}个任务")
            
        except Exception as e:
            logger.error(f"后台任务启动失败: {str(e)}")
            raise
    
    async def _data_push_loop(self):
        """数据推送循环"""
        logger.info("数据推送任务启动")
        
        try:
            while self.status == ServiceStatus.RUNNING:
                try:
                    current_time = datetime.now()
                    
                    # 检查需要推送的订阅
                    push_tasks = []
                    for subscription_id, subscription in self.data_subscriptions.items():
                        if not subscription.is_active:
                            continue
                        
                        # 检查推送间隔
                        if (subscription.last_push is None or 
                            (current_time - subscription.last_push).total_seconds() >= subscription.push_interval):
                            
                            task = asyncio.create_task(
                                self._push_subscription_data(subscription)
                            )
                            push_tasks.append(task)
                    
                    # 执行推送任务
                    if push_tasks:
                        await asyncio.gather(*push_tasks, return_exceptions=True)
                    
                    # 等待下一次推送
                    await asyncio.sleep(self.config.push_interval)
                    
                except Exception as e:
                    logger.error(f"数据推送循环异常: {str(e)}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logger.info("数据推送任务取消")
        except Exception as e:
            logger.error(f"数据推送任务失败: {str(e)}")
    
    async def _push_subscription_data(self, subscription: DataSubscription):
        """推送订阅数据"""
        try:
            # 创建数据请求
            request = DataRequest(
                request_id=f"push_{subscription.subscription_id}_{datetime.now().timestamp()}",
                data_source=subscription.data_source,
                data_type=subscription.data_type,
                symbols=subscription.symbols,
                parameters=subscription.parameters
            )
            
            # 获取数据
            response = await self.connector_manager.fetch_data(request)
            
            if response.status == DataStatus.SUCCESS and response.data is not None:
                # 确定订阅类型映射
                websocket_subscription_type = self._map_to_websocket_subscription(subscription.data_type)
                
                # 准备推送数据
                push_data = {
                    'subscription_id': subscription.subscription_id,
                    'data_source': subscription.data_source.value,
                    'data_type': subscription.data_type.value,
                    'symbols': subscription.symbols,
                    'timestamp': datetime.now().isoformat(),
                    'data': self._serialize_data(response.data)
                }
                
                # 通过WebSocket推送
                await self.websocket_server.broadcast_data(
                    websocket_subscription_type,
                    push_data
                )
                
                # 更新推送时间
                subscription.last_push = datetime.now()
                
                # 更新统计
                self.stats['successful_pushes'] += 1
                self.stats['last_activity'] = datetime.now()
                
                logger.debug(f"数据推送成功: {subscription.subscription_id}")
                
            else:
                logger.warning(f"数据获取失败: {subscription.subscription_id}, 状态: {response.status.value}")
                self.stats['failed_pushes'] += 1
            
            self.stats['total_data_requests'] += 1
            
        except Exception as e:
            logger.error(f"订阅数据推送失败: {subscription.subscription_id}, 错误: {str(e)}")
            self.stats['failed_pushes'] += 1
    
    def _map_to_websocket_subscription(self, data_type: DataType) -> SubscriptionType:
        """映射数据类型到WebSocket订阅类型"""
        mapping = {
            DataType.REAL_TIME_QUOTES: SubscriptionType.REAL_TIME_QUOTES,
            DataType.HISTORICAL_DATA: SubscriptionType.MARKET_DATA,
            DataType.FINANCIAL_DATA: SubscriptionType.MARKET_DATA,
            DataType.NEWS_DATA: SubscriptionType.NEWS_ALERTS,
            DataType.SENTIMENT_DATA: SubscriptionType.NEWS_ALERTS,
            DataType.MACRO_DATA: SubscriptionType.MARKET_DATA,
            DataType.ALTERNATIVE_DATA: SubscriptionType.CUSTOM
        }
        return mapping.get(data_type, SubscriptionType.CUSTOM)
    
    def _serialize_data(self, data: Any) -> Any:
        """序列化数据"""
        try:
            # 处理pandas DataFrame
            if hasattr(data, 'to_dict'):
                return data.to_dict('records')
            
            # 处理numpy数组
            if hasattr(data, 'tolist'):
                return data.tolist()
            
            # 处理字典和列表
            if isinstance(data, (dict, list, str, int, float, bool)):
                return data
            
            # 其他类型转换为字符串
            return str(data)
            
        except Exception as e:
            logger.error(f"数据序列化失败: {str(e)}")
            return str(data)
    
    async def _subscription_manager(self):
        """订阅管理任务"""
        logger.info("订阅管理任务启动")
        
        try:
            while self.status == ServiceStatus.RUNNING:
                try:
                    # 清理无效订阅
                    inactive_subscriptions = []
                    for subscription_id, subscription in self.data_subscriptions.items():
                        # 检查是否有对应的WebSocket订阅
                        has_websocket_subscription = any(
                            subscription_id in data_sub_ids 
                            for data_sub_ids in self.subscription_mapping.values()
                        )
                        
                        if not has_websocket_subscription:
                            inactive_subscriptions.append(subscription_id)
                    
                    # 移除无效订阅
                    for subscription_id in inactive_subscriptions:
                        del self.data_subscriptions[subscription_id]
                        logger.info(f"移除无效订阅: {subscription_id}")
                    
                    # 更新统计
                    self.stats['active_subscriptions'] = len(self.data_subscriptions)
                    
                    await asyncio.sleep(60)  # 每分钟检查一次
                    
                except Exception as e:
                    logger.error(f"订阅管理任务异常: {str(e)}")
                    await asyncio.sleep(10)
                    
        except asyncio.CancelledError:
            logger.info("订阅管理任务取消")
        except Exception as e:
            logger.error(f"订阅管理任务失败: {str(e)}")
    
    async def _stats_updater(self):
        """统计更新任务"""
        logger.info("统计更新任务启动")
        
        try:
            while self.status == ServiceStatus.RUNNING:
                try:
                    # 更新统计信息
                    self.stats['active_subscriptions'] = len(self.data_subscriptions)
                    
                    # 记录统计日志
                    if self.stats['total_data_requests'] > 0:
                        success_rate = (self.stats['successful_pushes'] / 
                                      self.stats['total_data_requests'] * 100)
                        logger.info(
                            f"服务统计 - 活跃订阅: {self.stats['active_subscriptions']}, "
                            f"总请求: {self.stats['total_data_requests']}, "
                            f"成功率: {success_rate:.1f}%"
                        )
                    
                    await asyncio.sleep(300)  # 每5分钟更新一次
                    
                except Exception as e:
                    logger.error(f"统计更新任务异常: {str(e)}")
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            logger.info("统计更新任务取消")
        except Exception as e:
            logger.error(f"统计更新任务失败: {str(e)}")
    
    def create_data_subscription(self, websocket_subscription_id: str, 
                               data_source: DataSource, data_type: DataType,
                               symbols: List[str] = None, 
                               parameters: Dict[str, Any] = None,
                               push_interval: int = 1) -> str:
        """创建数据订阅
        
        Args:
            websocket_subscription_id: WebSocket订阅ID
            data_source: 数据源
            data_type: 数据类型
            symbols: 股票代码列表
            parameters: 参数
            push_interval: 推送间隔（秒）
            
        Returns:
            数据订阅ID
        """
        try:
            subscription_id = f"data_{websocket_subscription_id}_{datetime.now().timestamp()}"
            
            subscription = DataSubscription(
                subscription_id=subscription_id,
                data_source=data_source,
                data_type=data_type,
                symbols=symbols or [],
                parameters=parameters or {},
                push_interval=push_interval
            )
            
            self.data_subscriptions[subscription_id] = subscription
            
            # 建立映射关系
            if websocket_subscription_id not in self.subscription_mapping:
                self.subscription_mapping[websocket_subscription_id] = set()
            self.subscription_mapping[websocket_subscription_id].add(subscription_id)
            
            logger.info(f"数据订阅创建成功: {subscription_id}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"数据订阅创建失败: {str(e)}")
            raise
    
    def remove_data_subscription(self, websocket_subscription_id: str):
        """移除数据订阅
        
        Args:
            websocket_subscription_id: WebSocket订阅ID
        """
        try:
            if websocket_subscription_id in self.subscription_mapping:
                data_subscription_ids = self.subscription_mapping[websocket_subscription_id]
                
                # 移除数据订阅
                for subscription_id in data_subscription_ids:
                    if subscription_id in self.data_subscriptions:
                        del self.data_subscriptions[subscription_id]
                        logger.info(f"数据订阅移除成功: {subscription_id}")
                
                # 移除映射
                del self.subscription_mapping[websocket_subscription_id]
                
        except Exception as e:
            logger.error(f"数据订阅移除失败: {str(e)}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        websocket_stats = self.websocket_server.get_server_stats()
        connector_stats = self.connector_manager.get_connector_stats()
        
        return {
            'status': self.status.value,
            'uptime': (datetime.now() - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0,
            'websocket': websocket_stats,
            'connectors': connector_stats,
            'subscriptions': {
                'total_data_subscriptions': len(self.data_subscriptions),
                'active_mappings': len(self.subscription_mapping)
            },
            'performance': {
                'total_requests': self.stats['total_data_requests'],
                'successful_pushes': self.stats['successful_pushes'],
                'failed_pushes': self.stats['failed_pushes'],
                'success_rate': (self.stats['successful_pushes'] / max(1, self.stats['total_data_requests'])) * 100,
                'last_activity': self.stats['last_activity'].isoformat() if self.stats['last_activity'] else None
            }
        }
    
    def get_subscription_info(self) -> List[Dict[str, Any]]:
        """获取订阅信息"""
        subscriptions = []
        for subscription_id, subscription in self.data_subscriptions.items():
            subscriptions.append({
                'subscription_id': subscription_id,
                'data_source': subscription.data_source.value,
                'data_type': subscription.data_type.value,
                'symbols': subscription.symbols,
                'push_interval': subscription.push_interval,
                'last_push': subscription.last_push.isoformat() if subscription.last_push else None,
                'is_active': subscription.is_active
            })
        return subscriptions

# 便捷函数
def create_realtime_service(config_dict: Dict[str, Any] = None) -> RealtimeDataService:
    """创建实时数据服务
    
    Args:
        config_dict: 配置字典
        
    Returns:
        实时数据服务实例
    """
    if config_dict is None:
        config_dict = {}
    
    # 默认配置
    default_config = {
        'websocket_host': 'localhost',
        'websocket_port': 8765,
        'max_connections': 1000,
        'heartbeat_interval': 30,
        'push_interval': 1,
        'batch_size': 100,
        'cache_enabled': True,
        'cache_ttl': 300,
        'enable_auth': False,
        'jwt_secret': 'your-secret-key',
        'log_level': 'INFO',
        'data_sources': {
            'akshare': {
                'name': 'AkShare',
                'enabled': True,
                'rate_limit': 100
            },
            'tushare': {
                'name': 'TuShare',
                'enabled': False,  # 需要token
                'token': '',
                'rate_limit': 200
            },
            'yahoo_finance': {
                'name': 'Yahoo Finance',
                'enabled': True,
                'rate_limit': 100
            }
        }
    }
    
    # 合并配置
    merged_config = {**default_config, **config_dict}
    config = RealtimeConfig(**merged_config)
    
    return RealtimeDataService(config)