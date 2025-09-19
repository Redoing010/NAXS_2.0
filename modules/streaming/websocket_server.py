# WebSocket服务器 - 实现实时数据推送

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError:
    websockets = None
    WebSocketServerProtocol = None
    ConnectionClosed = None
    WebSocketException = None

try:
    import jwt
except ImportError:
    jwt = None

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """消息类型枚举"""
    SUBSCRIBE = "subscribe"          # 订阅
    UNSUBSCRIBE = "unsubscribe"      # 取消订阅
    DATA = "data"                    # 数据推送
    HEARTBEAT = "heartbeat"          # 心跳
    AUTH = "auth"                    # 认证
    ERROR = "error"                  # 错误
    ACK = "ack"                      # 确认
    STATUS = "status"                # 状态

class SubscriptionType(Enum):
    """订阅类型枚举"""
    MARKET_DATA = "market_data"      # 市场数据
    REAL_TIME_QUOTES = "quotes"      # 实时报价
    TRADE_SIGNALS = "signals"        # 交易信号
    NEWS_ALERTS = "news"             # 新闻提醒
    FACTOR_VALUES = "factors"        # 因子值
    PORTFOLIO_UPDATES = "portfolio"  # 组合更新
    SYSTEM_ALERTS = "alerts"         # 系统告警
    CUSTOM = "custom"                # 自定义

@dataclass
class ClientConnection:
    """客户端连接信息"""
    connection_id: str
    websocket: WebSocketServerProtocol
    user_id: Optional[str] = None
    subscriptions: Set[str] = field(default_factory=set)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    connected_at: datetime = field(default_factory=datetime.now)
    ip_address: str = ""
    user_agent: str = ""
    is_authenticated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Subscription:
    """订阅信息"""
    subscription_id: str
    client_id: str
    subscription_type: SubscriptionType
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_data_sent: Optional[datetime] = None
    message_count: int = 0

@dataclass
class WebSocketMessage:
    """WebSocket消息"""
    message_type: MessageType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: Optional[str] = None
    client_id: Optional[str] = None

class WebSocketServer:
    """WebSocket服务器
    
    提供实时数据推送服务，支持：
    1. 客户端连接管理
    2. 订阅管理
    3. 实时数据推送
    4. 心跳检测
    5. 认证授权
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 8765)
        self.max_connections = config.get('max_connections', 1000)
        self.heartbeat_interval = config.get('heartbeat_interval', 30)  # 秒
        self.auth_required = config.get('auth_required', False)
        self.jwt_secret = config.get('jwt_secret', os.getenv('JWT_SECRET', 'your-secret-key'))
        
        # 连接管理
        self.connections: Dict[str, ClientConnection] = {}
        self.subscriptions: Dict[str, Subscription] = {}
        self.subscription_index: Dict[SubscriptionType, Set[str]] = defaultdict(set)
        
        # 数据处理器
        self.data_handlers: Dict[SubscriptionType, List[Callable]] = defaultdict(list)
        
        # 服务器状态
        self.server = None
        self.is_running = False
        self.background_tasks: Set[asyncio.Task] = set()
        
        # 统计信息
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'total_messages_sent': 0,
            'total_messages_received': 0,
            'start_time': None,
            'last_activity': None
        }
        
        if websockets is None:
            logger.error("websockets未安装，请安装: pip install websockets")
            raise ImportError("websockets library is required")
        
        logger.info(f"WebSocket服务器初始化完成，监听: {self.host}:{self.port}")
    
    async def start_server(self):
        """启动WebSocket服务器"""
        try:
            if self.is_running:
                logger.warning("WebSocket服务器已在运行")
                return
            
            # 启动WebSocket服务器
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                max_size=1024*1024,  # 1MB
                max_queue=100,
                compression=None,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.is_running = True
            self.stats['start_time'] = datetime.now()
            
            # 启动后台任务
            heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            cleanup_task = asyncio.create_task(self._cleanup_connections())
            
            self.background_tasks.add(heartbeat_task)
            self.background_tasks.add(cleanup_task)
            
            heartbeat_task.add_done_callback(self.background_tasks.discard)
            cleanup_task.add_done_callback(self.background_tasks.discard)
            
            logger.info(f"WebSocket服务器启动成功: ws://{self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"WebSocket服务器启动失败: {str(e)}")
            raise
    
    async def stop_server(self):
        """停止WebSocket服务器"""
        try:
            self.is_running = False
            
            # 关闭所有连接
            for connection in list(self.connections.values()):
                try:
                    await connection.websocket.close()
                except Exception:
                    pass
            
            self.connections.clear()
            self.subscriptions.clear()
            self.subscription_index.clear()
            
            # 取消后台任务
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.background_tasks.clear()
            
            # 关闭服务器
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            logger.info("WebSocket服务器已停止")
            
        except Exception as e:
            logger.error(f"WebSocket服务器停止失败: {str(e)}")
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """处理客户端连接"""
        connection_id = f"conn_{datetime.now().timestamp()}_{id(websocket)}"
        
        try:
            # 检查连接数限制
            if len(self.connections) >= self.max_connections:
                await websocket.close(code=1013, reason="Server overloaded")
                return
            
            # 创建连接对象
            connection = ClientConnection(
                connection_id=connection_id,
                websocket=websocket,
                ip_address=websocket.remote_address[0] if websocket.remote_address else "",
                user_agent=websocket.request_headers.get('User-Agent', '')
            )
            
            self.connections[connection_id] = connection
            self.stats['total_connections'] += 1
            self.stats['active_connections'] = len(self.connections)
            
            logger.info(f"客户端连接: {connection_id} from {connection.ip_address}")
            
            # 发送欢迎消息
            welcome_msg = WebSocketMessage(
                message_type=MessageType.STATUS,
                data={
                    'status': 'connected',
                    'connection_id': connection_id,
                    'server_time': datetime.now().isoformat(),
                    'auth_required': self.auth_required
                }
            )
            await self._send_message(connection, welcome_msg)
            
            # 处理消息循环
            async for message in websocket:
                try:
                    await self._handle_message(connection, message)
                    self.stats['total_messages_received'] += 1
                    self.stats['last_activity'] = datetime.now()
                except Exception as e:
                    logger.error(f"消息处理失败: {str(e)}")
                    await self._send_error(connection, str(e))
        
        except ConnectionClosed:
            logger.info(f"客户端断开连接: {connection_id}")
        except WebSocketException as e:
            logger.error(f"WebSocket异常: {connection_id}, 错误: {str(e)}")
        except Exception as e:
            logger.error(f"客户端处理异常: {connection_id}, 错误: {str(e)}")
        finally:
            # 清理连接
            await self._cleanup_connection(connection_id)
    
    async def _handle_message(self, connection: ClientConnection, raw_message: str):
        """处理客户端消息"""
        try:
            # 解析消息
            message_data = json.loads(raw_message)
            message_type = MessageType(message_data.get('type', 'data'))
            
            message = WebSocketMessage(
                message_type=message_type,
                data=message_data.get('data', {}),
                message_id=message_data.get('message_id'),
                client_id=connection.connection_id
            )
            
            # 更新心跳时间
            connection.last_heartbeat = datetime.now()
            
            # 根据消息类型处理
            if message_type == MessageType.AUTH:
                await self._handle_auth(connection, message)
            elif message_type == MessageType.SUBSCRIBE:
                await self._handle_subscribe(connection, message)
            elif message_type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(connection, message)
            elif message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(connection, message)
            else:
                logger.warning(f"未知消息类型: {message_type}")
                await self._send_error(connection, f"Unknown message type: {message_type.value}")
            
        except json.JSONDecodeError:
            await self._send_error(connection, "Invalid JSON format")
        except ValueError as e:
            await self._send_error(connection, f"Invalid message: {str(e)}")
        except Exception as e:
            logger.error(f"消息处理异常: {str(e)}")
            await self._send_error(connection, "Internal server error")
    
    async def _handle_auth(self, connection: ClientConnection, message: WebSocketMessage):
        """处理认证消息"""
        try:
            if not self.auth_required:
                connection.is_authenticated = True
                await self._send_ack(connection, "Authentication not required")
                return
            
            token = message.data.get('token')
            if not token:
                await self._send_error(connection, "Token required")
                return
            
            if jwt is None:
                logger.error("JWT库未安装，无法进行认证")
                await self._send_error(connection, "Authentication service unavailable")
                return
            
            # 验证JWT token
            try:
                payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
                connection.user_id = payload.get('user_id')
                connection.is_authenticated = True
                connection.metadata.update(payload)
                
                await self._send_ack(connection, "Authentication successful")
                logger.info(f"用户认证成功: {connection.user_id}")
                
            except jwt.ExpiredSignatureError:
                await self._send_error(connection, "Token expired")
            except jwt.InvalidTokenError:
                await self._send_error(connection, "Invalid token")
            
        except Exception as e:
            logger.error(f"认证处理失败: {str(e)}")
            await self._send_error(connection, "Authentication failed")
    
    async def _handle_subscribe(self, connection: ClientConnection, message: WebSocketMessage):
        """处理订阅消息"""
        try:
            if self.auth_required and not connection.is_authenticated:
                await self._send_error(connection, "Authentication required")
                return
            
            subscription_type_str = message.data.get('subscription_type')
            if not subscription_type_str:
                await self._send_error(connection, "Subscription type required")
                return
            
            try:
                subscription_type = SubscriptionType(subscription_type_str)
            except ValueError:
                await self._send_error(connection, f"Invalid subscription type: {subscription_type_str}")
                return
            
            # 创建订阅
            subscription_id = f"sub_{connection.connection_id}_{subscription_type.value}_{datetime.now().timestamp()}"
            
            subscription = Subscription(
                subscription_id=subscription_id,
                client_id=connection.connection_id,
                subscription_type=subscription_type,
                filters=message.data.get('filters', {})
            )
            
            # 存储订阅
            self.subscriptions[subscription_id] = subscription
            connection.subscriptions.add(subscription_id)
            self.subscription_index[subscription_type].add(subscription_id)
            
            # 发送确认
            await self._send_ack(connection, f"Subscribed to {subscription_type.value}", {
                'subscription_id': subscription_id,
                'subscription_type': subscription_type.value
            })
            
            logger.info(f"客户端订阅: {connection.connection_id} -> {subscription_type.value}")
            
        except Exception as e:
            logger.error(f"订阅处理失败: {str(e)}")
            await self._send_error(connection, "Subscription failed")
    
    async def _handle_unsubscribe(self, connection: ClientConnection, message: WebSocketMessage):
        """处理取消订阅消息"""
        try:
            subscription_id = message.data.get('subscription_id')
            if not subscription_id:
                await self._send_error(connection, "Subscription ID required")
                return
            
            if subscription_id not in self.subscriptions:
                await self._send_error(connection, "Subscription not found")
                return
            
            subscription = self.subscriptions[subscription_id]
            
            # 验证订阅所有权
            if subscription.client_id != connection.connection_id:
                await self._send_error(connection, "Access denied")
                return
            
            # 移除订阅
            del self.subscriptions[subscription_id]
            connection.subscriptions.discard(subscription_id)
            self.subscription_index[subscription.subscription_type].discard(subscription_id)
            
            # 发送确认
            await self._send_ack(connection, f"Unsubscribed from {subscription.subscription_type.value}")
            
            logger.info(f"客户端取消订阅: {connection.connection_id} -> {subscription.subscription_type.value}")
            
        except Exception as e:
            logger.error(f"取消订阅处理失败: {str(e)}")
            await self._send_error(connection, "Unsubscribe failed")
    
    async def _handle_heartbeat(self, connection: ClientConnection, message: WebSocketMessage):
        """处理心跳消息"""
        try:
            # 更新心跳时间
            connection.last_heartbeat = datetime.now()
            
            # 发送心跳响应
            heartbeat_response = WebSocketMessage(
                message_type=MessageType.HEARTBEAT,
                data={
                    'server_time': datetime.now().isoformat(),
                    'client_time': message.data.get('client_time')
                }
            )
            
            await self._send_message(connection, heartbeat_response)
            
        except Exception as e:
            logger.error(f"心跳处理失败: {str(e)}")
    
    async def _send_message(self, connection: ClientConnection, message: WebSocketMessage):
        """发送消息给客户端"""
        try:
            message_data = {
                'type': message.message_type.value,
                'data': message.data,
                'timestamp': message.timestamp.isoformat()
            }
            
            if message.message_id:
                message_data['message_id'] = message.message_id
            
            await connection.websocket.send(json.dumps(message_data))
            self.stats['total_messages_sent'] += 1
            
        except ConnectionClosed:
            logger.debug(f"连接已关闭: {connection.connection_id}")
            await self._cleanup_connection(connection.connection_id)
        except Exception as e:
            logger.error(f"消息发送失败: {connection.connection_id}, 错误: {str(e)}")
    
    async def _send_ack(self, connection: ClientConnection, message: str, data: Dict[str, Any] = None):
        """发送确认消息"""
        ack_message = WebSocketMessage(
            message_type=MessageType.ACK,
            data={
                'message': message,
                'data': data or {}
            }
        )
        await self._send_message(connection, ack_message)
    
    async def _send_error(self, connection: ClientConnection, error_message: str):
        """发送错误消息"""
        error_msg = WebSocketMessage(
            message_type=MessageType.ERROR,
            data={
                'error': error_message,
                'timestamp': datetime.now().isoformat()
            }
        )
        await self._send_message(connection, error_msg)
    
    async def broadcast_data(self, subscription_type: SubscriptionType, data: Dict[str, Any], 
                           filters: Dict[str, Any] = None):
        """广播数据给订阅者
        
        Args:
            subscription_type: 订阅类型
            data: 要广播的数据
            filters: 过滤条件
        """
        try:
            subscription_ids = self.subscription_index.get(subscription_type, set())
            
            if not subscription_ids:
                return
            
            # 创建数据消息
            data_message = WebSocketMessage(
                message_type=MessageType.DATA,
                data={
                    'subscription_type': subscription_type.value,
                    'payload': data
                }
            )
            
            # 发送给所有订阅者
            send_tasks = []
            for subscription_id in list(subscription_ids):
                if subscription_id in self.subscriptions:
                    subscription = self.subscriptions[subscription_id]
                    
                    # 应用过滤器
                    if filters and not self._match_filters(data, subscription.filters, filters):
                        continue
                    
                    # 获取连接
                    connection = self.connections.get(subscription.client_id)
                    if connection:
                        task = asyncio.create_task(self._send_message(connection, data_message))
                        send_tasks.append(task)
                        
                        # 更新订阅统计
                        subscription.last_data_sent = datetime.now()
                        subscription.message_count += 1
            
            # 等待所有发送完成
            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)
                logger.debug(f"广播数据完成: {subscription_type.value}, 发送给{len(send_tasks)}个客户端")
            
        except Exception as e:
            logger.error(f"数据广播失败: {str(e)}")
    
    def _match_filters(self, data: Dict[str, Any], subscription_filters: Dict[str, Any], 
                      broadcast_filters: Dict[str, Any]) -> bool:
        """匹配过滤条件"""
        try:
            # 简单的过滤器匹配逻辑
            for key, value in subscription_filters.items():
                if key in data:
                    if isinstance(value, list):
                        if data[key] not in value:
                            return False
                    else:
                        if data[key] != value:
                            return False
            
            # 应用广播过滤器
            for key, value in broadcast_filters.items():
                if key in data and data[key] != value:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"过滤器匹配失败: {str(e)}")
            return True  # 默认通过
    
    async def _heartbeat_monitor(self):
        """心跳监控任务"""
        try:
            while self.is_running:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.heartbeat_interval * 2)
                
                # 检查超时连接
                timeout_connections = []
                for connection_id, connection in self.connections.items():
                    if connection.last_heartbeat < timeout_threshold:
                        timeout_connections.append(connection_id)
                
                # 关闭超时连接
                for connection_id in timeout_connections:
                    logger.info(f"连接超时，关闭连接: {connection_id}")
                    await self._cleanup_connection(connection_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
        except Exception as e:
            logger.error(f"心跳监控任务失败: {str(e)}")
    
    async def _cleanup_connections(self):
        """清理连接任务"""
        try:
            while self.is_running:
                # 每5分钟清理一次
                await asyncio.sleep(300)
                
                # 清理已断开的连接
                disconnected_connections = []
                for connection_id, connection in self.connections.items():
                    if connection.websocket.closed:
                        disconnected_connections.append(connection_id)
                
                for connection_id in disconnected_connections:
                    await self._cleanup_connection(connection_id)
                
                logger.debug(f"连接清理完成，清理了{len(disconnected_connections)}个连接")
                
        except Exception as e:
            logger.error(f"连接清理任务失败: {str(e)}")
    
    async def _cleanup_connection(self, connection_id: str):
        """清理单个连接"""
        try:
            if connection_id not in self.connections:
                return
            
            connection = self.connections[connection_id]
            
            # 清理订阅
            for subscription_id in list(connection.subscriptions):
                if subscription_id in self.subscriptions:
                    subscription = self.subscriptions[subscription_id]
                    self.subscription_index[subscription.subscription_type].discard(subscription_id)
                    del self.subscriptions[subscription_id]
            
            # 关闭WebSocket连接
            try:
                if not connection.websocket.closed:
                    await connection.websocket.close()
            except Exception:
                pass
            
            # 移除连接
            del self.connections[connection_id]
            self.stats['active_connections'] = len(self.connections)
            
            logger.debug(f"连接清理完成: {connection_id}")
            
        except Exception as e:
            logger.error(f"连接清理失败: {connection_id}, 错误: {str(e)}")
    
    def register_data_handler(self, subscription_type: SubscriptionType, handler: Callable):
        """注册数据处理器
        
        Args:
            subscription_type: 订阅类型
            handler: 处理函数
        """
        self.data_handlers[subscription_type].append(handler)
        logger.info(f"数据处理器注册成功: {subscription_type.value}")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        current_time = datetime.now()
        uptime = (current_time - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0
        
        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'total_connections': self.stats['total_connections'],
            'active_connections': self.stats['active_connections'],
            'total_subscriptions': len(self.subscriptions),
            'messages_sent': self.stats['total_messages_sent'],
            'messages_received': self.stats['total_messages_received'],
            'last_activity': self.stats['last_activity'].isoformat() if self.stats['last_activity'] else None,
            'subscription_types': {
                sub_type.value: len(sub_ids)
                for sub_type, sub_ids in self.subscription_index.items()
            }
        }
    
    def get_connection_info(self) -> List[Dict[str, Any]]:
        """获取连接信息"""
        return [
            {
                'connection_id': conn.connection_id,
                'user_id': conn.user_id,
                'ip_address': conn.ip_address,
                'connected_at': conn.connected_at.isoformat(),
                'last_heartbeat': conn.last_heartbeat.isoformat(),
                'is_authenticated': conn.is_authenticated,
                'subscriptions': len(conn.subscriptions)
            }
            for conn in self.connections.values()
        ]