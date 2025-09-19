# 流处理器 - 实时数据流处理和分发

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

try:
    import redis
except ImportError:
    redis = None

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
except ImportError:
    KafkaProducer = None
    KafkaConsumer = None
    KafkaError = None

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)

class StreamType(Enum):
    """数据流类型枚举"""
    MARKET_DATA = "market_data"      # 市场行情数据
    NEWS = "news"                    # 新闻数据
    SOCIAL = "social"                # 社交媒体数据
    FUNDAMENTAL = "fundamental"      # 基本面数据
    ALTERNATIVE = "alternative"      # 另类数据
    FACTOR = "factor"                # 因子数据
    SIGNAL = "signal"                # 交易信号
    ALERT = "alert"                  # 告警数据

class ProcessingMode(Enum):
    """处理模式枚举"""
    REAL_TIME = "real_time"          # 实时处理
    BATCH = "batch"                  # 批处理
    MICRO_BATCH = "micro_batch"      # 微批处理
    STREAMING = "streaming"          # 流处理

@dataclass
class StreamMessage:
    """流消息"""
    message_id: str
    stream_type: StreamType
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    priority: int = 1  # 1-10, 10最高
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    retry_count: int = 0

@dataclass
class StreamConfig:
    """流配置"""
    stream_name: str
    stream_type: StreamType
    processing_mode: ProcessingMode
    batch_size: int = 100
    batch_timeout: float = 1.0  # 秒
    max_retries: int = 3
    enable_dlq: bool = True  # 死信队列
    compression: str = "gzip"
    serialization: str = "json"
    partitions: int = 1
    replication_factor: int = 1

class StreamProcessor:
    """流处理器
    
    负责实时数据流的接收、处理、转换和分发
    支持多种消息队列后端（Kafka、Redis）
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = config.get('backend', 'redis')  # kafka, redis
        
        # 流配置
        self.stream_configs: Dict[str, StreamConfig] = {}
        
        # 消息处理器
        self.message_handlers: Dict[StreamType, List[Callable]] = defaultdict(list)
        
        # 消息缓冲区
        self.message_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # 处理统计
        self.processing_stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'processing_time_total': 0.0,
            'last_processed': None
        }
        
        # 运行状态
        self.is_running = False
        self.processing_tasks: Set[asyncio.Task] = set()
        
        # 初始化后端
        self._init_backend()
        
        logger.info(f"流处理器初始化完成，后端: {self.backend}")
    
    def _init_backend(self):
        """初始化消息队列后端"""
        if self.backend == 'kafka':
            self._init_kafka()
        elif self.backend == 'redis':
            self._init_redis()
        else:
            logger.error(f"不支持的后端: {self.backend}")
    
    def _init_kafka(self):
        """初始化Kafka"""
        try:
            if KafkaProducer is None or KafkaConsumer is None:
                logger.error("kafka-python未安装，请安装: pip install kafka-python")
                return
            
            kafka_config = self.config.get('kafka', {})
            bootstrap_servers = kafka_config.get('bootstrap_servers', ['localhost:9092'])
            
            # 创建生产者
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                compression_type='gzip',
                retries=3,
                acks='all'
            )
            
            # 消费者将在订阅时创建
            self.kafka_consumers: Dict[str, KafkaConsumer] = {}
            
            logger.info("Kafka初始化成功")
            
        except Exception as e:
            logger.error(f"Kafka初始化失败: {str(e)}")
            self.kafka_producer = None
    
    def _init_redis(self):
        """初始化Redis"""
        try:
            if redis is None:
                logger.error("redis未安装，请安装: pip install redis")
                return
            
            redis_config = self.config.get('redis', {})
            
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                password=redis_config.get('password'),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # 测试连接
            self.redis_client.ping()
            
            logger.info("Redis初始化成功")
            
        except Exception as e:
            logger.error(f"Redis初始化失败: {str(e)}")
            self.redis_client = None
    
    def register_stream(self, stream_config: StreamConfig):
        """注册数据流
        
        Args:
            stream_config: 流配置
        """
        try:
            stream_name = stream_config.stream_name
            self.stream_configs[stream_name] = stream_config
            
            # 创建消息缓冲区
            if stream_name not in self.message_buffers:
                self.message_buffers[stream_name] = deque(maxlen=10000)
            
            logger.info(f"数据流注册成功: {stream_name}")
            
        except Exception as e:
            logger.error(f"数据流注册失败: {str(e)}")
    
    def register_handler(self, stream_type: StreamType, handler: Callable):
        """注册消息处理器
        
        Args:
            stream_type: 流类型
            handler: 处理函数
        """
        try:
            self.message_handlers[stream_type].append(handler)
            logger.info(f"处理器注册成功: {stream_type.value}")
            
        except Exception as e:
            logger.error(f"处理器注册失败: {str(e)}")
    
    async def start_processing(self):
        """开始处理数据流"""
        try:
            if self.is_running:
                logger.warning("流处理器已在运行")
                return
            
            self.is_running = True
            
            # 启动各个流的处理任务
            for stream_name, stream_config in self.stream_configs.items():
                if stream_config.processing_mode == ProcessingMode.REAL_TIME:
                    task = asyncio.create_task(self._process_real_time_stream(stream_name))
                elif stream_config.processing_mode == ProcessingMode.BATCH:
                    task = asyncio.create_task(self._process_batch_stream(stream_name))
                elif stream_config.processing_mode == ProcessingMode.MICRO_BATCH:
                    task = asyncio.create_task(self._process_micro_batch_stream(stream_name))
                else:
                    task = asyncio.create_task(self._process_streaming(stream_name))
                
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard)
            
            # 启动消费者任务
            if self.backend == 'kafka':
                consumer_task = asyncio.create_task(self._consume_kafka_messages())
                self.processing_tasks.add(consumer_task)
                consumer_task.add_done_callback(self.processing_tasks.discard)
            elif self.backend == 'redis':
                consumer_task = asyncio.create_task(self._consume_redis_messages())
                self.processing_tasks.add(consumer_task)
                consumer_task.add_done_callback(self.processing_tasks.discard)
            
            logger.info(f"流处理器启动成功，处理{len(self.stream_configs)}个数据流")
            
        except Exception as e:
            logger.error(f"流处理器启动失败: {str(e)}")
            self.is_running = False
    
    async def stop_processing(self):
        """停止处理数据流"""
        try:
            self.is_running = False
            
            # 取消所有处理任务
            for task in self.processing_tasks:
                if not task.done():
                    task.cancel()
            
            # 等待任务完成
            if self.processing_tasks:
                await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            
            self.processing_tasks.clear()
            
            # 关闭连接
            if hasattr(self, 'kafka_producer') and self.kafka_producer:
                self.kafka_producer.close()
            
            for consumer in getattr(self, 'kafka_consumers', {}).values():
                consumer.close()
            
            if hasattr(self, 'redis_client') and self.redis_client:
                self.redis_client.close()
            
            logger.info("流处理器已停止")
            
        except Exception as e:
            logger.error(f"流处理器停止失败: {str(e)}")
    
    async def send_message(self, stream_name: str, message: StreamMessage):
        """发送消息到数据流
        
        Args:
            stream_name: 流名称
            message: 流消息
        """
        try:
            if stream_name not in self.stream_configs:
                logger.error(f"未知的数据流: {stream_name}")
                return
            
            # 序列化消息
            message_data = {
                'message_id': message.message_id,
                'stream_type': message.stream_type.value,
                'timestamp': message.timestamp.isoformat(),
                'data': message.data,
                'source': message.source,
                'priority': message.priority,
                'metadata': message.metadata
            }
            
            if self.backend == 'kafka':
                await self._send_kafka_message(stream_name, message_data)
            elif self.backend == 'redis':
                await self._send_redis_message(stream_name, message_data)
            
            self.processing_stats['messages_received'] += 1
            
        except Exception as e:
            logger.error(f"消息发送失败: {str(e)}")
    
    async def _send_kafka_message(self, stream_name: str, message_data: Dict[str, Any]):
        """发送Kafka消息"""
        try:
            if not hasattr(self, 'kafka_producer') or not self.kafka_producer:
                logger.error("Kafka生产者未初始化")
                return
            
            # 异步发送
            future = self.kafka_producer.send(
                topic=stream_name,
                key=message_data.get('message_id'),
                value=message_data
            )
            
            # 等待发送完成
            record_metadata = future.get(timeout=10)
            logger.debug(f"Kafka消息发送成功: {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
            
        except Exception as e:
            logger.error(f"Kafka消息发送失败: {str(e)}")
    
    async def _send_redis_message(self, stream_name: str, message_data: Dict[str, Any]):
        """发送Redis消息"""
        try:
            if not hasattr(self, 'redis_client') or not self.redis_client:
                logger.error("Redis客户端未初始化")
                return
            
            # 使用Redis Stream
            stream_key = f"stream:{stream_name}"
            message_id = self.redis_client.xadd(stream_key, message_data)
            
            logger.debug(f"Redis消息发送成功: {stream_key}:{message_id}")
            
        except Exception as e:
            logger.error(f"Redis消息发送失败: {str(e)}")
    
    async def _consume_kafka_messages(self):
        """消费Kafka消息"""
        try:
            if not hasattr(self, 'kafka_producer'):
                return
            
            # 为每个流创建消费者
            for stream_name in self.stream_configs.keys():
                consumer = KafkaConsumer(
                    stream_name,
                    bootstrap_servers=self.config.get('kafka', {}).get('bootstrap_servers', ['localhost:9092']),
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    key_deserializer=lambda k: k.decode('utf-8') if k else None,
                    group_id=f"naxs_processor_{stream_name}",
                    auto_offset_reset='latest',
                    enable_auto_commit=True
                )
                
                self.kafka_consumers[stream_name] = consumer
            
            # 消费消息
            while self.is_running:
                for stream_name, consumer in self.kafka_consumers.items():
                    try:
                        message_pack = consumer.poll(timeout_ms=100)
                        
                        for topic_partition, messages in message_pack.items():
                            for message in messages:
                                await self._handle_received_message(stream_name, message.value)
                    
                    except Exception as e:
                        logger.error(f"Kafka消息消费失败: {stream_name}, 错误: {str(e)}")
                
                await asyncio.sleep(0.01)  # 避免CPU占用过高
                
        except Exception as e:
            logger.error(f"Kafka消费者失败: {str(e)}")
    
    async def _consume_redis_messages(self):
        """消费Redis消息"""
        try:
            if not hasattr(self, 'redis_client'):
                return
            
            # 为每个流创建消费者组
            consumer_group = "naxs_processor"
            consumer_name = "processor_1"
            
            for stream_name in self.stream_configs.keys():
                stream_key = f"stream:{stream_name}"
                
                try:
                    # 创建消费者组
                    self.redis_client.xgroup_create(stream_key, consumer_group, id='0', mkstream=True)
                except Exception:
                    pass  # 组可能已存在
            
            # 消费消息
            while self.is_running:
                for stream_name in self.stream_configs.keys():
                    stream_key = f"stream:{stream_name}"
                    
                    try:
                        # 读取新消息
                        messages = self.redis_client.xreadgroup(
                            consumer_group,
                            consumer_name,
                            {stream_key: '>'},
                            count=10,
                            block=100
                        )
                        
                        for stream, msgs in messages:
                            for msg_id, fields in msgs:
                                await self._handle_received_message(stream_name, fields)
                                
                                # 确认消息处理完成
                                self.redis_client.xack(stream_key, consumer_group, msg_id)
                    
                    except Exception as e:
                        logger.error(f"Redis消息消费失败: {stream_name}, 错误: {str(e)}")
                
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Redis消费者失败: {str(e)}")
    
    async def _handle_received_message(self, stream_name: str, message_data: Dict[str, Any]):
        """处理接收到的消息"""
        try:
            # 反序列化消息
            stream_message = StreamMessage(
                message_id=message_data.get('message_id', ''),
                stream_type=StreamType(message_data.get('stream_type', 'market_data')),
                timestamp=datetime.fromisoformat(message_data.get('timestamp', datetime.now().isoformat())),
                data=message_data.get('data', {}),
                source=message_data.get('source', ''),
                priority=message_data.get('priority', 1),
                metadata=message_data.get('metadata', {})
            )
            
            # 添加到缓冲区
            self.message_buffers[stream_name].append(stream_message)
            
            # 调用处理器
            handlers = self.message_handlers.get(stream_message.stream_type, [])
            for handler in handlers:
                try:
                    await handler(stream_message)
                except Exception as e:
                    logger.error(f"消息处理器执行失败: {str(e)}")
            
            self.processing_stats['messages_processed'] += 1
            self.processing_stats['last_processed'] = datetime.now()
            
        except Exception as e:
            logger.error(f"消息处理失败: {str(e)}")
            self.processing_stats['messages_failed'] += 1
    
    async def _process_real_time_stream(self, stream_name: str):
        """处理实时流"""
        try:
            while self.is_running:
                buffer = self.message_buffers[stream_name]
                
                if buffer:
                    message = buffer.popleft()
                    start_time = datetime.now()
                    
                    # 实时处理单条消息
                    await self._process_single_message(message)
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    self.processing_stats['processing_time_total'] += processing_time
                else:
                    await asyncio.sleep(0.001)  # 1ms
                    
        except Exception as e:
            logger.error(f"实时流处理失败: {stream_name}, 错误: {str(e)}")
    
    async def _process_batch_stream(self, stream_name: str):
        """处理批流"""
        try:
            stream_config = self.stream_configs[stream_name]
            batch_size = stream_config.batch_size
            batch_timeout = stream_config.batch_timeout
            
            while self.is_running:
                buffer = self.message_buffers[stream_name]
                batch = []
                
                # 收集批次消息
                start_time = datetime.now()
                while len(batch) < batch_size and (datetime.now() - start_time).total_seconds() < batch_timeout:
                    if buffer:
                        batch.append(buffer.popleft())
                    else:
                        await asyncio.sleep(0.01)
                
                # 处理批次
                if batch:
                    await self._process_message_batch(batch)
                else:
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"批流处理失败: {stream_name}, 错误: {str(e)}")
    
    async def _process_micro_batch_stream(self, stream_name: str):
        """处理微批流"""
        try:
            stream_config = self.stream_configs[stream_name]
            batch_size = min(stream_config.batch_size, 10)  # 微批次较小
            batch_timeout = min(stream_config.batch_timeout, 0.1)  # 超时较短
            
            while self.is_running:
                buffer = self.message_buffers[stream_name]
                batch = []
                
                # 收集微批次消息
                start_time = datetime.now()
                while len(batch) < batch_size and (datetime.now() - start_time).total_seconds() < batch_timeout:
                    if buffer:
                        batch.append(buffer.popleft())
                    else:
                        await asyncio.sleep(0.001)
                
                # 处理微批次
                if batch:
                    await self._process_message_batch(batch)
                else:
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"微批流处理失败: {stream_name}, 错误: {str(e)}")
    
    async def _process_streaming(self, stream_name: str):
        """处理流式数据"""
        try:
            # 流式处理结合了实时和批处理的特点
            window_size = 100
            window_timeout = 1.0
            
            while self.is_running:
                buffer = self.message_buffers[stream_name]
                window = []
                
                # 收集窗口数据
                start_time = datetime.now()
                while len(window) < window_size and (datetime.now() - start_time).total_seconds() < window_timeout:
                    if buffer:
                        window.append(buffer.popleft())
                    else:
                        await asyncio.sleep(0.01)
                
                # 流式处理窗口数据
                if window:
                    await self._process_streaming_window(window)
                else:
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"流式处理失败: {stream_name}, 错误: {str(e)}")
    
    async def _process_single_message(self, message: StreamMessage):
        """处理单条消息"""
        try:
            # 标记为已处理
            message.processed = True
            
            # 这里可以添加具体的业务逻辑
            logger.debug(f"处理消息: {message.message_id}, 类型: {message.stream_type.value}")
            
        except Exception as e:
            logger.error(f"单条消息处理失败: {str(e)}")
    
    async def _process_message_batch(self, batch: List[StreamMessage]):
        """处理消息批次"""
        try:
            # 批量处理逻辑
            for message in batch:
                message.processed = True
            
            logger.debug(f"处理消息批次: {len(batch)}条消息")
            
        except Exception as e:
            logger.error(f"批次消息处理失败: {str(e)}")
    
    async def _process_streaming_window(self, window: List[StreamMessage]):
        """处理流式窗口数据"""
        try:
            # 窗口处理逻辑
            for message in window:
                message.processed = True
            
            logger.debug(f"处理流式窗口: {len(window)}条消息")
            
        except Exception as e:
            logger.error(f"流式窗口处理失败: {str(e)}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.processing_stats.copy()
        
        # 计算平均处理时间
        if stats['messages_processed'] > 0:
            stats['avg_processing_time'] = stats['processing_time_total'] / stats['messages_processed']
        else:
            stats['avg_processing_time'] = 0.0
        
        # 添加缓冲区状态
        stats['buffer_status'] = {
            stream_name: len(buffer)
            for stream_name, buffer in self.message_buffers.items()
        }
        
        # 添加流配置信息
        stats['stream_configs'] = {
            stream_name: {
                'type': config.stream_type.value,
                'mode': config.processing_mode.value,
                'batch_size': config.batch_size
            }
            for stream_name, config in self.stream_configs.items()
        }
        
        return stats
    
    def get_stream_health(self) -> Dict[str, Any]:
        """获取流健康状态"""
        health = {
            'overall_status': 'healthy' if self.is_running else 'stopped',
            'backend_status': 'unknown',
            'streams': {},
            'processing_tasks': len(self.processing_tasks),
            'last_activity': self.processing_stats.get('last_processed')
        }
        
        # 检查后端状态
        try:
            if self.backend == 'redis' and hasattr(self, 'redis_client'):
                self.redis_client.ping()
                health['backend_status'] = 'healthy'
            elif self.backend == 'kafka' and hasattr(self, 'kafka_producer'):
                # Kafka健康检查较复杂，这里简化
                health['backend_status'] = 'healthy' if self.kafka_producer else 'unhealthy'
        except Exception:
            health['backend_status'] = 'unhealthy'
        
        # 检查各流状态
        for stream_name, buffer in self.message_buffers.items():
            health['streams'][stream_name] = {
                'buffer_size': len(buffer),
                'status': 'active' if len(buffer) > 0 or self.is_running else 'idle'
            }
        
        return health
    
    async def flush_buffers(self):
        """刷新所有缓冲区"""
        try:
            total_flushed = 0
            
            for stream_name, buffer in self.message_buffers.items():
                flushed_count = len(buffer)
                buffer.clear()
                total_flushed += flushed_count
            
            logger.info(f"缓冲区刷新完成，清理了{total_flushed}条消息")
            
        except Exception as e:
            logger.error(f"缓冲区刷新失败: {str(e)}")