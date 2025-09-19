# 实时数据流服务启动器
# 整合WebSocket服务、实时数据推送、流处理等功能

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional
from datetime import datetime

from .realtime_service import RealtimeDataService, RealtimeConfig, create_realtime_service
from .stream_processor import StreamProcessor, StreamConfig, StreamType, ProcessingMode
from .data_connector import DataSource, DataType

logger = logging.getLogger(__name__)

class StreamingServiceManager:
    """实时数据流服务管理器
    
    统一管理和协调所有实时数据流相关服务
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # 服务实例
        self.realtime_service: Optional[RealtimeDataService] = None
        self.stream_processor: Optional[StreamProcessor] = None
        
        # 运行状态
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        logger.info("实时数据流服务管理器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # WebSocket配置
            'websocket': {
                'host': 'localhost',
                'port': 8765,
                'max_connections': 1000,
                'heartbeat_interval': 30
            },
            
            # 数据源配置
            'data_sources': {
                'akshare': {
                    'name': 'AkShare',
                    'enabled': True,
                    'rate_limit': 100,
                    'timeout': 30
                },
                'tushare': {
                    'name': 'TuShare',
                    'enabled': False,  # 需要配置token
                    'token': '',
                    'rate_limit': 200,
                    'timeout': 30
                },
                'yahoo_finance': {
                    'name': 'Yahoo Finance',
                    'enabled': True,
                    'rate_limit': 100,
                    'timeout': 30
                }
            },
            
            # 流处理配置
            'stream_processor': {
                'backend': 'redis',  # redis, kafka
                'batch_size': 100,
                'redis': {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0,
                    'password': None
                },
                'kafka': {
                    'bootstrap_servers': ['localhost:9092'],
                    'group_id': 'naxs_streaming'
                }
            },
            
            # 推送配置
            'push': {
                'interval': 1,  # 秒
                'batch_size': 100,
                'enable_compression': True
            },
            
            # 缓存配置
            'cache': {
                'enabled': True,
                'ttl': 300,  # 秒
                'max_size': 10000
            },
            
            # 认证配置
            'auth': {
                'enabled': False,
                'jwt_secret': 'your-secret-key',
                'token_expiry': 3600  # 秒
            },
            
            # 日志配置
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    async def start(self):
        """启动所有服务"""
        try:
            if self.is_running:
                logger.warning("服务已在运行")
                return
            
            logger.info("启动实时数据流服务...")
            
            # 配置日志
            self._setup_logging()
            
            # 初始化流处理器
            await self._init_stream_processor()
            
            # 初始化实时数据服务
            await self._init_realtime_service()
            
            # 启动服务
            await self._start_services()
            
            # 注册信号处理器
            self._setup_signal_handlers()
            
            self.is_running = True
            logger.info("实时数据流服务启动成功")
            
            # 等待关闭信号
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"服务启动失败: {str(e)}")
            raise
    
    async def stop(self):
        """停止所有服务"""
        try:
            if not self.is_running:
                logger.warning("服务未运行")
                return
            
            logger.info("停止实时数据流服务...")
            
            # 停止实时数据服务
            if self.realtime_service:
                await self.realtime_service.stop()
            
            # 停止流处理器
            if self.stream_processor:
                await self.stream_processor.stop_processing()
            
            self.is_running = False
            logger.info("实时数据流服务已停止")
            
        except Exception as e:
            logger.error(f"服务停止失败: {str(e)}")
    
    def _setup_logging(self):
        """配置日志"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('streaming_service.log')
            ]
        )
    
    async def _init_stream_processor(self):
        """初始化流处理器"""
        try:
            stream_config = self.config.get('stream_processor', {})
            self.stream_processor = StreamProcessor(stream_config)
            
            # 注册默认数据流
            await self._register_default_streams()
            
            # 启动流处理
            await self.stream_processor.start_processing()
            
            logger.info("流处理器初始化成功")
            
        except Exception as e:
            logger.error(f"流处理器初始化失败: {str(e)}")
            raise
    
    async def _register_default_streams(self):
        """注册默认数据流"""
        try:
            # 市场数据流
            market_stream = StreamConfig(
                stream_name="market_data",
                stream_type=StreamType.MARKET_DATA,
                processing_mode=ProcessingMode.REAL_TIME,
                batch_size=100,
                batch_timeout=1.0
            )
            self.stream_processor.register_stream(market_stream)
            
            # 新闻数据流
            news_stream = StreamConfig(
                stream_name="news_data",
                stream_type=StreamType.NEWS,
                processing_mode=ProcessingMode.BATCH,
                batch_size=50,
                batch_timeout=5.0
            )
            self.stream_processor.register_stream(news_stream)
            
            # 因子数据流
            factor_stream = StreamConfig(
                stream_name="factor_data",
                stream_type=StreamType.FACTOR,
                processing_mode=ProcessingMode.MICRO_BATCH,
                batch_size=200,
                batch_timeout=2.0
            )
            self.stream_processor.register_stream(factor_stream)
            
            # 交易信号流
            signal_stream = StreamConfig(
                stream_name="trading_signals",
                stream_type=StreamType.SIGNAL,
                processing_mode=ProcessingMode.REAL_TIME,
                batch_size=10,
                batch_timeout=0.5
            )
            self.stream_processor.register_stream(signal_stream)
            
            # 告警数据流
            alert_stream = StreamConfig(
                stream_name="alerts",
                stream_type=StreamType.ALERT,
                processing_mode=ProcessingMode.REAL_TIME,
                batch_size=1,
                batch_timeout=0.1
            )
            self.stream_processor.register_stream(alert_stream)
            
            logger.info("默认数据流注册完成")
            
        except Exception as e:
            logger.error(f"默认数据流注册失败: {str(e)}")
    
    async def _init_realtime_service(self):
        """初始化实时数据服务"""
        try:
            # 构建实时服务配置
            realtime_config = {
                'websocket_host': self.config['websocket']['host'],
                'websocket_port': self.config['websocket']['port'],
                'max_connections': self.config['websocket']['max_connections'],
                'heartbeat_interval': self.config['websocket']['heartbeat_interval'],
                'push_interval': self.config['push']['interval'],
                'batch_size': self.config['push']['batch_size'],
                'cache_enabled': self.config['cache']['enabled'],
                'cache_ttl': self.config['cache']['ttl'],
                'enable_auth': self.config['auth']['enabled'],
                'jwt_secret': self.config['auth']['jwt_secret'],
                'data_sources': self.config['data_sources']
            }
            
            self.realtime_service = create_realtime_service(realtime_config)
            
            logger.info("实时数据服务初始化成功")
            
        except Exception as e:
            logger.error(f"实时数据服务初始化失败: {str(e)}")
            raise
    
    async def _start_services(self):
        """启动所有服务"""
        try:
            # 启动实时数据服务
            if self.realtime_service:
                await self.realtime_service.start()
            
            logger.info("所有服务启动完成")
            
        except Exception as e:
            logger.error(f"服务启动失败: {str(e)}")
            raise
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，准备关闭服务...")
            asyncio.create_task(self._shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _shutdown(self):
        """关闭服务"""
        try:
            await self.stop()
            self.shutdown_event.set()
        except Exception as e:
            logger.error(f"关闭服务失败: {str(e)}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        status = {
            'is_running': self.is_running,
            'start_time': datetime.now().isoformat(),
            'services': {}
        }
        
        if self.realtime_service:
            status['services']['realtime'] = self.realtime_service.get_service_status()
        
        if self.stream_processor:
            status['services']['stream_processor'] = {
                'is_running': self.stream_processor.is_running,
                'processing_stats': self.stream_processor.processing_stats,
                'registered_streams': len(self.stream_processor.stream_configs)
            }
        
        return status
    
    def create_market_data_subscription(self, symbols: list, push_interval: int = 1) -> str:
        """创建市场数据订阅
        
        Args:
            symbols: 股票代码列表
            push_interval: 推送间隔（秒）
            
        Returns:
            订阅ID
        """
        if not self.realtime_service:
            raise RuntimeError("实时数据服务未初始化")
        
        # 创建WebSocket订阅ID
        websocket_subscription_id = f"market_{datetime.now().timestamp()}"
        
        # 创建数据订阅
        return self.realtime_service.create_data_subscription(
            websocket_subscription_id=websocket_subscription_id,
            data_source=DataSource.AKSHARE,
            data_type=DataType.REAL_TIME_QUOTES,
            symbols=symbols,
            push_interval=push_interval
        )
    
    def create_news_subscription(self, keywords: list = None) -> str:
        """创建新闻数据订阅
        
        Args:
            keywords: 关键词列表
            
        Returns:
            订阅ID
        """
        if not self.realtime_service:
            raise RuntimeError("实时数据服务未初始化")
        
        websocket_subscription_id = f"news_{datetime.now().timestamp()}"
        
        return self.realtime_service.create_data_subscription(
            websocket_subscription_id=websocket_subscription_id,
            data_source=DataSource.AKSHARE,
            data_type=DataType.NEWS_DATA,
            parameters={'keywords': keywords or []},
            push_interval=60  # 新闻数据1分钟推送一次
        )

# 便捷函数
def create_streaming_service(config_file: str = None) -> StreamingServiceManager:
    """创建实时数据流服务
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        服务管理器实例
    """
    config = None
    if config_file:
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"配置文件加载失败: {str(e)}")
    
    return StreamingServiceManager(config)

async def main():
    """主函数"""
    try:
        # 创建服务管理器
        service_manager = create_streaming_service()
        
        # 启动服务
        await service_manager.start()
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务...")
    except Exception as e:
        logger.error(f"服务运行异常: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())