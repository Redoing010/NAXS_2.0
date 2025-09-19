# 实时数据流模块
# 提供WebSocket实时推送、数据连接器、流处理等功能

from .websocket_server import (
    WebSocketServer, 
    MessageType, 
    SubscriptionType, 
    ClientConnection, 
    Subscription, 
    WebSocketMessage
)

from .realtime_service import (
    RealtimeDataService,
    RealtimeConfig,
    DataSubscription,
    ServiceStatus,
    create_realtime_service
)

from .stream_processor import (
    StreamProcessor,
    StreamType,
    ProcessingMode,
    StreamMessage,
    StreamConfig
)

from .data_connector import (
    DataConnectorManager,
    DataSource,
    DataType,
    DataRequest,
    DataResponse,
    DataStatus,
    AkShareConnector,
    TuShareConnector,
    YahooFinanceConnector
)

__all__ = [
    # WebSocket服务
    'WebSocketServer',
    'MessageType',
    'SubscriptionType', 
    'ClientConnection',
    'Subscription',
    'WebSocketMessage',
    
    # 实时数据服务
    'RealtimeDataService',
    'RealtimeConfig',
    'DataSubscription',
    'ServiceStatus',
    'create_realtime_service',
    
    # 流处理器
    'StreamProcessor',
    'StreamType',
    'ProcessingMode',
    'StreamMessage',
    'StreamConfig',
    
    # 数据连接器
    'DataConnectorManager',
    'DataSource',
    'DataType',
    'DataRequest',
    'DataResponse',
    'DataStatus',
    'AkShareConnector',
    'TuShareConnector',
    'YahooFinanceConnector'
]

__version__ = '1.0.0'
__description__ = 'NAXS实时数据流处理模块'