# 多源数据连接器模块
# 统一接入各种数据源：券商、东方财富、AkShare、TuShare、聚宽、Yahoo等

from .base_connector import BaseConnector, DataSourceConfig, DataPoint, DataBatch
from .akshare_connector import AkShareConnector
from .tushare_connector import TuShareConnector
from .eastmoney_connector import EastMoneyConnector
from .yahoo_connector import YahooConnector
from .jukuan_connector import JuKuanConnector
from .broker_connector import BrokerConnector
from .connector_manager import ConnectorManager, DataSourceType
from .data_normalizer import DataNormalizer, NormalizationRule
from .quality_checker import DataQualityChecker, QualityRule
from .cache_manager import CacheManager, CacheConfig
from .rate_limiter import RateLimiter, RateLimit

# 数据源类型
__all__ = [
    # 基础类
    'BaseConnector',
    'DataSourceConfig',
    'DataPoint',
    'DataBatch',
    
    # 连接器实现
    'AkShareConnector',
    'TuShareConnector', 
    'EastMoneyConnector',
    'YahooConnector',
    'JuKuanConnector',
    'BrokerConnector',
    
    # 管理器
    'ConnectorManager',
    'DataSourceType',
    
    # 数据处理
    'DataNormalizer',
    'NormalizationRule',
    'DataQualityChecker',
    'QualityRule',
    
    # 缓存和限流
    'CacheManager',
    'CacheConfig',
    'RateLimiter',
    'RateLimit',
]

__version__ = '1.0.0'
__description__ = 'Multi-source data connectors for financial data integration'