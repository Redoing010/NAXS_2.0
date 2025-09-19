# 特征存储核心类 - 实现特征的存储、检索和管理

import logging
import json
import pickle
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import redis
except ImportError:
    redis = None

try:
    import sqlite3
except ImportError:
    sqlite3 = None

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """特征类型枚举"""
    NUMERICAL = "numerical"          # 数值型
    CATEGORICAL = "categorical"      # 分类型
    BOOLEAN = "boolean"              # 布尔型
    TEXT = "text"                    # 文本型
    DATETIME = "datetime"            # 时间型
    VECTOR = "vector"                # 向量型
    JSON = "json"                    # JSON型
    BINARY = "binary"                # 二进制型

class StorageBackend(Enum):
    """存储后端枚举"""
    MEMORY = "memory"                # 内存存储
    REDIS = "redis"                  # Redis存储
    SQLITE = "sqlite"                # SQLite存储
    POSTGRESQL = "postgresql"        # PostgreSQL存储
    CLICKHOUSE = "clickhouse"        # ClickHouse存储
    PARQUET = "parquet"              # Parquet文件存储

@dataclass
class Feature:
    """特征定义"""
    name: str
    feature_type: FeatureType
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    owner: str = ""
    
    # 特征约束
    nullable: bool = True
    default_value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    
    # 统计信息
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def validate_value(self, value: Any) -> bool:
        """验证特征值"""
        try:
            # 空值检查
            if value is None:
                return self.nullable
            
            # 类型检查
            if self.feature_type == FeatureType.NUMERICAL:
                if not isinstance(value, (int, float)):
                    return False
                if self.min_value is not None and value < self.min_value:
                    return False
                if self.max_value is not None and value > self.max_value:
                    return False
            
            elif self.feature_type == FeatureType.CATEGORICAL:
                if self.allowed_values and value not in self.allowed_values:
                    return False
            
            elif self.feature_type == FeatureType.BOOLEAN:
                if not isinstance(value, bool):
                    return False
            
            elif self.feature_type == FeatureType.DATETIME:
                if not isinstance(value, (datetime, str)):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"特征值验证失败: {str(e)}")
            return False

@dataclass
class FeatureGroup:
    """特征组定义"""
    name: str
    description: str = ""
    features: List[Feature] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    owner: str = ""
    
    # 组配置
    primary_key: Optional[str] = None
    event_timestamp: Optional[str] = None
    ttl: Optional[int] = None  # 生存时间（秒）
    
    def add_feature(self, feature: Feature):
        """添加特征"""
        if feature.name not in [f.name for f in self.features]:
            self.features.append(feature)
            self.updated_at = datetime.now()
        else:
            raise ValueError(f"特征 {feature.name} 已存在")
    
    def remove_feature(self, feature_name: str):
        """移除特征"""
        self.features = [f for f in self.features if f.name != feature_name]
        self.updated_at = datetime.now()
    
    def get_feature(self, feature_name: str) -> Optional[Feature]:
        """获取特征"""
        for feature in self.features:
            if feature.name == feature_name:
                return feature
        return None
    
    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证记录"""
        errors = []
        
        # 检查主键
        if self.primary_key and self.primary_key not in record:
            errors.append(f"缺少主键: {self.primary_key}")
        
        # 检查事件时间戳
        if self.event_timestamp and self.event_timestamp not in record:
            errors.append(f"缺少事件时间戳: {self.event_timestamp}")
        
        # 验证每个特征
        for feature in self.features:
            if feature.name in record:
                if not feature.validate_value(record[feature.name]):
                    errors.append(f"特征 {feature.name} 值无效: {record[feature.name]}")
            elif not feature.nullable and feature.default_value is None:
                errors.append(f"特征 {feature.name} 不能为空")
        
        return len(errors) == 0, errors

class BaseStorage(ABC):
    """存储后端基类"""
    
    @abstractmethod
    async def put_features(self, feature_group: str, records: List[Dict[str, Any]]) -> bool:
        """存储特征数据"""
        pass
    
    @abstractmethod
    async def get_features(self, feature_group: str, keys: List[str], 
                          feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取特征数据"""
        pass
    
    @abstractmethod
    async def get_historical_features(self, feature_group: str, 
                                    start_time: datetime, end_time: datetime,
                                    feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取历史特征数据"""
        pass
    
    @abstractmethod
    async def delete_features(self, feature_group: str, keys: List[str]) -> bool:
        """删除特征数据"""
        pass
    
    @abstractmethod
    async def list_feature_groups(self) -> List[str]:
        """列出所有特征组"""
        pass

class MemoryStorage(BaseStorage):
    """内存存储后端"""
    
    def __init__(self):
        self.data: Dict[str, Dict[str, Dict[str, Any]]] = {}  # {feature_group: {key: {feature: value}}}
        self.lock = threading.RLock()
    
    async def put_features(self, feature_group: str, records: List[Dict[str, Any]]) -> bool:
        """存储特征数据"""
        try:
            with self.lock:
                if feature_group not in self.data:
                    self.data[feature_group] = {}
                
                for record in records:
                    # 假设第一个字段是主键
                    if not record:
                        continue
                    
                    key = str(list(record.values())[0])
                    self.data[feature_group][key] = {
                        **record,
                        '_timestamp': datetime.now().isoformat()
                    }
            
            return True
            
        except Exception as e:
            logger.error(f"内存存储失败: {str(e)}")
            return False
    
    async def get_features(self, feature_group: str, keys: List[str], 
                          feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取特征数据"""
        try:
            with self.lock:
                if feature_group not in self.data:
                    return []
                
                results = []
                for key in keys:
                    if key in self.data[feature_group]:
                        record = self.data[feature_group][key].copy()
                        
                        # 过滤特征
                        if feature_names:
                            filtered_record = {}
                            for name in feature_names:
                                if name in record:
                                    filtered_record[name] = record[name]
                            record = filtered_record
                        
                        results.append(record)
                
                return results
                
        except Exception as e:
            logger.error(f"内存获取失败: {str(e)}")
            return []
    
    async def get_historical_features(self, feature_group: str, 
                                    start_time: datetime, end_time: datetime,
                                    feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取历史特征数据"""
        try:
            with self.lock:
                if feature_group not in self.data:
                    return []
                
                results = []
                for key, record in self.data[feature_group].items():
                    # 简单的时间过滤（基于存储时间戳）
                    if '_timestamp' in record:
                        timestamp = datetime.fromisoformat(record['_timestamp'])
                        if start_time <= timestamp <= end_time:
                            filtered_record = record.copy()
                            
                            # 过滤特征
                            if feature_names:
                                temp_record = {}
                                for name in feature_names:
                                    if name in filtered_record:
                                        temp_record[name] = filtered_record[name]
                                filtered_record = temp_record
                            
                            results.append(filtered_record)
                
                return results
                
        except Exception as e:
            logger.error(f"历史数据获取失败: {str(e)}")
            return []
    
    async def delete_features(self, feature_group: str, keys: List[str]) -> bool:
        """删除特征数据"""
        try:
            with self.lock:
                if feature_group not in self.data:
                    return True
                
                for key in keys:
                    if key in self.data[feature_group]:
                        del self.data[feature_group][key]
            
            return True
            
        except Exception as e:
            logger.error(f"内存删除失败: {str(e)}")
            return False
    
    async def list_feature_groups(self) -> List[str]:
        """列出所有特征组"""
        with self.lock:
            return list(self.data.keys())

class RedisStorage(BaseStorage):
    """Redis存储后端"""
    
    def __init__(self, config: Dict[str, Any]):
        if redis is None:
            raise ImportError("Redis未安装，请安装: pip install redis")
        
        self.config = config
        self.client = redis.Redis(
            host=config.get('host', 'localhost'),
            port=config.get('port', 6379),
            db=config.get('db', 0),
            password=config.get('password'),
            decode_responses=True
        )
        
        # 测试连接
        try:
            self.client.ping()
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {str(e)}")
            raise
    
    def _get_key(self, feature_group: str, key: str) -> str:
        """生成Redis键"""
        return f"features:{feature_group}:{key}"
    
    def _get_group_key(self, feature_group: str) -> str:
        """生成特征组键"""
        return f"feature_groups:{feature_group}"
    
    async def put_features(self, feature_group: str, records: List[Dict[str, Any]]) -> bool:
        """存储特征数据"""
        try:
            pipe = self.client.pipeline()
            
            for record in records:
                if not record:
                    continue
                
                # 假设第一个字段是主键
                key = str(list(record.values())[0])
                redis_key = self._get_key(feature_group, key)
                
                # 添加时间戳
                record_with_timestamp = {
                    **record,
                    '_timestamp': datetime.now().isoformat()
                }
                
                # 存储为JSON
                pipe.set(redis_key, json.dumps(record_with_timestamp, default=str))
                
                # 添加到特征组集合
                pipe.sadd(self._get_group_key(feature_group), key)
            
            pipe.execute()
            return True
            
        except Exception as e:
            logger.error(f"Redis存储失败: {str(e)}")
            return False
    
    async def get_features(self, feature_group: str, keys: List[str], 
                          feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取特征数据"""
        try:
            pipe = self.client.pipeline()
            
            for key in keys:
                redis_key = self._get_key(feature_group, key)
                pipe.get(redis_key)
            
            results = pipe.execute()
            
            features = []
            for result in results:
                if result:
                    record = json.loads(result)
                    
                    # 过滤特征
                    if feature_names:
                        filtered_record = {}
                        for name in feature_names:
                            if name in record:
                                filtered_record[name] = record[name]
                        record = filtered_record
                    
                    features.append(record)
            
            return features
            
        except Exception as e:
            logger.error(f"Redis获取失败: {str(e)}")
            return []
    
    async def get_historical_features(self, feature_group: str, 
                                    start_time: datetime, end_time: datetime,
                                    feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取历史特征数据"""
        try:
            # 获取特征组中的所有键
            group_key = self._get_group_key(feature_group)
            keys = self.client.smembers(group_key)
            
            if not keys:
                return []
            
            # 批量获取数据
            pipe = self.client.pipeline()
            for key in keys:
                redis_key = self._get_key(feature_group, key)
                pipe.get(redis_key)
            
            results = pipe.execute()
            
            features = []
            for result in results:
                if result:
                    record = json.loads(result)
                    
                    # 时间过滤
                    if '_timestamp' in record:
                        timestamp = datetime.fromisoformat(record['_timestamp'])
                        if start_time <= timestamp <= end_time:
                            # 过滤特征
                            if feature_names:
                                filtered_record = {}
                                for name in feature_names:
                                    if name in record:
                                        filtered_record[name] = record[name]
                                record = filtered_record
                            
                            features.append(record)
            
            return features
            
        except Exception as e:
            logger.error(f"Redis历史数据获取失败: {str(e)}")
            return []
    
    async def delete_features(self, feature_group: str, keys: List[str]) -> bool:
        """删除特征数据"""
        try:
            pipe = self.client.pipeline()
            
            for key in keys:
                redis_key = self._get_key(feature_group, key)
                pipe.delete(redis_key)
                
                # 从特征组集合中移除
                group_key = self._get_group_key(feature_group)
                pipe.srem(group_key, key)
            
            pipe.execute()
            return True
            
        except Exception as e:
            logger.error(f"Redis删除失败: {str(e)}")
            return False
    
    async def list_feature_groups(self) -> List[str]:
        """列出所有特征组"""
        try:
            pattern = "feature_groups:*"
            keys = self.client.keys(pattern)
            return [key.replace("feature_groups:", "") for key in keys]
            
        except Exception as e:
            logger.error(f"Redis列出特征组失败: {str(e)}")
            return []

class FeatureStore:
    """特征存储主类
    
    提供特征的存储、检索、管理功能，支持：
    1. 多种存储后端
    2. 特征组管理
    3. 特征版本控制
    4. 数据验证
    5. 批量操作
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_groups: Dict[str, FeatureGroup] = {}
        
        # 初始化存储后端
        backend_type = StorageBackend(config.get('backend', 'memory'))
        if backend_type == StorageBackend.MEMORY:
            self.storage = MemoryStorage()
        elif backend_type == StorageBackend.REDIS:
            self.storage = RedisStorage(config.get('redis', {}))
        else:
            raise ValueError(f"不支持的存储后端: {backend_type}")
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 10))
        
        # 统计信息
        self.stats = {
            'total_features_stored': 0,
            'total_features_retrieved': 0,
            'total_feature_groups': 0,
            'last_activity': None
        }
        
        logger.info(f"特征存储初始化完成，后端: {backend_type.value}")
    
    def register_feature_group(self, feature_group: FeatureGroup):
        """注册特征组"""
        try:
            if feature_group.name in self.feature_groups:
                logger.warning(f"特征组 {feature_group.name} 已存在，将被覆盖")
            
            self.feature_groups[feature_group.name] = feature_group
            self.stats['total_feature_groups'] = len(self.feature_groups)
            
            logger.info(f"特征组注册成功: {feature_group.name}")
            
        except Exception as e:
            logger.error(f"特征组注册失败: {str(e)}")
            raise
    
    def get_feature_group(self, name: str) -> Optional[FeatureGroup]:
        """获取特征组"""
        return self.feature_groups.get(name)
    
    def list_feature_groups(self) -> List[str]:
        """列出所有特征组"""
        return list(self.feature_groups.keys())
    
    async def put_features(self, feature_group_name: str, records: List[Dict[str, Any]], 
                          validate: bool = True) -> Tuple[bool, List[str]]:
        """存储特征数据
        
        Args:
            feature_group_name: 特征组名称
            records: 特征记录列表
            validate: 是否验证数据
            
        Returns:
            (成功标志, 错误信息列表)
        """
        try:
            # 获取特征组
            feature_group = self.feature_groups.get(feature_group_name)
            if not feature_group:
                return False, [f"特征组 {feature_group_name} 不存在"]
            
            errors = []
            valid_records = []
            
            # 验证数据
            if validate:
                for i, record in enumerate(records):
                    is_valid, record_errors = feature_group.validate_record(record)
                    if is_valid:
                        valid_records.append(record)
                    else:
                        errors.extend([f"记录 {i}: {error}" for error in record_errors])
            else:
                valid_records = records
            
            # 存储有效记录
            if valid_records:
                success = await self.storage.put_features(feature_group_name, valid_records)
                if success:
                    self.stats['total_features_stored'] += len(valid_records)
                    self.stats['last_activity'] = datetime.now()
                    
                    logger.info(f"特征存储成功: {feature_group_name}, 记录数: {len(valid_records)}")
                    return True, errors
                else:
                    errors.append("存储操作失败")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"特征存储异常: {str(e)}")
            return False, [str(e)]
    
    async def get_features(self, feature_group_name: str, keys: List[str], 
                          feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取特征数据
        
        Args:
            feature_group_name: 特征组名称
            keys: 主键列表
            feature_names: 要获取的特征名称列表
            
        Returns:
            特征记录列表
        """
        try:
            # 检查特征组是否存在
            if feature_group_name not in self.feature_groups:
                logger.warning(f"特征组 {feature_group_name} 不存在")
                return []
            
            # 获取数据
            results = await self.storage.get_features(feature_group_name, keys, feature_names)
            
            self.stats['total_features_retrieved'] += len(results)
            self.stats['last_activity'] = datetime.now()
            
            logger.debug(f"特征获取成功: {feature_group_name}, 记录数: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"特征获取异常: {str(e)}")
            return []
    
    async def get_historical_features(self, feature_group_name: str, 
                                    start_time: datetime, end_time: datetime,
                                    feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取历史特征数据
        
        Args:
            feature_group_name: 特征组名称
            start_time: 开始时间
            end_time: 结束时间
            feature_names: 要获取的特征名称列表
            
        Returns:
            历史特征记录列表
        """
        try:
            # 检查特征组是否存在
            if feature_group_name not in self.feature_groups:
                logger.warning(f"特征组 {feature_group_name} 不存在")
                return []
            
            # 获取历史数据
            results = await self.storage.get_historical_features(
                feature_group_name, start_time, end_time, feature_names
            )
            
            self.stats['total_features_retrieved'] += len(results)
            self.stats['last_activity'] = datetime.now()
            
            logger.debug(f"历史特征获取成功: {feature_group_name}, 记录数: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"历史特征获取异常: {str(e)}")
            return []
    
    async def delete_features(self, feature_group_name: str, keys: List[str]) -> bool:
        """删除特征数据
        
        Args:
            feature_group_name: 特征组名称
            keys: 要删除的主键列表
            
        Returns:
            删除是否成功
        """
        try:
            # 检查特征组是否存在
            if feature_group_name not in self.feature_groups:
                logger.warning(f"特征组 {feature_group_name} 不存在")
                return False
            
            # 删除数据
            success = await self.storage.delete_features(feature_group_name, keys)
            
            if success:
                self.stats['last_activity'] = datetime.now()
                logger.info(f"特征删除成功: {feature_group_name}, 键数: {len(keys)}")
            
            return success
            
        except Exception as e:
            logger.error(f"特征删除异常: {str(e)}")
            return False
    
    def get_feature_group_info(self, feature_group_name: str) -> Optional[Dict[str, Any]]:
        """获取特征组信息"""
        feature_group = self.feature_groups.get(feature_group_name)
        if not feature_group:
            return None
        
        return {
            'name': feature_group.name,
            'description': feature_group.description,
            'features': [
                {
                    'name': f.name,
                    'type': f.feature_type.value,
                    'description': f.description,
                    'nullable': f.nullable,
                    'tags': f.tags
                }
                for f in feature_group.features
            ],
            'tags': feature_group.tags,
            'version': feature_group.version,
            'owner': feature_group.owner,
            'created_at': feature_group.created_at.isoformat(),
            'updated_at': feature_group.updated_at.isoformat(),
            'primary_key': feature_group.primary_key,
            'event_timestamp': feature_group.event_timestamp,
            'ttl': feature_group.ttl
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'last_activity': self.stats['last_activity'].isoformat() if self.stats['last_activity'] else None,
            'backend': self.config.get('backend', 'memory'),
            'feature_groups': list(self.feature_groups.keys())
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试存储后端
            test_groups = await self.storage.list_feature_groups()
            
            return {
                'status': 'healthy',
                'backend': self.config.get('backend', 'memory'),
                'feature_groups_count': len(self.feature_groups),
                'storage_groups_count': len(test_groups),
                'last_activity': self.stats['last_activity'].isoformat() if self.stats['last_activity'] else None
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'backend': self.config.get('backend', 'memory')
            }
    
    def close(self):
        """关闭特征存储"""
        try:
            self.executor.shutdown(wait=True)
            logger.info("特征存储已关闭")
        except Exception as e:
            logger.error(f"特征存储关闭失败: {str(e)}")

# 便捷函数
def create_feature_store(config: Dict[str, Any] = None) -> FeatureStore:
    """创建特征存储实例
    
    Args:
        config: 配置字典
        
    Returns:
        特征存储实例
    """
    if config is None:
        config = {
            'backend': 'memory',
            'max_workers': 10
        }
    
    return FeatureStore(config)