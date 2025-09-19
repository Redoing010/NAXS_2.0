# 缓存管理器
# 提供数据缓存功能，提高数据访问性能

import asyncio
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import redis
import aioredis
from pathlib import Path

from .base_connector import DataBatch, DataPoint

class CacheBackend(Enum):
    """缓存后端类型"""
    MEMORY = "memory"      # 内存缓存
    REDIS = "redis"        # Redis缓存
    FILE = "file"          # 文件缓存
    HYBRID = "hybrid"      # 混合缓存

class SerializationFormat(Enum):
    """序列化格式"""
    JSON = "json"          # JSON格式
    PICKLE = "pickle"      # Pickle格式
    MSGPACK = "msgpack"    # MessagePack格式

@dataclass
class CacheConfig:
    """缓存配置"""
    backend: CacheBackend = CacheBackend.MEMORY
    default_ttl: int = 300  # 默认TTL（秒）
    max_size: int = 1000    # 最大缓存条目数
    serialization: SerializationFormat = SerializationFormat.PICKLE
    
    # Redis配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # 文件缓存配置
    cache_dir: str = "./cache"
    
    # 性能配置
    compression: bool = True
    async_write: bool = True
    
    # 清理配置
    cleanup_interval: int = 3600  # 清理间隔（秒）
    max_memory_usage: int = 100 * 1024 * 1024  # 最大内存使用（字节）

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    @property
    def age(self) -> timedelta:
        """缓存年龄"""
        return datetime.now() - self.created_at
    
    def touch(self):
        """更新访问时间"""
        self.access_count += 1
        self.last_accessed = datetime.now()

class CacheStats:
    """缓存统计"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0
        self.total_size = 0
        self.start_time = datetime.now()
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
    
    @property
    def uptime(self) -> timedelta:
        """运行时间"""
        return datetime.now() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'evictions': self.evictions,
            'errors': self.errors,
            'hit_rate': self.hit_rate,
            'total_size': self.total_size,
            'uptime_seconds': self.uptime.total_seconds(),
        }

class MemoryCache:
    """内存缓存实现"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        async with self.lock:
            entry = self.cache.get(key)
            if entry is None or entry.is_expired:
                if entry and entry.is_expired:
                    del self.cache[key]
                return None
            
            entry.touch()
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        async with self.lock:
            ttl = ttl or self.config.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
            
            # 计算大小
            size = self._calculate_size(value)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                size=size
            )
            
            # 检查容量限制
            if len(self.cache) >= self.config.max_size:
                await self._evict_lru()
            
            self.cache[key] = entry
            return True
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> bool:
        """清空缓存"""
        async with self.lock:
            self.cache.clear()
            return True
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        async with self.lock:
            entry = self.cache.get(key)
            return entry is not None and not entry.is_expired
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """获取键列表"""
        async with self.lock:
            if pattern == "*":
                return list(self.cache.keys())
            else:
                # 简单的模式匹配
                import fnmatch
                return [key for key in self.cache.keys() if fnmatch.fnmatch(key, pattern)]
    
    async def cleanup(self) -> int:
        """清理过期条目"""
        async with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
    
    async def _evict_lru(self):
        """LRU淘汰"""
        if not self.cache:
            return
        
        # 找到最少使用的条目
        lru_key = min(self.cache.keys(), 
                     key=lambda k: (self.cache[k].access_count, self.cache[k].last_accessed))
        del self.cache[lru_key]
    
    def _calculate_size(self, value: Any) -> int:
        """计算值的大小"""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return 1024  # 默认大小
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_size = sum(entry.size for entry in self.cache.values())
        expired_count = sum(1 for entry in self.cache.values() if entry.is_expired)
        
        return {
            'total_entries': len(self.cache),
            'total_size': total_size,
            'expired_entries': expired_count,
            'max_size': self.config.max_size,
        }

class RedisCache:
    """Redis缓存实现"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
        self.key_prefix = "naxs:cache:"
    
    async def connect(self):
        """连接Redis"""
        try:
            self.redis = await aioredis.from_url(
                f"redis://{self.config.redis_host}:{self.config.redis_port}/{self.config.redis_db}",
                password=self.config.redis_password,
                decode_responses=False
            )
            # 测试连接
            await self.redis.ping()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
    
    async def disconnect(self):
        """断开连接"""
        if self.redis:
            await self.redis.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self.redis:
            return None
        
        try:
            data = await self.redis.get(self._make_key(key))
            if data is None:
                return None
            
            return self._deserialize(data)
        except Exception:
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        if not self.redis:
            return False
        
        try:
            ttl = ttl or self.config.default_ttl
            data = self._serialize(value)
            
            if ttl > 0:
                await self.redis.setex(self._make_key(key), ttl, data)
            else:
                await self.redis.set(self._make_key(key), data)
            
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        if not self.redis:
            return False
        
        try:
            result = await self.redis.delete(self._make_key(key))
            return result > 0
        except Exception:
            return False
    
    async def clear(self) -> bool:
        """清空缓存"""
        if not self.redis:
            return False
        
        try:
            keys = await self.redis.keys(self._make_key("*"))
            if keys:
                await self.redis.delete(*keys)
            return True
        except Exception:
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self.redis:
            return False
        
        try:
            result = await self.redis.exists(self._make_key(key))
            return result > 0
        except Exception:
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """获取键列表"""
        if not self.redis:
            return []
        
        try:
            keys = await self.redis.keys(self._make_key(pattern))
            return [key.decode().replace(self.key_prefix, "") for key in keys]
        except Exception:
            return []
    
    def _make_key(self, key: str) -> str:
        """生成完整键名"""
        return f"{self.key_prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """序列化值"""
        if self.config.serialization == SerializationFormat.JSON:
            return json.dumps(value, default=str).encode()
        elif self.config.serialization == SerializationFormat.PICKLE:
            return pickle.dumps(value)
        else:
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """反序列化值"""
        if self.config.serialization == SerializationFormat.JSON:
            return json.loads(data.decode())
        elif self.config.serialization == SerializationFormat.PICKLE:
            return pickle.loads(data)
        else:
            return pickle.loads(data)

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.logger = logging.getLogger("CacheManager")
        self.stats = CacheStats()
        
        # 初始化缓存后端
        if self.config.backend == CacheBackend.MEMORY:
            self.cache = MemoryCache(self.config)
        elif self.config.backend == CacheBackend.REDIS:
            self.cache = RedisCache(self.config)
        else:
            self.cache = MemoryCache(self.config)  # 默认使用内存缓存
        
        # 启动清理任务
        self._cleanup_task = None
        self._start_cleanup_task()
    
    async def connect(self):
        """连接缓存后端"""
        if hasattr(self.cache, 'connect'):
            await self.cache.connect()
    
    async def disconnect(self):
        """断开连接"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        if hasattr(self.cache, 'disconnect'):
            await self.cache.disconnect()
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            value = await self.cache.get(key)
            if value is not None:
                self.stats.hits += 1
                self.logger.debug(f"Cache hit: {key}")
            else:
                self.stats.misses += 1
                self.logger.debug(f"Cache miss: {key}")
            return value
        except Exception as e:
            self.stats.errors += 1
            self.logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            result = await self.cache.set(key, value, ttl)
            if result:
                self.stats.sets += 1
                self.logger.debug(f"Cache set: {key}")
            return result
        except Exception as e:
            self.stats.errors += 1
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            result = await self.cache.delete(key)
            if result:
                self.stats.deletes += 1
                self.logger.debug(f"Cache delete: {key}")
            return result
        except Exception as e:
            self.stats.errors += 1
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """清空缓存"""
        try:
            result = await self.cache.clear()
            if result:
                self.logger.info("Cache cleared")
            return result
        except Exception as e:
            self.stats.errors += 1
            self.logger.error(f"Cache clear error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            return await self.cache.exists(key)
        except Exception as e:
            self.stats.errors += 1
            self.logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """获取键列表"""
        try:
            return await self.cache.keys(pattern)
        except Exception as e:
            self.stats.errors += 1
            self.logger.error(f"Cache keys error: {e}")
            return []
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """获取缓存值，如果不存在则通过工厂函数创建"""
        value = await self.get(key)
        if value is not None:
            return value
        
        # 生成新值
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        # 设置缓存
        await self.set(key, value, ttl)
        return value
    
    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取"""
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    async def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """批量设置"""
        success_count = 0
        for key, value in mapping.items():
            if await self.set(key, value, ttl):
                success_count += 1
        
        return success_count == len(mapping)
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """按模式失效缓存"""
        keys = await self.keys(pattern)
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
        return count
    
    def generate_key(self, *parts: Any) -> str:
        """生成缓存键"""
        # 将所有部分转换为字符串并连接
        key_parts = []
        for part in parts:
            if isinstance(part, (dict, list)):
                key_parts.append(json.dumps(part, sort_keys=True))
            else:
                key_parts.append(str(part))
        
        key_string = ":".join(key_parts)
        
        # 如果键太长，使用哈希
        if len(key_string) > 200:
            return hashlib.md5(key_string.encode()).hexdigest()
        
        return key_string
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        stats = self.stats.to_dict()
        
        # 添加后端特定统计
        if hasattr(self.cache, 'get_stats'):
            backend_stats = self.cache.get_stats()
            stats.update(backend_stats)
        
        return stats
    
    def _start_cleanup_task(self):
        """启动清理任务"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.cleanup_interval)
                    if hasattr(self.cache, 'cleanup'):
                        cleaned = await self.cache.cleanup()
                        if cleaned > 0:
                            self.stats.evictions += cleaned
                            self.logger.debug(f"Cleaned up {cleaned} expired entries")
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())

# 便捷函数
def create_cache_manager(
    backend: CacheBackend = CacheBackend.MEMORY,
    **config_kwargs
) -> CacheManager:
    """创建缓存管理器"""
    config = CacheConfig(backend=backend, **config_kwargs)
    return CacheManager(config)

def create_cache_key(*parts: Any) -> str:
    """创建缓存键"""
    manager = CacheManager()
    return manager.generate_key(*parts)