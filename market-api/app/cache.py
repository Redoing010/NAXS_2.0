import time
import json
import pickle
import asyncio
import logging
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps
from cachetools import TTLCache, LRUCache
import aioredis
from .config import settings

logger = logging.getLogger(__name__)

class CacheManager:
    """多级缓存管理器"""
    
    def __init__(self):
        # 内存缓存 (L1)
        self.memory_cache = TTLCache(
            maxsize=1000,
            ttl=60  # 1分钟TTL
        )
        
        # LRU缓存 (L1.5)
        self.lru_cache = LRUCache(maxsize=500)
        
        # Redis连接池 (L2)
        self.redis_pool = None
        self.redis_client = None
        
        # 缓存统计
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'redis_hits': 0,
            'redis_misses': 0,
            'total_sets': 0,
            'total_deletes': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """初始化Redis连接"""
        try:
            self.redis_pool = aioredis.ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=settings.REDIS_POOL_SIZE,
                socket_timeout=settings.REDIS_TIMEOUT,
                retry_on_timeout=settings.REDIS_RETRY_ON_TIMEOUT
            )
            self.redis_client = aioredis.Redis(connection_pool=self.redis_pool)
            
            # 测试连接
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}. Using memory cache only.")
            self.redis_client = None
    
    def _make_key(self, key: str) -> str:
        """生成缓存键"""
        return f"{settings.CACHE_PREFIX}{key}"
    
    async def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值 (多级缓存策略)"""
        cache_key = self._make_key(key)
        
        try:
            # L1: 检查内存缓存
            if cache_key in self.memory_cache:
                self.stats['memory_hits'] += 1
                return self.memory_cache[cache_key]
            
            # L1.5: 检查LRU缓存
            if cache_key in self.lru_cache:
                value = self.lru_cache[cache_key]
                # 回写到内存缓存
                self.memory_cache[cache_key] = value
                self.stats['memory_hits'] += 1
                return value
            
            # L2: 检查Redis缓存
            if self.redis_client:
                try:
                    redis_value = await self.redis_client.get(cache_key)
                    if redis_value is not None:
                        # 反序列化
                        value = pickle.loads(redis_value)
                        
                        # 回写到内存缓存
                        self.memory_cache[cache_key] = value
                        self.lru_cache[cache_key] = value
                        
                        self.stats['redis_hits'] += 1
                        return value
                except Exception as e:
                    logger.error(f"Redis get error: {e}")
                    self.stats['errors'] += 1
            
            # 缓存未命中
            self.stats['memory_misses'] += 1
            if self.redis_client:
                self.stats['redis_misses'] += 1
            
            return default
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.stats['errors'] += 1
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        cache_key = self._make_key(key)
        
        try:
            # 设置内存缓存
            self.memory_cache[cache_key] = value
            self.lru_cache[cache_key] = value
            
            # 设置Redis缓存
            if self.redis_client:
                try:
                    serialized_value = pickle.dumps(value)
                    if ttl:
                        await self.redis_client.setex(
                            cache_key, 
                            int(ttl), 
                            serialized_value
                        )
                    else:
                        await self.redis_client.set(cache_key, serialized_value)
                except Exception as e:
                    logger.error(f"Redis set error: {e}")
                    self.stats['errors'] += 1
            
            self.stats['total_sets'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.stats['errors'] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        cache_key = self._make_key(key)
        
        try:
            # 删除内存缓存
            self.memory_cache.pop(cache_key, None)
            self.lru_cache.pop(cache_key, None)
            
            # 删除Redis缓存
            if self.redis_client:
                try:
                    await self.redis_client.delete(cache_key)
                except Exception as e:
                    logger.error(f"Redis delete error: {e}")
                    self.stats['errors'] += 1
            
            self.stats['total_deletes'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self.stats['errors'] += 1
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> bool:
        """清空缓存"""
        try:
            # 清空内存缓存
            if pattern:
                keys_to_remove = [
                    k for k in self.memory_cache.keys() 
                    if pattern in k
                ]
                for k in keys_to_remove:
                    self.memory_cache.pop(k, None)
                    self.lru_cache.pop(k, None)
            else:
                self.memory_cache.clear()
                self.lru_cache.clear()
            
            # 清空Redis缓存
            if self.redis_client:
                try:
                    if pattern:
                        pattern_key = self._make_key(f"*{pattern}*")
                        keys = await self.redis_client.keys(pattern_key)
                        if keys:
                            await self.redis_client.delete(*keys)
                    else:
                        await self.redis_client.flushdb()
                except Exception as e:
                    logger.error(f"Redis clear error: {e}")
                    self.stats['errors'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self.stats['errors'] += 1
            return False
    
    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        total_requests = (
            self.stats['memory_hits'] + 
            self.stats['memory_misses']
        )
        
        memory_hit_rate = (
            self.stats['memory_hits'] / total_requests 
            if total_requests > 0 else 0
        )
        
        redis_requests = (
            self.stats['redis_hits'] + 
            self.stats['redis_misses']
        )
        
        redis_hit_rate = (
            self.stats['redis_hits'] / redis_requests 
            if redis_requests > 0 else 0
        )
        
        return {
            **self.stats,
            'memory_hit_rate': memory_hit_rate,
            'redis_hit_rate': redis_hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'lru_cache_size': len(self.lru_cache)
        }
    
    async def health_check(self) -> dict:
        """缓存健康检查"""
        result = {
            'memory_cache': True,
            'redis_cache': False
        }
        
        # 测试Redis连接
        if self.redis_client:
            try:
                await self.redis_client.ping()
                result['redis_cache'] = True
            except Exception as e:
                logger.error(f"Redis health check failed: {e}")
        
        return result
    
    async def cleanup(self):
        """清理资源"""
        if self.redis_client:
            await self.redis_client.close()
        if self.redis_pool:
            await self.redis_pool.disconnect()

# 全局缓存管理器实例
cache_manager = CacheManager()

# 缓存装饰器
def cached(ttl: float = 300, key_prefix: str = ""):
    """缓存装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 存储到缓存
            await cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# 兼容性函数 (保持向后兼容)
_CACHE: Dict[str, Dict[str, Any]] = {}

def get_cache(key: str, ttl: float) -> Optional[Any]:
    """同步缓存获取 (向后兼容)"""
    node = _CACHE.get(key)
    if not node:
        return None
    if time.time() - node["ts"] <= ttl:
        return node["data"]
    return None

def set_cache(key: str, data: Any) -> None:
    """同步缓存设置 (向后兼容)"""
    _CACHE[key] = {"ts": time.time(), "data": data}

# 初始化函数
async def init_cache():
    """初始化缓存"""
    await cache_manager.initialize()

# 清理函数
async def cleanup_cache():
    """清理缓存"""
    await cache_manager.cleanup()





