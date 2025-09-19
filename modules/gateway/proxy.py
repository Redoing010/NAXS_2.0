# 代理模块 - 实现请求转发和负载均衡功能

import logging
import time
import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin, urlparse
import threading
from collections import defaultdict
import random
import hashlib

logger = logging.getLogger(__name__)

class ProxyMethod(Enum):
    """代理方法"""
    FORWARD = "forward"        # 直接转发
    LOAD_BALANCE = "load_balance"  # 负载均衡
    FAILOVER = "failover"      # 故障转移
    CIRCUIT_BREAK = "circuit_break"  # 熔断

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"        # 健康
    UNHEALTHY = "unhealthy"    # 不健康
    UNKNOWN = "unknown"        # 未知

class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"    # 轮询
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # 加权轮询
    LEAST_CONNECTIONS = "least_connections"  # 最少连接
    RANDOM = "random"              # 随机
    HASH = "hash"                  # 哈希
    LEAST_RESPONSE_TIME = "least_response_time"  # 最短响应时间

@dataclass
class UpstreamServer:
    """上游服务器"""
    host: str
    port: int
    weight: int = 1
    max_connections: int = 100
    health_check_url: str = "/health"
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    
    # 运行时状态
    status: HealthStatus = HealthStatus.UNKNOWN
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = 0.0
    last_error: Optional[str] = None
    
    @property
    def url(self) -> str:
        """服务器URL"""
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_check_full_url(self) -> str:
        """健康检查完整URL"""
        return urljoin(self.url, self.health_check_url)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'host': self.host,
            'port': self.port,
            'weight': self.weight,
            'max_connections': self.max_connections,
            'url': self.url,
            'status': self.status.value,
            'current_connections': self.current_connections,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.success_rate,
            'avg_response_time': self.avg_response_time,
            'last_health_check': self.last_health_check,
            'last_error': self.last_error
        }

@dataclass
class ProxyConfig:
    """代理配置"""
    # 负载均衡配置
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    
    # 健康检查配置
    health_check_enabled: bool = True
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    health_check_retries: int = 3
    
    # 连接配置
    connection_timeout: float = 10.0
    read_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 熔断配置
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    
    # 缓存配置
    cache_enabled: bool = False
    cache_ttl: float = 300.0
    
    # 请求头配置
    preserve_host_header: bool = False
    add_forwarded_headers: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'load_balance_strategy': self.load_balance_strategy.value,
            'health_check_enabled': self.health_check_enabled,
            'health_check_interval': self.health_check_interval,
            'health_check_timeout': self.health_check_timeout,
            'health_check_retries': self.health_check_retries,
            'connection_timeout': self.connection_timeout,
            'read_timeout': self.read_timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'circuit_breaker_enabled': self.circuit_breaker_enabled,
            'failure_threshold': self.failure_threshold,
            'recovery_timeout': self.recovery_timeout,
            'cache_enabled': self.cache_enabled,
            'cache_ttl': self.cache_ttl,
            'preserve_host_header': self.preserve_host_header,
            'add_forwarded_headers': self.add_forwarded_headers
        }

@dataclass
class ProxyRequest:
    """代理请求"""
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    client_ip: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'method': self.method,
            'path': self.path,
            'headers': self.headers,
            'query_params': self.query_params,
            'body': self.body.decode('utf-8') if self.body else None,
            'client_ip': self.client_ip
        }

@dataclass
class ProxyResponse:
    """代理响应"""
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    upstream_server: Optional[str] = None
    response_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'status_code': self.status_code,
            'headers': self.headers,
            'body': self.body.decode('utf-8') if self.body else None,
            'upstream_server': self.upstream_server,
            'response_time': self.response_time
        }

class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.round_robin_index = 0
        self.lock = threading.RLock()
        
        logger.debug(f"负载均衡器初始化: {strategy.value}")
    
    def select_server(self, servers: List[UpstreamServer], request: ProxyRequest = None) -> Optional[UpstreamServer]:
        """选择服务器
        
        Args:
            servers: 可用服务器列表
            request: 代理请求
            
        Returns:
            选中的服务器
        """
        # 过滤健康的服务器
        healthy_servers = [s for s in servers if s.status == HealthStatus.HEALTHY]
        
        if not healthy_servers:
            logger.warning("没有健康的服务器可用")
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin(healthy_servers)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(healthy_servers)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_servers)
        elif self.strategy == LoadBalanceStrategy.RANDOM:
            return self._random(healthy_servers)
        elif self.strategy == LoadBalanceStrategy.HASH:
            return self._hash(healthy_servers, request)
        elif self.strategy == LoadBalanceStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time(healthy_servers)
        else:
            return self._round_robin(healthy_servers)
    
    def _round_robin(self, servers: List[UpstreamServer]) -> UpstreamServer:
        """轮询策略"""
        with self.lock:
            server = servers[self.round_robin_index % len(servers)]
            self.round_robin_index += 1
            return server
    
    def _weighted_round_robin(self, servers: List[UpstreamServer]) -> UpstreamServer:
        """加权轮询策略"""
        total_weight = sum(s.weight for s in servers)
        if total_weight == 0:
            return self._round_robin(servers)
        
        with self.lock:
            target = self.round_robin_index % total_weight
            self.round_robin_index += 1
            
            current_weight = 0
            for server in servers:
                current_weight += server.weight
                if target < current_weight:
                    return server
            
            return servers[0]
    
    def _least_connections(self, servers: List[UpstreamServer]) -> UpstreamServer:
        """最少连接策略"""
        return min(servers, key=lambda s: s.current_connections)
    
    def _random(self, servers: List[UpstreamServer]) -> UpstreamServer:
        """随机策略"""
        return random.choice(servers)
    
    def _hash(self, servers: List[UpstreamServer], request: ProxyRequest) -> UpstreamServer:
        """哈希策略"""
        if not request or not request.client_ip:
            return self._round_robin(servers)
        
        hash_value = hashlib.md5(request.client_ip.encode()).hexdigest()
        index = int(hash_value, 16) % len(servers)
        return servers[index]
    
    def _least_response_time(self, servers: List[UpstreamServer]) -> UpstreamServer:
        """最短响应时间策略"""
        return min(servers, key=lambda s: s.avg_response_time)

class HealthChecker:
    """健康检查器"""
    
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        self.check_tasks: Dict[str, asyncio.Task] = {}
        
        logger.debug("健康检查器初始化完成")
    
    async def start(self):
        """启动健康检查"""
        if not self.config.health_check_enabled:
            return
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.health_check_timeout)
        )
        self.running = True
        
        logger.info("健康检查器启动")
    
    async def stop(self):
        """停止健康检查"""
        self.running = False
        
        # 取消所有检查任务
        for task in self.check_tasks.values():
            task.cancel()
        
        # 等待任务完成
        if self.check_tasks:
            await asyncio.gather(*self.check_tasks.values(), return_exceptions=True)
        
        # 关闭会话
        if self.session:
            await self.session.close()
        
        logger.info("健康检查器停止")
    
    def start_checking(self, server: UpstreamServer):
        """开始检查服务器
        
        Args:
            server: 上游服务器
        """
        if not self.config.health_check_enabled or not self.running:
            return
        
        server_key = f"{server.host}:{server.port}"
        
        if server_key not in self.check_tasks:
            task = asyncio.create_task(self._check_server_health(server))
            self.check_tasks[server_key] = task
            
            logger.debug(f"开始健康检查: {server_key}")
    
    def stop_checking(self, server: UpstreamServer):
        """停止检查服务器
        
        Args:
            server: 上游服务器
        """
        server_key = f"{server.host}:{server.port}"
        
        if server_key in self.check_tasks:
            self.check_tasks[server_key].cancel()
            del self.check_tasks[server_key]
            
            logger.debug(f"停止健康检查: {server_key}")
    
    async def _check_server_health(self, server: UpstreamServer):
        """检查服务器健康状态
        
        Args:
            server: 上游服务器
        """
        while self.running:
            try:
                start_time = time.time()
                
                async with self.session.get(server.health_check_full_url) as response:
                    if response.status == 200:
                        server.status = HealthStatus.HEALTHY
                        server.last_error = None
                    else:
                        server.status = HealthStatus.UNHEALTHY
                        server.last_error = f"HTTP {response.status}"
                
                server.last_health_check = time.time()
                
                logger.debug(f"健康检查完成: {server.host}:{server.port} - {server.status.value}")
                
            except Exception as e:
                server.status = HealthStatus.UNHEALTHY
                server.last_error = str(e)
                server.last_health_check = time.time()
                
                logger.warning(f"健康检查失败: {server.host}:{server.port} - {e}")
            
            # 等待下次检查
            await asyncio.sleep(server.health_check_interval)

class RequestCache:
    """请求缓存"""
    
    def __init__(self, ttl: float = 300.0):
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        
        logger.debug(f"请求缓存初始化: TTL={ttl}s")
    
    def get(self, key: str) -> Optional[ProxyResponse]:
        """获取缓存响应
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的响应
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # 检查是否过期
            if time.time() - entry['timestamp'] > self.ttl:
                del self.cache[key]
                return None
            
            return entry['response']
    
    def set(self, key: str, response: ProxyResponse):
        """设置缓存响应
        
        Args:
            key: 缓存键
            response: 响应对象
        """
        with self.lock:
            self.cache[key] = {
                'response': response,
                'timestamp': time.time()
            }
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            logger.debug("请求缓存已清空")
    
    def _generate_cache_key(self, request: ProxyRequest) -> str:
        """生成缓存键
        
        Args:
            request: 代理请求
            
        Returns:
            缓存键
        """
        key_data = f"{request.method}:{request.path}:{json.dumps(request.query_params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

class ReverseProxy:
    """反向代理
    
    实现反向代理功能：
    1. 请求转发和负载均衡
    2. 健康检查和故障转移
    3. 连接池管理
    4. 请求缓存
    5. 熔断保护
    """
    
    def __init__(self, name: str, config: ProxyConfig = None):
        self.name = name
        self.config = config or ProxyConfig()
        
        # 上游服务器
        self.upstream_servers: List[UpstreamServer] = []
        
        # 组件
        self.load_balancer = LoadBalancer(self.config.load_balance_strategy)
        self.health_checker = HealthChecker(self.config)
        self.request_cache = RequestCache(self.config.cache_ttl) if self.config.cache_enabled else None
        
        # HTTP会话
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info(f"反向代理初始化完成: {self.name}")
    
    async def start(self):
        """启动代理"""
        # 创建HTTP会话
        timeout = aiohttp.ClientTimeout(
            connect=self.config.connection_timeout,
            total=self.config.read_timeout
        )
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # 启动健康检查
        await self.health_checker.start()
        
        # 开始检查所有服务器
        for server in self.upstream_servers:
            self.health_checker.start_checking(server)
        
        logger.info(f"反向代理启动: {self.name}")
    
    async def stop(self):
        """停止代理"""
        # 停止健康检查
        await self.health_checker.stop()
        
        # 关闭HTTP会话
        if self.session:
            await self.session.close()
        
        logger.info(f"反向代理停止: {self.name}")
    
    def add_upstream_server(self, server: UpstreamServer):
        """添加上游服务器
        
        Args:
            server: 上游服务器
        """
        with self.lock:
            self.upstream_servers.append(server)
            
            # 如果健康检查器已启动，开始检查新服务器
            if self.health_checker.running:
                self.health_checker.start_checking(server)
            
            logger.info(f"添加上游服务器: {server.host}:{server.port}")
    
    def remove_upstream_server(self, host: str, port: int) -> bool:
        """移除上游服务器
        
        Args:
            host: 服务器主机
            port: 服务器端口
            
        Returns:
            是否成功移除
        """
        with self.lock:
            for i, server in enumerate(self.upstream_servers):
                if server.host == host and server.port == port:
                    # 停止健康检查
                    self.health_checker.stop_checking(server)
                    
                    # 移除服务器
                    del self.upstream_servers[i]
                    
                    logger.info(f"移除上游服务器: {host}:{port}")
                    return True
            
            return False
    
    async def proxy_request(self, request: ProxyRequest) -> ProxyResponse:
        """代理请求
        
        Args:
            request: 代理请求
            
        Returns:
            代理响应
        """
        with self.lock:
            self.stats['total_requests'] += 1
        
        # 检查缓存
        if self.request_cache and request.method.upper() == 'GET':
            cache_key = self.request_cache._generate_cache_key(request)
            cached_response = self.request_cache.get(cache_key)
            
            if cached_response:
                with self.lock:
                    self.stats['cache_hits'] += 1
                logger.debug(f"缓存命中: {request.path}")
                return cached_response
            else:
                with self.lock:
                    self.stats['cache_misses'] += 1
        
        # 选择上游服务器
        server = self.load_balancer.select_server(self.upstream_servers, request)
        if not server:
            with self.lock:
                self.stats['failed_requests'] += 1
            raise Exception("没有可用的上游服务器")
        
        # 执行请求
        try:
            response = await self._forward_request(server, request)
            
            with self.lock:
                self.stats['successful_requests'] += 1
            
            # 缓存响应
            if (self.request_cache and request.method.upper() == 'GET' and 
                response.status_code == 200):
                cache_key = self.request_cache._generate_cache_key(request)
                self.request_cache.set(cache_key, response)
            
            return response
            
        except Exception as e:
            with self.lock:
                self.stats['failed_requests'] += 1
                server.failed_requests += 1
            
            logger.error(f"请求转发失败: {server.host}:{server.port} - {e}")
            raise
    
    async def _forward_request(self, server: UpstreamServer, request: ProxyRequest) -> ProxyResponse:
        """转发请求到上游服务器
        
        Args:
            server: 上游服务器
            request: 代理请求
            
        Returns:
            代理响应
        """
        # 更新服务器连接数
        server.current_connections += 1
        server.total_requests += 1
        
        try:
            # 构建请求URL
            url = urljoin(server.url, request.path)
            
            # 准备请求头
            headers = request.headers.copy()
            
            if self.config.add_forwarded_headers:
                headers['X-Forwarded-For'] = request.client_ip or 'unknown'
                headers['X-Forwarded-Proto'] = 'http'
                headers['X-Forwarded-Host'] = headers.get('Host', 'unknown')
            
            if not self.config.preserve_host_header:
                headers['Host'] = f"{server.host}:{server.port}"
            
            # 执行请求
            start_time = time.time()
            
            async with self.session.request(
                method=request.method,
                url=url,
                headers=headers,
                params=request.query_params,
                data=request.body
            ) as response:
                response_body = await response.read()
                response_time = time.time() - start_time
                
                # 更新服务器统计
                server.avg_response_time = (
                    (server.avg_response_time * (server.total_requests - 1) + response_time) /
                    server.total_requests
                )
                
                # 构建响应
                proxy_response = ProxyResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=response_body,
                    upstream_server=f"{server.host}:{server.port}",
                    response_time=response_time
                )
                
                logger.debug(f"请求转发成功: {server.host}:{server.port} - {response.status} ({response_time:.3f}s)")
                
                return proxy_response
        
        finally:
            # 更新服务器连接数
            server.current_connections -= 1
    
    def get_upstream_servers(self) -> List[Dict[str, Any]]:
        """获取上游服务器列表
        
        Returns:
            服务器信息列表
        """
        return [server.to_dict() for server in self.upstream_servers]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息
        """
        with self.lock:
            return {
                **self.stats,
                'upstream_servers': len(self.upstream_servers),
                'healthy_servers': len([s for s in self.upstream_servers if s.status == HealthStatus.HEALTHY])
            }
    
    def clear_cache(self):
        """清空缓存"""
        if self.request_cache:
            self.request_cache.clear()
            logger.info(f"代理缓存已清空: {self.name}")

class ProxyManager:
    """代理管理器
    
    管理多个反向代理实例
    """
    
    def __init__(self):
        self.proxies: Dict[str, ReverseProxy] = {}
        self.lock = threading.RLock()
        
        logger.info("代理管理器初始化完成")
    
    def create_proxy(self, name: str, config: ProxyConfig = None) -> ReverseProxy:
        """创建代理
        
        Args:
            name: 代理名称
            config: 代理配置
            
        Returns:
            代理实例
        """
        with self.lock:
            if name in self.proxies:
                raise ValueError(f"代理已存在: {name}")
            
            proxy = ReverseProxy(name, config)
            self.proxies[name] = proxy
            
            logger.info(f"创建代理: {name}")
            return proxy
    
    def get_proxy(self, name: str) -> Optional[ReverseProxy]:
        """获取代理
        
        Args:
            name: 代理名称
            
        Returns:
            代理实例
        """
        return self.proxies.get(name)
    
    def remove_proxy(self, name: str) -> bool:
        """移除代理
        
        Args:
            name: 代理名称
            
        Returns:
            是否成功移除
        """
        with self.lock:
            if name in self.proxies:
                del self.proxies[name]
                logger.info(f"移除代理: {name}")
                return True
            return False
    
    def list_proxies(self) -> List[str]:
        """列出所有代理名称
        
        Returns:
            代理名称列表
        """
        return list(self.proxies.keys())
    
    async def start_all(self):
        """启动所有代理"""
        for proxy in self.proxies.values():
            await proxy.start()
        logger.info("所有代理已启动")
    
    async def stop_all(self):
        """停止所有代理"""
        for proxy in self.proxies.values():
            await proxy.stop()
        logger.info("所有代理已停止")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有代理的统计信息
        
        Returns:
            统计信息字典
        """
        return {
            name: proxy.get_stats()
            for name, proxy in self.proxies.items()
        }

# 便捷函数
def create_upstream_server(host: str, port: int, **kwargs) -> UpstreamServer:
    """创建上游服务器
    
    Args:
        host: 服务器主机
        port: 服务器端口
        **kwargs: 其他配置参数
        
    Returns:
        上游服务器实例
    """
    return UpstreamServer(host=host, port=port, **kwargs)

def create_proxy_config(**kwargs) -> ProxyConfig:
    """创建代理配置
    
    Args:
        **kwargs: 配置参数
        
    Returns:
        代理配置
    """
    return ProxyConfig(**kwargs)

def create_reverse_proxy(name: str, config: ProxyConfig = None) -> ReverseProxy:
    """创建反向代理
    
    Args:
        name: 代理名称
        config: 代理配置
        
    Returns:
        反向代理实例
    """
    return ReverseProxy(name, config)

def create_proxy_manager() -> ProxyManager:
    """创建代理管理器
    
    Returns:
        代理管理器实例
    """
    return ProxyManager()