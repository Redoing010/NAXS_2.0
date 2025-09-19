# 服务发现 - 实现服务注册、健康检查和负载均衡

import logging
import asyncio
import time
import random
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict
import threading
import aiohttp
from urllib.parse import urljoin
import json

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """服务状态"""
    HEALTHY = "healthy"        # 健康
    UNHEALTHY = "unhealthy"    # 不健康
    UNKNOWN = "unknown"        # 未知
    MAINTENANCE = "maintenance" # 维护中

class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"        # 轮询
    WEIGHTED_ROUND_ROBIN = "weighted"  # 加权轮询
    RANDOM = "random"                  # 随机
    LEAST_CONNECTIONS = "least_conn"   # 最少连接
    IP_HASH = "ip_hash"                # IP哈希
    CONSISTENT_HASH = "consistent"     # 一致性哈希

@dataclass
class ServiceHealth:
    """服务健康状态"""
    status: ServiceStatus
    last_check: datetime
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'status': self.status.value,
            'last_check': self.last_check.isoformat(),
            'response_time': self.response_time,
            'error_message': self.error_message,
            'consecutive_failures': self.consecutive_failures,
            'consecutive_successes': self.consecutive_successes,
            'metadata': self.metadata
        }

@dataclass
class ServiceInstance:
    """服务实例"""
    id: str
    service_name: str
    host: str
    port: int
    protocol: str = "http"
    weight: int = 1
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_check_url: Optional[str] = None
    health_check_interval: int = 30  # 秒
    health_check_timeout: int = 5    # 秒
    health_check_retries: int = 3
    
    # 运行时状态
    health: ServiceHealth = field(default_factory=lambda: ServiceHealth(
        status=ServiceStatus.UNKNOWN,
        last_check=datetime.now()
    ))
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    active_connections: int = 0
    
    @property
    def url(self) -> str:
        """获取服务URL"""
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        """是否健康"""
        return self.health.status == ServiceStatus.HEALTHY
    
    def update_heartbeat(self):
        """更新心跳时间"""
        self.last_heartbeat = datetime.now()
    
    def increment_connections(self):
        """增加连接数"""
        self.active_connections += 1
    
    def decrement_connections(self):
        """减少连接数"""
        if self.active_connections > 0:
            self.active_connections -= 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'service_name': self.service_name,
            'host': self.host,
            'port': self.port,
            'protocol': self.protocol,
            'weight': self.weight,
            'tags': self.tags,
            'metadata': self.metadata,
            'health_check_url': self.health_check_url,
            'health_check_interval': self.health_check_interval,
            'health_check_timeout': self.health_check_timeout,
            'health_check_retries': self.health_check_retries,
            'health': self.health.to_dict(),
            'registered_at': self.registered_at.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'active_connections': self.active_connections,
            'url': self.url,
            'is_healthy': self.is_healthy
        }

class HealthChecker:
    """健康检查器
    
    定期检查服务实例的健康状态
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.check_interval = self.config.get('check_interval', 10)  # 秒
        self.default_timeout = self.config.get('default_timeout', 5)  # 秒
        self.running = False
        self.checker_task: Optional[asyncio.Task] = None
        self.client_session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"健康检查器初始化完成，检查间隔: {self.check_interval}秒")
    
    async def start(self):
        """启动健康检查"""
        if self.running:
            logger.warning("健康检查器已在运行")
            return
        
        self.running = True
        self.client_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.default_timeout)
        )
        
        self.checker_task = asyncio.create_task(self._check_loop())
        logger.info("健康检查器启动")
    
    async def stop(self):
        """停止健康检查"""
        if not self.running:
            logger.warning("健康检查器未在运行")
            return
        
        self.running = False
        
        if self.checker_task:
            self.checker_task.cancel()
            try:
                await self.checker_task
            except asyncio.CancelledError:
                pass
        
        if self.client_session:
            await self.client_session.close()
        
        logger.info("健康检查器停止")
    
    async def check_instance(self, instance: ServiceInstance) -> ServiceHealth:
        """检查单个服务实例
        
        Args:
            instance: 服务实例
            
        Returns:
            健康状态
        """
        start_time = time.time()
        
        try:
            # 确定健康检查URL
            health_url = instance.health_check_url
            if not health_url:
                health_url = f"{instance.url}/health"
            
            # 执行健康检查
            timeout = aiohttp.ClientTimeout(total=instance.health_check_timeout)
            
            async with self.client_session.get(health_url, timeout=timeout) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    # 健康检查成功
                    health = ServiceHealth(
                        status=ServiceStatus.HEALTHY,
                        last_check=datetime.now(),
                        response_time=response_time,
                        consecutive_successes=instance.health.consecutive_successes + 1,
                        consecutive_failures=0
                    )
                else:
                    # 健康检查失败
                    health = ServiceHealth(
                        status=ServiceStatus.UNHEALTHY,
                        last_check=datetime.now(),
                        response_time=response_time,
                        error_message=f"HTTP {response.status}",
                        consecutive_failures=instance.health.consecutive_failures + 1,
                        consecutive_successes=0
                    )
                
                return health
                
        except Exception as e:
            # 健康检查异常
            response_time = time.time() - start_time
            
            health = ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                last_check=datetime.now(),
                response_time=response_time,
                error_message=str(e),
                consecutive_failures=instance.health.consecutive_failures + 1,
                consecutive_successes=0
            )
            
            return health
    
    async def _check_loop(self):
        """健康检查循环"""
        logger.info("健康检查循环启动")
        
        while self.running:
            try:
                await asyncio.sleep(self.check_interval)
                
                # 这里需要从服务注册表获取实例列表
                # 由于循环依赖，这个方法会被ServiceRegistry调用
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查循环异常: {str(e)}")
        
        logger.info("健康检查循环结束")

class LoadBalancer(ABC):
    """负载均衡器基类"""
    
    def __init__(self, strategy: LoadBalanceStrategy):
        self.strategy = strategy
    
    @abstractmethod
    def select_instance(self, instances: List[ServiceInstance], 
                       client_info: Dict[str, Any] = None) -> Optional[ServiceInstance]:
        """选择服务实例
        
        Args:
            instances: 可用的服务实例列表
            client_info: 客户端信息
            
        Returns:
            选中的服务实例
        """
        pass
    
    def filter_healthy_instances(self, instances: List[ServiceInstance]) -> List[ServiceInstance]:
        """过滤健康的实例"""
        return [instance for instance in instances if instance.is_healthy]

class RoundRobinBalancer(LoadBalancer):
    """轮询负载均衡器"""
    
    def __init__(self):
        super().__init__(LoadBalanceStrategy.ROUND_ROBIN)
        self.counters: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()
    
    def select_instance(self, instances: List[ServiceInstance], 
                       client_info: Dict[str, Any] = None) -> Optional[ServiceInstance]:
        """轮询选择实例"""
        healthy_instances = self.filter_healthy_instances(instances)
        if not healthy_instances:
            return None
        
        with self.lock:
            # 使用服务名作为计数器键
            service_name = healthy_instances[0].service_name
            index = self.counters[service_name] % len(healthy_instances)
            self.counters[service_name] += 1
            
            return healthy_instances[index]

class WeightedBalancer(LoadBalancer):
    """加权轮询负载均衡器"""
    
    def __init__(self):
        super().__init__(LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN)
        self.current_weights: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.lock = threading.RLock()
    
    def select_instance(self, instances: List[ServiceInstance], 
                       client_info: Dict[str, Any] = None) -> Optional[ServiceInstance]:
        """加权轮询选择实例"""
        healthy_instances = self.filter_healthy_instances(instances)
        if not healthy_instances:
            return None
        
        with self.lock:
            service_name = healthy_instances[0].service_name
            current_weights = self.current_weights[service_name]
            
            # 初始化权重
            for instance in healthy_instances:
                if instance.id not in current_weights:
                    current_weights[instance.id] = 0
            
            # 计算总权重
            total_weight = sum(instance.weight for instance in healthy_instances)
            if total_weight == 0:
                # 如果所有权重都为0，使用轮询
                return healthy_instances[0]
            
            # 增加当前权重
            for instance in healthy_instances:
                current_weights[instance.id] += instance.weight
            
            # 选择权重最大的实例
            selected_instance = max(healthy_instances, 
                                  key=lambda x: current_weights[x.id])
            
            # 减少选中实例的权重
            current_weights[selected_instance.id] -= total_weight
            
            return selected_instance

class RandomBalancer(LoadBalancer):
    """随机负载均衡器"""
    
    def __init__(self):
        super().__init__(LoadBalanceStrategy.RANDOM)
    
    def select_instance(self, instances: List[ServiceInstance], 
                       client_info: Dict[str, Any] = None) -> Optional[ServiceInstance]:
        """随机选择实例"""
        healthy_instances = self.filter_healthy_instances(instances)
        if not healthy_instances:
            return None
        
        return random.choice(healthy_instances)

class LeastConnectionsBalancer(LoadBalancer):
    """最少连接负载均衡器"""
    
    def __init__(self):
        super().__init__(LoadBalanceStrategy.LEAST_CONNECTIONS)
    
    def select_instance(self, instances: List[ServiceInstance], 
                       client_info: Dict[str, Any] = None) -> Optional[ServiceInstance]:
        """选择连接数最少的实例"""
        healthy_instances = self.filter_healthy_instances(instances)
        if not healthy_instances:
            return None
        
        return min(healthy_instances, key=lambda x: x.active_connections)

class IPHashBalancer(LoadBalancer):
    """IP哈希负载均衡器"""
    
    def __init__(self):
        super().__init__(LoadBalanceStrategy.IP_HASH)
    
    def select_instance(self, instances: List[ServiceInstance], 
                       client_info: Dict[str, Any] = None) -> Optional[ServiceInstance]:
        """基于客户端IP哈希选择实例"""
        healthy_instances = self.filter_healthy_instances(instances)
        if not healthy_instances:
            return None
        
        client_ip = client_info.get('ip', '127.0.0.1') if client_info else '127.0.0.1'
        hash_value = hash(client_ip)
        index = hash_value % len(healthy_instances)
        
        return healthy_instances[index]

class ServiceRegistry:
    """服务注册表
    
    管理服务实例的注册、发现和健康检查
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.instances: Dict[str, ServiceInstance] = {}  # instance_id -> instance
        self.services: Dict[str, List[str]] = defaultdict(list)  # service_name -> [instance_ids]
        self.load_balancers: Dict[str, LoadBalancer] = {}  # service_name -> balancer
        self.health_checker = HealthChecker(self.config.get('health_check', {}))
        
        # 配置参数
        self.heartbeat_timeout = self.config.get('heartbeat_timeout', 60)  # 秒
        self.cleanup_interval = self.config.get('cleanup_interval', 30)   # 秒
        
        # 默认负载均衡策略
        self.default_lb_strategy = LoadBalanceStrategy(
            self.config.get('default_lb_strategy', 'round_robin')
        )
        
        # 清理任务
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info(f"服务注册表初始化完成")
    
    async def start(self):
        """启动服务注册表"""
        if self.running:
            logger.warning("服务注册表已在运行")
            return
        
        self.running = True
        
        # 启动健康检查器
        await self.health_checker.start()
        
        # 启动清理任务
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("服务注册表启动")
    
    async def stop(self):
        """停止服务注册表"""
        if not self.running:
            logger.warning("服务注册表未在运行")
            return
        
        self.running = False
        
        # 停止健康检查器
        await self.health_checker.stop()
        
        # 停止清理任务
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("服务注册表停止")
    
    def register_instance(self, instance: ServiceInstance) -> bool:
        """注册服务实例
        
        Args:
            instance: 服务实例
            
        Returns:
            是否成功注册
        """
        try:
            with self.lock:
                # 检查实例是否已存在
                if instance.id in self.instances:
                    logger.warning(f"服务实例已存在: {instance.id}")
                    return False
                
                # 注册实例
                self.instances[instance.id] = instance
                self.services[instance.service_name].append(instance.id)
                
                # 创建负载均衡器（如果不存在）
                if instance.service_name not in self.load_balancers:
                    self.load_balancers[instance.service_name] = self._create_load_balancer(
                        self.default_lb_strategy
                    )
                
                logger.info(f"注册服务实例: {instance.service_name}@{instance.id} ({instance.url})")
                return True
                
        except Exception as e:
            logger.error(f"注册服务实例失败: {instance.id}, 错误: {str(e)}")
            return False
    
    def deregister_instance(self, instance_id: str) -> bool:
        """注销服务实例
        
        Args:
            instance_id: 实例ID
            
        Returns:
            是否成功注销
        """
        try:
            with self.lock:
                if instance_id not in self.instances:
                    logger.warning(f"服务实例不存在: {instance_id}")
                    return False
                
                instance = self.instances[instance_id]
                service_name = instance.service_name
                
                # 移除实例
                del self.instances[instance_id]
                if instance_id in self.services[service_name]:
                    self.services[service_name].remove(instance_id)
                
                # 如果服务没有实例了，移除负载均衡器
                if not self.services[service_name]:
                    del self.services[service_name]
                    if service_name in self.load_balancers:
                        del self.load_balancers[service_name]
                
                logger.info(f"注销服务实例: {service_name}@{instance_id}")
                return True
                
        except Exception as e:
            logger.error(f"注销服务实例失败: {instance_id}, 错误: {str(e)}")
            return False
    
    def get_instance(self, instance_id: str) -> Optional[ServiceInstance]:
        """获取服务实例
        
        Args:
            instance_id: 实例ID
            
        Returns:
            服务实例
        """
        return self.instances.get(instance_id)
    
    def get_service_instances(self, service_name: str, 
                            healthy_only: bool = True) -> List[ServiceInstance]:
        """获取服务的所有实例
        
        Args:
            service_name: 服务名称
            healthy_only: 是否只返回健康的实例
            
        Returns:
            服务实例列表
        """
        with self.lock:
            instance_ids = self.services.get(service_name, [])
            instances = [self.instances[instance_id] for instance_id in instance_ids 
                        if instance_id in self.instances]
            
            if healthy_only:
                instances = [instance for instance in instances if instance.is_healthy]
            
            return instances
    
    def discover_service(self, service_name: str, 
                        client_info: Dict[str, Any] = None) -> Optional[ServiceInstance]:
        """发现服务实例
        
        Args:
            service_name: 服务名称
            client_info: 客户端信息
            
        Returns:
            选中的服务实例
        """
        instances = self.get_service_instances(service_name, healthy_only=True)
        if not instances:
            logger.warning(f"没有可用的服务实例: {service_name}")
            return None
        
        # 使用负载均衡器选择实例
        load_balancer = self.load_balancers.get(service_name)
        if load_balancer:
            selected_instance = load_balancer.select_instance(instances, client_info)
            if selected_instance:
                selected_instance.increment_connections()
                logger.debug(f"选择服务实例: {service_name}@{selected_instance.id}")
            return selected_instance
        
        # 如果没有负载均衡器，返回第一个实例
        return instances[0]
    
    def release_instance(self, instance_id: str):
        """释放服务实例连接
        
        Args:
            instance_id: 实例ID
        """
        instance = self.get_instance(instance_id)
        if instance:
            instance.decrement_connections()
    
    def heartbeat(self, instance_id: str) -> bool:
        """更新实例心跳
        
        Args:
            instance_id: 实例ID
            
        Returns:
            是否成功更新
        """
        instance = self.get_instance(instance_id)
        if instance:
            instance.update_heartbeat()
            logger.debug(f"更新心跳: {instance.service_name}@{instance_id}")
            return True
        return False
    
    def set_load_balancer(self, service_name: str, strategy: LoadBalanceStrategy):
        """设置服务的负载均衡策略
        
        Args:
            service_name: 服务名称
            strategy: 负载均衡策略
        """
        with self.lock:
            self.load_balancers[service_name] = self._create_load_balancer(strategy)
            logger.info(f"设置负载均衡策略: {service_name} -> {strategy.value}")
    
    def _create_load_balancer(self, strategy: LoadBalanceStrategy) -> LoadBalancer:
        """创建负载均衡器"""
        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return RoundRobinBalancer()
        elif strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return WeightedBalancer()
        elif strategy == LoadBalanceStrategy.RANDOM:
            return RandomBalancer()
        elif strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return LeastConnectionsBalancer()
        elif strategy == LoadBalanceStrategy.IP_HASH:
            return IPHashBalancer()
        else:
            logger.warning(f"不支持的负载均衡策略: {strategy}, 使用轮询")
            return RoundRobinBalancer()
    
    async def _cleanup_loop(self):
        """清理循环"""
        logger.info("服务清理循环启动")
        
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_instances()
                await self._run_health_checks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"服务清理循环异常: {str(e)}")
        
        logger.info("服务清理循环结束")
    
    async def _cleanup_expired_instances(self):
        """清理过期的实例"""
        current_time = datetime.now()
        expired_instances = []
        
        with self.lock:
            for instance_id, instance in self.instances.items():
                # 检查心跳超时
                if (current_time - instance.last_heartbeat).total_seconds() > self.heartbeat_timeout:
                    expired_instances.append(instance_id)
        
        # 移除过期实例
        for instance_id in expired_instances:
            logger.info(f"清理过期实例: {instance_id}")
            self.deregister_instance(instance_id)
    
    async def _run_health_checks(self):
        """运行健康检查"""
        instances_to_check = []
        
        with self.lock:
            current_time = datetime.now()
            for instance in self.instances.values():
                # 检查是否需要健康检查
                time_since_check = (current_time - instance.health.last_check).total_seconds()
                if time_since_check >= instance.health_check_interval:
                    instances_to_check.append(instance)
        
        # 并发执行健康检查
        if instances_to_check:
            tasks = [self.health_checker.check_instance(instance) 
                    for instance in instances_to_check]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 更新健康状态
            for instance, result in zip(instances_to_check, results):
                if isinstance(result, ServiceHealth):
                    instance.health = result
                    logger.debug(f"健康检查完成: {instance.service_name}@{instance.id} - {result.status.value}")
                else:
                    logger.error(f"健康检查异常: {instance.service_name}@{instance.id} - {str(result)}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息
        
        Returns:
            统计信息
        """
        with self.lock:
            service_stats = {}
            for service_name, instance_ids in self.services.items():
                instances = [self.instances[instance_id] for instance_id in instance_ids 
                           if instance_id in self.instances]
                
                healthy_count = len([i for i in instances if i.is_healthy])
                
                service_stats[service_name] = {
                    'total_instances': len(instances),
                    'healthy_instances': healthy_count,
                    'unhealthy_instances': len(instances) - healthy_count,
                    'load_balancer': self.load_balancers.get(service_name, {}).strategy.value if service_name in self.load_balancers else None
                }
            
            return {
                'total_services': len(self.services),
                'total_instances': len(self.instances),
                'services': service_stats,
                'running': self.running
            }
    
    def list_services(self) -> List[str]:
        """列出所有服务
        
        Returns:
            服务名称列表
        """
        return list(self.services.keys())
    
    def get_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """获取服务信息
        
        Args:
            service_name: 服务名称
            
        Returns:
            服务信息
        """
        if service_name not in self.services:
            return None
        
        instances = self.get_service_instances(service_name, healthy_only=False)
        healthy_instances = [i for i in instances if i.is_healthy]
        
        return {
            'name': service_name,
            'total_instances': len(instances),
            'healthy_instances': len(healthy_instances),
            'load_balancer': self.load_balancers.get(service_name, {}).strategy.value if service_name in self.load_balancers else None,
            'instances': [instance.to_dict() for instance in instances]
        }

# 便捷函数
def create_service_registry(config: Dict[str, Any] = None) -> ServiceRegistry:
    """创建服务注册表实例
    
    Args:
        config: 配置字典
        
    Returns:
        服务注册表实例
    """
    return ServiceRegistry(config)

def create_service_instance(service_name: str, host: str, port: int,
                          instance_id: str = None, protocol: str = "http",
                          **kwargs) -> ServiceInstance:
    """创建服务实例
    
    Args:
        service_name: 服务名称
        host: 主机地址
        port: 端口号
        instance_id: 实例ID
        protocol: 协议
        **kwargs: 其他参数
        
    Returns:
        服务实例
    """
    if not instance_id:
        instance_id = f"{service_name}-{host}-{port}-{uuid.uuid4().hex[:8]}"
    
    return ServiceInstance(
        id=instance_id,
        service_name=service_name,
        host=host,
        port=port,
        protocol=protocol,
        **kwargs
    )

def create_round_robin_balancer() -> RoundRobinBalancer:
    """创建轮询负载均衡器"""
    return RoundRobinBalancer()

def create_weighted_balancer() -> WeightedBalancer:
    """创建加权负载均衡器"""
    return WeightedBalancer()

def create_random_balancer() -> RandomBalancer:
    """创建随机负载均衡器"""
    return RandomBalancer()

def create_least_connections_balancer() -> LeastConnectionsBalancer:
    """创建最少连接负载均衡器"""
    return LeastConnectionsBalancer()