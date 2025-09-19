#!/usr/bin/env python3
"""
NAXS Market API 压力测试脚本

使用 locust 进行压力测试，验证系统性能优化效果
"""

import time
import random
import json
from locust import HttpUser, task, between
from locust.exception import RescheduleTask

class MarketAPIUser(HttpUser):
    """市场API用户模拟"""
    
    wait_time = between(1, 3)  # 用户请求间隔1-3秒
    
    def on_start(self):
        """用户开始时的初始化"""
        self.client.verify = False  # 忽略SSL证书验证
        
        # 预热请求
        try:
            response = self.client.get("/health")
            if response.status_code != 200:
                print(f"Health check failed: {response.status_code}")
        except Exception as e:
            print(f"Health check error: {e}")
    
    @task(10)
    def health_check(self):
        """健康检查 - 高频率"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(8)
    def detailed_health_check(self):
        """详细健康检查"""
        with self.client.get("/health/detailed", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get('status') == 'ok':
                        response.success()
                    else:
                        response.failure(f"Unhealthy status: {data.get('status')}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Detailed health check failed: {response.status_code}")
    
    @task(6)
    def system_stats(self):
        """系统统计信息"""
        with self.client.get("/system/stats", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # 验证关键字段存在
                    required_fields = ['database', 'cache', 'health']
                    for field in required_fields:
                        if field not in data:
                            response.failure(f"Missing field: {field}")
                            return
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"System stats failed: {response.status_code}")
    
    @task(5)
    def system_health(self):
        """系统健康状态"""
        with self.client.get("/system/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'components' in data:
                        response.success()
                    else:
                        response.failure("Missing components in health response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"System health failed: {response.status_code}")
    
    @task(4)
    def metrics_endpoint(self):
        """Prometheus指标"""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                # 检查是否是Prometheus格式
                if 'http_requests_total' in response.text or 'Metrics not enabled' in response.text:
                    response.success()
                else:
                    response.failure("Invalid metrics format")
            elif response.status_code == 404:
                # 指标未启用也是正常的
                response.success()
            else:
                response.failure(f"Metrics failed: {response.status_code}")
    
    @task(3)
    def health_database(self):
        """数据库健康检查"""
        with self.client.get("/health/database", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'status' in data:
                        response.success()
                    else:
                        response.failure("Missing status in database health response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Database health failed: {response.status_code}")
    
    @task(3)
    def health_cache(self):
        """缓存健康检查"""
        with self.client.get("/health/cache", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'status' in data:
                        response.success()
                    else:
                        response.failure("Missing status in cache health response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Cache health failed: {response.status_code}")
    
    @task(2)
    def system_alerts(self):
        """系统告警"""
        with self.client.get("/system/alerts", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'active_alerts' in data and 'stats' in data:
                        response.success()
                    else:
                        response.failure("Missing fields in alerts response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"System alerts failed: {response.status_code}")
    
    @task(1)
    def create_test_alert(self):
        """创建测试告警"""
        with self.client.post("/system/alerts/test", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'alert' in data and 'message' in data:
                        response.success()
                    else:
                        response.failure("Missing fields in test alert response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Test alert creation failed: {response.status_code}")
    
    @task(2)
    def api_root(self):
        """API根路径"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'name' in data and 'status' in data:
                        response.success()
                    else:
                        response.failure("Missing fields in root response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"API root failed: {response.status_code}")

class HighLoadUser(HttpUser):
    """高负载用户模拟"""
    
    wait_time = between(0.1, 0.5)  # 更短的等待时间
    
    @task(20)
    def rapid_health_checks(self):
        """快速健康检查"""
        endpoints = ["/health", "/health/system", "/health/metrics"]
        endpoint = random.choice(endpoints)
        
        with self.client.get(endpoint, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Rapid check failed: {response.status_code}")
    
    @task(10)
    def concurrent_stats_requests(self):
        """并发统计请求"""
        with self.client.get("/system/stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # 触发限流是预期的
                response.success()
            else:
                response.failure(f"Stats request failed: {response.status_code}")

class BurstTrafficUser(HttpUser):
    """突发流量用户模拟"""
    
    wait_time = between(0, 0.1)  # 几乎无等待时间
    
    def on_start(self):
        """突发流量开始"""
        # 模拟突发流量场景
        self.burst_count = 0
        self.max_burst = random.randint(50, 100)
    
    @task
    def burst_requests(self):
        """突发请求"""
        if self.burst_count >= self.max_burst:
            # 突发结束，暂停一段时间
            time.sleep(random.uniform(5, 10))
            self.burst_count = 0
            self.max_burst = random.randint(50, 100)
            return
        
        self.burst_count += 1
        
        # 随机选择端点
        endpoints = [
            "/health",
            "/health/detailed", 
            "/system/stats",
            "/system/health"
        ]
        endpoint = random.choice(endpoints)
        
        with self.client.get(endpoint, catch_response=True) as response:
            if response.status_code in [200, 429, 503]:
                # 200: 成功, 429: 限流, 503: 服务不可用 都是可接受的
                response.success()
            else:
                response.failure(f"Burst request failed: {response.status_code}")

class MemoryStressUser(HttpUser):
    """内存压力测试用户"""
    
    wait_time = between(0.5, 1.0)
    
    def on_start(self):
        """初始化大量数据"""
        self.large_data = 'x' * 1024 * 1024  # 1MB 数据
    
    @task(5)
    def memory_intensive_requests(self):
        """内存密集型请求"""
        # 发送带有大量数据的请求（如果API支持）
        with self.client.get("/system/stats", catch_response=True) as response:
            if response.status_code == 200:
                # 处理响应数据，模拟内存使用
                data = response.json()
                processed_data = json.dumps(data) * 10  # 放大数据
                response.success()
            else:
                response.failure(f"Memory stress request failed: {response.status_code}")

if __name__ == "__main__":
    print("""
    NAXS Market API 压力测试脚本
    
    使用方法:
    1. 安装 locust: pip install locust
    2. 启动API服务: uvicorn app.main:app --host 0.0.0.0 --port 8000
    3. 运行压力测试: locust -f stress_test.py --host http://localhost:8000
    4. 打开浏览器访问: http://localhost:8089
    
    测试场景:
    - MarketAPIUser: 正常用户行为模拟
    - HighLoadUser: 高负载场景
    - BurstTrafficUser: 突发流量场景
    - MemoryStressUser: 内存压力测试
    
    建议测试参数:
    - 用户数: 100-500
    - 每秒新增用户: 10-50
    - 测试时间: 10-30分钟
    """)