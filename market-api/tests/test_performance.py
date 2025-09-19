import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import psutil
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.database import db_manager
from app.cache import cache_manager
from app.monitoring import performance_monitor

class PerformanceTestSuite:
    """性能测试套件"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    async def test_api_response_time(self, endpoint: str, num_requests: int = 100) -> Dict[str, Any]:
        """测试API响应时间"""
        response_times = []
        errors = 0
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for _ in range(num_requests):
                task = self._make_request(session, endpoint)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                else:
                    response_times.append(result)
        
        if response_times:
            stats = {
                'endpoint': endpoint,
                'total_requests': num_requests,
                'successful_requests': len(response_times),
                'errors': errors,
                'min_time': min(response_times),
                'max_time': max(response_times),
                'avg_time': statistics.mean(response_times),
                'median_time': statistics.median(response_times),
                'p95_time': self._percentile(response_times, 95),
                'p99_time': self._percentile(response_times, 99),
                'requests_per_second': len(response_times) / sum(response_times) if sum(response_times) > 0 else 0
            }
        else:
            stats = {
                'endpoint': endpoint,
                'total_requests': num_requests,
                'successful_requests': 0,
                'errors': errors,
                'error_rate': 100.0
            }
        
        return stats
    
    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str) -> float:
        """发送单个请求并测量响应时间"""
        start_time = time.time()
        
        try:
            async with session.get(f"{self.base_url}{endpoint}") as response:
                await response.text()
                return time.time() - start_time
        except Exception as e:
            raise e
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def test_concurrent_load(self, endpoint: str, concurrent_users: int = 50, duration: int = 60) -> Dict[str, Any]:
        """测试并发负载"""
        start_time = time.time()
        end_time = start_time + duration
        
        response_times = []
        errors = 0
        total_requests = 0
        
        async with aiohttp.ClientSession() as session:
            while time.time() < end_time:
                tasks = []
                
                # 创建并发请求
                for _ in range(concurrent_users):
                    task = self._make_request(session, endpoint)
                    tasks.append(task)
                
                # 等待所有请求完成
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    total_requests += 1
                    if isinstance(result, Exception):
                        errors += 1
                    else:
                        response_times.append(result)
                
                # 短暂休息避免过载
                await asyncio.sleep(0.1)
        
        actual_duration = time.time() - start_time
        
        return {
            'endpoint': endpoint,
            'concurrent_users': concurrent_users,
            'duration': actual_duration,
            'total_requests': total_requests,
            'successful_requests': len(response_times),
            'errors': errors,
            'error_rate': (errors / total_requests * 100) if total_requests > 0 else 0,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'requests_per_second': total_requests / actual_duration,
            'successful_rps': len(response_times) / actual_duration
        }
    
    async def test_memory_usage(self, test_duration: int = 30) -> Dict[str, Any]:
        """测试内存使用情况"""
        process = psutil.Process()
        memory_samples = []
        
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            memory_info = process.memory_info()
            memory_samples.append({
                'timestamp': time.time(),
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': process.memory_percent()
            })
            
            await asyncio.sleep(1)
        
        rss_values = [sample['rss'] for sample in memory_samples]
        percent_values = [sample['percent'] for sample in memory_samples]
        
        return {
            'duration': test_duration,
            'samples': len(memory_samples),
            'memory_rss': {
                'min': min(rss_values),
                'max': max(rss_values),
                'avg': statistics.mean(rss_values),
                'growth': max(rss_values) - min(rss_values)
            },
            'memory_percent': {
                'min': min(percent_values),
                'max': max(percent_values),
                'avg': statistics.mean(percent_values)
            }
        }
    
    async def test_database_performance(self, num_operations: int = 1000) -> Dict[str, Any]:
        """测试数据库性能"""
        connection_times = []
        query_times = []
        errors = 0
        
        for _ in range(num_operations):
            try:
                # 测试连接时间
                start_time = time.time()
                with db_manager.get_db_session() as session:
                    connection_time = time.time() - start_time
                    connection_times.append(connection_time)
                    
                    # 测试查询时间
                    query_start = time.time()
                    session.execute("SELECT 1")
                    query_time = time.time() - query_start
                    query_times.append(query_time)
                    
            except Exception as e:
                errors += 1
        
        return {
            'total_operations': num_operations,
            'errors': errors,
            'success_rate': ((num_operations - errors) / num_operations * 100) if num_operations > 0 else 0,
            'connection_times': {
                'min': min(connection_times) if connection_times else 0,
                'max': max(connection_times) if connection_times else 0,
                'avg': statistics.mean(connection_times) if connection_times else 0
            },
            'query_times': {
                'min': min(query_times) if query_times else 0,
                'max': max(query_times) if query_times else 0,
                'avg': statistics.mean(query_times) if query_times else 0
            },
            'connection_stats': db_manager.get_connection_stats()
        }
    
    async def test_cache_performance(self, num_operations: int = 10000) -> Dict[str, Any]:
        """测试缓存性能"""
        set_times = []
        get_times = []
        errors = 0
        
        # 测试缓存写入性能
        for i in range(num_operations):
            try:
                start_time = time.time()
                await cache_manager.set(f"test_key_{i}", f"test_value_{i}", ttl=60)
                set_time = time.time() - start_time
                set_times.append(set_time)
            except Exception as e:
                errors += 1
        
        # 测试缓存读取性能
        for i in range(num_operations):
            try:
                start_time = time.time()
                await cache_manager.get(f"test_key_{i}")
                get_time = time.time() - start_time
                get_times.append(get_time)
            except Exception as e:
                errors += 1
        
        # 清理测试数据
        for i in range(num_operations):
            try:
                await cache_manager.delete(f"test_key_{i}")
            except:
                pass
        
        return {
            'total_operations': num_operations * 2,  # set + get
            'errors': errors,
            'set_performance': {
                'min': min(set_times) if set_times else 0,
                'max': max(set_times) if set_times else 0,
                'avg': statistics.mean(set_times) if set_times else 0,
                'ops_per_second': len(set_times) / sum(set_times) if sum(set_times) > 0 else 0
            },
            'get_performance': {
                'min': min(get_times) if get_times else 0,
                'max': max(get_times) if get_times else 0,
                'avg': statistics.mean(get_times) if get_times else 0,
                'ops_per_second': len(get_times) / sum(get_times) if sum(get_times) > 0 else 0
            },
            'cache_stats': cache_manager.get_stats()
        }
    
    async def run_full_performance_test(self) -> Dict[str, Any]:
        """运行完整的性能测试套件"""
        print("开始性能测试...")
        
        results = {
            'test_start_time': time.time(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'python_version': sys.version
            }
        }
        
        # API响应时间测试
        print("测试API响应时间...")
        api_endpoints = [
            '/health',
            '/health/detailed',
            '/health/system',
            '/health/metrics'
        ]
        
        api_results = []
        for endpoint in api_endpoints:
            try:
                result = await self.test_api_response_time(endpoint, 100)
                api_results.append(result)
                print(f"  {endpoint}: 平均响应时间 {result.get('avg_time', 0):.3f}s")
            except Exception as e:
                print(f"  {endpoint}: 测试失败 - {e}")
        
        results['api_performance'] = api_results
        
        # 并发负载测试
        print("测试并发负载...")
        try:
            load_result = await self.test_concurrent_load('/health', concurrent_users=20, duration=30)
            results['load_test'] = load_result
            print(f"  并发测试: {load_result.get('successful_rps', 0):.2f} RPS")
        except Exception as e:
            print(f"  并发测试失败: {e}")
        
        # 内存使用测试
        print("测试内存使用...")
        try:
            memory_result = await self.test_memory_usage(30)
            results['memory_test'] = memory_result
            print(f"  内存测试: 平均使用 {memory_result['memory_percent']['avg']:.2f}%")
        except Exception as e:
            print(f"  内存测试失败: {e}")
        
        # 数据库性能测试
        print("测试数据库性能...")
        try:
            db_result = await self.test_database_performance(500)
            results['database_test'] = db_result
            print(f"  数据库测试: 平均查询时间 {db_result['query_times']['avg']:.4f}s")
        except Exception as e:
            print(f"  数据库测试失败: {e}")
        
        # 缓存性能测试
        print("测试缓存性能...")
        try:
            cache_result = await self.test_cache_performance(1000)
            results['cache_test'] = cache_result
            print(f"  缓存测试: 读取 {cache_result['get_performance']['ops_per_second']:.0f} ops/s")
        except Exception as e:
            print(f"  缓存测试失败: {e}")
        
        results['test_end_time'] = time.time()
        results['total_duration'] = results['test_end_time'] - results['test_start_time']
        
        print(f"性能测试完成，总耗时: {results['total_duration']:.2f}秒")
        
        return results

# pytest测试函数
@pytest.mark.asyncio
async def test_api_performance():
    """API性能测试"""
    suite = PerformanceTestSuite()
    result = await suite.test_api_response_time('/health', 50)
    
    # 断言性能要求
    assert result['avg_time'] < 0.1, f"平均响应时间过长: {result['avg_time']:.3f}s"
    assert result['p95_time'] < 0.2, f"P95响应时间过长: {result['p95_time']:.3f}s"
    assert result['errors'] == 0, f"存在错误请求: {result['errors']}"

@pytest.mark.asyncio
async def test_database_performance():
    """数据库性能测试"""
    suite = PerformanceTestSuite()
    result = await suite.test_database_performance(100)
    
    # 断言性能要求
    assert result['success_rate'] > 95, f"数据库成功率过低: {result['success_rate']:.2f}%"
    assert result['query_times']['avg'] < 0.01, f"平均查询时间过长: {result['query_times']['avg']:.4f}s"

@pytest.mark.asyncio
async def test_cache_performance():
    """缓存性能测试"""
    suite = PerformanceTestSuite()
    result = await suite.test_cache_performance(1000)
    
    # 断言性能要求
    assert result['get_performance']['ops_per_second'] > 1000, f"缓存读取性能过低: {result['get_performance']['ops_per_second']:.0f} ops/s"
    assert result['set_performance']['ops_per_second'] > 500, f"缓存写入性能过低: {result['set_performance']['ops_per_second']:.0f} ops/s"

if __name__ == "__main__":
    # 运行完整性能测试
    async def main():
        suite = PerformanceTestSuite()
        results = await suite.run_full_performance_test()
        
        # 保存结果到文件
        import json
        with open('performance_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\n性能测试结果已保存到 performance_test_results.json")
    
    asyncio.run(main())