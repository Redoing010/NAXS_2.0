#!/usr/bin/env python3
"""
快速性能测试脚本
"""

import asyncio
import aiohttp
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import json

class QuickPerformanceTest:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.results = []
    
    async def single_request(self, session, endpoint):
        """发送单个请求"""
        start_time = time.time()
        try:
            async with session.get(f"{self.base_url}{endpoint}") as response:
                await response.text()
                return time.time() - start_time, response.status
        except Exception as e:
            return time.time() - start_time, 0
    
    async def concurrent_test(self, endpoint, num_requests=100, concurrency=10):
        """并发测试"""
        print(f"🔥 Testing {endpoint} with {num_requests} requests, {concurrency} concurrent")
        
        connector = aiohttp.TCPConnector(limit=concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            # 创建任务
            tasks = []
            for _ in range(num_requests):
                task = self.single_request(session, endpoint)
                tasks.append(task)
            
            # 执行任务
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # 分析结果
            response_times = [r[0] for r in results]
            status_codes = [r[1] for r in results]
            
            successful_requests = len([s for s in status_codes if s == 200])
            
            stats = {
                'endpoint': endpoint,
                'total_requests': num_requests,
                'successful_requests': successful_requests,
                'success_rate': successful_requests / num_requests * 100,
                'total_time': total_time,
                'requests_per_second': num_requests / total_time,
                'avg_response_time': statistics.mean(response_times) * 1000,  # ms
                'min_response_time': min(response_times) * 1000,
                'max_response_time': max(response_times) * 1000,
                'p95_response_time': self.percentile(response_times, 95) * 1000,
                'p99_response_time': self.percentile(response_times, 99) * 1000
            }
            
            return stats
    
    def percentile(self, data, percentile):
        """计算百分位数"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def run_tests(self):
        """运行所有测试"""
        print("🚀 Starting NAXS API Performance Tests...\n")
        
        # 测试端点
        endpoints = [
            '/health',
            '/system/stats',
            '/performance/test'
        ]
        
        all_results = []
        
        for endpoint in endpoints:
            # 轻量测试
            result = await self.concurrent_test(endpoint, num_requests=50, concurrency=5)
            all_results.append(result)
            
            print(f"📊 Results for {endpoint}:")
            print(f"  ✅ Success Rate: {result['success_rate']:.1f}%")
            print(f"  ⚡ Requests/sec: {result['requests_per_second']:.1f}")
            print(f"  📈 Avg Response: {result['avg_response_time']:.2f}ms")
            print(f"  📊 P95 Response: {result['p95_response_time']:.2f}ms")
            print(f"  📊 P99 Response: {result['p99_response_time']:.2f}ms")
            print()
        
        # 高并发测试
        print("🔥 High Concurrency Test...")
        high_load_result = await self.concurrent_test('/health', num_requests=200, concurrency=20)
        all_results.append(high_load_result)
        
        print(f"📊 High Load Results:")
        print(f"  ✅ Success Rate: {high_load_result['success_rate']:.1f}%")
        print(f"  ⚡ Requests/sec: {high_load_result['requests_per_second']:.1f}")
        print(f"  📈 Avg Response: {high_load_result['avg_response_time']:.2f}ms")
        print(f"  📊 P95 Response: {high_load_result['p95_response_time']:.2f}ms")
        print()
        
        # 总结
        print("📋 Performance Summary:")
        avg_rps = statistics.mean([r['requests_per_second'] for r in all_results])
        avg_response = statistics.mean([r['avg_response_time'] for r in all_results])
        overall_success = statistics.mean([r['success_rate'] for r in all_results])
        
        print(f"  🎯 Overall Success Rate: {overall_success:.1f}%")
        print(f"  ⚡ Average RPS: {avg_rps:.1f}")
        print(f"  📈 Average Response Time: {avg_response:.2f}ms")
        
        # 性能评级
        if avg_response < 50 and avg_rps > 500 and overall_success > 99:
            grade = "🏆 EXCELLENT"
        elif avg_response < 100 and avg_rps > 200 and overall_success > 95:
            grade = "🥇 GOOD"
        elif avg_response < 200 and avg_rps > 100 and overall_success > 90:
            grade = "🥈 FAIR"
        else:
            grade = "🥉 NEEDS IMPROVEMENT"
        
        print(f"  🎖️ Performance Grade: {grade}")
        
        return all_results

async def main():
    tester = QuickPerformanceTest()
    results = await tester.run_tests()
    
    # 保存结果
    with open('performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Results saved to performance_results.json")
    print("\n🎉 Performance testing completed!")

if __name__ == "__main__":
    asyncio.run(main())