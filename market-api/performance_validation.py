#!/usr/bin/env python3
"""
NAXS API 性能验证脚本
用于验证系统优化后的性能表现
"""

import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Any

class PerformanceValidator:
    """性能验证器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def test_endpoint(self, session: aiohttp.ClientSession, endpoint: str, 
                           num_requests: int = 10) -> Dict[str, Any]:
        """测试单个端点"""
        print(f"🔍 Testing {endpoint} with {num_requests} requests...")
        
        response_times = []
        success_count = 0
        error_count = 0
        
        start_time = time.time()
        
        for i in range(num_requests):
            request_start = time.time()
            try:
                async with session.get(f"{self.base_url}{endpoint}") as response:
                    await response.text()
                    request_time = (time.time() - request_start) * 1000  # ms
                    response_times.append(request_time)
                    
                    if response.status == 200:
                        success_count += 1
                    else:
                        error_count += 1
                        print(f"  ❌ Request {i+1}: HTTP {response.status}")
                        
            except Exception as e:
                error_count += 1
                print(f"  ❌ Request {i+1}: {str(e)}")
        
        total_time = time.time() - start_time
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # 计算百分位数
            sorted_times = sorted(response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
            p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else sorted_times[-1]
        else:
            avg_response_time = 0
            min_response_time = 0
            max_response_time = 0
            p95_response_time = 0
            p99_response_time = 0
        
        success_rate = (success_count / num_requests) * 100
        rps = num_requests / total_time if total_time > 0 else 0
        
        result = {
            'endpoint': endpoint,
            'total_requests': num_requests,
            'successful_requests': success_count,
            'failed_requests': error_count,
            'success_rate': success_rate,
            'total_time': total_time,
            'requests_per_second': rps,
            'avg_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time,
            'p95_response_time': p95_response_time,
            'p99_response_time': p99_response_time
        }
        
        # 打印结果
        print(f"  ✅ Success Rate: {success_rate:.1f}%")
        print(f"  ⚡ Requests/sec: {rps:.1f}")
        print(f"  📈 Avg Response: {avg_response_time:.2f}ms")
        print(f"  📊 P95 Response: {p95_response_time:.2f}ms")
        print(f"  📊 P99 Response: {p99_response_time:.2f}ms")
        print()
        
        return result
    
    async def run_validation(self):
        """运行性能验证"""
        print("🚀 Starting NAXS API Performance Validation...\n")
        
        # 测试端点列表
        endpoints = [
            '/health',
            '/system/stats',
            '/performance/test'
        ]
        
        async with aiohttp.ClientSession() as session:
            # 基础性能测试
            print("📋 Basic Performance Tests:")
            for endpoint in endpoints:
                try:
                    result = await self.test_endpoint(session, endpoint, 10)
                    self.results.append(result)
                except Exception as e:
                    print(f"❌ Failed to test {endpoint}: {e}\n")
            
            # 重点测试 /system/stats 端点（之前的问题端点）
            print("🎯 Focused /system/stats Test:")
            try:
                result = await self.test_endpoint(session, '/system/stats', 20)
                self.results.append({
                    **result,
                    'test_type': 'focused_system_stats'
                })
            except Exception as e:
                print(f"❌ Failed focused test: {e}\n")
        
        # 生成总结报告
        self.generate_summary()
    
    def generate_summary(self):
        """生成性能总结报告"""
        print("📊 Performance Validation Summary:")
        print("=" * 50)
        
        if not self.results:
            print("❌ No test results available")
            return
        
        # 计算整体指标
        total_requests = sum(r['total_requests'] for r in self.results)
        total_successful = sum(r['successful_requests'] for r in self.results)
        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        
        avg_rps = sum(r['requests_per_second'] for r in self.results) / len(self.results)
        avg_response_time = sum(r['avg_response_time'] for r in self.results) / len(self.results)
        
        print(f"🎯 Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"⚡ Average RPS: {avg_rps:.1f}")
        print(f"📈 Average Response Time: {avg_response_time:.2f}ms")
        
        # 性能等级评定
        if overall_success_rate >= 99 and avg_response_time < 100:
            grade = "🏆 EXCELLENT"
        elif overall_success_rate >= 95 and avg_response_time < 500:
            grade = "🥇 GOOD"
        elif overall_success_rate >= 90 and avg_response_time < 1000:
            grade = "🥈 FAIR"
        else:
            grade = "🥉 NEEDS IMPROVEMENT"
        
        print(f"🎖️ Performance Grade: {grade}")
        
        # 重点关注 /system/stats 的改进
        system_stats_results = [r for r in self.results if r['endpoint'] == '/system/stats']
        if system_stats_results:
            print("\n🎯 /system/stats Endpoint Analysis:")
            for result in system_stats_results:
                print(f"  📊 Response Time: {result['avg_response_time']:.2f}ms")
                print(f"  ✅ Success Rate: {result['success_rate']:.1f}%")
                print(f"  ⚡ RPS: {result['requests_per_second']:.1f}")
                
                if result['avg_response_time'] < 1000:  # 小于1秒
                    print("  🎉 OPTIMIZATION SUCCESS: Response time significantly improved!")
                else:
                    print("  ⚠️ Still needs optimization")
        
        # 保存结果
        with open('performance_validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\n💾 Results saved to performance_validation_results.json")
        print("\n🎉 Performance validation completed!")

async def main():
    """主函数"""
    validator = PerformanceValidator()
    await validator.run_validation()

if __name__ == "__main__":
    asyncio.run(main())