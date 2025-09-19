#!/usr/bin/env python3
"""
系统监控功能测试脚本
用于验证新创建的系统监控API端点是否正常工作
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

# API基础URL
BASE_URL = "http://localhost:8000"

async def test_system_metrics():
    """测试系统指标API"""
    print("\n=== 测试系统指标API ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/system/metrics") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ 系统指标API正常")
                    print(f"   - API响应时间: {data['api']['responseTime']}ms")
                    print(f"   - API成功率: {data['api']['successRate']}%")
                    print(f"   - CPU使用率: {data['system']['cpuUsage']}%")
                    print(f"   - 内存使用率: {data['system']['memoryUsage']}%")
                    print(f"   - 数据库状态: {data['database']['status']}")
                    return True
                else:
                    print(f"❌ 系统指标API失败: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 系统指标API异常: {e}")
            return False

async def test_test_results():
    """测试测试结果API"""
    print("\n=== 测试测试结果API ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/system/tests/results") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ 测试结果API正常")
                    print(f"   - 测试结果数量: {len(data)}")
                    
                    for result in data[:3]:  # 显示前3个结果
                        print(f"   - {result['name']}: {result['status']} ({result['duration']}ms)")
                    
                    return True
                else:
                    print(f"❌ 测试结果API失败: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 测试结果API异常: {e}")
            return False

async def test_run_test():
    """测试运行测试API"""
    print("\n=== 测试运行测试API ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            # 运行API测试
            async with session.post(f"{BASE_URL}/system/tests/run/api") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ 运行测试API正常")
                    print(f"   - 测试ID: {data['test_id']}")
                    print(f"   - 状态: {data['status']}")
                    print(f"   - 消息: {data['message']}")
                    
                    # 等待几秒钟让测试完成
                    print("   - 等待测试完成...")
                    await asyncio.sleep(3)
                    
                    # 检查测试结果
                    async with session.get(f"{BASE_URL}/system/tests/results") as results_response:
                        if results_response.status == 200:
                            results_data = await results_response.json()
                            # 查找刚刚运行的测试
                            for result in results_data:
                                if result['id'] == data['test_id']:
                                    print(f"   - 测试完成状态: {result['status']}")
                                    break
                    
                    return True
                else:
                    print(f"❌ 运行测试API失败: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 运行测试API异常: {e}")
            return False

async def test_system_status():
    """测试系统状态API"""
    print("\n=== 测试系统状态API ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/system/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ 系统状态API正常")
                    print(f"   - CPU使用率: {data['system']['cpu_percent']}%")
                    print(f"   - 内存使用率: {data['system']['memory_percent']}%")
                    print(f"   - 磁盘使用率: {data['system']['disk_percent']:.1f}%")
                    print(f"   - 数据库健康: {data['database']['healthy']}")
                    return True
                else:
                    print(f"❌ 系统状态API失败: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 系统状态API异常: {e}")
            return False

async def test_websocket_stats():
    """测试WebSocket统计API"""
    print("\n=== 测试WebSocket统计API ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            # 先模拟一些WebSocket活动
            async with session.post(f"{BASE_URL}/system/websocket/simulate", 
                                   json={"connections": 25, "message_rate": 100}) as sim_response:
                if sim_response.status == 200:
                    print("✅ WebSocket活动模拟成功")
                
            # 获取WebSocket统计
            async with session.get(f"{BASE_URL}/system/websocket/stats") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ WebSocket统计API正常")
                    print(f"   - 连接状态: {'已连接' if data['connected'] else '未连接'}")
                    print(f"   - 活跃连接数: {data['active_connections']}")
                    print(f"   - 消息速率: {data['message_rate']}/秒")
                    return True
                else:
                    print(f"❌ WebSocket统计API失败: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"❌ WebSocket统计API异常: {e}")
            return False

async def test_health_check():
    """测试健康检查API"""
    print("\n=== 测试健康检查API ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/health/detailed") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ 健康检查API正常")
                    print(f"   - 整体状态: {data['status']}")
                    print(f"   - API状态: {data['api_status']}")
                    
                    if 'components' in data:
                        for component, status in data['components'].items():
                            print(f"   - {component}: {status['status']}")
                    
                    return True
                else:
                    print(f"❌ 健康检查API失败: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 健康检查API异常: {e}")
            return False

async def main():
    """主测试函数"""
    print("🚀 开始系统监控功能测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API地址: {BASE_URL}")
    
    # 等待服务器启动
    print("\n⏳ 等待服务器启动...")
    await asyncio.sleep(2)
    
    # 运行所有测试
    tests = [
        ("健康检查", test_health_check),
        ("系统指标", test_system_metrics),
        ("系统状态", test_system_status),
        ("测试结果", test_test_results),
        ("运行测试", test_run_test),
        ("WebSocket统计", test_websocket_stats),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "="*50)
    print("📊 测试结果总结")
    print("="*50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {len(results)} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
    print(f"成功率: {(passed/len(results)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过！系统监控功能正常工作。")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查系统配置。")
    
    return failed == 0

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
        exit(1)
    except Exception as e:
        print(f"\n\n💥 测试执行异常: {e}")
        exit(1)