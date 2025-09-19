#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健康检查模块

为系统各组件提供健康状态检查功能
"""

import os
import time
import psutil
import logging
import requests
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class HealthCheckRegistry:
    """健康检查注册器"""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}
        self._lock = threading.Lock()
        
        # 注册默认健康检查
        self._register_default_checks()
        
        logger.info("健康检查注册器初始化完成")
    
    def _register_default_checks(self):
        """注册默认健康检查"""
        self.register("system_resources", self._check_system_resources)
        self.register("disk_space", self._check_disk_space)
        self.register("data_directories", self._check_data_directories)
        self.register("parquet_store", self._check_parquet_store)
        self.register("akshare_connection", self._check_akshare_connection)
        self.register("trading_calendar", self._check_trading_calendar)
    
    def register(self, name: str, check_func: Callable[[], bool]):
        """注册健康检查"""
        with self._lock:
            self.checks[name] = check_func
        logger.info(f"注册健康检查: {name}")
    
    def unregister(self, name: str):
        """取消注册健康检查"""
        with self._lock:
            if name in self.checks:
                del self.checks[name]
                logger.info(f"取消注册健康检查: {name}")
    
    def get_all_checks(self) -> Dict[str, Callable[[], bool]]:
        """获取所有健康检查"""
        with self._lock:
            return self.checks.copy()
    
    def _check_system_resources(self) -> bool:
        """检查系统资源"""
        try:
            # 检查CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                logger.warning(f"CPU使用率过高: {cpu_percent}%")
                return False
            
            # 检查内存使用率
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"内存使用率过高: {memory.percent}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"系统资源检查失败: {e}")
            return False
    
    def _check_disk_space(self) -> bool:
        """检查磁盘空间"""
        try:
            # 检查根目录磁盘空间
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024 ** 3)
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            if free_gb < 1.0:  # 少于1GB
                logger.warning(f"磁盘剩余空间不足: {free_gb:.1f}GB")
                return False
            
            if usage_percent > 95:  # 使用率超过95%
                logger.warning(f"磁盘使用率过高: {usage_percent:.1f}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"磁盘空间检查失败: {e}")
            return False
    
    def _check_data_directories(self) -> bool:
        """检查数据目录"""
        try:
            # 检查关键目录是否存在且可写
            directories = [
                "data",
                "data/parquet", 
                "data/qlib",
                "data/cache",
                "logs"
            ]
            
            for dir_path in directories:
                path = Path(dir_path)
                
                # 创建目录（如果不存在）
                path.mkdir(parents=True, exist_ok=True)
                
                # 检查是否可写
                if not os.access(path, os.W_OK):
                    logger.warning(f"目录不可写: {path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"数据目录检查失败: {e}")
            return False
    
    def _check_parquet_store(self) -> bool:
        """检查Parquet存储"""
        try:
            # 导入并测试ParquetStore
            import sys
            from pathlib import Path
            
            # 添加项目根目录到Python路径
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))
            
            from modules.data.parquet_store import ParquetStore
            
            store = ParquetStore()
            
            # 测试基本功能
            test_root = "data/parquet"
            symbols = store.list_symbols(test_root)
            
            # 如果有数据，尝试读取一个样本
            if symbols:
                test_symbol = symbols[0]
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                
                try:
                    df = store.read_daily_bars(test_symbol, start_date, end_date, test_root)
                    # 读取成功即可，不要求有数据
                except Exception:
                    pass  # 读取失败不影响健康检查
            
            return True
            
        except Exception as e:
            logger.error(f"Parquet存储检查失败: {e}")
            return False
    
    def _check_akshare_connection(self) -> bool:
        """检查AkShare连接"""
        try:
            # 导入AkShare并测试连接
            import akshare as ak
            
            # 简单测试：获取股票列表（限制数量）
            start_time = time.time()
            stock_list = ak.stock_info_a_code_name()
            elapsed_time = time.time() - start_time
            
            if stock_list is None or stock_list.empty:
                logger.warning("AkShare返回空数据")
                return False
            
            if elapsed_time > 30:  # 超过30秒认为连接慢
                logger.warning(f"AkShare响应较慢: {elapsed_time:.1f}秒")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"AkShare连接检查失败: {e}")
            return False
    
    def _check_trading_calendar(self) -> bool:
        """检查交易日历"""
        try:
            # 导入并测试交易日历
            import sys
            from pathlib import Path
            
            # 添加项目根目录到Python路径
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))
            
            from modules.data.calendar import get_trading_calendar
            
            calendar = get_trading_calendar()
            
            # 测试基本功能
            current_year = datetime.now().year
            trading_dates = calendar.fetch_trading_dates(current_year, current_year)
            
            if not trading_dates:
                logger.warning("交易日历为空")
                return False
            
            # 测试日期判断功能
            test_date = "2024-01-02"
            is_trading = calendar.is_trading_day(test_date)
            
            return True
            
        except Exception as e:
            logger.error(f"交易日历检查失败: {e}")
            return False


class APIHealthChecker:
    """API健康检查器"""
    
    def __init__(self, base_url: str = "http://localhost:3001"):
        self.base_url = base_url
        self.timeout = 10
        
    def check_api_health(self) -> bool:
        """检查API健康状态"""
        try:
            # 检查健康检查端点
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.warning(f"API健康检查返回状态码: {response.status_code}")
                return False
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API健康检查失败: {e}")
            return False
    
    def check_data_endpoints(self) -> bool:
        """检查数据端点"""
        try:
            # 检查数据端点（不需要实际数据）
            response = requests.get(
                f"{self.base_url}/api/bars",
                params={
                    "code": "000001.SZ",
                    "start": "2024-01-01",
                    "end": "2024-01-02",
                    "freq": "D"
                },
                timeout=self.timeout
            )
            
            # 200或404都可以接受（404表示没有数据但端点正常）
            if response.status_code not in [200, 404]:
                logger.warning(f"数据端点返回状态码: {response.status_code}")
                return False
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"数据端点检查失败: {e}")
            return False
    
    def check_admin_endpoints(self) -> bool:
        """检查管理端点"""
        try:
            # 检查系统状态端点（不需要认证）
            response = requests.get(
                f"{self.base_url}/api/admin/get_system_status",
                timeout=self.timeout
            )
            
            # 401表示需要认证，这是正常的
            if response.status_code not in [200, 401]:
                logger.warning(f"管理端点返回状态码: {response.status_code}")
                return False
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"管理端点检查失败: {e}")
            return False


class DatabaseHealthChecker:
    """数据库健康检查器"""
    
    def check_parquet_files(self, data_root: str = "data/parquet") -> bool:
        """检查Parquet文件完整性"""
        try:
            import pandas as pd
            
            data_path = Path(data_root)
            if not data_path.exists():
                logger.warning(f"Parquet数据目录不存在: {data_path}")
                return True  # 目录不存在不算错误
            
            # 检查是否有Parquet文件
            parquet_files = list(data_path.rglob("*.parquet"))
            if not parquet_files:
                logger.info("没有找到Parquet文件")
                return True  # 没有文件不算错误
            
            # 随机检查几个文件
            import random
            sample_files = random.sample(parquet_files, min(3, len(parquet_files)))
            
            for file_path in sample_files:
                try:
                    df = pd.read_parquet(file_path)
                    if df.empty:
                        logger.warning(f"Parquet文件为空: {file_path}")
                        continue
                    
                    # 检查必要列
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        logger.warning(f"Parquet文件缺少列 {missing_cols}: {file_path}")
                        return False
                        
                except Exception as e:
                    logger.error(f"读取Parquet文件失败 {file_path}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Parquet文件检查失败: {e}")
            return False
    
    def check_qlib_data(self, qlib_root: str = "data/qlib") -> bool:
        """检查Qlib数据完整性"""
        try:
            qlib_path = Path(qlib_root)
            if not qlib_path.exists():
                logger.info(f"Qlib数据目录不存在: {qlib_path}")
                return True  # 目录不存在不算错误
            
            # 检查Qlib数据结构
            expected_dirs = ['features', 'calendars']
            for dir_name in expected_dirs:
                dir_path = qlib_path / dir_name
                if dir_path.exists() and not any(dir_path.iterdir()):
                    logger.warning(f"Qlib目录为空: {dir_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Qlib数据检查失败: {e}")
            return False


# 全局健康检查注册器
_health_registry = None


def get_health_registry() -> HealthCheckRegistry:
    """获取健康检查注册器单例"""
    global _health_registry
    if _health_registry is None:
        _health_registry = HealthCheckRegistry()
    return _health_registry


def register_health_check(name: str, check_func: Callable[[], bool]):
    """注册健康检查"""
    get_health_registry().register(name, check_func)


def run_all_health_checks() -> Dict[str, bool]:
    """运行所有健康检查"""
    registry = get_health_registry()
    checks = registry.get_all_checks()
    results = {}
    
    for name, check_func in checks.items():
        try:
            start_time = time.time()
            result = check_func()
            elapsed_time = time.time() - start_time
            
            results[name] = result
            
            if result:
                logger.debug(f"健康检查 {name} 通过 ({elapsed_time:.2f}s)")
            else:
                logger.warning(f"健康检查 {name} 失败 ({elapsed_time:.2f}s)")
                
        except Exception as e:
            logger.error(f"健康检查 {name} 异常: {e}")
            results[name] = False
    
    return results


def get_system_health_summary() -> Dict[str, Any]:
    """获取系统健康摘要"""
    check_results = run_all_health_checks()
    
    total_checks = len(check_results)
    passed_checks = sum(1 for result in check_results.values() if result)
    failed_checks = total_checks - passed_checks
    
    # 确定整体健康状态
    if failed_checks == 0:
        overall_status = "healthy"
    elif failed_checks <= total_checks * 0.3:  # 30%以下失败
        overall_status = "warning"
    else:
        overall_status = "critical"
    
    return {
        'timestamp': datetime.now().isoformat(),
        'overall_status': overall_status,
        'is_healthy': failed_checks == 0,
        'total_checks': total_checks,
        'passed_checks': passed_checks,
        'failed_checks': failed_checks,
        'check_results': check_results,
        'failed_check_names': [name for name, result in check_results.items() if not result]
    }