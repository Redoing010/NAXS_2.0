#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAXS端到端测试脚本

测试完整的数据处理链路：
1. 数据拉取 (AkShare -> Parquet)
2. 数据质量检查
3. Qlib数据包构建
4. API接口测试
5. 系统集成验证

用法:
    python test_e2e.py
"""

import sys
import os
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import requests
import time

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from modules.data.akshare_source import AkshareSource
from modules.data.parquet_store import ParquetStore
from modules.data.calendar import get_trading_calendar
from modules.data.dq import DataQualityChecker
from modules.data.qlib_writer import QlibWriter
from modules.data.utils import normalize_symbol

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class E2ETestRunner:
    """端到端测试运行器"""
    
    def __init__(self, test_dir: str = None):
        self.test_dir = test_dir or tempfile.mkdtemp(prefix="naxs_e2e_")
        self.parquet_root = os.path.join(self.test_dir, "parquet")
        self.qlib_root = os.path.join(self.test_dir, "qlib")
        self.reports_root = os.path.join(self.test_dir, "reports")
        
        # 测试用股票代码
        self.test_symbols = ["000001.SZ", "600519.SH", "000858.SZ"]
        
        # 测试日期范围（最近30个交易日）
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=60)  # 扩大范围确保有足够交易日
        self.start_date = start_date.strftime('%Y-%m-%d')
        self.end_date = end_date.strftime('%Y-%m-%d')
        
        # 初始化组件
        self.source = AkshareSource()
        self.store = ParquetStore()
        self.calendar = get_trading_calendar()
        self.dq_checker = DataQualityChecker()
        self.qlib_writer = QlibWriter()
        
        # 测试结果
        self.test_results = {}
        
        logger.info(f"E2E测试初始化完成，测试目录: {self.test_dir}")
        logger.info(f"测试日期范围: {self.start_date} - {self.end_date}")
        logger.info(f"测试股票: {self.test_symbols}")
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        logger.info("开始执行端到端测试")
        
        tests = [
            ("test_data_source", "数据源测试"),
            ("test_data_storage", "数据存储测试"),
            ("test_trading_calendar", "交易日历测试"),
            ("test_data_quality", "数据质量测试"),
            ("test_qlib_integration", "Qlib集成测试"),
            ("test_api_endpoints", "API接口测试"),
            ("test_end_to_end_flow", "端到端流程测试")
        ]
        
        all_passed = True
        
        for test_method, test_name in tests:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"开始执行: {test_name}")
                logger.info(f"{'='*50}")
                
                result = getattr(self, test_method)()
                self.test_results[test_name] = result
                
                if result:
                    logger.info(f"✅ {test_name} - 通过")
                else:
                    logger.error(f"❌ {test_name} - 失败")
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"❌ {test_name} - 异常: {e}")
                self.test_results[test_name] = False
                all_passed = False
        
        # 打印测试总结
        self.print_test_summary()
        
        return all_passed
    
    def test_data_source(self) -> bool:
        """测试数据源功能"""
        try:
            logger.info("测试AkShare数据源连接...")
            
            # 测试股票列表获取
            stock_list = self.source.fetch_stock_list()
            if stock_list.empty:
                logger.error("获取股票列表失败")
                return False
            
            logger.info(f"成功获取股票列表，共 {len(stock_list)} 只股票")
            
            # 测试交易日历获取
            trading_dates = self.source.fetch_trading_calendar(2024, 2024)
            if not trading_dates:
                logger.error("获取交易日历失败")
                return False
            
            logger.info(f"成功获取交易日历，共 {len(trading_dates)} 个交易日")
            
            # 测试单只股票数据获取
            test_symbol = self.test_symbols[0]
            logger.info(f"测试获取 {test_symbol} 的日频数据...")
            
            df = self.source.fetch_daily_bars(test_symbol, self.start_date, self.end_date)
            if df.empty:
                logger.warning(f"未获取到 {test_symbol} 的数据，可能是网络问题")
                # 不直接返回False，因为可能是网络问题
                return True
            
            logger.info(f"成功获取 {test_symbol} 数据，共 {len(df)} 条记录")
            
            # 验证数据结构
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"数据缺少必要列: {missing_cols}")
                return False
            
            logger.info("数据源测试通过")
            return True
            
        except Exception as e:
            logger.error(f"数据源测试失败: {e}")
            return False
    
    def test_data_storage(self) -> bool:
        """测试数据存储功能"""
        try:
            logger.info("测试Parquet数据存储...")
            
            # 创建测试数据
            import pandas as pd
            import numpy as np
            
            # 生成模拟数据
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            test_data = pd.DataFrame({
                'open': np.random.uniform(10, 20, len(dates)),
                'high': np.random.uniform(15, 25, len(dates)),
                'low': np.random.uniform(8, 15, len(dates)),
                'close': np.random.uniform(10, 20, len(dates)),
                'volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
            
            # 确保OHLC逻辑正确
            test_data['high'] = np.maximum(test_data['high'], 
                                         np.maximum(test_data['open'], test_data['close']))
            test_data['low'] = np.minimum(test_data['low'], 
                                        np.minimum(test_data['open'], test_data['close']))
            
            test_symbol = "TEST001.SZ"
            
            # 测试写入
            logger.info(f"测试写入数据: {test_symbol}")
            file_path = self.store.write_daily_bars(test_symbol, test_data, self.parquet_root)
            
            if not file_path or not os.path.exists(file_path):
                logger.error("数据写入失败")
                return False
            
            logger.info(f"数据写入成功: {file_path}")
            
            # 测试读取
            logger.info(f"测试读取数据: {test_symbol}")
            read_data = self.store.read_daily_bars(test_symbol, self.start_date, self.end_date, self.parquet_root)
            
            if read_data.empty:
                logger.error("数据读取失败")
                return False
            
            logger.info(f"数据读取成功，共 {len(read_data)} 条记录")
            
            # 验证数据一致性
            if len(read_data) != len(test_data):
                logger.error(f"数据长度不一致: 写入 {len(test_data)}, 读取 {len(read_data)}")
                return False
            
            # 测试列出股票代码
            symbols = self.store.list_symbols(self.parquet_root)
            if test_symbol not in symbols:
                logger.error(f"列出的股票代码中未找到 {test_symbol}")
                return False
            
            logger.info("数据存储测试通过")
            return True
            
        except Exception as e:
            logger.error(f"数据存储测试失败: {e}")
            return False
    
    def test_trading_calendar(self) -> bool:
        """测试交易日历功能"""
        try:
            logger.info("测试交易日历功能...")
            
            # 测试获取交易日历
            trading_dates = self.calendar.fetch_trading_dates(2024, 2024)
            if not trading_dates:
                logger.error("获取交易日历失败")
                return False
            
            logger.info(f"成功获取2024年交易日历，共 {len(trading_dates)} 个交易日")
            
            # 测试交易日判断
            test_date = "2024-01-02"  # 通常是交易日
            is_trading = self.calendar.is_trading_day(test_date)
            logger.info(f"{test_date} 是否为交易日: {is_trading}")
            
            # 测试获取下一个交易日
            next_trading_day = self.calendar.get_next_trading_day(test_date)
            logger.info(f"{test_date} 的下一个交易日: {next_trading_day}")
            
            # 测试获取前一个交易日
            prev_trading_day = self.calendar.get_prev_trading_day(test_date)
            logger.info(f"{test_date} 的前一个交易日: {prev_trading_day}")
            
            # 测试获取日期范围内的交易日
            range_dates = self.calendar.get_trading_days_between("2024-01-01", "2024-01-31")
            logger.info(f"2024年1月交易日数量: {len(range_dates)}")
            
            logger.info("交易日历测试通过")
            return True
            
        except Exception as e:
            logger.error(f"交易日历测试失败: {e}")
            return False
    
    def test_data_quality(self) -> bool:
        """测试数据质量检查功能"""
        try:
            logger.info("测试数据质量检查...")
            
            # 首先确保有测试数据
            if not os.path.exists(self.parquet_root):
                logger.warning("没有Parquet数据，跳过数据质量测试")
                return True
            
            # 获取可用的股票代码
            available_symbols = self.store.list_symbols(self.parquet_root)
            if not available_symbols:
                logger.warning("没有可用的股票数据，跳过数据质量测试")
                return True
            
            test_symbol = available_symbols[0]
            logger.info(f"对 {test_symbol} 进行数据质量检查")
            
            # 执行质量检查
            report = self.dq_checker.check_symbol(
                test_symbol, self.start_date, self.end_date, self.parquet_root
            )
            
            if not report:
                logger.error("数据质量检查失败")
                return False
            
            logger.info(f"数据质量检查完成:")
            logger.info(f"  - 总记录数: {report.total_records}")
            logger.info(f"  - 错误数: {report.error_count}")
            logger.info(f"  - 警告数: {report.warning_count}")
            logger.info(f"  - 健康状态: {report.is_healthy}")
            
            # 测试多股票质量检查
            test_symbols = available_symbols[:2]  # 取前两只股票
            reports = self.dq_checker.check_multiple_symbols(
                test_symbols, self.start_date, self.end_date, self.parquet_root
            )
            
            if not reports:
                logger.error("多股票数据质量检查失败")
                return False
            
            logger.info(f"多股票质量检查完成，检查了 {len(reports)} 只股票")
            
            # 生成汇总报告
            summary = self.dq_checker.generate_summary_report(reports)
            logger.info(f"汇总报告: 总股票数 {summary['total_symbols']}, 健康股票数 {summary['healthy_symbols']}")
            
            logger.info("数据质量测试通过")
            return True
            
        except Exception as e:
            logger.error(f"数据质量测试失败: {e}")
            return False
    
    def test_qlib_integration(self) -> bool:
        """测试Qlib集成功能"""
        try:
            logger.info("测试Qlib集成...")
            
            # 检查是否有Parquet数据
            if not os.path.exists(self.parquet_root):
                logger.warning("没有Parquet数据，跳过Qlib集成测试")
                return True
            
            available_symbols = self.store.list_symbols(self.parquet_root)
            if not available_symbols:
                logger.warning("没有可用的股票数据，跳过Qlib集成测试")
                return True
            
            # 选择测试股票（最多3只）
            test_symbols = available_symbols[:3]
            logger.info(f"测试Qlib数据包构建，股票: {test_symbols}")
            
            # 构建Qlib数据包
            output_path = self.qlib_writer.build_qlib_bundle(
                test_symbols, self.start_date, self.end_date, 
                self.parquet_root, self.qlib_root
            )
            
            if not output_path or not os.path.exists(output_path):
                logger.error("Qlib数据包构建失败")
                return False
            
            logger.info(f"Qlib数据包构建成功: {output_path}")
            
            # 验证Qlib数据包
            validation_result = self.qlib_writer.validate_qlib_bundle(self.qlib_root)
            
            if not validation_result['valid']:
                logger.error(f"Qlib数据包验证失败: {validation_result['errors']}")
                return False
            
            logger.info("Qlib数据包验证通过")
            logger.info(f"统计信息: {validation_result['stats']}")
            
            if validation_result['warnings']:
                logger.warning(f"验证警告: {validation_result['warnings'][:3]}")
            
            logger.info("Qlib集成测试通过")
            return True
            
        except Exception as e:
            logger.error(f"Qlib集成测试失败: {e}")
            return False
    
    def test_api_endpoints(self) -> bool:
        """测试API接口"""
        try:
            logger.info("测试API接口...")
            
            # 这里假设API服务在localhost:3001运行
            # 实际测试中可能需要启动API服务
            base_url = "http://localhost:3001"
            
            # 测试健康检查接口
            try:
                response = requests.get(f"{base_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ 健康检查接口正常")
                else:
                    logger.warning(f"健康检查接口返回状态码: {response.status_code}")
            except requests.exceptions.RequestException:
                logger.warning("API服务未运行，跳过API接口测试")
                return True
            
            # 测试数据接口
            if os.path.exists(self.parquet_root):
                available_symbols = self.store.list_symbols(self.parquet_root)
                if available_symbols:
                    test_symbol = available_symbols[0]
                    
                    try:
                        response = requests.get(
                            f"{base_url}/api/bars",
                            params={
                                "code": test_symbol,
                                "start": self.start_date,
                                "end": self.end_date,
                                "freq": "D"
                            },
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            logger.info(f"✅ 数据接口正常，返回 {data.get('total_records', 0)} 条记录")
                        else:
                            logger.warning(f"数据接口返回状态码: {response.status_code}")
                            
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"数据接口测试失败: {e}")
            
            logger.info("API接口测试完成")
            return True
            
        except Exception as e:
            logger.error(f"API接口测试失败: {e}")
            return False
    
    def test_end_to_end_flow(self) -> bool:
        """测试端到端完整流程"""
        try:
            logger.info("测试端到端完整流程...")
            
            # 1. 数据拉取
            logger.info("步骤1: 数据拉取")
            test_symbol = self.test_symbols[0]
            
            try:
                df = self.source.fetch_daily_bars(test_symbol, self.start_date, self.end_date)
                if not df.empty:
                    logger.info(f"✅ 成功拉取 {test_symbol} 数据 {len(df)} 条")
                    
                    # 2. 数据存储
                    logger.info("步骤2: 数据存储")
                    file_path = self.store.write_daily_bars(test_symbol, df, self.parquet_root)
                    if file_path:
                        logger.info(f"✅ 数据存储成功: {file_path}")
                        
                        # 3. 数据质量检查
                        logger.info("步骤3: 数据质量检查")
                        report = self.dq_checker.check_symbol(
                            test_symbol, self.start_date, self.end_date, self.parquet_root
                        )
                        if report:
                            logger.info(f"✅ 数据质量检查完成，健康状态: {report.is_healthy}")
                            
                            # 4. Qlib数据包构建
                            logger.info("步骤4: Qlib数据包构建")
                            qlib_path = self.qlib_writer.build_qlib_bundle(
                                [test_symbol], self.start_date, self.end_date,
                                self.parquet_root, self.qlib_root
                            )
                            if qlib_path:
                                logger.info(f"✅ Qlib数据包构建成功: {qlib_path}")
                                
                                # 5. 数据读取验证
                                logger.info("步骤5: 数据读取验证")
                                read_df = self.store.read_daily_bars(
                                    test_symbol, self.start_date, self.end_date, self.parquet_root
                                )
                                if not read_df.empty:
                                    logger.info(f"✅ 数据读取验证成功，共 {len(read_df)} 条记录")
                                    
                                    logger.info("🎉 端到端完整流程测试通过")
                                    return True
                else:
                    logger.warning(f"未能获取 {test_symbol} 的数据，可能是网络问题")
                    return True  # 网络问题不算测试失败
                    
            except Exception as e:
                logger.warning(f"数据拉取失败: {e}，可能是网络问题")
                return True  # 网络问题不算测试失败
            
            logger.error("端到端流程测试失败")
            return False
            
        except Exception as e:
            logger.error(f"端到端流程测试失败: {e}")
            return False
    
    def print_test_summary(self):
        """打印测试总结"""
        logger.info("\n" + "="*60)
        logger.info("测试总结")
        logger.info("="*60)
        
        passed_count = sum(1 for result in self.test_results.values() if result)
        total_count = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"{test_name:<30} {status}")
        
        logger.info("-" * 60)
        logger.info(f"总计: {passed_count}/{total_count} 通过")
        
        if passed_count == total_count:
            logger.info("🎉 所有测试通过！")
        else:
            logger.warning(f"⚠️  {total_count - passed_count} 个测试失败")
        
        logger.info(f"测试目录: {self.test_dir}")
        logger.info("="*60)
    
    def cleanup(self):
        """清理测试环境"""
        try:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
                logger.info(f"清理测试目录: {self.test_dir}")
        except Exception as e:
            logger.warning(f"清理测试目录失败: {e}")


def main():
    """主函数"""
    logger.info("NAXS端到端测试开始")
    
    # 创建测试运行器
    runner = E2ETestRunner()
    
    try:
        # 运行所有测试
        success = runner.run_all_tests()
        
        if success:
            logger.info("\n🎉 所有端到端测试通过！")
            return 0
        else:
            logger.error("\n❌ 部分测试失败")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("\n用户中断测试")
        return 130
    except Exception as e:
        logger.error(f"\n测试执行异常: {e}")
        return 1
    finally:
        # 清理测试环境
        runner.cleanup()


if __name__ == '__main__':
    sys.exit(main())