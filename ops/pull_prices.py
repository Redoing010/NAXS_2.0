#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据拉取脚本

用法:
    python ops/pull_prices.py --market stock --codes 000001.SZ --codes 600519.SH \
        --start 2020-01-01 --end 2025-09-17 --freq D --out data/parquet
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.data.akshare_source import AkshareSource
from modules.data.parquet_store import ParquetStore
from modules.data.calendar import get_trading_calendar
from modules.data.utils import normalize_symbol

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/pull_prices.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='拉取股票价格数据')
    
    parser.add_argument('--market', type=str, default='stock',
                       choices=['stock'], help='市场类型')
    
    parser.add_argument('--codes', type=str, action='append', required=True,
                       help='股票代码，可多次指定')
    
    parser.add_argument('--start', type=str, required=True,
                       help='开始日期 YYYY-MM-DD')
    
    parser.add_argument('--end', type=str, required=True,
                       help='结束日期 YYYY-MM-DD')
    
    parser.add_argument('--freq', type=str, default='D',
                       choices=['D', '1m', '5m', '15m', '30m', '60m'],
                       help='数据频率')
    
    parser.add_argument('--out', type=str, default='data/parquet',
                       help='输出目录')
    
    parser.add_argument('--adjust', type=str, default='qfq',
                       choices=['qfq', 'hfq', 'none'],
                       help='复权类型')
    
    parser.add_argument('--batch-size', type=int, default=10,
                       help='批处理大小')
    
    parser.add_argument('--retry', type=int, default=3,
                       help='重试次数')
    
    parser.add_argument('--delay', type=float, default=0.5,
                       help='请求间隔（秒）')
    
    parser.add_argument('--force', action='store_true',
                       help='强制重新拉取（覆盖现有数据）')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='试运行模式（不实际拉取数据）')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    return parser.parse_args()


def setup_logging(log_level: str):
    """设置日志级别"""
    level = getattr(logging, log_level.upper())
    logging.getLogger().setLevel(level)
    
    # 确保日志目录存在
    Path('logs').mkdir(exist_ok=True)


def validate_date_range(start_date: str, end_date: str) -> bool:
    """验证日期范围"""
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_dt >= end_dt:
            logger.error(f"开始日期必须早于结束日期: {start_date} >= {end_date}")
            return False
        
        # 检查是否超过当前日期
        today = datetime.now().date()
        if end_dt.date() > today:
            logger.warning(f"结束日期超过当前日期，将调整为: {today}")
        
        return True
        
    except ValueError as e:
        logger.error(f"日期格式错误: {e}")
        return False


def get_stock_list() -> List[str]:
    """获取默认股票列表"""
    # 沪深300成分股示例
    default_stocks = [
        '000001.SZ', '000002.SZ', '000858.SZ', '000831.SZ',
        '600000.SH', '600036.SH', '600519.SH', '600887.SH',
        '000300.SZ', '399001.SZ'  # 指数
    ]
    return default_stocks


def pull_daily_data(source: AkshareSource, store: ParquetStore,
                   symbols: List[str], start_date: str, end_date: str,
                   output_dir: str, adjust: str = 'qfq',
                   retry_count: int = 3, delay: float = 0.5,
                   force: bool = False, dry_run: bool = False) -> dict:
    """拉取日频数据
    
    Args:
        source: 数据源
        store: 存储器
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        adjust: 复权类型
        retry_count: 重试次数
        delay: 请求间隔
        force: 是否强制重新拉取
        dry_run: 是否试运行
        
    Returns:
        拉取结果统计
    """
    import time
    
    stats = {
        'total': len(symbols),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'failed_symbols': [],
        'start_time': datetime.now(),
        'end_time': None
    }
    
    logger.info(f"开始拉取日频数据: {len(symbols)} 只股票, {start_date} - {end_date}")
    
    for i, symbol in enumerate(symbols):
        logger.info(f"处理 {symbol} ({i+1}/{len(symbols)})")
        
        if dry_run:
            logger.info(f"[试运行] 跳过实际拉取: {symbol}")
            stats['success'] += 1
            continue
        
        # 检查是否已存在数据（如果不强制重新拉取）
        if not force:
            try:
                existing_data = store.read_daily_bars(symbol, start_date, end_date, output_dir)
                if not existing_data.empty:
                    logger.info(f"数据已存在，跳过: {symbol}")
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass  # 如果读取失败，继续拉取
        
        # 重试机制
        success = False
        for attempt in range(retry_count):
            try:
                logger.debug(f"拉取 {symbol} 数据 (尝试 {attempt+1}/{retry_count})")
                
                # 拉取数据
                df = source.fetch_daily_bars(symbol, start_date, end_date, adjust)
                
                if df.empty:
                    logger.warning(f"未获取到数据: {symbol}")
                    break
                
                # 存储数据
                file_path = store.write_daily_bars(symbol, df, output_dir)
                
                if file_path:
                    logger.info(f"成功拉取并存储 {symbol}: {len(df)} 条记录")
                    stats['success'] += 1
                    success = True
                    break
                else:
                    logger.warning(f"存储失败: {symbol}")
                
            except Exception as e:
                logger.error(f"拉取 {symbol} 失败 (尝试 {attempt+1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(delay * (attempt + 1))  # 递增延迟
                continue
        
        if not success:
            stats['failed'] += 1
            stats['failed_symbols'].append(symbol)
        
        # 请求间隔
        if delay > 0 and i < len(symbols) - 1:
            time.sleep(delay)
    
    stats['end_time'] = datetime.now()
    stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
    
    return stats


def pull_minute_data(source: AkshareSource, store: ParquetStore,
                    symbols: List[str], start_date: str, end_date: str,
                    output_dir: str, period: str = '1',
                    adjust: str = 'qfq', retry_count: int = 3,
                    delay: float = 0.5, force: bool = False,
                    dry_run: bool = False) -> dict:
    """拉取分钟频数据
    
    Args:
        source: 数据源
        store: 存储器
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        period: 分钟周期
        adjust: 复权类型
        retry_count: 重试次数
        delay: 请求间隔
        force: 是否强制重新拉取
        dry_run: 是否试运行
        
    Returns:
        拉取结果统计
    """
    import time
    
    stats = {
        'total': len(symbols),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'failed_symbols': [],
        'start_time': datetime.now(),
        'end_time': None
    }
    
    logger.info(f"开始拉取{period}分钟数据: {len(symbols)} 只股票, {start_date} - {end_date}")
    
    for i, symbol in enumerate(symbols):
        logger.info(f"处理 {symbol} ({i+1}/{len(symbols)})")
        
        if dry_run:
            logger.info(f"[试运行] 跳过实际拉取: {symbol}")
            stats['success'] += 1
            continue
        
        # 重试机制
        success = False
        for attempt in range(retry_count):
            try:
                logger.debug(f"拉取 {symbol} {period}分钟数据 (尝试 {attempt+1}/{retry_count})")
                
                # 拉取数据
                df = source.fetch_minute_bars(symbol, start_date, end_date, period, adjust)
                
                if df.empty:
                    logger.warning(f"未获取到分钟数据: {symbol}")
                    break
                
                # 存储数据
                file_path = store.write_minute_bars(symbol, df, output_dir, period)
                
                if file_path:
                    logger.info(f"成功拉取并存储 {symbol} {period}分钟数据: {len(df)} 条记录")
                    stats['success'] += 1
                    success = True
                    break
                else:
                    logger.warning(f"存储分钟数据失败: {symbol}")
                
            except Exception as e:
                logger.error(f"拉取 {symbol} {period}分钟数据失败 (尝试 {attempt+1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(delay * (attempt + 1))
                continue
        
        if not success:
            stats['failed'] += 1
            stats['failed_symbols'].append(symbol)
        
        # 请求间隔
        if delay > 0 and i < len(symbols) - 1:
            time.sleep(delay)
    
    stats['end_time'] = datetime.now()
    stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
    
    return stats


def print_stats(stats: dict):
    """打印统计信息"""
    logger.info("=" * 50)
    logger.info("拉取统计:")
    logger.info(f"  总数: {stats['total']}")
    logger.info(f"  成功: {stats['success']}")
    logger.info(f"  失败: {stats['failed']}")
    logger.info(f"  跳过: {stats['skipped']}")
    logger.info(f"  耗时: {stats['duration']:.2f} 秒")
    
    if stats['failed_symbols']:
        logger.warning(f"失败的股票: {stats['failed_symbols'][:10]}")
        if len(stats['failed_symbols']) > 10:
            logger.warning(f"... 还有 {len(stats['failed_symbols']) - 10} 个")
    
    success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
    logger.info(f"成功率: {success_rate:.1f}%")
    logger.info("=" * 50)


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    logger.info(f"开始执行数据拉取任务: {args}")
    
    # 验证参数
    if not validate_date_range(args.start, args.end):
        sys.exit(1)
    
    # 标准化股票代码
    symbols = []
    for code in args.codes:
        try:
            std_symbol = normalize_symbol(code)
            symbols.append(std_symbol)
        except Exception as e:
            logger.error(f"无效的股票代码: {code}, {e}")
            continue
    
    if not symbols:
        logger.error("没有有效的股票代码")
        sys.exit(1)
    
    logger.info(f"处理股票代码: {symbols}")
    
    # 创建输出目录
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化数据源和存储器
    source = AkshareSource()
    store = ParquetStore()
    
    try:
        # 根据频率拉取数据
        if args.freq == 'D':
            stats = pull_daily_data(
                source, store, symbols, args.start, args.end,
                str(output_dir), args.adjust, args.retry, args.delay,
                args.force, args.dry_run
            )
        else:
            # 分钟数据
            period = args.freq.replace('m', '')
            stats = pull_minute_data(
                source, store, symbols, args.start, args.end,
                str(output_dir), period, args.adjust, args.retry,
                args.delay, args.force, args.dry_run
            )
        
        # 打印统计信息
        print_stats(stats)
        
        # 返回码
        if stats['failed'] > 0:
            logger.warning(f"部分拉取失败，失败数: {stats['failed']}")
            sys.exit(2)
        else:
            logger.info("所有数据拉取成功")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        sys.exit(130)
    except Exception as e:
        logger.error(f"拉取数据失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()