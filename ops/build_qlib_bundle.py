#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qlib数据包构建脚本

用法:
    python ops/build_qlib_bundle.py --start 2018-01-01 --end 2025-09-17 \
        --parquet-root data/parquet --qlib-out data/qlib_cn_daily
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.data.qlib_writer import QlibWriter
from modules.data.parquet_store import ParquetStore
from modules.data.calendar import get_trading_calendar
from modules.data.utils import normalize_symbol

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/build_qlib.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='构建Qlib数据包')
    
    parser.add_argument('--start', type=str, required=True,
                       help='开始日期 YYYY-MM-DD')
    
    parser.add_argument('--end', type=str, required=True,
                       help='结束日期 YYYY-MM-DD')
    
    parser.add_argument('--parquet-root', type=str, default='data/parquet',
                       help='Parquet数据根目录')
    
    parser.add_argument('--qlib-out', type=str, default='data/qlib_cn_daily',
                       help='Qlib输出目录')
    
    parser.add_argument('--symbols', type=str, nargs='*',
                       help='指定股票代码列表，不指定则使用所有可用股票')
    
    parser.add_argument('--symbols-file', type=str,
                       help='从文件读取股票代码列表（每行一个）')
    
    parser.add_argument('--market', type=str, default='all',
                       choices=['all', 'sh', 'sz'],
                       help='市场过滤')
    
    parser.add_argument('--min-days', type=int, default=100,
                       help='最少交易天数过滤')
    
    parser.add_argument('--update', action='store_true',
                       help='增量更新模式')
    
    parser.add_argument('--validate', action='store_true',
                       help='构建后验证数据包')
    
    parser.add_argument('--backup', action='store_true',
                       help='备份现有数据包')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='试运行模式（不实际构建）')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    parser.add_argument('--parallel', type=int, default=1,
                       help='并行处理数量')
    
    return parser.parse_args()


def setup_logging(log_level: str):
    """设置日志级别"""
    level = getattr(logging, log_level.upper())
    logging.getLogger().setLevel(level)
    
    # 确保日志目录存在
    Path('logs').mkdir(exist_ok=True)


def load_symbols_from_file(file_path: str) -> List[str]:
    """从文件加载股票代码列表
    
    Args:
        file_path: 文件路径
        
    Returns:
        股票代码列表
    """
    symbols = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    symbols.append(line)
        logger.info(f"从文件加载 {len(symbols)} 个股票代码: {file_path}")
        return symbols
    except Exception as e:
        logger.error(f"加载股票代码文件失败: {e}")
        return []


def get_available_symbols(parquet_root: str, market: str = 'all') -> List[str]:
    """获取可用的股票代码列表
    
    Args:
        parquet_root: Parquet数据根目录
        market: 市场过滤
        
    Returns:
        股票代码列表
    """
    try:
        store = ParquetStore()
        all_symbols = store.list_symbols(parquet_root)
        
        if market == 'all':
            return all_symbols
        elif market == 'sh':
            return [s for s in all_symbols if s.endswith('.SH')]
        elif market == 'sz':
            return [s for s in all_symbols if s.endswith('.SZ')]
        else:
            return all_symbols
            
    except Exception as e:
        logger.error(f"获取可用股票代码失败: {e}")
        return []


def filter_symbols_by_data_quality(symbols: List[str], parquet_root: str,
                                  start_date: str, end_date: str,
                                  min_days: int = 100) -> List[str]:
    """根据数据质量过滤股票代码
    
    Args:
        symbols: 股票代码列表
        parquet_root: Parquet数据根目录
        start_date: 开始日期
        end_date: 结束日期
        min_days: 最少交易天数
        
    Returns:
        过滤后的股票代码列表
    """
    logger.info(f"开始数据质量过滤，最少交易天数: {min_days}")
    
    store = ParquetStore()
    filtered_symbols = []
    
    for symbol in symbols:
        try:
            df = store.read_daily_bars(symbol, start_date, end_date, parquet_root)
            
            if df.empty:
                logger.debug(f"跳过无数据股票: {symbol}")
                continue
            
            # 检查数据天数
            if len(df) < min_days:
                logger.debug(f"跳过数据不足股票: {symbol} ({len(df)} < {min_days})")
                continue
            
            # 检查数据完整性
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.debug(f"跳过缺少列的股票: {symbol} (缺少: {missing_cols})")
                continue
            
            # 检查数据有效性
            null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if null_ratio > 0.5:  # 超过50%的数据缺失
                logger.debug(f"跳过数据缺失过多股票: {symbol} (缺失率: {null_ratio:.2%})")
                continue
            
            filtered_symbols.append(symbol)
            
        except Exception as e:
            logger.debug(f"检查股票数据失败: {symbol}, {e}")
            continue
    
    logger.info(f"数据质量过滤完成: {len(symbols)} -> {len(filtered_symbols)}")
    return filtered_symbols


def backup_existing_bundle(qlib_out: str) -> Optional[str]:
    """备份现有数据包
    
    Args:
        qlib_out: Qlib输出目录
        
    Returns:
        备份目录路径
    """
    try:
        qlib_path = Path(qlib_out)
        if not qlib_path.exists():
            logger.info("无现有数据包需要备份")
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = qlib_path.parent / f"{qlib_path.name}_backup_{timestamp}"
        
        logger.info(f"备份现有数据包: {qlib_path} -> {backup_path}")
        
        import shutil
        shutil.copytree(qlib_path, backup_path)
        
        logger.info(f"备份完成: {backup_path}")
        return str(backup_path)
        
    except Exception as e:
        logger.error(f"备份数据包失败: {e}")
        return None


def build_bundle(symbols: List[str], start_date: str, end_date: str,
                parquet_root: str, qlib_out: str, update: bool = False,
                dry_run: bool = False, parallel: int = 1) -> dict:
    """构建Qlib数据包
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        parquet_root: Parquet数据根目录
        qlib_out: Qlib输出目录
        update: 是否增量更新
        dry_run: 是否试运行
        parallel: 并行数量
        
    Returns:
        构建结果统计
    """
    stats = {
        'total_symbols': len(symbols),
        'success_symbols': 0,
        'failed_symbols': [],
        'start_time': datetime.now(),
        'end_time': None,
        'output_path': qlib_out
    }
    
    logger.info(f"开始构建Qlib数据包: {len(symbols)} 只股票")
    logger.info(f"日期范围: {start_date} - {end_date}")
    logger.info(f"输出目录: {qlib_out}")
    logger.info(f"更新模式: {update}")
    
    if dry_run:
        logger.info("[试运行模式] 跳过实际构建")
        stats['success_symbols'] = len(symbols)
        stats['end_time'] = datetime.now()
        return stats
    
    try:
        writer = QlibWriter()
        
        if update:
            # 增量更新模式
            output_path = writer.update_qlib_bundle(
                qlib_out, symbols, start_date, end_date, parquet_root
            )
        else:
            # 全量构建模式
            output_path = writer.build_qlib_bundle(
                symbols, start_date, end_date, parquet_root, qlib_out
            )
        
        if output_path:
            stats['success_symbols'] = len(symbols)  # 简化统计
            logger.info(f"Qlib数据包构建成功: {output_path}")
        else:
            logger.error("Qlib数据包构建失败")
            stats['failed_symbols'] = symbols
        
    except Exception as e:
        logger.error(f"构建Qlib数据包失败: {e}")
        stats['failed_symbols'] = symbols
    
    stats['end_time'] = datetime.now()
    stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
    
    return stats


def validate_bundle(qlib_out: str) -> dict:
    """验证Qlib数据包
    
    Args:
        qlib_out: Qlib输出目录
        
    Returns:
        验证结果
    """
    logger.info(f"开始验证Qlib数据包: {qlib_out}")
    
    try:
        writer = QlibWriter()
        result = writer.validate_qlib_bundle(qlib_out)
        
        if result['valid']:
            logger.info("数据包验证通过")
            logger.info(f"统计信息: {result['stats']}")
        else:
            logger.error("数据包验证失败")
            for error in result['errors']:
                logger.error(f"  错误: {error}")
        
        if result['warnings']:
            logger.warning("验证警告:")
            for warning in result['warnings'][:5]:  # 只显示前5个警告
                logger.warning(f"  警告: {warning}")
        
        return result
        
    except Exception as e:
        logger.error(f"验证数据包失败: {e}")
        return {'valid': False, 'errors': [str(e)], 'warnings': [], 'stats': {}}


def print_stats(stats: dict):
    """打印统计信息"""
    logger.info("=" * 50)
    logger.info("构建统计:")
    logger.info(f"  总股票数: {stats['total_symbols']}")
    logger.info(f"  成功数: {stats['success_symbols']}")
    logger.info(f"  失败数: {len(stats['failed_symbols'])}")
    logger.info(f"  耗时: {stats['duration']:.2f} 秒")
    logger.info(f"  输出路径: {stats['output_path']}")
    
    if stats['failed_symbols']:
        logger.warning(f"失败的股票: {stats['failed_symbols'][:10]}")
        if len(stats['failed_symbols']) > 10:
            logger.warning(f"... 还有 {len(stats['failed_symbols']) - 10} 个")
    
    success_rate = stats['success_symbols'] / stats['total_symbols'] * 100 if stats['total_symbols'] > 0 else 0
    logger.info(f"成功率: {success_rate:.1f}%")
    logger.info("=" * 50)


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    logger.info(f"开始执行Qlib数据包构建任务: {args}")
    
    # 验证日期范围
    try:
        start_dt = datetime.strptime(args.start, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end, '%Y-%m-%d')
        
        if start_dt >= end_dt:
            logger.error(f"开始日期必须早于结束日期: {args.start} >= {args.end}")
            sys.exit(1)
            
    except ValueError as e:
        logger.error(f"日期格式错误: {e}")
        sys.exit(1)
    
    # 确定股票代码列表
    symbols = []
    
    if args.symbols:
        # 使用命令行指定的股票代码
        symbols = [normalize_symbol(s) for s in args.symbols]
        logger.info(f"使用指定的股票代码: {len(symbols)} 个")
    elif args.symbols_file:
        # 从文件读取股票代码
        symbols = load_symbols_from_file(args.symbols_file)
        if not symbols:
            logger.error("从文件读取股票代码失败")
            sys.exit(1)
    else:
        # 使用所有可用的股票代码
        symbols = get_available_symbols(args.parquet_root, args.market)
        if not symbols:
            logger.error("未找到可用的股票代码")
            sys.exit(1)
        logger.info(f"使用所有可用股票代码: {len(symbols)} 个")
    
    # 数据质量过滤
    if args.min_days > 0:
        symbols = filter_symbols_by_data_quality(
            symbols, args.parquet_root, args.start, args.end, args.min_days
        )
        
        if not symbols:
            logger.error("经过数据质量过滤后，没有符合条件的股票")
            sys.exit(1)
    
    logger.info(f"最终处理股票数量: {len(symbols)}")
    
    # 备份现有数据包
    if args.backup and not args.dry_run:
        backup_path = backup_existing_bundle(args.qlib_out)
        if backup_path:
            logger.info(f"现有数据包已备份到: {backup_path}")
    
    try:
        # 构建数据包
        stats = build_bundle(
            symbols, args.start, args.end, args.parquet_root,
            args.qlib_out, args.update, args.dry_run, args.parallel
        )
        
        # 打印统计信息
        print_stats(stats)
        
        # 验证数据包
        if args.validate and not args.dry_run:
            validation_result = validate_bundle(args.qlib_out)
            if not validation_result['valid']:
                logger.error("数据包验证失败")
                sys.exit(3)
        
        # 返回码
        if len(stats['failed_symbols']) > 0:
            logger.warning(f"部分构建失败，失败数: {len(stats['failed_symbols'])}")
            sys.exit(2)
        else:
            logger.info("Qlib数据包构建成功")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        sys.exit(130)
    except Exception as e:
        logger.error(f"构建数据包失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()