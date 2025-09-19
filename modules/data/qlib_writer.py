# Qlib数据写入器模块

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime, date
from pathlib import Path
import logging
import pickle
import struct
from .interfaces import IQlibWriter
from .parquet_store import ParquetStore
from .calendar import get_trading_calendar
from .utils import normalize_symbol, get_exchange_from_symbol

logger = logging.getLogger(__name__)


class QlibWriter(IQlibWriter):
    """Qlib数据写入器实现类"""
    
    def __init__(self, parquet_store: Optional[ParquetStore] = None):
        self.store = parquet_store or ParquetStore()
        self.trading_calendar = get_trading_calendar()
        logger.info("QlibWriter初始化完成")
    
    def write_calendars(self, 
                       trading_dates: List[str], 
                       output_dir: str) -> str:
        """写入交易日历
        
        Args:
            trading_dates: 交易日期列表，格式为 'YYYY-MM-DD'
            output_dir: 输出目录
            
        Returns:
            写入的文件路径
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            calendar_file = output_path / "calendars" / "day.txt"
            calendar_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"写入交易日历，共 {len(trading_dates)} 个交易日")
            
            # 转换日期格式为Qlib要求的格式
            qlib_dates = []
            for date_str in trading_dates:
                # 转换为 YYYY-MM-DD 格式
                if len(date_str) == 8:  # YYYYMMDD格式
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                else:
                    formatted_date = date_str
                qlib_dates.append(formatted_date)
            
            # 写入文件
            with open(calendar_file, 'w', encoding='utf-8') as f:
                for date_str in sorted(qlib_dates):
                    f.write(f"{date_str}\n")
            
            logger.info(f"交易日历已写入: {calendar_file}")
            return str(calendar_file)
            
        except Exception as e:
            logger.error(f"写入交易日历失败: {e}")
            raise
    
    def write_instruments(self, 
                         symbols: List[str], 
                         output_dir: str) -> str:
        """写入股票列表
        
        Args:
            symbols: 股票代码列表
            output_dir: 输出目录
            
        Returns:
            写入的文件路径
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            instruments_file = output_path / "instruments" / "all.txt"
            instruments_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"写入股票列表，共 {len(symbols)} 只股票")
            
            # 标准化股票代码并按交易所分组
            instruments_data = []
            
            for symbol in symbols:
                try:
                    std_symbol = normalize_symbol(symbol)
                    exchange = get_exchange_from_symbol(std_symbol)
                    code = std_symbol.split('.')[0]
                    
                    # Qlib格式: SH000001 或 SZ000001
                    if exchange == 'SH':
                        qlib_symbol = f"SH{code}"
                    elif exchange == 'SZ':
                        qlib_symbol = f"SZ{code}"
                    else:
                        continue
                    
                    instruments_data.append({
                        'symbol': qlib_symbol,
                        'start_date': '2018-01-01',  # 默认开始日期
                        'end_date': '2030-12-31'     # 默认结束日期
                    })
                    
                except Exception as e:
                    logger.warning(f"处理股票代码 {symbol} 失败: {e}")
                    continue
            
            # 写入instruments文件
            with open(instruments_file, 'w', encoding='utf-8') as f:
                for item in sorted(instruments_data, key=lambda x: x['symbol']):
                    f.write(f"{item['symbol']}\t{item['start_date']}\t{item['end_date']}\n")
            
            logger.info(f"股票列表已写入: {instruments_file}")
            return str(instruments_file)
            
        except Exception as e:
            logger.error(f"写入股票列表失败: {e}")
            raise
    
    def write_features(self, 
                      symbol: str, 
                      df: pd.DataFrame, 
                      output_dir: str,
                      freq: str = "D") -> str:
        """写入特征数据
        
        Args:
            symbol: 股票代码
            df: 特征数据，index为DatetimeIndex
            output_dir: 输出目录
            freq: 频率
            
        Returns:
            写入的文件路径
        """
        try:
            if df.empty:
                logger.warning(f"数据为空，跳过写入: {symbol}")
                return ""
            
            # 标准化股票代码
            std_symbol = normalize_symbol(symbol)
            exchange = get_exchange_from_symbol(std_symbol)
            code = std_symbol.split('.')[0]
            
            # 转换为Qlib格式的股票代码
            if exchange == 'SH':
                qlib_symbol = f"SH{code}"
            elif exchange == 'SZ':
                qlib_symbol = f"SZ{code}"
            else:
                raise ValueError(f"不支持的交易所: {exchange}")
            
            logger.info(f"写入 {qlib_symbol} 特征数据，共 {len(df)} 条记录")
            
            # 创建输出目录
            output_path = Path(output_dir)
            features_dir = output_path / "features" / qlib_symbol.lower()
            features_dir.mkdir(parents=True, exist_ok=True)
            
            # 确保数据有正确的索引
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame索引必须是DatetimeIndex")
            
            # 转换时区到UTC
            df_copy = df.copy()
            if df_copy.index.tz is None:
                df_copy.index = df_copy.index.tz_localize('Asia/Shanghai')
            df_copy.index = df_copy.index.tz_convert('UTC')
            
            # 标准化列名映射
            column_mapping = {
                'open': '$open',
                'high': '$high', 
                'low': '$low',
                'close': '$close',
                'volume': '$volume',
                'amount': '$amount',
                'factor': '$factor'  # 复权因子
            }
            
            # 重命名列
            df_renamed = df_copy.rename(columns=column_mapping)
            
            # 确保必要的列存在
            required_cols = ['$open', '$high', '$low', '$close', '$volume']
            available_cols = [col for col in required_cols if col in df_renamed.columns]
            
            if len(available_cols) < 4:  # 至少需要OHLC
                logger.warning(f"缺少必要的OHLC列: {symbol}")
                return ""
            
            # 添加复权因子（如果不存在）
            if '$factor' not in df_renamed.columns:
                df_renamed['$factor'] = 1.0
            
            # 按列写入二进制文件
            written_files = []
            
            for col in df_renamed.columns:
                if col.startswith('$'):
                    col_file = features_dir / f"{col[1:]}.bin"  # 移除$前缀
                    
                    # 写入二进制数据
                    self._write_binary_feature(df_renamed[col], col_file)
                    written_files.append(str(col_file))
            
            # 写入日期索引
            dates_file = features_dir / "date.bin"
            self._write_date_index(df_renamed.index, dates_file)
            written_files.append(str(dates_file))
            
            logger.info(f"成功写入 {qlib_symbol} 特征数据到 {features_dir}")
            return str(features_dir)
            
        except Exception as e:
            logger.error(f"写入 {symbol} 特征数据失败: {e}")
            raise
    
    def _write_binary_feature(self, series: pd.Series, output_file: Path):
        """写入二进制特征文件
        
        Args:
            series: 数据序列
            output_file: 输出文件路径
        """
        try:
            # 转换为float32数组
            data = series.astype(np.float32).values
            
            # 处理NaN值
            data = np.nan_to_num(data, nan=0.0)
            
            # 写入二进制文件
            with open(output_file, 'wb') as f:
                # 写入数据长度
                f.write(struct.pack('I', len(data)))
                # 写入数据
                f.write(data.tobytes())
            
            logger.debug(f"写入特征文件: {output_file}")
            
        except Exception as e:
            logger.error(f"写入二进制特征失败: {e}")
            raise
    
    def _write_date_index(self, date_index: pd.DatetimeIndex, output_file: Path):
        """写入日期索引文件
        
        Args:
            date_index: 日期索引
            output_file: 输出文件路径
        """
        try:
            # 转换为时间戳（秒）
            timestamps = date_index.astype('int64') // 10**9
            timestamps = np.asarray(timestamps, dtype=np.int64)
            
            # 写入二进制文件
            with open(output_file, 'wb') as f:
                # 写入数据长度
                f.write(struct.pack('I', len(timestamps)))
                # 写入时间戳数据
                f.write(timestamps.tobytes())
            
            logger.debug(f"写入日期索引: {output_file}")
            
        except Exception as e:
            logger.error(f"写入日期索引失败: {e}")
            raise
    
    def build_qlib_bundle(self, 
                         symbols: List[str],
                         start_date: str,
                         end_date: str,
                         parquet_root: str,
                         output_dir: str) -> str:
        """构建完整的Qlib数据包
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            parquet_root: Parquet数据根目录
            output_dir: 输出目录
            
        Returns:
            输出目录路径
        """
        try:
            logger.info(f"开始构建Qlib数据包: {len(symbols)} 只股票, {start_date} - {end_date}")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 1. 写入交易日历
            trading_dates = self.trading_calendar.get_trading_days_between(start_date, end_date)
            self.write_calendars(trading_dates, output_dir)
            
            # 2. 写入股票列表
            self.write_instruments(symbols, output_dir)
            
            # 3. 写入特征数据
            success_count = 0
            failed_symbols = []
            
            for i, symbol in enumerate(symbols):
                try:
                    logger.info(f"处理 {symbol} ({i+1}/{len(symbols)})")
                    
                    # 从Parquet读取数据
                    df = self.store.read_daily_bars(symbol, start_date, end_date, parquet_root)
                    
                    if df.empty:
                        logger.warning(f"未找到 {symbol} 的数据")
                        failed_symbols.append(symbol)
                        continue
                    
                    # 写入特征数据
                    self.write_features(symbol, df, output_dir)
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"处理 {symbol} 失败: {e}")
                    failed_symbols.append(symbol)
                    continue
            
            # 4. 生成元数据
            metadata = {
                'created_at': datetime.now().isoformat(),
                'date_range': f"{start_date} - {end_date}",
                'total_symbols': len(symbols),
                'success_symbols': success_count,
                'failed_symbols': failed_symbols,
                'trading_days': len(trading_dates),
                'data_source': 'akshare',
                'frequency': 'daily'
            }
            
            metadata_file = output_path / "metadata.json"
            import json
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Qlib数据包构建完成: {output_dir}")
            logger.info(f"成功处理 {success_count}/{len(symbols)} 只股票")
            
            if failed_symbols:
                logger.warning(f"失败的股票: {failed_symbols[:10]}{'...' if len(failed_symbols) > 10 else ''}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"构建Qlib数据包失败: {e}")
            raise
    
    def validate_qlib_bundle(self, bundle_dir: str) -> Dict[str, any]:
        """验证Qlib数据包
        
        Args:
            bundle_dir: 数据包目录
            
        Returns:
            验证结果
        """
        try:
            bundle_path = Path(bundle_dir)
            
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'stats': {}
            }
            
            # 检查必要的目录和文件
            required_paths = [
                'calendars/day.txt',
                'instruments/all.txt',
                'features'
            ]
            
            for path in required_paths:
                full_path = bundle_path / path
                if not full_path.exists():
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"缺少必要文件/目录: {path}")
            
            # 检查交易日历
            calendar_file = bundle_path / "calendars" / "day.txt"
            if calendar_file.exists():
                with open(calendar_file, 'r') as f:
                    trading_days = [line.strip() for line in f if line.strip()]
                validation_result['stats']['trading_days'] = len(trading_days)
            
            # 检查股票列表
            instruments_file = bundle_path / "instruments" / "all.txt"
            if instruments_file.exists():
                with open(instruments_file, 'r') as f:
                    instruments = [line.strip() for line in f if line.strip()]
                validation_result['stats']['instruments'] = len(instruments)
            
            # 检查特征目录
            features_dir = bundle_path / "features"
            if features_dir.exists():
                symbol_dirs = [d for d in features_dir.iterdir() if d.is_dir()]
                validation_result['stats']['feature_symbols'] = len(symbol_dirs)
                
                # 检查特征文件完整性
                incomplete_symbols = []
                for symbol_dir in symbol_dirs[:10]:  # 只检查前10个
                    required_files = ['open.bin', 'high.bin', 'low.bin', 'close.bin', 'volume.bin']
                    missing_files = [f for f in required_files if not (symbol_dir / f).exists()]
                    if missing_files:
                        incomplete_symbols.append(f"{symbol_dir.name}: {missing_files}")
                
                if incomplete_symbols:
                    validation_result['warnings'].extend(incomplete_symbols)
            
            logger.info(f"Qlib数据包验证完成: {validation_result}")
            return validation_result
            
        except Exception as e:
            logger.error(f"验证Qlib数据包失败: {e}")
            return {
                'valid': False,
                'errors': [f"验证失败: {e}"],
                'warnings': [],
                'stats': {}
            }
    
    def update_qlib_bundle(self, 
                          bundle_dir: str,
                          symbols: List[str],
                          start_date: str,
                          end_date: str,
                          parquet_root: str) -> str:
        """增量更新Qlib数据包
        
        Args:
            bundle_dir: 现有数据包目录
            symbols: 要更新的股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            parquet_root: Parquet数据根目录
            
        Returns:
            更新后的目录路径
        """
        try:
            logger.info(f"增量更新Qlib数据包: {len(symbols)} 只股票")
            
            bundle_path = Path(bundle_dir)
            if not bundle_path.exists():
                logger.warning(f"数据包目录不存在，将创建新的数据包: {bundle_dir}")
                return self.build_qlib_bundle(symbols, start_date, end_date, parquet_root, bundle_dir)
            
            # 更新交易日历
            trading_dates = self.trading_calendar.get_trading_days_between(start_date, end_date)
            
            # 读取现有交易日历
            calendar_file = bundle_path / "calendars" / "day.txt"
            existing_dates = set()
            if calendar_file.exists():
                with open(calendar_file, 'r') as f:
                    existing_dates = {line.strip() for line in f if line.strip()}
            
            # 合并日期
            all_dates = sorted(existing_dates.union(set(trading_dates)))
            self.write_calendars(all_dates, bundle_dir)
            
            # 更新股票列表
            instruments_file = bundle_path / "instruments" / "all.txt"
            existing_symbols = set()
            if instruments_file.exists():
                with open(instruments_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            symbol = line.split('\t')[0]
                            existing_symbols.add(symbol)
            
            # 转换新股票代码格式
            new_qlib_symbols = []
            for symbol in symbols:
                std_symbol = normalize_symbol(symbol)
                exchange = get_exchange_from_symbol(std_symbol)
                code = std_symbol.split('.')[0]
                if exchange == 'SH':
                    new_qlib_symbols.append(f"SH{code}")
                elif exchange == 'SZ':
                    new_qlib_symbols.append(f"SZ{code}")
            
            # 合并股票列表
            all_symbols_qlib = list(existing_symbols.union(set(new_qlib_symbols)))
            
            # 转换回标准格式用于写入instruments
            all_symbols_std = []
            for qlib_symbol in all_symbols_qlib:
                if qlib_symbol.startswith('SH'):
                    all_symbols_std.append(f"{qlib_symbol[2:]}.SH")
                elif qlib_symbol.startswith('SZ'):
                    all_symbols_std.append(f"{qlib_symbol[2:]}.SZ")
            
            self.write_instruments(all_symbols_std, bundle_dir)
            
            # 更新特征数据
            updated_count = 0
            for symbol in symbols:
                try:
                    df = self.store.read_daily_bars(symbol, start_date, end_date, parquet_root)
                    if not df.empty:
                        self.write_features(symbol, df, bundle_dir)
                        updated_count += 1
                except Exception as e:
                    logger.error(f"更新 {symbol} 失败: {e}")
                    continue
            
            logger.info(f"增量更新完成，更新了 {updated_count} 只股票")
            return bundle_dir
            
        except Exception as e:
            logger.error(f"增量更新Qlib数据包失败: {e}")
            raise


# 便捷函数
def build_qlib_data(symbols: List[str], start_date: str, end_date: str,
                   parquet_root: str = "data/parquet", 
                   qlib_output: str = "data/qlib_cn_daily") -> str:
    """构建Qlib数据的便捷函数
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        parquet_root: Parquet数据根目录
        qlib_output: Qlib输出目录
        
    Returns:
        输出目录路径
    """
    writer = QlibWriter()
    return writer.build_qlib_bundle(symbols, start_date, end_date, parquet_root, qlib_output)


def validate_qlib_data(bundle_dir: str) -> Dict[str, any]:
    """验证Qlib数据的便捷函数
    
    Args:
        bundle_dir: 数据包目录
        
    Returns:
        验证结果
    """
    writer = QlibWriter()
    return writer.validate_qlib_bundle(bundle_dir)