# Parquet存储模块

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Optional, Union, Dict
from datetime import datetime, date
from pathlib import Path
import logging
from .interfaces import IDataStore
from .utils import normalize_symbol, get_exchange_from_symbol, ensure_directory
from .calendar import get_trading_calendar

logger = logging.getLogger(__name__)


class ParquetStore(IDataStore):
    """Parquet存储实现类"""
    
    def __init__(self, compression: str = "snappy"):
        self.compression = compression
        self.trading_calendar = get_trading_calendar()
        logger.info(f"ParquetStore初始化完成，压缩格式: {compression}")
    
    def write_daily_bars(self, 
                        symbol: str, 
                        df: pd.DataFrame, 
                        root_path: str) -> str:
        """写入日频数据
        
        Args:
            symbol: 股票代码
            df: 数据DataFrame，index为DatetimeIndex
            root_path: 根路径
            
        Returns:
            写入的文件路径
        """
        try:
            # 标准化股票代码
            std_symbol = normalize_symbol(symbol)
            exchange = get_exchange_from_symbol(std_symbol)
            code = std_symbol.split('.')[0]
            
            logger.info(f"写入 {std_symbol} 日频数据，共 {len(df)} 条记录")
            
            if df.empty:
                logger.warning(f"数据为空，跳过写入: {std_symbol}")
                return ""
            
            # 确保数据有正确的索引
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame索引必须是DatetimeIndex")
            
            # 添加元数据列
            df_copy = df.copy()
            df_copy['symbol'] = std_symbol
            df_copy['exchange'] = exchange
            
            # 添加可见性时间戳（T+1规则）
            df_copy['visibility_ts'] = df_copy.index + pd.Timedelta(days=1)
            
            # 重置索引，将日期作为列
            df_copy = df_copy.reset_index()
            if 'date' not in df_copy.columns:
                df_copy.rename(columns={df_copy.columns[0]: 'date'}, inplace=True)
            
            # 确保日期列为日期类型
            df_copy['date'] = pd.to_datetime(df_copy['date']).dt.date
            
            # 构建分区路径: root/freq/exchange/symbol.parquet
            file_path = Path(root_path) / "D" / exchange / f"{code}.parquet"
            ensure_directory(str(file_path))
            
            # 如果文件已存在，进行增量更新
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                
                # 合并数据，去重
                combined_df = pd.concat([existing_df, df_copy], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                combined_df = combined_df.sort_values('date')
                
                logger.info(f"增量更新 {std_symbol}，原有 {len(existing_df)} 条，新增 {len(df_copy)} 条，合并后 {len(combined_df)} 条")
                df_to_write = combined_df
            else:
                df_to_write = df_copy.sort_values('date')
                logger.info(f"首次写入 {std_symbol}，共 {len(df_to_write)} 条记录")
            
            # 写入Parquet文件
            df_to_write.to_parquet(
                file_path,
                compression=self.compression,
                index=False,
                engine='pyarrow'
            )
            
            logger.info(f"成功写入 {std_symbol} 到 {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"写入 {symbol} 日频数据失败: {e}")
            raise
    
    def read_daily_bars(self, 
                       symbol: str, 
                       start_date: str, 
                       end_date: str,
                       root_path: str) -> pd.DataFrame:
        """读取日频数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            root_path: 根路径
            
        Returns:
            DataFrame with OHLCV data，index为DatetimeIndex
        """
        try:
            # 标准化股票代码
            std_symbol = normalize_symbol(symbol)
            exchange = get_exchange_from_symbol(std_symbol)
            code = std_symbol.split('.')[0]
            
            # 构建文件路径
            file_path = Path(root_path) / "D" / exchange / f"{code}.parquet"
            
            if not file_path.exists():
                logger.warning(f"文件不存在: {file_path}")
                return pd.DataFrame()
            
            logger.info(f"读取 {std_symbol} 日频数据: {start_date} - {end_date}")
            
            # 读取Parquet文件
            df = pd.read_parquet(file_path)
            
            if df.empty:
                logger.warning(f"文件为空: {file_path}")
                return pd.DataFrame()
            
            # 确保日期列存在
            if 'date' not in df.columns:
                logger.error(f"文件缺少date列: {file_path}")
                return pd.DataFrame()
            
            # 转换日期列
            df['date'] = pd.to_datetime(df['date'])
            
            # 按日期范围过滤
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
            df_filtered = df[mask].copy()
            
            if df_filtered.empty:
                logger.warning(f"指定日期范围内无数据: {std_symbol} {start_date}-{end_date}")
                return pd.DataFrame()
            
            # 设置日期索引
            df_filtered = df_filtered.set_index('date')
            
            # 转换为UTC时区
            if df_filtered.index.tz is None:
                df_filtered.index = df_filtered.index.tz_localize('Asia/Shanghai')
            df_filtered.index = df_filtered.index.tz_convert('UTC')
            
            # 移除元数据列，只保留OHLCV数据
            data_cols = ['open', 'high', 'low', 'close', 'volume']
            if 'amount' in df_filtered.columns:
                data_cols.append('amount')
            
            available_cols = [col for col in data_cols if col in df_filtered.columns]
            df_result = df_filtered[available_cols]
            
            logger.info(f"成功读取 {std_symbol} 数据 {len(df_result)} 条记录")
            return df_result
            
        except Exception as e:
            logger.error(f"读取 {symbol} 日频数据失败: {e}")
            return pd.DataFrame()
    
    def list_symbols(self, root_path: str) -> List[str]:
        """列出所有可用的股票代码
        
        Args:
            root_path: 根路径
            
        Returns:
            List of available symbols
        """
        try:
            symbols = []
            root_dir = Path(root_path) / "D"
            
            if not root_dir.exists():
                logger.warning(f"目录不存在: {root_dir}")
                return []
            
            # 遍历交易所目录
            for exchange_dir in root_dir.iterdir():
                if not exchange_dir.is_dir():
                    continue
                    
                exchange = exchange_dir.name
                
                # 遍历股票文件
                for file_path in exchange_dir.glob("*.parquet"):
                    code = file_path.stem
                    symbol = f"{code}.{exchange}"
                    symbols.append(symbol)
            
            symbols.sort()
            logger.info(f"找到 {len(symbols)} 个股票代码")
            return symbols
            
        except Exception as e:
            logger.error(f"列出股票代码失败: {e}")
            return []
    
    def write_minute_bars(self, 
                         symbol: str, 
                         df: pd.DataFrame, 
                         root_path: str,
                         period: str = "1") -> str:
        """写入分钟频数据
        
        Args:
            symbol: 股票代码
            df: 数据DataFrame
            root_path: 根路径
            period: 分钟周期
            
        Returns:
            写入的文件路径
        """
        try:
            std_symbol = normalize_symbol(symbol)
            exchange = get_exchange_from_symbol(std_symbol)
            code = std_symbol.split('.')[0]
            
            logger.info(f"写入 {std_symbol} {period}分钟数据，共 {len(df)} 条记录")
            
            if df.empty:
                logger.warning(f"数据为空，跳过写入: {std_symbol}")
                return ""
            
            # 添加元数据
            df_copy = df.copy()
            df_copy['symbol'] = std_symbol
            df_copy['exchange'] = exchange
            
            # 重置索引
            df_copy = df_copy.reset_index()
            if 'datetime' not in df_copy.columns:
                df_copy.rename(columns={df_copy.columns[0]: 'datetime'}, inplace=True)
            
            # 按日期分区存储
            df_copy['date'] = pd.to_datetime(df_copy['datetime']).dt.date
            
            # 按日期分组写入
            for date_val, group_df in df_copy.groupby('date'):
                date_str = date_val.strftime('%Y-%m-%d')
                
                # 构建路径: root/freq/exchange/date/symbol.parquet
                file_path = Path(root_path) / f"{period}m" / exchange / date_str / f"{code}.parquet"
                ensure_directory(str(file_path))
                
                # 移除临时的date列
                group_df = group_df.drop('date', axis=1)
                
                # 写入文件
                group_df.to_parquet(
                    file_path,
                    compression=self.compression,
                    index=False,
                    engine='pyarrow'
                )
            
            logger.info(f"成功写入 {std_symbol} {period}分钟数据")
            return str(file_path.parent)
            
        except Exception as e:
            logger.error(f"写入 {symbol} 分钟数据失败: {e}")
            raise
    
    def get_data_info(self, root_path: str) -> Dict:
        """获取数据存储信息
        
        Args:
            root_path: 根路径
            
        Returns:
            数据信息字典
        """
        try:
            info = {
                'total_symbols': 0,
                'exchanges': {},
                'frequencies': [],
                'date_range': {'start': None, 'end': None},
                'total_files': 0,
                'total_size_mb': 0
            }
            
            root_dir = Path(root_path)
            if not root_dir.exists():
                return info
            
            # 遍历频率目录
            for freq_dir in root_dir.iterdir():
                if not freq_dir.is_dir():
                    continue
                    
                freq = freq_dir.name
                info['frequencies'].append(freq)
                
                # 遍历交易所目录
                for exchange_dir in freq_dir.iterdir():
                    if not exchange_dir.is_dir():
                        continue
                        
                    exchange = exchange_dir.name
                    if exchange not in info['exchanges']:
                        info['exchanges'][exchange] = 0
                    
                    # 统计文件
                    for file_path in exchange_dir.rglob("*.parquet"):
                        info['total_files'] += 1
                        info['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                        
                        # 统计股票数量
                        if freq == 'D':  # 只在日频数据中统计
                            info['exchanges'][exchange] += 1
            
            info['total_symbols'] = sum(info['exchanges'].values())
            
            logger.info(f"数据存储信息: {info}")
            return info
            
        except Exception as e:
            logger.error(f"获取数据信息失败: {e}")
            return {}
    
    def cleanup_old_data(self, root_path: str, keep_days: int = 365):
        """清理旧数据
        
        Args:
            root_path: 根路径
            keep_days: 保留天数
        """
        try:
            cutoff_date = datetime.now().date() - pd.Timedelta(days=keep_days)
            logger.info(f"清理 {cutoff_date} 之前的数据")
            
            root_dir = Path(root_path)
            deleted_files = 0
            
            # 遍历分钟数据目录
            for freq_dir in root_dir.iterdir():
                if not freq_dir.is_dir() or freq_dir.name == 'D':
                    continue  # 跳过日频数据
                
                for exchange_dir in freq_dir.iterdir():
                    if not exchange_dir.is_dir():
                        continue
                    
                    # 遍历日期目录
                    for date_dir in exchange_dir.iterdir():
                        if not date_dir.is_dir():
                            continue
                        
                        try:
                            date_val = datetime.strptime(date_dir.name, '%Y-%m-%d').date()
                            if date_val < cutoff_date:
                                # 删除整个日期目录
                                import shutil
                                shutil.rmtree(date_dir)
                                deleted_files += 1
                                logger.debug(f"删除目录: {date_dir}")
                        except ValueError:
                            continue
            
            logger.info(f"清理完成，删除了 {deleted_files} 个目录")
            
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")


# 便捷函数
def read_code(symbol: str, start_date: str, end_date: str, 
              root: str = "data/parquet", freq: str = "D") -> pd.DataFrame:
    """读取股票数据的便捷函数
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        root: 数据根目录
        freq: 频率
        
    Returns:
        股票数据DataFrame
    """
    store = ParquetStore()
    if freq == "D":
        return store.read_daily_bars(symbol, start_date, end_date, root)
    else:
        # 分钟数据读取逻辑（待实现）
        logger.warning(f"暂不支持频率: {freq}")
        return pd.DataFrame()


def write_code(symbol: str, df: pd.DataFrame, root: str = "data/parquet", 
               freq: str = "D") -> str:
    """写入股票数据的便捷函数
    
    Args:
        symbol: 股票代码
        df: 数据DataFrame
        root: 数据根目录
        freq: 频率
        
    Returns:
        写入的文件路径
    """
    store = ParquetStore()
    if freq == "D":
        return store.write_daily_bars(symbol, df, root)
    else:
        # 分钟数据写入逻辑
        period = freq.replace('m', '')
        return store.write_minute_bars(symbol, df, root, period)