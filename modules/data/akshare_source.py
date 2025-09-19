# AkShare数据源实现

import akshare as ak
import pandas as pd
from typing import List, Union
from datetime import datetime, date
import logging
from .interfaces import IDataSource
from .utils import normalize_symbol, akshare_to_standard_symbol

logger = logging.getLogger(__name__)


class AkshareSource(IDataSource):
    """AkShare数据源实现类"""
    
    def __init__(self):
        self.name = "akshare"
        logger.info("AkShare数据源初始化完成")
    
    def fetch_daily_bars(self, 
                        symbol: str, 
                        start_date: Union[str, date, datetime], 
                        end_date: Union[str, date, datetime],
                        adjust: str = "qfq") -> pd.DataFrame:
        """获取日频OHLCV数据
        
        Args:
            symbol: 标准股票代码，如 "000001.SZ"
            start_date: 开始日期
            end_date: 结束日期  
            adjust: 复权类型 "qfq"/"hfq"/"none"
            
        Returns:
            DataFrame with columns: [open, high, low, close, volume, amount]
            Index: DatetimeIndex (UTC timezone)
        """
        try:
            # 转换为AkShare格式的代码
            ak_symbol = normalize_symbol(symbol, target="akshare")
            
            # 格式化日期
            if isinstance(start_date, (date, datetime)):
                start_str = start_date.strftime("%Y%m%d")
            else:
                start_str = pd.to_datetime(start_date).strftime("%Y%m%d")
                
            if isinstance(end_date, (date, datetime)):
                end_str = end_date.strftime("%Y%m%d")
            else:
                end_str = pd.to_datetime(end_date).strftime("%Y%m%d")
            
            logger.info(f"获取 {symbol} 日频数据: {start_str} - {end_str}, 复权: {adjust}")
            
            # 调用AkShare接口
            df = ak.stock_zh_a_hist(
                symbol=ak_symbol,
                period="daily", 
                start_date=start_str,
                end_date=end_str,
                adjust=adjust
            )
            
            if df is None or df.empty:
                logger.warning(f"未获取到 {symbol} 的数据")
                return pd.DataFrame()
            
            # 标准化列名
            column_mapping = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close', 
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_chg',
                '涨跌额': 'change',
                '换手率': 'turnover'
            }
            
            df = df.rename(columns=column_mapping)
            
            # 确保必要的列存在
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少必要的列: {missing_cols}")
            
            # 处理日期索引
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # 转换为UTC时区
            if df.index.tz is None:
                df.index = df.index.tz_localize('Asia/Shanghai')
            df.index = df.index.tz_convert('UTC')
            
            # 数据类型转换
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            if 'amount' in df.columns:
                numeric_cols.append('amount')
                
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 排序并去重
            df = df.sort_index().drop_duplicates()
            
            # 只返回核心字段
            core_cols = ['open', 'high', 'low', 'close', 'volume']
            if 'amount' in df.columns:
                core_cols.append('amount')
            
            df = df[core_cols]
            
            logger.info(f"成功获取 {symbol} 数据 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取 {symbol} 日频数据失败: {e}")
            return pd.DataFrame()
    
    def fetch_minute_bars(self,
                         symbol: str,
                         start_date: Union[str, date, datetime],
                         end_date: Union[str, date, datetime], 
                         period: str = "1",
                         adjust: str = "qfq") -> pd.DataFrame:
        """获取分钟频数据"""
        try:
            ak_symbol = normalize_symbol(symbol, target="akshare")
            
            logger.info(f"获取 {symbol} 分钟数据: period={period}, adjust={adjust}")
            
            # AkShare分钟数据接口
            df = ak.stock_zh_a_hist_min_em(
                symbol=ak_symbol,
                period=period,
                adjust=adjust
            )
            
            if df is None or df.empty:
                logger.warning(f"未获取到 {symbol} 的分钟数据")
                return pd.DataFrame()
            
            # 标准化列名
            column_mapping = {
                '时间': 'datetime',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high', 
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            }
            
            df = df.rename(columns=column_mapping)
            
            # 处理时间索引
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            
            # 转换时区
            if df.index.tz is None:
                df.index = df.index.tz_localize('Asia/Shanghai')
            df.index = df.index.tz_convert('UTC')
            
            # 数据类型转换
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 按时间范围过滤
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            end_dt = pd.to_datetime(end_date).tz_localize('UTC')
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            df = df.sort_index().drop_duplicates()
            
            logger.info(f"成功获取 {symbol} 分钟数据 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取 {symbol} 分钟数据失败: {e}")
            return pd.DataFrame()
    
    def fetch_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        try:
            logger.info("获取A股股票列表")
            
            # 获取A股列表
            df = ak.stock_info_a_code_name()
            
            if df is None or df.empty:
                logger.warning("未获取到股票列表")
                return pd.DataFrame()
            
            # 标准化列名
            column_mapping = {
                'code': 'symbol',
                'name': 'name'
            }
            
            df = df.rename(columns=column_mapping)
            
            # 转换为标准股票代码格式
            df['symbol'] = df['symbol'].apply(
                lambda x: akshare_to_standard_symbol(x)
            )
            
            # 添加交易所信息
            df['exchange'] = df['symbol'].apply(
                lambda x: 'SZ' if x.endswith('.SZ') else 'SH'
            )
            
            # 添加默认上市日期（实际项目中应该从其他接口获取）
            df['list_date'] = None
            df['delist_date'] = None
            
            logger.info(f"成功获取股票列表 {len(df)} 只")
            return df
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def fetch_trading_calendar(self, 
                              start_year: int = 2018, 
                              end_year: int = 2025) -> List[str]:
        """获取交易日历"""
        try:
            logger.info(f"获取交易日历: {start_year} - {end_year}")
            
            # 获取交易日历
            df = ak.tool_trade_date_hist_sina()
            
            if df is None or df.empty:
                logger.warning("未获取到交易日历")
                return []
            
            # 转换日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 按年份过滤
            df = df[
                (df['trade_date'].dt.year >= start_year) & 
                (df['trade_date'].dt.year <= end_year)
            ]
            
            # 转换为字符串列表
            trading_dates = df['trade_date'].dt.strftime('%Y-%m-%d').tolist()
            trading_dates.sort()
            
            logger.info(f"成功获取交易日历 {len(trading_dates)} 个交易日")
            return trading_dates
            
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []
    
    def fetch_realtime_quotes(self, symbols: List[str] = None) -> pd.DataFrame:
        """获取实时行情快照（用于实时数据增强）
        
        Args:
            symbols: 股票代码列表，如果为None则获取全市场
            
        Returns:
            DataFrame with realtime quote data
        """
        try:
            logger.info("获取实时行情快照")
            
            # 获取全市场快照
            df = ak.stock_zh_a_spot()
            
            if df is None or df.empty:
                logger.warning("未获取到实时行情")
                return pd.DataFrame()
            
            # 标准化列名
            column_mapping = {
                '代码': 'symbol',
                '名称': 'name',
                '最新价': 'price',
                '涨跌幅': 'pct_chg',
                '涨跌额': 'change',
                '成交量': 'volume',
                '成交额': 'amount',
                '今开': 'open',
                '昨收': 'prev_close',
                '最高': 'high',
                '最低': 'low'
            }
            
            df = df.rename(columns=column_mapping)
            
            # 转换股票代码格式
            df['symbol'] = df['symbol'].apply(
                lambda x: akshare_to_standard_symbol(x)
            )
            
            # 如果指定了股票列表，则过滤
            if symbols:
                df = df[df['symbol'].isin(symbols)]
            
            # 数据类型转换
            numeric_cols = ['price', 'pct_chg', 'change', 'volume', 'amount', 
                          'open', 'prev_close', 'high', 'low']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 添加时间戳
            df['timestamp'] = pd.Timestamp.now(tz='UTC')
            
            logger.info(f"成功获取实时行情 {len(df)} 只股票")
            return df
            
        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
            return pd.DataFrame()