# Data Layer Interfaces

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime, date


class IDataSource(ABC):
    """数据源接口抽象类"""
    
    @abstractmethod
    def fetch_daily_bars(self, 
                        symbol: str, 
                        start_date: Union[str, date, datetime], 
                        end_date: Union[str, date, datetime],
                        adjust: str = "qfq") -> pd.DataFrame:
        """获取日频OHLCV数据
        
        Args:
            symbol: 股票代码，如 "000001.SZ"
            start_date: 开始日期
            end_date: 结束日期  
            adjust: 复权类型 "qfq"/"hfq"/"none"
            
        Returns:
            DataFrame with columns: [open, high, low, close, volume, amount]
            Index: DatetimeIndex (UTC timezone)
        """
        pass
    
    @abstractmethod
    def fetch_minute_bars(self,
                         symbol: str,
                         start_date: Union[str, date, datetime],
                         end_date: Union[str, date, datetime], 
                         period: str = "1",
                         adjust: str = "qfq") -> pd.DataFrame:
        """获取分钟频数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 周期 "1"/"5"/"15"/"30"/"60"
            adjust: 复权类型
            
        Returns:
            DataFrame with minute-level OHLCV data
        """
        pass
    
    @abstractmethod
    def fetch_stock_list(self) -> pd.DataFrame:
        """获取股票列表
        
        Returns:
            DataFrame with columns: [symbol, name, list_date, delist_date, exchange]
        """
        pass
    
    @abstractmethod
    def fetch_trading_calendar(self, 
                              start_year: int = 2018, 
                              end_year: int = 2025) -> List[str]:
        """获取交易日历
        
        Args:
            start_year: 开始年份
            end_year: 结束年份
            
        Returns:
            List of trading dates in 'YYYY-MM-DD' format
        """
        pass


class IDataStore(ABC):
    """数据存储接口抽象类"""
    
    @abstractmethod
    def write_daily_bars(self, 
                        symbol: str, 
                        df: pd.DataFrame, 
                        root_path: str) -> str:
        """写入日频数据
        
        Args:
            symbol: 股票代码
            df: 数据DataFrame
            root_path: 根路径
            
        Returns:
            写入的文件路径
        """
        pass
    
    @abstractmethod
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
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def list_symbols(self, root_path: str) -> List[str]:
        """列出所有可用的股票代码
        
        Args:
            root_path: 根路径
            
        Returns:
            List of available symbols
        """
        pass


class IQlibWriter(ABC):
    """Qlib数据写入接口"""
    
    @abstractmethod
    def write_calendars(self, 
                       trading_dates: List[str], 
                       output_dir: str) -> str:
        """写入交易日历
        
        Args:
            trading_dates: 交易日期列表
            output_dir: 输出目录
            
        Returns:
            写入的文件路径
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def write_features(self, 
                      symbol: str, 
                      df: pd.DataFrame, 
                      output_dir: str,
                      freq: str = "D") -> str:
        """写入特征数据
        
        Args:
            symbol: 股票代码
            df: 特征数据
            output_dir: 输出目录
            freq: 频率
            
        Returns:
            写入的文件路径
        """
        pass