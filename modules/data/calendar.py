# 交易日历模块

from __future__ import annotations

import importlib
from importlib import util as importlib_util
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Set, Union

import pandas as pd

logger = logging.getLogger(__name__)

_akshare_spec = importlib_util.find_spec("akshare")
if _akshare_spec is not None:
    ak = importlib.import_module("akshare")  # type: ignore[import-untyped]
else:  # pragma: no cover - 仅在缺少AkShare时执行
    ak = None  # type: ignore[assignment]
    logger.warning(
        "AkShare package is not available; TradingCalendar will rely on fallback calendars."
    )


class TradingCalendar:
    """交易日历管理类"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "trading_calendar.json"
        self._trading_dates_cache = None
        
    def fetch_trading_dates(self, start_year: int = 2018, end_year: int = 2025) -> List[str]:
        """获取交易日期列表
        
        Args:
            start_year: 开始年份
            end_year: 结束年份
            
        Returns:
            交易日期列表，格式为 'YYYY-MM-DD'
        """
        try:
            logger.info(f"获取交易日历: {start_year}-{end_year}")

            # 尝试从缓存加载
            cached_dates = self._load_from_cache(start_year, end_year)
            if cached_dates:
                logger.info(f"从缓存加载交易日历，共 {len(cached_dates)} 个交易日")
                return cached_dates

            if ak is None:
                logger.info("AkShare不可用，使用备用交易日历")
                return self._generate_fallback_calendar(start_year, end_year)

            df = ak.tool_trade_date_hist_sina()  # type: ignore[attr-defined]
            if df is None or df.empty:
                logger.warning("未获取到交易日历，使用备用交易日历")
                return self._generate_fallback_calendar(start_year, end_year)

            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df[
                (df['trade_date'].dt.year >= start_year) &
                (df['trade_date'].dt.year <= end_year)
            ]

            trading_dates = (
                df['trade_date']
                .dt.strftime('%Y-%m-%d')
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            self._save_to_cache(trading_dates, start_year, end_year)

            logger.info(f"成功获取交易日历，共 {len(trading_dates)} 个交易日")
            return trading_dates

        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            fallback = self._generate_fallback_calendar(start_year, end_year)
            logger.info("使用备用交易日历，共 %d 个交易日", len(fallback))
            return fallback
    
    def _load_from_cache(self, start_year: int, end_year: int) -> List[str]:
        """从缓存加载交易日历"""
        try:
            if not self.cache_file.exists():
                return []
                
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 检查缓存是否覆盖所需年份
            cached_start = cache_data.get('start_year', 9999)
            cached_end = cache_data.get('end_year', 0)
            
            if cached_start <= start_year and cached_end >= end_year:
                dates = cache_data.get('trading_dates', [])
                # 过滤指定年份范围的日期
                filtered_dates = [
                    d for d in dates 
                    if start_year <= int(d[:4]) <= end_year
                ]
                return filtered_dates
                
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            
        return []
    
    def _save_to_cache(self, trading_dates: List[str], start_year: int, end_year: int):
        """保存交易日历到缓存"""
        try:
            cache_data = {
                'start_year': start_year,
                'end_year': end_year,
                'trading_dates': trading_dates,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"交易日历已缓存到 {self.cache_file}")
            
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _generate_fallback_calendar(self, start_year: int, end_year: int) -> List[str]:
        """生成备用交易日历（工作日）"""
        logger.warning("使用备用交易日历（工作日）")
        
        trading_dates = []
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        current_date = start_date
        while current_date <= end_date:
            # 排除周末
            if current_date.weekday() < 5:  # 0-4 为周一到周五
                trading_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        return trading_dates
    
    def is_trading_day(self, date_str: Union[str, date, datetime]) -> bool:
        """判断是否为交易日
        
        Args:
            date_str: 日期
            
        Returns:
            是否为交易日
        """
        if isinstance(date_str, (date, datetime)):
            date_str = date_str.strftime('%Y-%m-%d')
        
        # 懒加载交易日历
        if self._trading_dates_cache is None:
            year = int(date_str[:4])
            self._trading_dates_cache = set(self.fetch_trading_dates(year - 1, year + 1))
        
        return date_str in self._trading_dates_cache
    
    def get_next_trading_day(self, date_str: Union[str, date, datetime]) -> str:
        """获取下一个交易日
        
        Args:
            date_str: 当前日期
            
        Returns:
            下一个交易日，格式为 'YYYY-MM-DD'
        """
        if isinstance(date_str, (date, datetime)):
            current_date = date_str
        else:
            current_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # 获取交易日历
        year = current_date.year
        trading_dates = self.fetch_trading_dates(year, year + 1)
        trading_dates_set = set(trading_dates)
        
        # 查找下一个交易日
        next_date = current_date + timedelta(days=1)
        max_attempts = 10  # 最多查找10天
        attempts = 0
        
        while attempts < max_attempts:
            next_date_str = next_date.strftime('%Y-%m-%d')
            if next_date_str in trading_dates_set:
                return next_date_str
            next_date += timedelta(days=1)
            attempts += 1
        
        # 如果找不到，返回当前日期后的第一个工作日
        next_date = current_date + timedelta(days=1)
        while next_date.weekday() >= 5:  # 跳过周末
            next_date += timedelta(days=1)
        
        return next_date.strftime('%Y-%m-%d')
    
    def get_prev_trading_day(self, date_str: Union[str, date, datetime]) -> str:
        """获取前一个交易日
        
        Args:
            date_str: 当前日期
            
        Returns:
            前一个交易日，格式为 'YYYY-MM-DD'
        """
        if isinstance(date_str, (date, datetime)):
            current_date = date_str
        else:
            current_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # 获取交易日历
        year = current_date.year
        trading_dates = self.fetch_trading_dates(year - 1, year)
        trading_dates_set = set(trading_dates)
        
        # 查找前一个交易日
        prev_date = current_date - timedelta(days=1)
        max_attempts = 10  # 最多查找10天
        attempts = 0
        
        while attempts < max_attempts:
            prev_date_str = prev_date.strftime('%Y-%m-%d')
            if prev_date_str in trading_dates_set:
                return prev_date_str
            prev_date -= timedelta(days=1)
            attempts += 1
        
        # 如果找不到，返回当前日期前的第一个工作日
        prev_date = current_date - timedelta(days=1)
        while prev_date.weekday() >= 5:  # 跳过周末
            prev_date -= timedelta(days=1)
        
        return prev_date.strftime('%Y-%m-%d')
    
    def get_trading_days_between(self, start_date: Union[str, date, datetime], 
                                end_date: Union[str, date, datetime]) -> List[str]:
        """获取两个日期之间的交易日
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日列表
        """
        if isinstance(start_date, (date, datetime)):
            start_str = start_date.strftime('%Y-%m-%d')
        else:
            start_str = start_date
            
        if isinstance(end_date, (date, datetime)):
            end_str = end_date.strftime('%Y-%m-%d')
        else:
            end_str = end_date
        
        start_year = int(start_str[:4])
        end_year = int(end_str[:4])
        
        # 获取覆盖时间范围的交易日历
        all_trading_dates = self.fetch_trading_dates(start_year, end_year)
        
        # 过滤指定范围的交易日
        filtered_dates = [
            d for d in all_trading_dates 
            if start_str <= d <= end_str
        ]
        
        return filtered_dates
    
    def clear_cache(self):
        """清除缓存"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info("交易日历缓存已清除")
            self._trading_dates_cache = None
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")


# 全局实例
_calendar_instance = None


def get_trading_calendar() -> TradingCalendar:
    """获取交易日历单例"""
    global _calendar_instance
    if _calendar_instance is None:
        _calendar_instance = TradingCalendar()
    return _calendar_instance


# 便捷函数
def is_trading_day(date_str: Union[str, date, datetime]) -> bool:
    """判断是否为交易日"""
    return get_trading_calendar().is_trading_day(date_str)


def get_next_trading_day(date_str: Union[str, date, datetime]) -> str:
    """获取下一个交易日"""
    return get_trading_calendar().get_next_trading_day(date_str)


def get_prev_trading_day(date_str: Union[str, date, datetime]) -> str:
    """获取前一个交易日"""
    return get_trading_calendar().get_prev_trading_day(date_str)


def get_trading_days_between(start_date: Union[str, date, datetime], 
                           end_date: Union[str, date, datetime]) -> List[str]:
    """获取两个日期之间的交易日"""
    return get_trading_calendar().get_trading_days_between(start_date, end_date)