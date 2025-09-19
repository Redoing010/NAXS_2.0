# TuShare数据连接器
# 实现TuShare Pro数据源的统一接入

import tushare as ts
import pandas as pd
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import logging
import time

from .base_connector import (
    BaseConnector, DataSourceConfig, DataPoint, DataBatch,
    DataType, DataFrequency, DataSourceStatus, ConnectorFactory
)

class TuShareConnector(BaseConnector):
    """TuShare Pro数据连接器"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.logger = logging.getLogger("TuShareConnector")
        self.pro_api = None
        
        # TuShare支持的数据类型
        self._supported_data_types = [
            DataType.STOCK_PRICE,
            DataType.STOCK_INFO,
            DataType.FINANCIAL_REPORT,
            DataType.NEWS,
            DataType.MARKET_INDEX,
            DataType.SECTOR_DATA,
            DataType.MACRO_DATA,
        ]
        
        # 频率映射
        self._frequency_map = {
            DataFrequency.DAILY: 'D',
            DataFrequency.WEEKLY: 'W',
            DataFrequency.MONTHLY: 'M',
            DataFrequency.MINUTE_1: '1min',
            DataFrequency.MINUTE_5: '5min',
            DataFrequency.MINUTE_15: '15min',
            DataFrequency.MINUTE_30: '30min',
            DataFrequency.HOUR_1: '60min',
        }
        
        # API调用限制
        self._last_call_time = 0
        self._call_interval = 0.2  # 200ms间隔，避免频率限制
    
    async def connect(self) -> bool:
        """连接TuShare Pro"""
        try:
            if not self.config.api_key:
                raise ValueError("TuShare API key is required")
            
            # 设置token
            ts.set_token(self.config.api_key)
            self.pro_api = ts.pro_api()
            
            # 测试连接
            await self._run_in_executor(self.pro_api.stock_basic, exchange='', list_status='L', fields='ts_code,symbol,name')
            
            self.status = DataSourceStatus.ACTIVE
            self.logger.info("TuShare connector connected successfully")
            return True
            
        except Exception as e:
            self.status = DataSourceStatus.ERROR
            self.logger.error(f"Failed to connect to TuShare: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """断开连接"""
        self.pro_api = None
        self.status = DataSourceStatus.INACTIVE
        return True
    
    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            if not self.pro_api:
                return False
            
            # 测试获取股票列表
            await self._run_in_executor(
                self.pro_api.stock_basic,
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name'
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    async def get_stock_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY,
        fields: Optional[List[str]] = None
    ) -> DataBatch:
        """获取股票数据"""
        try:
            ts_code = self._convert_symbol(symbol)
            freq = self._frequency_map.get(frequency, 'D')
            
            # 默认字段
            if not fields:
                fields = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
            
            # 根据频率选择API
            if frequency == DataFrequency.DAILY:
                df = await self._api_call(
                    self.pro_api.daily,
                    ts_code=ts_code,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    fields=','.join(fields)
                )
            elif frequency in [DataFrequency.MINUTE_1, DataFrequency.MINUTE_5, 
                             DataFrequency.MINUTE_15, DataFrequency.MINUTE_30, DataFrequency.HOUR_1]:
                # 分钟级数据需要逐日获取
                df = await self._get_intraday_data(ts_code, start_date, end_date, freq)
            else:
                # 周线、月线数据
                df = await self._api_call(
                    self.pro_api.daily,
                    ts_code=ts_code,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    fields=','.join(fields)
                )
                # 这里可以添加周线、月线的重采样逻辑
            
            # 转换为标准格式
            data_points = self._convert_stock_data(df, symbol)
            
            return DataBatch(
                data_points=data_points,
                batch_id=f"tushare_stock_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source="tushare",
                data_type=DataType.STOCK_PRICE,
                metadata={
                    'symbol': symbol,
                    'frequency': frequency.value,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get stock data for {symbol}: {e}")
            raise
    
    async def get_market_data(
        self,
        symbols: List[str],
        data_type: DataType,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        **kwargs
    ) -> DataBatch:
        """获取市场数据"""
        try:
            if data_type == DataType.MARKET_INDEX:
                return await self._get_index_data(symbols, start_date, end_date)
            elif data_type == DataType.STOCK_INFO:
                return await self._get_stock_info(symbols)
            elif data_type == DataType.SECTOR_DATA:
                return await self._get_sector_data(symbols)
            elif data_type == DataType.MACRO_DATA:
                return await self._get_macro_data(kwargs.get('indicators', []))
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            raise
    
    async def get_financial_data(
        self,
        symbol: str,
        report_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> DataBatch:
        """获取财务数据"""
        try:
            ts_code = self._convert_symbol(symbol)
            
            # 根据报表类型选择API
            if report_type == "income":
                df = await self._api_call(
                    self.pro_api.income,
                    ts_code=ts_code,
                    start_date=start_date.strftime('%Y%m%d') if start_date else None,
                    end_date=end_date.strftime('%Y%m%d') if end_date else None
                )
            elif report_type == "balancesheet":
                df = await self._api_call(
                    self.pro_api.balancesheet,
                    ts_code=ts_code,
                    start_date=start_date.strftime('%Y%m%d') if start_date else None,
                    end_date=end_date.strftime('%Y%m%d') if end_date else None
                )
            elif report_type == "cashflow":
                df = await self._api_call(
                    self.pro_api.cashflow,
                    ts_code=ts_code,
                    start_date=start_date.strftime('%Y%m%d') if start_date else None,
                    end_date=end_date.strftime('%Y%m%d') if end_date else None
                )
            elif report_type == "fina_indicator":
                df = await self._api_call(
                    self.pro_api.fina_indicator,
                    ts_code=ts_code,
                    start_date=start_date.strftime('%Y%m%d') if start_date else None,
                    end_date=end_date.strftime('%Y%m%d') if end_date else None
                )
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            # 转换为标准格式
            data_points = self._convert_financial_data(df, symbol, report_type)
            
            return DataBatch(
                data_points=data_points,
                batch_id=f"tushare_financial_{symbol}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source="tushare",
                data_type=DataType.FINANCIAL_REPORT,
                metadata={
                    'symbol': symbol,
                    'report_type': report_type,
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get financial data for {symbol}: {e}")
            raise
    
    async def get_news_data(
        self,
        symbols: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> DataBatch:
        """获取新闻数据"""
        try:
            # TuShare的新闻数据API
            df = await self._api_call(
                self.pro_api.news,
                start_date=start_date.strftime('%Y%m%d') if start_date else None,
                end_date=end_date.strftime('%Y%m%d') if end_date else None,
                limit=limit
            )
            
            # 转换为标准格式
            data_points = self._convert_news_data(df, symbols, keywords)
            
            return DataBatch(
                data_points=data_points,
                batch_id=f"tushare_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source="tushare",
                data_type=DataType.NEWS,
                metadata={
                    'symbols': symbols,
                    'keywords': keywords,
                    'limit': limit,
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get news data: {e}")
            raise
    
    def get_supported_symbols(self) -> List[str]:
        """获取支持的股票代码列表"""
        try:
            if not self.pro_api:
                return []
            
            # 获取股票基本信息
            df = self.pro_api.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')
            return df['ts_code'].tolist()
            
        except Exception as e:
            self.logger.error(f"Failed to get supported symbols: {e}")
            return []
    
    def get_supported_data_types(self) -> List[DataType]:
        """获取支持的数据类型"""
        return self._supported_data_types
    
    async def _get_index_data(
        self,
        symbols: List[str],
        start_date: Optional[date],
        end_date: Optional[date]
    ) -> DataBatch:
        """获取指数数据"""
        data_points = []
        
        for symbol in symbols:
            try:
                ts_code = self._convert_index_symbol(symbol)
                
                df = await self._api_call(
                    self.pro_api.index_daily,
                    ts_code=ts_code,
                    start_date=start_date.strftime('%Y%m%d') if start_date else None,
                    end_date=end_date.strftime('%Y%m%d') if end_date else None
                )
                
                # 转换数据
                for _, row in df.iterrows():
                    point = DataPoint(
                        timestamp=pd.to_datetime(row['trade_date']),
                        symbol=symbol,
                        data_type=DataType.MARKET_INDEX,
                        values={
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['vol']),
                            'amount': float(row['amount']),
                        },
                        source="tushare"
                    )
                    data_points.append(point)
                    
            except Exception as e:
                self.logger.warning(f"Failed to get index data for {symbol}: {e}")
        
        return DataBatch(
            data_points=data_points,
            batch_id=f"tushare_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source="tushare",
            data_type=DataType.MARKET_INDEX
        )
    
    async def _get_stock_info(self, symbols: List[str]) -> DataBatch:
        """获取股票基本信息"""
        try:
            # 获取股票基本信息
            df = await self._api_call(
                self.pro_api.stock_basic,
                exchange='',
                list_status='L'
            )
            
            # 过滤指定股票
            if symbols:
                ts_codes = [self._convert_symbol(s) for s in symbols]
                df = df[df['ts_code'].isin(ts_codes)]
            
            data_points = []
            for _, row in df.iterrows():
                point = DataPoint(
                    timestamp=datetime.now(),
                    symbol=row['ts_code'],
                    data_type=DataType.STOCK_INFO,
                    values={
                        'symbol': row['symbol'],
                        'name': row['name'],
                        'area': row.get('area', ''),
                        'industry': row.get('industry', ''),
                        'market': row.get('market', ''),
                        'list_date': row.get('list_date', ''),
                    },
                    source="tushare"
                )
                data_points.append(point)
            
            return DataBatch(
                data_points=data_points,
                batch_id=f"tushare_stock_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source="tushare",
                data_type=DataType.STOCK_INFO
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get stock info: {e}")
            raise
    
    async def _get_sector_data(self, sectors: List[str]) -> DataBatch:
        """获取行业数据"""
        try:
            # 获取行业分类
            df = await self._api_call(
                self.pro_api.index_classify,
                level='L2',
                src='SW2021'
            )
            
            data_points = []
            for _, row in df.iterrows():
                point = DataPoint(
                    timestamp=datetime.now(),
                    symbol=row['index_code'],
                    data_type=DataType.SECTOR_DATA,
                    values={
                        'name': row['industry_name'],
                        'code': row['index_code'],
                        'level': row['level'],
                        'src': row['src'],
                    },
                    source="tushare"
                )
                data_points.append(point)
            
            return DataBatch(
                data_points=data_points,
                batch_id=f"tushare_sector_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source="tushare",
                data_type=DataType.SECTOR_DATA
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get sector data: {e}")
            raise
    
    async def _get_macro_data(self, indicators: List[str]) -> DataBatch:
        """获取宏观数据"""
        data_points = []
        
        for indicator in indicators:
            try:
                # 根据指标类型选择API
                if indicator == 'gdp':
                    df = await self._api_call(self.pro_api.cn_gdp)
                elif indicator == 'cpi':
                    df = await self._api_call(self.pro_api.cn_cpi)
                elif indicator == 'ppi':
                    df = await self._api_call(self.pro_api.cn_ppi)
                elif indicator == 'm2':
                    df = await self._api_call(self.pro_api.cn_m)
                else:
                    continue
                
                # 转换数据
                for _, row in df.iterrows():
                    point = DataPoint(
                        timestamp=pd.to_datetime(row.get('month', row.get('quarter', datetime.now()))),
                        symbol=indicator,
                        data_type=DataType.MACRO_DATA,
                        values=dict(row),
                        source="tushare"
                    )
                    data_points.append(point)
                    
            except Exception as e:
                self.logger.warning(f"Failed to get macro data for {indicator}: {e}")
        
        return DataBatch(
            data_points=data_points,
            batch_id=f"tushare_macro_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source="tushare",
            data_type=DataType.MACRO_DATA
        )
    
    async def _get_intraday_data(
        self,
        ts_code: str,
        start_date: date,
        end_date: date,
        freq: str
    ) -> pd.DataFrame:
        """获取分钟级数据"""
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                df = await self._api_call(
                    self.pro_api.stk_mins,
                    ts_code=ts_code,
                    trade_date=current_date.strftime('%Y%m%d'),
                    freq=freq
                )
                if not df.empty:
                    all_data.append(df)
                    
            except Exception as e:
                self.logger.warning(f"Failed to get intraday data for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    def _convert_symbol(self, symbol: str) -> str:
        """转换股票代码格式为TuShare格式"""
        if '.' in symbol:
            code, exchange = symbol.split('.')
            if exchange.upper() == 'SH':
                return f"{code}.SH"
            elif exchange.upper() == 'SZ':
                return f"{code}.SZ"
        
        # 根据代码判断交易所
        if symbol.startswith('6'):
            return f"{symbol}.SH"
        elif symbol.startswith(('0', '3')):
            return f"{symbol}.SZ"
        
        return symbol
    
    def _convert_index_symbol(self, symbol: str) -> str:
        """转换指数代码格式"""
        index_map = {
            '000001': '000001.SH',  # 上证指数
            '399001': '399001.SZ',  # 深证成指
            '399006': '399006.SZ',  # 创业板指
            '000300': '000300.SH',  # 沪深300
            '000905': '000905.SH',  # 中证500
        }
        return index_map.get(symbol, symbol)
    
    def _convert_stock_data(self, df: pd.DataFrame, symbol: str) -> List[DataPoint]:
        """转换股票数据格式"""
        data_points = []
        
        for _, row in df.iterrows():
            point = DataPoint(
                timestamp=pd.to_datetime(row['trade_date']),
                symbol=symbol,
                data_type=DataType.STOCK_PRICE,
                values={
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['vol']),
                    'amount': float(row['amount']),
                },
                source="tushare"
            )
            data_points.append(point)
        
        return data_points
    
    def _convert_financial_data(self, df: pd.DataFrame, symbol: str, report_type: str) -> List[DataPoint]:
        """转换财务数据格式"""
        data_points = []
        
        for _, row in df.iterrows():
            # 提取财务指标
            values = {}
            for col in df.columns:
                if col not in ['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type']:
                    try:
                        values[col] = float(row[col]) if pd.notna(row[col]) else 0.0
                    except (ValueError, TypeError):
                        values[col] = str(row[col])
            
            point = DataPoint(
                timestamp=pd.to_datetime(row['end_date']),
                symbol=symbol,
                data_type=DataType.FINANCIAL_REPORT,
                values=values,
                metadata={
                    'report_type': report_type,
                    'ann_date': row.get('ann_date'),
                    'f_ann_date': row.get('f_ann_date'),
                },
                source="tushare"
            )
            data_points.append(point)
        
        return data_points
    
    def _convert_news_data(
        self,
        df: pd.DataFrame,
        symbols: Optional[List[str]],
        keywords: Optional[List[str]]
    ) -> List[DataPoint]:
        """转换新闻数据格式"""
        data_points = []
        
        for _, row in df.iterrows():
            # 过滤相关股票
            if symbols:
                content = str(row.get('content', '')) + str(row.get('title', ''))
                if not any(symbol in content for symbol in symbols):
                    continue
            
            # 过滤关键词
            if keywords:
                content = str(row.get('content', '')) + str(row.get('title', ''))
                if not any(keyword in content for keyword in keywords):
                    continue
            
            point = DataPoint(
                timestamp=pd.to_datetime(row.get('datetime', datetime.now())),
                symbol="",  # 新闻可能涉及多个股票
                data_type=DataType.NEWS,
                values={
                    'title': row.get('title', ''),
                    'content': row.get('content', ''),
                    'source': row.get('source', ''),
                    'url': row.get('url', ''),
                },
                source="tushare"
            )
            data_points.append(point)
        
        return data_points
    
    async def _api_call(self, func, **kwargs):
        """API调用包装，处理频率限制"""
        # 频率限制
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < self._call_interval:
            await asyncio.sleep(self._call_interval - time_since_last_call)
        
        # 执行API调用
        result = await self._run_in_executor(func, **kwargs)
        self._last_call_time = time.time()
        
        return result
    
    async def _run_in_executor(self, func, *args, **kwargs):
        """在线程池中运行同步函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

# 注册连接器
ConnectorFactory.register("tushare", TuShareConnector)