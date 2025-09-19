# AkShare数据连接器
# 实现AkShare数据源的统一接入

import akshare as ak
import pandas as pd
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import logging

from .base_connector import (
    BaseConnector, DataSourceConfig, DataPoint, DataBatch,
    DataType, DataFrequency, DataSourceStatus, ConnectorFactory
)

class AkShareConnector(BaseConnector):
    """AkShare数据连接器"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.logger = logging.getLogger("AkShareConnector")
        
        # AkShare支持的数据类型
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
            DataFrequency.DAILY: 'daily',
            DataFrequency.WEEKLY: 'weekly',
            DataFrequency.MONTHLY: 'monthly',
            DataFrequency.MINUTE_1: '1',
            DataFrequency.MINUTE_5: '5',
            DataFrequency.MINUTE_15: '15',
            DataFrequency.MINUTE_30: '30',
            DataFrequency.HOUR_1: '60',
        }
    
    async def connect(self) -> bool:
        """连接AkShare（无需认证）"""
        try:
            # AkShare不需要显式连接，测试一个简单的API调用
            await self._run_in_executor(ak.stock_zh_a_spot_em)
            self.status = DataSourceStatus.ACTIVE
            self.logger.info("AkShare connector connected successfully")
            return True
        except Exception as e:
            self.status = DataSourceStatus.ERROR
            self.logger.error(f"Failed to connect to AkShare: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """断开连接（AkShare无需断开）"""
        self.status = DataSourceStatus.INACTIVE
        return True
    
    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            # 测试获取股票列表
            await self._run_in_executor(ak.stock_zh_a_spot_em)
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
            # 转换股票代码格式
            ak_symbol = self._convert_symbol(symbol)
            period = self._frequency_map.get(frequency, 'daily')
            
            # 根据频率选择不同的API
            if frequency == DataFrequency.DAILY:
                df = await self._run_in_executor(
                    ak.stock_zh_a_hist,
                    symbol=ak_symbol,
                    period=period,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"  # 前复权
                )
            else:
                # 分钟级数据
                df = await self._run_in_executor(
                    ak.stock_zh_a_hist_min_em,
                    symbol=ak_symbol,
                    period=period,
                    start_date=start_date.strftime('%Y-%m-%d %H:%M:%S'),
                    end_date=end_date.strftime('%Y-%m-%d %H:%M:%S'),
                    adjust="qfq"
                )
            
            # 转换为标准格式
            data_points = self._convert_stock_data(df, symbol)
            
            return DataBatch(
                data_points=data_points,
                batch_id=f"akshare_stock_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source="akshare",
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
            ak_symbol = self._convert_symbol(symbol)
            
            # 根据报表类型选择API
            if report_type == "balance_sheet":
                df = await self._run_in_executor(
                    ak.stock_balance_sheet_by_report_em,
                    symbol=ak_symbol
                )
            elif report_type == "income_statement":
                df = await self._run_in_executor(
                    ak.stock_profit_sheet_by_report_em,
                    symbol=ak_symbol
                )
            elif report_type == "cash_flow":
                df = await self._run_in_executor(
                    ak.stock_cash_flow_sheet_by_report_em,
                    symbol=ak_symbol
                )
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            # 转换为标准格式
            data_points = self._convert_financial_data(df, symbol, report_type)
            
            return DataBatch(
                data_points=data_points,
                batch_id=f"akshare_financial_{symbol}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source="akshare",
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
            # 获取财经新闻
            df = await self._run_in_executor(
                ak.stock_news_em,
                symbol="全部"
            )
            
            # 限制数量
            if len(df) > limit:
                df = df.head(limit)
            
            # 转换为标准格式
            data_points = self._convert_news_data(df, symbols, keywords)
            
            return DataBatch(
                data_points=data_points,
                batch_id=f"akshare_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source="akshare",
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
            # 获取A股列表
            df = ak.stock_zh_a_spot_em()
            return df['代码'].tolist()
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
                # 获取指数历史数据
                df = await self._run_in_executor(
                    ak.stock_zh_index_daily,
                    symbol=symbol
                )
                
                # 日期过滤
                if start_date:
                    df = df[df['date'] >= start_date.strftime('%Y-%m-%d')]
                if end_date:
                    df = df[df['date'] <= end_date.strftime('%Y-%m-%d')]
                
                # 转换数据
                for _, row in df.iterrows():
                    point = DataPoint(
                        timestamp=pd.to_datetime(row['date']),
                        symbol=symbol,
                        data_type=DataType.MARKET_INDEX,
                        values={
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']),
                        },
                        source="akshare"
                    )
                    data_points.append(point)
                    
            except Exception as e:
                self.logger.warning(f"Failed to get index data for {symbol}: {e}")
        
        return DataBatch(
            data_points=data_points,
            batch_id=f"akshare_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source="akshare",
            data_type=DataType.MARKET_INDEX
        )
    
    async def _get_stock_info(self, symbols: List[str]) -> DataBatch:
        """获取股票基本信息"""
        try:
            # 获取股票基本信息
            df = await self._run_in_executor(ak.stock_zh_a_spot_em)
            
            # 过滤指定股票
            if symbols:
                df = df[df['代码'].isin([self._convert_symbol(s) for s in symbols])]
            
            data_points = []
            for _, row in df.iterrows():
                point = DataPoint(
                    timestamp=datetime.now(),
                    symbol=row['代码'],
                    data_type=DataType.STOCK_INFO,
                    values={
                        'name': row['名称'],
                        'price': float(row['最新价']),
                        'change_pct': float(row['涨跌幅']),
                        'change_amount': float(row['涨跌额']),
                        'volume': float(row['成交量']),
                        'turnover': float(row['成交额']),
                        'amplitude': float(row['振幅']),
                        'high': float(row['最高']),
                        'low': float(row['最低']),
                        'open': float(row['今开']),
                        'prev_close': float(row['昨收']),
                    },
                    source="akshare"
                )
                data_points.append(point)
            
            return DataBatch(
                data_points=data_points,
                batch_id=f"akshare_stock_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source="akshare",
                data_type=DataType.STOCK_INFO
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get stock info: {e}")
            raise
    
    async def _get_sector_data(self, sectors: List[str]) -> DataBatch:
        """获取行业数据"""
        try:
            # 获取行业数据
            df = await self._run_in_executor(ak.stock_board_industry_name_em)
            
            data_points = []
            for _, row in df.iterrows():
                point = DataPoint(
                    timestamp=datetime.now(),
                    symbol=row['板块名称'],
                    data_type=DataType.SECTOR_DATA,
                    values={
                        'name': row['板块名称'],
                        'code': row['板块代码'],
                        'stock_count': int(row['公司家数']),
                        'avg_price': float(row['平均价格']),
                        'change_pct': float(row['涨跌幅']),
                        'turnover': float(row['总成交量']),
                        'market_cap': float(row['总市值']),
                    },
                    source="akshare"
                )
                data_points.append(point)
            
            return DataBatch(
                data_points=data_points,
                batch_id=f"akshare_sector_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source="akshare",
                data_type=DataType.SECTOR_DATA
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get sector data: {e}")
            raise
    
    def _convert_symbol(self, symbol: str) -> str:
        """转换股票代码格式"""
        # 移除前缀，AkShare使用纯数字代码
        if '.' in symbol:
            return symbol.split('.')[0]
        return symbol
    
    def _convert_stock_data(self, df: pd.DataFrame, symbol: str) -> List[DataPoint]:
        """转换股票数据格式"""
        data_points = []
        
        for _, row in df.iterrows():
            # 处理不同的列名格式
            date_col = '日期' if '日期' in df.columns else 'date'
            open_col = '开盘' if '开盘' in df.columns else 'open'
            high_col = '最高' if '最高' in df.columns else 'high'
            low_col = '最低' if '最低' in df.columns else 'low'
            close_col = '收盘' if '收盘' in df.columns else 'close'
            volume_col = '成交量' if '成交量' in df.columns else 'volume'
            
            point = DataPoint(
                timestamp=pd.to_datetime(row[date_col]),
                symbol=symbol,
                data_type=DataType.STOCK_PRICE,
                values={
                    'open': float(row[open_col]),
                    'high': float(row[high_col]),
                    'low': float(row[low_col]),
                    'close': float(row[close_col]),
                    'volume': float(row[volume_col]),
                    'amount': float(row.get('成交额', row.get('amount', 0))),
                },
                source="akshare"
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
                if col not in ['股票代码', '股票简称', '公告日期', '报告期']:
                    try:
                        values[col] = float(row[col]) if pd.notna(row[col]) else 0.0
                    except (ValueError, TypeError):
                        values[col] = str(row[col])
            
            point = DataPoint(
                timestamp=pd.to_datetime(row.get('报告期', datetime.now())),
                symbol=symbol,
                data_type=DataType.FINANCIAL_REPORT,
                values=values,
                metadata={
                    'report_type': report_type,
                    'announce_date': row.get('公告日期'),
                },
                source="akshare"
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
                content = str(row.get('新闻内容', '')) + str(row.get('新闻标题', ''))
                if not any(symbol in content for symbol in symbols):
                    continue
            
            # 过滤关键词
            if keywords:
                content = str(row.get('新闻内容', '')) + str(row.get('新闻标题', ''))
                if not any(keyword in content for keyword in keywords):
                    continue
            
            point = DataPoint(
                timestamp=pd.to_datetime(row.get('发布时间', datetime.now())),
                symbol="",  # 新闻可能涉及多个股票
                data_type=DataType.NEWS,
                values={
                    'title': row.get('新闻标题', ''),
                    'content': row.get('新闻内容', ''),
                    'source': row.get('新闻来源', ''),
                    'url': row.get('新闻链接', ''),
                },
                source="akshare"
            )
            data_points.append(point)
        
        return data_points
    
    async def _run_in_executor(self, func, *args, **kwargs):
        """在线程池中运行同步函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

# 注册连接器
ConnectorFactory.register("akshare", AkShareConnector)