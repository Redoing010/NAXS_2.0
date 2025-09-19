# 数据层工具函数

import re
from typing import Dict, Optional


def normalize_symbol(symbol: str, target: str = "standard") -> str:
    """标准化股票代码格式
    
    Args:
        symbol: 输入的股票代码
        target: 目标格式 "standard"/"akshare"/"tushare"
        
    Returns:
        标准化后的股票代码
    """
    # 移除空格和特殊字符
    symbol = symbol.strip().upper()
    
    # 提取数字部分和交易所部分
    if '.' in symbol:
        code, exchange = symbol.split('.', 1)
    else:
        # 如果没有交易所后缀，根据代码判断
        code = symbol
        if code.startswith(('000', '001', '002', '003', '300')):
            exchange = 'SZ'
        elif code.startswith(('600', '601', '603', '605', '688')):
            exchange = 'SH'
        else:
            exchange = 'SZ'  # 默认深交所
    
    # 确保代码是6位数字
    code = code.zfill(6)
    
    if target == "standard":
        return f"{code}.{exchange}"
    elif target == "akshare":
        return code  # AkShare只需要6位数字代码
    elif target == "tushare":
        return f"{code}.{exchange}"
    else:
        return f"{code}.{exchange}"


def akshare_to_standard_symbol(ak_symbol: str) -> str:
    """将AkShare格式代码转换为标准格式
    
    Args:
        ak_symbol: AkShare格式代码（6位数字）
        
    Returns:
        标准格式代码（如 000001.SZ）
    """
    code = ak_symbol.strip()
    
    # 判断交易所
    if code.startswith(('000', '001', '002', '003', '300')):
        return f"{code}.SZ"
    elif code.startswith(('600', '601', '603', '605', '688')):
        return f"{code}.SH"
    else:
        return f"{code}.SZ"  # 默认深交所


def map_stock_code(code: str) -> str:
    """映射股票代码到标准格式
    
    Args:
        code: 输入代码
        
    Returns:
        标准格式代码
    """
    return normalize_symbol(code, "standard")


def validate_date_format(date_str: str) -> bool:
    """验证日期格式是否为 YYYY-MM-DD
    
    Args:
        date_str: 日期字符串
        
    Returns:
        是否为有效格式
    """
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    return bool(re.match(pattern, date_str))


def get_exchange_from_symbol(symbol: str) -> str:
    """从股票代码获取交易所
    
    Args:
        symbol: 股票代码
        
    Returns:
        交易所代码 'SH'/'SZ'
    """
    if '.' in symbol:
        return symbol.split('.')[1]
    
    # 根据代码前缀判断
    code = symbol.strip()
    if code.startswith(('000', '001', '002', '003', '300')):
        return 'SZ'
    elif code.startswith(('600', '601', '603', '605', '688')):
        return 'SH'
    else:
        return 'SZ'


def format_parquet_path(root_path: str, symbol: str, freq: str = "D") -> str:
    """格式化Parquet文件路径
    
    Args:
        root_path: 根路径
        symbol: 股票代码
        freq: 频率
        
    Returns:
        完整的文件路径
    """
    import os
    
    # 标准化代码
    std_symbol = normalize_symbol(symbol)
    exchange = get_exchange_from_symbol(std_symbol)
    code = std_symbol.split('.')[0]
    
    # 构建路径: root/freq/exchange/code.parquet
    return os.path.join(root_path, freq, exchange, f"{code}.parquet")


def ensure_directory(file_path: str) -> None:
    """确保目录存在
    
    Args:
        file_path: 文件路径
    """
    import os
    
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# 数据字段映射
STANDARD_COLUMNS = {
    'OHLCV': ['open', 'high', 'low', 'close', 'volume'],
    'OHLCVA': ['open', 'high', 'low', 'close', 'volume', 'amount'],
    'EXTENDED': ['open', 'high', 'low', 'close', 'volume', 'amount', 
                'pct_chg', 'change', 'turnover', 'amplitude']
}


# AkShare字段映射
AKSHARE_COLUMN_MAPPING = {
    '日期': 'date',
    '时间': 'datetime', 
    '开盘': 'open',
    '收盘': 'close',
    '最高': 'high',
    '最低': 'low',
    '成交量': 'volume',
    '成交额': 'amount',
    '振幅': 'amplitude',
    '涨跌幅': 'pct_chg',
    '涨跌额': 'change',
    '换手率': 'turnover',
    '代码': 'symbol',
    '名称': 'name',
    '最新价': 'price',
    '昨收': 'prev_close'
}


def standardize_dataframe_columns(df, source: str = "akshare"):
    """标准化DataFrame列名
    
    Args:
        df: 输入DataFrame
        source: 数据源类型
        
    Returns:
        标准化后的DataFrame
    """
    if source == "akshare":
        mapping = AKSHARE_COLUMN_MAPPING
    else:
        mapping = {}
    
    # 重命名列
    df_renamed = df.rename(columns=mapping)
    
    return df_renamed