import pandas as pd
from typing import Dict, Iterable


SPOT_COLS_MAP: Dict[str, str] = {
    "代码": "symbol",
    "名称": "name",
    "最新价": "price",
    "涨跌幅": "pct_chg",
    "涨跌额": "chg",
    "成交量": "vol",
    "成交额": "amount",
    "振幅": "amplitude",
    "最高": "high",
    "最低": "low",
    "今开": "open",
    "昨收": "prev_close",
    "市盈率-动态": "pe_ttm",
    "换手率": "turnover_rate",
    "量比": "volume_ratio",
    "总市值": "market_cap",
    "流通市值": "float_market_cap",
    "上市时间": "list_date",
}


MINUTE_COLS_MAP: Dict[str, str] = {
    "时间": "datetime",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def normalize_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns={c: mapping.get(c, c) for c in df.columns})


def to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df





