from typing import List, Optional
from pydantic import BaseModel


class SpotRow(BaseModel):
    symbol: str
    name: str
    price: Optional[float] = None
    pct_chg: Optional[float] = None
    chg: Optional[float] = None
    vol: Optional[float] = None
    amount: Optional[float] = None
    amplitude: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    prev_close: Optional[float] = None
    pe_ttm: Optional[float] = None
    turnover_rate: Optional[float] = None
    volume_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    float_market_cap: Optional[float] = None
    list_date: Optional[str] = None


class SpotPage(BaseModel):
    total: int
    page: int
    page_size: int
    rows: List[SpotRow]


class MinuteRow(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


class MinuteResp(BaseModel):
    symbol: str
    rows: List[MinuteRow]





