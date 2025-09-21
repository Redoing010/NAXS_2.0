try:
    import akshare as ak
except ImportError:  # pragma: no cover - optional dependency
    ak = None  # type: ignore
from fastapi import APIRouter, Query, HTTPException
from ..config import settings
from ..cache import get_cache, set_cache
from ..utils import normalize_columns, MINUTE_COLS_MAP
from ..schemas import MinuteResp


router = APIRouter(prefix=f"{settings.API_PREFIX}", tags=["minute"])


@router.get("/minute", response_model=MinuteResp)
def get_minute(
    symbol: str = Query(..., description="6位代码，如 000001"),
    period: str = Query("1", description="1/5/15/30/60 分钟"),
    adjust: str = Query("qfq", description="qfq/hfq/None"),
    limit: int = Query(240, ge=10, le=2000),
):
    cache_key = f"min_{symbol}_{period}_{adjust}"
    df = get_cache(cache_key, settings.TTL_MINUTE)

    if ak is None:
        raise HTTPException(status_code=503, detail="akshare dependency is not installed")

    if df is None:
        df = ak.stock_zh_a_hist_min_em(symbol=symbol, period=period, adjust=adjust)
        df = normalize_columns(df, MINUTE_COLS_MAP).tail(limit)
        set_cache(cache_key, df)

    return {"symbol": symbol, "rows": df.to_dict(orient="records")}





