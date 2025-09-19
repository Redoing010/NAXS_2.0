import pandas as pd
import akshare as ak
from fastapi import APIRouter, Query, Depends
from ..config import settings
from ..cache import get_cache, set_cache
from ..utils import normalize_columns, to_numeric, SPOT_COLS_MAP
from ..schemas import SpotPage
from ..deps import pagination


router = APIRouter(prefix=f"{settings.API_PREFIX}", tags=["spot"])


@router.get("/spot", response_model=SpotPage)
def get_spot(
    sort_by: str = Query("amount"),
    descending: bool = Query(True),
    pg=Depends(pagination),
):
    cache_key = "spot_all"
    df = get_cache(cache_key, settings.TTL_SPOT_ALL)
    if df is None:
        raw = ak.stock_zh_a_spot()
        df = normalize_columns(raw, SPOT_COLS_MAP)
        df = to_numeric(
            df,
            [
                "price",
                "pct_chg",
                "chg",
                "vol",
                "amount",
                "amplitude",
                "high",
                "low",
                "open",
                "prev_close",
                "pe_ttm",
                "turnover_rate",
                "volume_ratio",
                "market_cap",
                "float_market_cap",
            ],
        )
        set_cache(cache_key, df)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending)

    start = (pg["page"] - 1) * pg["page_size"]
    end = start + pg["page_size"]
    page_df = df.iloc[start:end]

    return {
        "total": int(len(df)),
        "page": pg["page"],
        "page_size": pg["page_size"],
        "rows": page_df.to_dict(orient="records"),
    }





