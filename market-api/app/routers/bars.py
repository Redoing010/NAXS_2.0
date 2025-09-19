from fastapi import APIRouter, Query, HTTPException
from ..config import settings
from ..schemas import MinuteRow
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.data.parquet_store import read_code
from modules.data.utils import normalize_symbol

router = APIRouter(prefix=f"{settings.API_PREFIX}", tags=["bars"])


@router.get("/bars")
def get_bars(
    code: str = Query(..., description="如 000831.SZ"),
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    freq: str = Query("D", description="暂仅 D"),
):
    """获取股票K线数据
    
    从Parquet存储中读取权威数据，支持日频数据查询。
    这是系统的权威数据接口，供前端和回测使用。
    """
    try:
        # 标准化股票代码
        std_code = normalize_symbol(code)
        
        # 使用数据层模块读取数据
        df = read_code(std_code, start, end, root=settings.PARQUET_ROOT, freq=freq)
        
        if df.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for {std_code} in range {start} to {end}"
            )
        
        # 转换为API响应格式
        df_reset = df.reset_index()
        
        # 确保日期列名正确
        if df.index.name:
            df_reset = df_reset.rename(columns={df.index.name: "datetime"})
        elif "date" in df_reset.columns:
            df_reset = df_reset.rename(columns={"date": "datetime"})
        
        # 转换为字典列表
        rows = df_reset.to_dict(orient="records")
        
        return {
            "code": std_code,
            "freq": freq.upper(),
            "start_date": start,
            "end_date": end,
            "total_records": len(rows),
            "rows": rows
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch data for {code}: {str(e)}"
        )





