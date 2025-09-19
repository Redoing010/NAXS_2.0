import subprocess
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from ..config import settings
from ..deps import admin_auth
import logging
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.data.parquet_store import ParquetStore
from modules.data.dq import DataQualityChecker
from modules.data.qlib_writer import QlibWriter

logger = logging.getLogger(__name__)
router = APIRouter(prefix=f"{settings.API_PREFIX}/admin", tags=["admin"]) 


@router.post("/pull")
def pull_prices(
    codes: str = Query("600519.SH,000831.SZ", description="股票代码，逗号分隔"),
    start: str = Query(..., description="开始日期 YYYY-MM-DD"),
    end: str = Query(..., description="结束日期 YYYY-MM-DD"),
    freq: str = Query("D", description="数据频率"),
    force: bool = Query(False, description="强制重新拉取"),
    retry: int = Query(3, description="重试次数"),
    auth=Depends(admin_auth),
):
    """拉取股票价格数据
    
    执行数据拉取任务，将数据存储到Parquet格式。
    支持增量更新和强制重新拉取。
    """
    try:
        logger.info(f"开始拉取数据: codes={codes}, start={start}, end={end}")
        
        cmd = [
            "python",
            "ops/pull_prices.py",
            "--market", "stock",
            "--start", start,
            "--end", end,
            "--freq", freq,
            "--out", settings.PARQUET_ROOT,
            "--retry", str(retry),
        ]
        
        # 添加股票代码
        for code in codes.split(","):
            if code.strip():
                cmd.extend(["--codes", code.strip()])
        
        # 强制重新拉取
        if force:
            cmd.append("--force")
        
        # 执行命令
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=3600  # 1小时超时
        )
        
        response = {
            "ok": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": " ".join(cmd),
            "timestamp": datetime.now().isoformat()
        }
        
        if result.returncode != 0:
            logger.error(f"数据拉取失败: {result.stderr}")
            raise HTTPException(
                status_code=500, 
                detail=f"Data pull failed with return code {result.returncode}: {result.stderr}"
            )
        
        logger.info(f"数据拉取成功: {codes}")
        return response
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Data pull timeout")
    except Exception as e:
        logger.error(f"数据拉取异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dq")
def dq_report(
    codes: str = Query("", description="股票代码，逗号分隔，空则检查所有"),
    start: str = Query(..., description="开始日期 YYYY-MM-DD"),
    end: str = Query(..., description="结束日期 YYYY-MM-DD"),
    out_dir: str = Query("reports/dq/latest", description="报告输出目录"),
    format: str = Query("json", description="报告格式: json/html/markdown"),
    sample_size: int = Query(0, description="随机采样数量，0表示全部"),
    auth=Depends(admin_auth),
):
    """生成数据质量报告
    
    检查指定股票的数据质量，生成详细报告。
    支持多种输出格式和随机采样。
    """
    try:
        logger.info(f"开始生成数据质量报告: start={start}, end={end}")
        
        cmd = [
            "python",
            "ops/dq_report.py",
            "--start", start,
            "--end", end,
            "--freq", "D",
            "--parquet-root", settings.PARQUET_ROOT,
            "--out", out_dir,
            "--format", format,
        ]
        
        # 添加采样参数
        if sample_size > 0:
            cmd.extend(["--sample-size", str(sample_size)])
        
        # 添加股票代码
        if codes.strip():
            for code in codes.split(","):
                if code.strip():
                    cmd.extend(["--symbols", code.strip()])
        
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30分钟超时
        )
        
        response = {
            "ok": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output_dir": out_dir,
            "format": format,
            "timestamp": datetime.now().isoformat()
        }
        
        if result.returncode != 0:
            logger.error(f"数据质量报告生成失败: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"DQ report failed with return code {result.returncode}: {result.stderr}"
            )
        
        logger.info(f"数据质量报告生成成功: {out_dir}")
        return response
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="DQ report timeout")
    except Exception as e:
        logger.error(f"数据质量报告异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qlib/build")
def build_qlib(
    start: str = Query(..., description="开始日期 YYYY-MM-DD"),
    end: str = Query(..., description="结束日期 YYYY-MM-DD"),
    qlib_out: str = Query("data/qlib_cn_daily", description="Qlib输出目录"),
    update: bool = Query(False, description="增量更新模式"),
    validate: bool = Query(True, description="构建后验证"),
    backup: bool = Query(False, description="备份现有数据包"),
    min_days: int = Query(100, description="最少交易天数过滤"),
    auth=Depends(admin_auth),
):
    """构建Qlib数据包
    
    将Parquet数据转换为Qlib格式，支持增量更新和数据验证。
    """
    try:
        logger.info(f"开始构建Qlib数据包: start={start}, end={end}, update={update}")
        
        cmd = [
            "python",
            "ops/build_qlib_bundle.py",
            "--start", start,
            "--end", end,
            "--parquet-root", settings.PARQUET_ROOT,
            "--qlib-out", qlib_out,
            "--min-days", str(min_days),
        ]
        
        # 添加可选参数
        if update:
            cmd.append("--update")
        if validate:
            cmd.append("--validate")
        if backup:
            cmd.append("--backup")
        
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        response = {
            "ok": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "qlib_output": qlib_out,
            "update_mode": update,
            "timestamp": datetime.now().isoformat()
        }
        
        if result.returncode != 0:
            logger.error(f"Qlib数据包构建失败: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"Qlib build failed with return code {result.returncode}: {result.stderr}"
            )
        
        logger.info(f"Qlib数据包构建成功: {qlib_out}")
        return response
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Qlib build timeout")
    except Exception as e:
        logger.error(f"Qlib构建异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def get_system_status(auth=Depends(admin_auth)):
    """获取系统状态
    
    返回数据存储、处理任务等系统状态信息。
    """
    try:
        store = ParquetStore()
        
        # 获取数据存储信息
        data_info = store.get_data_info(settings.PARQUET_ROOT)
        
        # 系统状态
        status = {
            "timestamp": datetime.now().isoformat(),
            "parquet_root": settings.PARQUET_ROOT,
            "data_info": data_info,
            "system": {
                "python_version": sys.version,
                "platform": sys.platform,
            }
        }
        
        return status
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols")
def list_symbols(
    market: str = Query("all", description="市场过滤: all/sh/sz"),
    auth=Depends(admin_auth)
):
    """列出可用的股票代码
    
    返回Parquet存储中所有可用的股票代码列表。
    """
    try:
        store = ParquetStore()
        all_symbols = store.list_symbols(settings.PARQUET_ROOT)
        
        # 市场过滤
        if market == "sh":
            symbols = [s for s in all_symbols if s.endswith('.SH')]
        elif market == "sz":
            symbols = [s for s in all_symbols if s.endswith('.SZ')]
        else:
            symbols = all_symbols
        
        return {
            "total": len(symbols),
            "market": market,
            "symbols": sorted(symbols),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"列出股票代码失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/data/cleanup")
def cleanup_old_data(
    keep_days: int = Query(365, description="保留天数"),
    auth=Depends(admin_auth)
):
    """清理旧数据
    
    删除超过指定天数的分钟级数据，保留日频数据。
    """
    try:
        logger.info(f"开始清理旧数据，保留 {keep_days} 天")
        
        store = ParquetStore()
        store.cleanup_old_data(settings.PARQUET_ROOT, keep_days)
        
        return {
            "ok": True,
            "message": f"Old data cleanup completed, kept {keep_days} days",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"清理旧数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))





