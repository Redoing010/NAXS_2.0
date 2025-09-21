"""Dashboard data orchestrator."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

from .user_profile import get_user_preferences, get_user_profile


def _risk_comment(score: int) -> str:
    if score <= 3:
        return "保守型：以资金安全为第一目标，可配置更多固定收益产品。"
    if score <= 6:
        return "平衡型：可在稳健与成长之间均衡配置，关注回撤控制。"
    if score <= 8:
        return "进取型：接受阶段性波动，适合配置成长和趋势类策略。"
    return "激进型：追求高增长，可结合量化趋势和主题投资策略。"


def _generate_performance_series(base_value: float = 3860.0) -> List[Dict[str, Any]]:
    today = datetime.now().date()
    series: List[Dict[str, Any]] = []
    for days_ago in range(9, -1, -1):
        date = today - timedelta(days=days_ago)
        offset = (days_ago - 4.5) / 4.5
        portfolio = base_value + offset * 45 + (days_ago % 3) * 18
        benchmark = base_value - 40 + offset * 30 + (days_ago % 2) * 12
        series.append(
            {
                "date": date.isoformat(),
                "portfolio": round(portfolio, 2),
                "benchmark": round(benchmark, 2),
            }
        )
    return series


def _generate_flow_series() -> List[Dict[str, Any]]:
    base = datetime.now().date()
    result: List[Dict[str, Any]] = []
    volumes = [5200, 6100, 4800, 6600, 5900, 7100, 6400]
    forecast = [4800, 5800, 5100, 6300, 5600, 6800, 6200]
    for idx, volume in enumerate(volumes):
        day = base - timedelta(days=6 - idx)
        result.append(
            {
                "date": day.strftime("%m-%d"),
                "net_flow": volume,
                "forecast": forecast[idx],
            }
        )
    return result


def _generate_ai_insights(preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    focus = preferences.get("focus_industries", [])
    first_focus = focus[0] if focus else "科技"
    return [
        {
            "title": "组合诊断",
            "detail": f"本周组合波动保持在可控范围内，风险得分 {preferences.get('risk_score', 5)}/10，仍处于平衡区间。",
        },
        {
            "title": "行业机会",
            "detail": f"{first_focus}行业资金面保持净流入，建议继续跟踪核心资产，并关注板块轮动节奏。",
        },
        {
            "title": "AI策略建议",
            "detail": "建议结合趋势追踪与因子多因子策略，设置动态止盈以锁定阶段收益。",
        },
    ]


def _generate_prompts(preferences: Dict[str, Any]) -> List[str]:
    horizon = preferences.get("investment_horizon", "中期 (1-3年)")
    focus = preferences.get("focus_industries", [])
    focus_display = "、".join(focus[:3]) if focus else "科技与新能源"
    return [
        f"基于{horizon}目标，如何优化当前组合的风险收益结构？",
        f"{focus_display}板块未来一周的资金面与关键事件有哪些？",
        "请给我一份包含止盈止损参数的再平衡执行清单。",
    ]


def get_dashboard_overview() -> Dict[str, Any]:
    profile = get_user_profile()
    preferences = get_user_preferences()
    risk_score = int(preferences.get("risk_score", 5))

    overview: Dict[str, Any] = {
        "headline": "AI 驱动的下一代智能投研平台",
        "subheadline": "整合市场数据、AI洞察与个性化投研助手，帮助你快速制定投资决策。",
        "timestamp": datetime.now().astimezone().isoformat(),
        "account": {
            "net_asset": 3876.34,
            "week_pnl": 13215.46,
            "available_cash": 3147.35,
            "score": "B-",
            "score_trend": 0.12,
            "goal": {"completed": 8, "target": 12},
            "risk_level": "中度",
            "risk_score": risk_score,
            "risk_comment": _risk_comment(risk_score),
        },
        "market_heat": {
            "score": 10.8,
            "change": 1.2,
            "grade": "良好",
            "north_bound": 6.1,
            "ai_sentiment": "偏积极",
        },
        "performance_trend": _generate_performance_series(),
        "fund_flow": _generate_flow_series(),
        "ai_insights": _generate_ai_insights(preferences),
        "quick_prompts": _generate_prompts(preferences),
        "profile": {
            "name": profile.get("name", "投资者"),
            "avatar": profile.get("avatar"),
            "badges": profile.get("badges", []),
            "greeting": profile.get("greeting"),
        },
        "news": [
            {
                "title": "A股主要指数集体收涨，成交额重回万亿",
                "source": "NAXS 量化研究中心",
                "timestamp": datetime.now().strftime("%m-%d %H:%M"),
            },
            {
                "title": "北向资金连续四日净流入，科技成长方向领涨",
                "source": "NAXS 智能资讯",
                "timestamp": (datetime.now() - timedelta(hours=3)).strftime("%m-%d %H:%M"),
            },
        ],
    }
    return overview


def get_daily_briefing() -> Dict[str, Any]:
    preferences = get_user_preferences()
    focus = preferences.get("focus_industries", [])
    briefing_focus = "、".join(focus[:2]) if focus else "科技与新能源"
    return {
        "title": "今日投研速览",
        "highlights": [
            "风险指标维持在平衡区间，组合回撤控制良好。",
            f"重点关注{briefing_focus}行业的资金流入与政策动向。",
            "AI策略建议关注短期突破信号，并结合量化止盈策略。",
        ],
    }

