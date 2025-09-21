"""Rule-based assistant responses for the UI prototype."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from .dashboard import get_dashboard_overview
from .user_profile import get_user_preferences


_KEYWORD_RESPONSES = {
    "风险": "当前组合风险得分维持在平衡区间，我会继续监控波动率与回撤水平，建议保持分散配置。",
    "止盈": "建议结合动态止盈机制：当单笔收益超过 12% 或组合创出近三十日新高时，可逐步锁定利润。",
    "行业": "重点关注近期资金净流入的核心行业，尤其是新能源与半导体方向。",
    "再平衡": "建议按照 60/40 的权益与固收权重执行再平衡，同时保留 10% 机动仓位应对突发机会。",
}


def _match_keywords(message: str) -> List[str]:
    suggestions: List[str] = []
    for keyword, reply in _KEYWORD_RESPONSES.items():
        if keyword in message:
            suggestions.append(reply)
    return suggestions


def _compose_summary(message: str) -> str:
    overview = get_dashboard_overview()
    prefs = get_user_preferences()
    focus = "、".join(prefs.get("focus_industries", [])[:2]) or "重点行业"
    return (
        "基于最新的账户数据，组合本周保持稳健运行。"
        "我建议重点关注{focus}板块的量价配合，并结合AI策略信号做出调整。"
        "若需要，我可以进一步拆解执行步骤或生成跟踪清单。"
    ).format(focus=focus)


def generate_assistant_reply(message: str) -> Dict[str, Any]:
    extra = _match_keywords(message)
    base_summary = _compose_summary(message)

    reply_parts = [
        f"收到，我已记录你的需求：{message.strip()}。",
        base_summary,
    ]
    if extra:
        reply_parts.append("另外，根据关键词我有以下提醒：")
        for idx, item in enumerate(extra, start=1):
            reply_parts.append(f"{idx}. {item}")

    return {
        "reply": "\n".join(reply_parts),
        "insights": extra,
        "timestamp": datetime.now().astimezone().isoformat(),
    }


def get_quick_prompts() -> List[str]:
    overview = get_dashboard_overview()
    prompts = overview.get("quick_prompts")
    if isinstance(prompts, list):
        return prompts
    return [
        "当前组合的资金流向和风险敞口如何？",
        "有哪些AI策略信号值得重点跟踪？",
        "为我生成下周的调仓执行计划。",
    ]

