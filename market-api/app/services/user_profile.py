"""User profile and personalization persistence layer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
PROFILE_FILE = DATA_DIR / "user_profile.json"
PREFERENCES_FILE = DATA_DIR / "user_preferences.json"

DEFAULT_PROFILE: Dict[str, Any] = {
    "id": "investor-001",
    "name": "张三",
    "phone": "186****8899",
    "avatar": "https://api.dicebear.com/7.x/miniavs/svg?seed=naxs",
    "city": "上海",
    "role": "资深智能投研顾问",
    "greeting": "早上好，张三！AI投研助手已经为您准备好了今日的市场洞察。",
    "badges": [
        "稳健型投资者",
        "风险偏好：平衡",
        "年度目标收益：12%",
    ],
    "updated_at": "2025-02-18T08:30:00+08:00",
}

DEFAULT_PREFERENCES: Dict[str, Any] = {
    "experience_level": "experienced",
    "experience_label": "有一定经验",
    "asset_scale": "100k-1m",
    "risk_score": 5,
    "risk_attitude": "平衡",
    "investment_horizon": "中期 (1-3年)",
    "strategy_style": "balanced",
    "focus_industries": ["新能源", "半导体", "人工智能", "消费升级"],
    "excluded_industries": ["煤炭", "钢铁"],
    "ai_auto_adjust": True,
    "notes": "偏好可控回撤的稳健成长组合，关注科技与高端制造机会。",
    "updated_at": "2025-02-18T08:30:00+08:00",
}


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_data_dir()
    if not path.exists():
        _write_json(path, default)
        return default

    try:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            _write_json(path, default)
            return default
        data = json.loads(content)
        if isinstance(data, dict):
            return {**default, **data}
        return default
    except Exception:
        _write_json(path, default)
        return default


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    _ensure_data_dir()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_user_profile() -> Dict[str, Any]:
    """Return persisted user profile information."""
    profile = _load_json(PROFILE_FILE, DEFAULT_PROFILE)
    if "updated_at" not in profile:
        profile["updated_at"] = datetime.now().astimezone().isoformat()
    return profile


def get_user_preferences() -> Dict[str, Any]:
    """Return personalization preferences."""
    prefs = _load_json(PREFERENCES_FILE, DEFAULT_PREFERENCES)
    if "updated_at" not in prefs:
        prefs["updated_at"] = datetime.now().astimezone().isoformat()
    return prefs


def update_user_preferences(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Persist user preferences and return the merged result."""
    current = get_user_preferences()
    merged = {**current, **updates}
    merged["updated_at"] = datetime.now().astimezone().isoformat()
    _write_json(PREFERENCES_FILE, merged)
    return merged


def reset_user_preferences() -> Dict[str, Any]:
    """Reset personalization preferences to defaults."""
    _write_json(PREFERENCES_FILE, DEFAULT_PREFERENCES)
    return get_user_preferences()

