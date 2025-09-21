"""UI centric endpoints for the NAXS assistant experience."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..schemas import (
    DashboardBriefing,
    DashboardOverview,
    UserPreferences,
    UserPreferencesUpdate,
    UserProfile,
)
from ..services.dashboard import get_daily_briefing, get_dashboard_overview
from ..services.user_profile import (
    get_user_preferences,
    get_user_profile,
    reset_user_preferences,
    update_user_preferences,
)

router = APIRouter(prefix=f"{settings.API_PREFIX}", tags=["ui"])


@router.get("/dashboard/overview", response_model=DashboardOverview)
def dashboard_overview() -> DashboardOverview:
    return DashboardOverview.model_validate(get_dashboard_overview())


@router.get("/dashboard/briefing", response_model=DashboardBriefing)
def dashboard_briefing() -> DashboardBriefing:
    return DashboardBriefing.model_validate(get_daily_briefing())


@router.get("/user/profile", response_model=UserProfile)
def fetch_profile() -> UserProfile:
    return UserProfile.model_validate(get_user_profile())


@router.get("/user/preferences", response_model=UserPreferences)
def fetch_preferences() -> UserPreferences:
    return UserPreferences.model_validate(get_user_preferences())


@router.post("/user/preferences", response_model=UserPreferences)
def persist_preferences(payload: UserPreferencesUpdate) -> UserPreferences:
    updated = update_user_preferences(payload.model_dump(exclude_none=True))
    return UserPreferences.model_validate(updated)


@router.post("/user/preferences/reset", response_model=UserPreferences)
def reset_preferences() -> UserPreferences:
    return UserPreferences.model_validate(reset_user_preferences())

