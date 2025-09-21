"""Pydantic schema definitions for API responses."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


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


class DashboardGoal(BaseModel):
    completed: int
    target: int


class DashboardAccount(BaseModel):
    net_asset: float
    week_pnl: float
    available_cash: float
    score: str
    score_trend: float
    goal: DashboardGoal
    risk_level: str
    risk_score: int
    risk_comment: str


class DashboardMarketHeat(BaseModel):
    score: float
    change: float
    grade: str
    north_bound: float
    ai_sentiment: str


class DashboardPerformancePoint(BaseModel):
    date: str
    portfolio: float
    benchmark: float


class DashboardFundFlowPoint(BaseModel):
    date: str
    net_flow: float
    forecast: float


class DashboardInsight(BaseModel):
    title: str
    detail: str


class DashboardNews(BaseModel):
    title: str
    source: str
    timestamp: str


class DashboardProfile(BaseModel):
    name: str
    avatar: Optional[str] = None
    badges: List[str] = Field(default_factory=list)
    greeting: Optional[str] = None


class DashboardOverview(BaseModel):
    headline: str
    subheadline: str
    timestamp: str
    account: DashboardAccount
    market_heat: DashboardMarketHeat
    performance_trend: List[DashboardPerformancePoint]
    fund_flow: List[DashboardFundFlowPoint]
    ai_insights: List[DashboardInsight]
    quick_prompts: List[str]
    profile: DashboardProfile
    news: List[DashboardNews]


class DashboardBriefing(BaseModel):
    title: str
    highlights: List[str]


class UserProfile(BaseModel):
    id: Optional[str] = None
    name: str
    phone: Optional[str] = None
    avatar: Optional[str] = None
    city: Optional[str] = None
    role: Optional[str] = None
    greeting: Optional[str] = None
    badges: List[str] = Field(default_factory=list)
    updated_at: Optional[str] = None


class UserPreferences(BaseModel):
    experience_level: Literal["newbie", "experienced", "advanced", "expert"] | str
    experience_label: str
    asset_scale: Literal["<100k", "100k-1m", ">1m"] | str
    risk_score: int = Field(ge=1, le=10)
    risk_attitude: str
    investment_horizon: str
    strategy_style: Literal["conservative", "balanced", "growth", "aggressive"] | str
    focus_industries: List[str] = Field(default_factory=list)
    excluded_industries: List[str] = Field(default_factory=list)
    ai_auto_adjust: bool = True
    notes: Optional[str] = None
    updated_at: Optional[str] = None


class UserPreferencesUpdate(BaseModel):
    experience_level: Optional[str] = None
    experience_label: Optional[str] = None
    asset_scale: Optional[str] = None
    risk_score: Optional[int] = Field(default=None, ge=1, le=10)
    risk_attitude: Optional[str] = None
    investment_horizon: Optional[str] = None
    strategy_style: Optional[str] = None
    focus_industries: Optional[List[str]] = None
    excluded_industries: Optional[List[str]] = None
    ai_auto_adjust: Optional[bool] = None
    notes: Optional[str] = None


class AssistantChatRequest(BaseModel):
    message: str


class AssistantChatResponse(BaseModel):
    reply: str
    insights: List[str] = Field(default_factory=list)
    timestamp: str


class QuickPrompts(BaseModel):
    items: List[str]

