"""Assistant chat endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..schemas import AssistantChatRequest, AssistantChatResponse, QuickPrompts
from ..services.assistant import generate_assistant_reply, get_quick_prompts

router = APIRouter(prefix=f"{settings.API_PREFIX}/assistant", tags=["assistant"])


@router.get("/prompts", response_model=QuickPrompts)
def quick_prompts() -> QuickPrompts:
    return QuickPrompts.model_validate({"items": get_quick_prompts()})


@router.post("/chat", response_model=AssistantChatResponse)
def chat(request: AssistantChatRequest) -> AssistantChatResponse:
    result = generate_assistant_reply(request.message)
    return AssistantChatResponse.model_validate(result)

