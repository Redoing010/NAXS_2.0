from fastapi import Query, Header, HTTPException
from .config import settings


def pagination(
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=10, le=settings.MAX_PAGE_SIZE),
):
    return {"page": page, "page_size": page_size}


def admin_auth(x_admin_token: str = Header(None)):
    if not settings.ADMIN_TOKEN:
        return
    if x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid X-Admin-Token")
    return





