import os
import sys

from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", ".."))

from app.main import app


client = TestClient(app)


def test_dashboard_overview():
    response = client.get("/api/dashboard/overview")
    assert response.status_code == 200
    payload = response.json()
    assert payload["headline"]
    assert payload["account"]["net_asset"] > 0
    assert len(payload["performance_trend"]) >= 5


def test_user_preferences_roundtrip():
    initial = client.get("/api/user/preferences")
    assert initial.status_code == 200

    update_payload = {
        "risk_score": 6,
        "risk_attitude": "平衡",
        "focus_industries": ["新能源", "人工智能"],
    }
    updated = client.post("/api/user/preferences", json=update_payload)
    assert updated.status_code == 200
    data = updated.json()
    assert data["risk_score"] == 6
    assert "人工智能" in data["focus_industries"]

    reset = client.post("/api/user/preferences/reset")
    assert reset.status_code == 200


def test_assistant_chat():
    response = client.post("/api/assistant/chat", json={"message": "请分析一下当前风险情况"})
    assert response.status_code == 200
    payload = response.json()
    assert "风险" in payload["reply"]
    assert payload["timestamp"]

    prompts = client.get("/api/assistant/prompts")
    assert prompts.status_code == 200
    assert len(prompts.json()["items"]) > 0

