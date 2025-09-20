"""Minimal qlib adapter for backtest execution."""
from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BacktestResult:
    metrics: Dict[str, Any]
    evidence_id: str


def run_backtest(params: Dict[str, Any], profile_id: str) -> BacktestResult:
    """Simulate a qlib backtest run.

    The adapter produces deterministic metrics based on the provided profile and
    parameter hash. This keeps the MVP deterministic without depending on
    external data.
    """
    seed = f"{profile_id}:{json.dumps(params, sort_keys=True)}"
    derived = uuid.uuid5(uuid.NAMESPACE_OID, seed)

    metrics = {
        "annRet": round((derived.int % 1000) / 100.0, 4),
        "sharpe": round(((derived.int // 10) % 500) / 100.0, 4),
        "maxDD": round(((derived.int // 100) % 300) / 100.0, 4),
        "turnover": round(((derived.int // 1000) % 400) / 100.0, 4),
    }

    evidence_id = f"evi-{derived.hex}"

    return BacktestResult(metrics=metrics, evidence_id=evidence_id)


def _main() -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    params = payload.get("params", {})
    profile_id = payload.get("profileId", "default")

    result = run_backtest(params=params, profile_id=profile_id)

    print(
        json.dumps(
            {
                "metrics": result.metrics,
                "evidenceId": result.evidence_id,
            }
        )
    )


if __name__ == "__main__":
    _main()
