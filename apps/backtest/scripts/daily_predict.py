from __future__ import annotations

import os
import sys


def main() -> int:
    # 为确保能找到项目内模块
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from apps.backtest.qlib_adapter.pipeline import run_predict_scores

    out = run_predict_scores({"work_dir": "apps/backtest/out"})
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


