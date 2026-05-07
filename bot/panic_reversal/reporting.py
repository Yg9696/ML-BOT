from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_quality_report(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def print_run_summary(summary: dict[str, object]) -> None:
    print(json.dumps(summary, indent=2, default=str))
