from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BotConfig


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    return str(value)


class StateStore:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.processed_path = self.config.state_dir / "processed_signals.json"
        self.positions_path = self.config.state_dir / "positions.json"
        self.accepted_path = self.config.state_dir / "accepted_signals.jsonl"
        self.skipped_path = self.config.state_dir / "skipped_signals.jsonl"
        self.closed_path = self.config.state_dir / "closed_positions.jsonl"
        self.runs_path = self.config.log_dir / "scheduler_runs.jsonl"

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")

    def append_jsonl(self, path: Path, row: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=_json_default) + "\n")

    def processed_keys(self) -> set[str]:
        return set(self._read_json(self.processed_path, []))

    def mark_processed(self, key: str) -> None:
        keys = sorted(self.processed_keys() | {key})
        self._write_json(self.processed_path, keys)

    def load_positions(self) -> list[dict[str, object]]:
        return self._read_json(self.positions_path, [])

    def save_positions(self, positions: list[dict[str, object]]) -> None:
        self._write_json(self.positions_path, positions)

    def log_accepted(self, row: dict[str, object]) -> None:
        self.append_jsonl(self.accepted_path, row)

    def log_skipped(self, row: dict[str, object]) -> None:
        self.append_jsonl(self.skipped_path, row)

    def log_closed(self, row: dict[str, object]) -> None:
        self.append_jsonl(self.closed_path, row)

    def log_run(self, row: dict[str, object]) -> None:
        self.append_jsonl(self.runs_path, row)
