from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .config import BotConfig
from .data_fetcher import fetch_latest_completed_candles
from .execution_client import ExecutionClientStub
from .position_manager import PositionManager
from .reporting import print_run_summary, write_quality_report
from .risk_engine import RiskEngine
from .signal_engine import generate_candidates
from .state_store import StateStore


class DryRunScheduler:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.config.assert_dry_run_safe()
        self.store = StateStore(config)
        self.execution = ExecutionClientStub(config)
        self.positions = PositionManager(config, self.store, self.execution)
        self.risk = RiskEngine(config)

    def run_once(self) -> dict[str, object]:
        batch = fetch_latest_completed_candles(self.config)
        if batch.aligned_timestamp is None:
            raise RuntimeError("Could not align latest completed candles across all configured symbols.")
        decision_time = batch.aligned_timestamp + pd.Timedelta(minutes=self.config.candle_minutes)
        due_counts = self.positions.process_due_positions(batch.frames, decision_time)
        candidates = generate_candidates(batch.frames, batch.aligned_timestamp, self.config)
        decisions = self.risk.apply(candidates, self.store.load_positions())
        self.positions.record_decisions(decisions)
        quality_path = self.config.log_dir / "latest_data_quality.csv"
        write_quality_report(batch.quality, quality_path)
        accepted_count = sum(1 for decision in decisions if decision.accepted)
        skipped_count = len(decisions) - accepted_count
        summary = {
            "run_time_utc": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.bot_mode,
            "live_trading_enabled": self.config.live_trading_enabled,
            "data_source": self.config.data_source,
            "aligned_candle_open_time": batch.aligned_timestamp.isoformat(),
            "decision_time": decision_time.isoformat(),
            "candidate_count": len(candidates),
            "accepted_count": accepted_count,
            "skipped_count": skipped_count,
            "opened_positions": due_counts["opened"],
            "closed_positions": due_counts["closed"],
            "active_positions": due_counts["active_positions"],
            "quality_report": str(quality_path),
        }
        self.store.log_run(summary)
        print_run_summary(summary)
        return summary
