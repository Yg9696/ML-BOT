from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

import pandas as pd

from .config import BotConfig
from .execution_client import ExecutionClientStub
from .risk_engine import RiskDecision
from .state_store import StateStore


class PositionManager:
    def __init__(self, config: BotConfig, store: StateStore, execution: ExecutionClientStub) -> None:
        self.config = config
        self.store = store
        self.execution = execution

    def _signal_key(self, symbol: str, event_time: str) -> str:
        return f"{self.config.config_version}|{symbol}|{event_time}"

    def record_decisions(self, decisions: list[RiskDecision]) -> None:
        positions = self.store.load_positions()
        processed = self.store.processed_keys()
        for decision in decisions:
            candidate = decision.candidate
            event_time = candidate.event_candle_open_time.isoformat()
            key = self._signal_key(candidate.symbol, event_time)
            base = {
                **asdict(candidate),
                "config_version": self.config.config_version,
                "processed_key": key,
                "decision_reason": decision.reason,
                "active_positions_before_decision": decision.active_positions,
                "active_symbol_positions_before_decision": decision.active_symbol_positions,
                "allocation_fraction": decision.allocation_fraction,
            }
            if key in processed:
                self.store.log_skipped({**base, "skip_reason": "duplicate_signal"})
                continue
            self.store.mark_processed(key)
            if not decision.accepted:
                self.store.log_skipped({**base, "skip_reason": decision.reason})
                continue
            position = {
                **base,
                "position_id": str(uuid4()),
                "side": "long",
                "status": "entry_pending",
                "entry_time_semantic": "next_candle_close",
                "quantity": None,
                "entry_price": None,
                "exit_price": None,
                "created_at": pd.Timestamp.utcnow().isoformat(),
            }
            positions.append(position)
            self.store.log_accepted(position)
        self.store.save_positions(positions)

    def _close_at_or_before(self, frames: dict[str, pd.DataFrame], symbol: str, due_time: pd.Timestamp) -> float | None:
        candle_open = due_time - pd.Timedelta(minutes=self.config.candle_minutes)
        frame = frames.get(symbol)
        if frame is None:
            return None
        row = frame[frame["timestamp"] == candle_open]
        if row.empty:
            return None
        return float(row.iloc[0]["close"])

    def process_due_positions(self, frames: dict[str, pd.DataFrame], decision_time: pd.Timestamp) -> dict[str, int]:
        positions = self.store.load_positions()
        opened = 0
        closed = 0
        retained: list[dict[str, object]] = []
        for position in positions:
            status = position.get("status")
            symbol = str(position["symbol"])
            if status == "entry_pending":
                entry_due = pd.Timestamp(position["entry_due_time"])
                if entry_due <= decision_time:
                    price = self._close_at_or_before(frames, symbol, entry_due)
                    if price is not None:
                        quantity = float(position["allocation_fraction"]) / price
                        fill = self.execution.simulate_market_order(symbol, "BUY", quantity, price)
                        position.update({"status": "open", "entry_price": price, "quantity": quantity, "entry_fill": asdict(fill)})
                        opened += 1
            if position.get("status") == "open":
                exit_due = pd.Timestamp(position["exit_due_time"])
                if exit_due <= decision_time:
                    price = self._close_at_or_before(frames, symbol, exit_due)
                    if price is not None:
                        quantity = float(position["quantity"])
                        fill = self.execution.simulate_market_order(symbol, "SELL", quantity, price)
                        entry_price = float(position["entry_price"])
                        net_return = price / entry_price - 1.0
                        position.update({"status": "closed", "exit_price": price, "exit_fill": asdict(fill), "net_return": net_return})
                        self.store.log_closed(position)
                        closed += 1
            if position.get("status") != "closed":
                retained.append(position)
        self.store.save_positions(retained)
        return {"opened": opened, "closed": closed, "active_positions": len(retained)}
