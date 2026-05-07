from __future__ import annotations

from dataclasses import dataclass

from .config import BotConfig
from .signal_engine import SignalCandidate


@dataclass(frozen=True)
class RiskDecision:
    candidate: SignalCandidate
    accepted: bool
    reason: str
    active_positions: int
    active_symbol_positions: int
    allocation_fraction: float


class RiskEngine:
    def __init__(self, config: BotConfig) -> None:
        self.config = config

    def apply(self, candidates: list[SignalCandidate], positions: list[dict[str, object]]) -> list[RiskDecision]:
        active = [pos for pos in positions if pos.get("status") in {"entry_pending", "open", "exit_pending"}]
        active_by_symbol: dict[str, int] = {}
        for pos in active:
            symbol = str(pos["symbol"])
            active_by_symbol[symbol] = active_by_symbol.get(symbol, 0) + 1

        ordered = sorted(candidates, key=lambda item: (-item.abs_drop, self.config.symbols.index(item.symbol), item.symbol))
        decisions: list[RiskDecision] = []
        active_count = len(active)
        for candidate in ordered:
            symbol_active = active_by_symbol.get(candidate.symbol, 0)
            reason = "accepted"
            accepted = True
            if active_count >= self.config.concurrent_cap:
                accepted = False
                reason = "concurrent_cap"
            elif symbol_active >= self.config.symbol_cap:
                accepted = False
                reason = "symbol_cap"
            elif (active_count + 1) * self.config.allocation_per_position > self.config.max_exposure_fraction + 1e-12:
                accepted = False
                reason = "max_exposure"

            decisions.append(
                RiskDecision(
                    candidate=candidate,
                    accepted=accepted,
                    reason=reason,
                    active_positions=active_count,
                    active_symbol_positions=symbol_active,
                    allocation_fraction=self.config.allocation_per_position if accepted else 0.0,
                )
            )
            if accepted:
                active_count += 1
                active_by_symbol[candidate.symbol] = symbol_active + 1
        return decisions
