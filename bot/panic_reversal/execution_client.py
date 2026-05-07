from __future__ import annotations

from dataclasses import dataclass

from .config import BotConfig


@dataclass(frozen=True)
class SimulatedFill:
    symbol: str
    side: str
    quantity: float
    price: float
    order_type: str
    mode: str


class ExecutionClientStub:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.config.assert_dry_run_safe()

    def place_order(self, *_: object, **__: object) -> None:
        raise RuntimeError("Real order placement is disabled in the dry-run skeleton.")

    def simulate_market_order(self, symbol: str, side: str, quantity: float, price: float) -> SimulatedFill:
        self.config.assert_dry_run_safe()
        return SimulatedFill(symbol=symbol, side=side, quantity=quantity, price=price, order_type="market", mode="dry_run")
