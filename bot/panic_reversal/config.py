from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
]


@dataclass(frozen=True)
class BotConfig:
    symbols: list[str] = field(default_factory=lambda: SYMBOLS.copy())
    timeframe: str = "15m"
    candle_minutes: int = 15
    event_return_threshold: float = -0.03
    breadth_return_threshold: float = -0.02
    min_breadth_count: int = 4
    btc_24h_bars: int = 96
    ema50_period: int = 50
    ema200_period: int = 200
    ema_slope_bars: int = 24
    realized_vol_lookback_bars: int = 96
    realized_vol_min_periods: int = 48
    vol_threshold_lookback_bars: int = 96 * 30
    vol_threshold_min_periods: int = 96 * 7
    vol_threshold_quantile: float = 0.67
    entry_mode: str = "next_close"
    comparison_entry_mode: str = "next_open"
    exit_model: str = "pure_horizon_12"
    hold_bars: int = 12
    stacking_enabled: bool = True
    concurrent_cap: int = 10
    symbol_cap: int = 2
    selection_policy: str = "largest_drop"
    allocation_per_position: float = 0.02
    max_exposure_fraction: float = 0.20
    bot_mode: str = "dry_run"
    live_trading_enabled: bool = False
    data_source: str = "local"
    data_dir: Path = Path("data") / "binance_um_ohlcv" / "15m"
    state_dir: Path = Path("bot") / "panic_reversal" / "state"
    log_dir: Path = Path("bot") / "panic_reversal" / "logs"
    config_version: str = "panic_reversal_v1_dry_run"
    binance_base_url: str = "https://fapi.binance.com"
    binance_klines_path: str = "/fapi/v1/klines"
    binance_request_limit: int = 1500
    poll_after_close_delay_seconds: int = 5

    @classmethod
    def from_env(cls) -> "BotConfig":
        mode = os.getenv("BOT_MODE", "dry_run").strip().lower()
        live_enabled = os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() == "true"
        return cls(
            bot_mode=mode,
            live_trading_enabled=live_enabled,
            data_source=os.getenv("PANIC_BOT_DATA_SOURCE", "local").strip().lower(),
            data_dir=Path(os.getenv("PANIC_BOT_DATA_DIR", str(Path("data") / "binance_um_ohlcv" / "15m"))),
            state_dir=Path(os.getenv("PANIC_BOT_STATE_DIR", str(Path("bot") / "panic_reversal" / "state"))),
            log_dir=Path(os.getenv("PANIC_BOT_LOG_DIR", str(Path("bot") / "panic_reversal" / "logs"))),
        )

    @property
    def candle_delta_minutes(self) -> int:
        return self.candle_minutes

    def assert_dry_run_safe(self) -> None:
        if self.bot_mode != "dry_run":
            raise RuntimeError("Refusing to run: BOT_MODE must be dry_run for this skeleton.")
        if self.live_trading_enabled:
            raise RuntimeError("Refusing to run: LIVE_TRADING_ENABLED must remain false.")
