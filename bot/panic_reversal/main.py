from __future__ import annotations

import argparse
from dataclasses import replace

from .config import BotConfig
from .scheduler import DryRunScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panic-Reversal Bot dry-run skeleton.")
    parser.add_argument("--once", action="store_true", help="Run one dry-run scheduler cycle.")
    parser.add_argument("--data-source", choices=["local", "binance"], help="Override PANIC_BOT_DATA_SOURCE.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BotConfig.from_env()
    if args.data_source:
        config = replace(config, data_source=args.data_source)
    config.assert_dry_run_safe()
    scheduler = DryRunScheduler(config)
    if not args.once:
        raise SystemExit("Only --once is implemented in this dry-run skeleton.")
    scheduler.run_once()


if __name__ == "__main__":
    main()
