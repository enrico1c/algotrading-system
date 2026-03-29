"""
Signal-Only Executor — default mode when no broker API is configured.
Logs all signals and prints eToro manual execution instructions.
No real trades are placed.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from execution.base import ExecutionBase
from strategies.base import Signal, SignalType
from utils.logger import get_logger

logger = get_logger("execution.signal_only")

SIGNALS_FILE = Path("reports") / "signals_log.json"


class SignalOnlyExecutor(ExecutionBase):
    """
    Logs signals to console + JSON file.
    Provides step-by-step eToro manual execution instructions.
    """

    def __init__(self):
        self._virtual_positions: list = []
        self._balance = 0.0  # not tracked in signal-only mode

    def execute(self, signal: Signal, size_eur: float) -> dict:
        logger.info(
            f"SIGNAL | {signal.signal_type.value} | {signal.ticker} | "
            f"Strategy: {signal.strategy_name} | "
            f"Price: {signal.price:.4f} | Size: {size_eur:.2f} EUR"
        )

        if signal.signal_type == SignalType.BUY:
            self._print_etoro_buy(signal, size_eur)
        elif signal.signal_type in (SignalType.SELL, SignalType.CLOSE):
            self._print_etoro_sell(signal)

        self._append_to_log(signal, size_eur)

        return {
            "mode": "signal_only",
            "status": "logged",
            "signal": signal.signal_type.value,
            "ticker": signal.ticker,
            "size_eur": size_eur,
            "timestamp": str(signal.timestamp),
        }

    def get_open_positions(self) -> list:
        return self._virtual_positions

    def get_account_balance(self) -> float:
        return self._balance

    # ──────────────────────────────────────────

    def _print_etoro_buy(self, signal: Signal, size_eur: float) -> None:
        print(f"\n{'='*60}")
        print(f"  ACTION REQUIRED — OPEN POSITION ON ETORO")
        print(f"{'='*60}")
        print(f"  Ticker   : {signal.ticker}")
        print(f"  Action   : BUY (Long)")
        print(f"  Amount   : {size_eur:.2f} EUR")
        print(f"  Strategy : {signal.strategy_name}")
        print(f"  Ref Price: {signal.price:.4f}")
        print(f"  Confidence: {signal.confidence:.0%}")
        if signal.metadata:
            print(f"  Indicators: {signal.metadata}")
        print(f"\n  STEPS:")
        print(f"  1. Go to eToro → search '{signal.ticker}'")
        print(f"  2. Click TRADE → BUY")
        print(f"  3. Set amount: {size_eur:.2f} EUR")
        print(f"  4. Leverage: x1 (no leverage)")
        print(f"  5. Confirm order")
        print(f"{'='*60}\n")

    def _print_etoro_sell(self, signal: Signal) -> None:
        print(f"\n{'='*60}")
        print(f"  ACTION REQUIRED — CLOSE POSITION ON ETORO")
        print(f"{'='*60}")
        print(f"  Ticker   : {signal.ticker}")
        print(f"  Action   : CLOSE / SELL")
        print(f"  Strategy : {signal.strategy_name}")
        print(f"  Ref Price: {signal.price:.4f}")
        print(f"\n  STEPS:")
        print(f"  1. Go to eToro → Portfolio")
        print(f"  2. Find open position for '{signal.ticker}'")
        print(f"  3. Click CLOSE TRADE")
        print(f"  4. Confirm close at market price")
        print(f"{'='*60}\n")

    def _append_to_log(self, signal: Signal, size_eur: float) -> None:
        SIGNALS_FILE.parent.mkdir(exist_ok=True)
        entry = {
            "ts": datetime.now().isoformat(),
            "strategy": signal.strategy_name,
            "ticker": signal.ticker,
            "action": signal.signal_type.value,
            "price": signal.price,
            "size_eur": size_eur,
            "confidence": signal.confidence,
            "meta": signal.metadata,
        }
        entries = []
        if SIGNALS_FILE.exists():
            try:
                with open(SIGNALS_FILE) as f:
                    entries = json.load(f)
            except Exception:
                pass
        entries.append(entry)
        with open(SIGNALS_FILE, "w") as f:
            json.dump(entries, f, indent=2, default=str)
