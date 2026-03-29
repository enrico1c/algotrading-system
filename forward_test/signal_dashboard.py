"""
Signal Dashboard — console display for forward-test / live mode.
Renders current signals, portfolio state, and eToro action instructions.
Compatible with eToro's manual execution workflow (no API key required).
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

from strategies.base import Signal, SignalType
from portfolio.allocator import AllocationState

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORS = True
except ImportError:
    COLORS = False


class SignalDashboard:
    """
    Console dashboard for live signal display.
    Shows: signals, portfolio weights, eToro execution instructions.
    """

    def render(
        self,
        signals: List[Signal],
        state: Optional[AllocationState],
        paper_equity: Dict[str, float] = None,
    ) -> None:
        self._clear()
        self._header()
        self._portfolio_section(state, paper_equity)
        self._signals_section(signals)
        self._etoro_instructions(signals, state)
        self._footer()

    # ──────────────────────────────────────────

    def _header(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(self._bold("=" * 70))
        print(self._bold(f"  ALGO TRADING DASHBOARD  |  {now}"))
        print(self._bold("=" * 70))

    def _portfolio_section(
        self, state: Optional[AllocationState], paper_equity: Dict[str, float]
    ) -> None:
        print(f"\n{self._bold('PORTFOLIO')}")
        print("-" * 50)
        if state is None:
            print("  Not initialized")
            return

        total = state.total_capital
        print(f"  Total Capital : {total:.2f} EUR")
        print(f"  Strategies    : {state.n_strategies}")
        print()

        for name, weight in state.weights.items():
            cap = state.allocated_capital.get(name, 0.0)
            max_trade = cap * 0.10
            live_eq = (paper_equity or {}).get(name, cap)
            pnl = live_eq - cap
            pnl_str = self._color(f"{pnl:+.2f}", "green" if pnl >= 0 else "red")
            print(
                f"  {name:<35} {weight:.1%}  "
                f"Cap:{cap:.2f} EUR  MaxTrade:{max_trade:.2f} EUR  PnL:{pnl_str} EUR"
            )

    def _signals_section(self, signals: List[Signal]) -> None:
        print(f"\n{self._bold('SIGNALS')}")
        print("-" * 50)
        if not signals:
            print(f"  {self._color('No new signals', 'yellow')}")
            return

        for sig in signals:
            action_color = {
                SignalType.BUY: "green",
                SignalType.SELL: "red",
                SignalType.CLOSE: "yellow",
                SignalType.HOLD: "white",
            }.get(sig.signal_type, "white")

            action = self._color(f"[{sig.signal_type.value}]", action_color)
            meta = " | ".join(f"{k}={v}" for k, v in (sig.metadata or {}).items())
            print(
                f"  {action:<20}  {sig.strategy_name:<30}  "
                f"{sig.ticker:<10}  @ {sig.price:.4f}  conf={sig.confidence:.0%}"
            )
            if meta:
                print(f"    └─ {meta}")

    def _etoro_instructions(
        self, signals: List[Signal], state: Optional[AllocationState]
    ) -> None:
        buy_sell = [s for s in signals if s.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.CLOSE)]
        if not buy_sell or state is None:
            return

        print(f"\n{self._bold('ETORO EXECUTION INSTRUCTIONS')}")
        print("-" * 50)
        print("  (Execute these manually on eToro until API key is obtained)")
        print()

        for sig in buy_sell:
            cap = state.allocated_capital.get(sig.strategy_name, 0.0)
            max_trade_eur = cap * 0.10
            action = sig.signal_type.value

            if sig.signal_type == SignalType.BUY:
                print(
                    f"  1. Search for '{sig.ticker}' on eToro")
                print(
                    f"  2. Click TRADE → BUY")
                print(
                    f"  3. Amount: {max_trade_eur:.2f} EUR (max 10% of {sig.strategy_name} capital)")
                print(
                    f"  4. Set Stop-Loss: -15%  |  Take-Profit: +10%")
                print(
                    f"  5. Confirm order (leverage=x1 recommended)")

            elif sig.signal_type in (SignalType.SELL, SignalType.CLOSE):
                print(
                    f"  1. Go to Portfolio → find open {sig.ticker} position")
                print(
                    f"  2. Click CLOSE TRADE")
                print(
                    f"  3. Confirm close at market price")

            print(f"  Strategy: {sig.strategy_name} | Signal: {action} | "
                  f"Price ref: {sig.price:.4f}")
            print()

    def _footer(self) -> None:
        print("=" * 70)
        print("  Refresh: automatic | Mode: SIGNAL_ONLY | Data: yfinance (free)")
        print("=" * 70)

    # ──────────────────────────────────────────
    # FORMATTING HELPERS
    # ──────────────────────────────────────────

    @staticmethod
    def _clear() -> None:
        os.system("cls" if os.name == "nt" else "clear")

    @staticmethod
    def _bold(text: str) -> str:
        if COLORS:
            return f"{Style.BRIGHT}{text}{Style.RESET_ALL}"
        return text

    @staticmethod
    def _color(text: str, color: str) -> str:
        if not COLORS:
            return text
        color_map = {
            "green": Fore.GREEN, "red": Fore.RED, "yellow": Fore.YELLOW,
            "white": Fore.WHITE, "cyan": Fore.CYAN,
        }
        return f"{color_map.get(color, '')}{text}{Style.RESET_ALL}"
