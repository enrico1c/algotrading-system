"""
Forward Test Runner — live signal generation & paper trading simulation.
Fetches real-time data via yfinance, generates signals from all strategies,
applies risk checks, and logs recommended actions for eToro execution.

eToro compatibility note (from vault):
  Since eToro API requires developer key (not yet obtained),
  this runner operates in SIGNAL_ONLY mode by default:
  signals are printed, logged, and saved — human executes on eToro manually.
  When API key is available, set EXECUTION_CONFIG.mode = "etoro".
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.settings import EXECUTION_CONFIG, Settings
from data.fetcher import DataFetcher
from execution.base import ExecutionBase
from portfolio.allocator import AllocationState, PortfolioAllocator
from portfolio.risk_manager import RiskManager
from strategies.base import Signal
from strategies.registry import registry
from utils.logger import get_logger

logger = get_logger("forward_test.runner")


class ForwardTestRunner:
    """
    Runs all strategies in forward-test (paper/live signal) mode.

    Modes:
      signal_only — generate signals, log them, no automated execution
      paper       — simulate execution locally, track virtual P&L
      etoro       — route approved signals to eToro API (requires key)
    """

    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.fetcher = DataFetcher()
        self.allocator = PortfolioAllocator(settings.portfolio if settings else None)
        self.risk_manager = RiskManager()
        self._executor: Optional[ExecutionBase] = None
        self._paper_positions: Dict[str, Dict] = {}  # strategy → open position
        self._paper_equity: Dict[str, float] = {}    # strategy → current equity
        self._peak_equity: Dict[str, float] = {}     # for drawdown tracking
        self._signal_log: List[Dict] = []

    def initialize(self) -> None:
        """Set up strategies, allocator, and execution layer."""
        registry.auto_discover()
        strategy_names = registry.names()
        self.allocator.initialize(strategy_names)

        # Initialize paper equity tracking
        for name in strategy_names:
            cap = self.allocator.state.allocated_capital.get(name, 0.0)
            self._paper_equity[name] = cap
            self._peak_equity[name] = cap

        # Set up execution layer
        self._executor = self._build_executor()
        logger.info(
            f"ForwardTestRunner initialized | mode={self.settings.execution.mode} | "
            f"strategies={strategy_names} | "
            f"total_capital={self.settings.portfolio.initial_capital:.2f} EUR"
        )

    def run_once(self) -> List[Signal]:
        """
        Run one full cycle: fetch data → generate signals → risk check → execute/log.
        Call this from a scheduler or loop.
        """
        if self.allocator.state is None:
            self.initialize()

        all_signals: List[Signal] = []
        strategies = registry.instantiate_all({
            "rsi2_mean_reversion": self.settings.rsi2,
            "triple_rsi": self.settings.triple_rsi,
            "cointegration_pairs_vecm": self.settings.pairs_vecm,
        })

        # Rebalance check
        if self.allocator.should_rebalance():
            self.allocator.rebalance(registry.names())

        for strat in strategies:
            try:
                data = self._load_live_data(strat)
                if data.empty:
                    continue
                signals = strat.generate_signals(data)
                for sig in signals:
                    approved_sig = self._process_signal(sig)
                    if approved_sig:
                        all_signals.append(approved_sig)
            except Exception as e:
                logger.error(f"Error in {strat.name}: {e}")

        self._update_equity_tracking()
        return all_signals

    def run_loop(self, interval_seconds: int = None) -> None:
        """
        Continuous forward-test loop.
        Runs indefinitely; Ctrl+C to stop.
        """
        interval = interval_seconds or self.settings.execution.dashboard_refresh_seconds
        logger.info(f"Starting forward-test loop (interval={interval}s). Press Ctrl+C to stop.")
        self.initialize()

        from forward_test.signal_dashboard import SignalDashboard
        dashboard = SignalDashboard()

        try:
            while True:
                signals = self.run_once()
                dashboard.render(signals, self.allocator.state, self._paper_equity)
                if signals:
                    self._save_signals(signals)
                logger.info(f"Cycle complete. Sleeping {interval}s...")
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Forward test stopped by user.")

    # ──────────────────────────────────────────
    # INTERNALS
    # ──────────────────────────────────────────

    def _process_signal(self, signal: Signal) -> Optional[Signal]:
        """Apply risk check and route to executor."""
        state = self.allocator.state
        trade_size = self.allocator.get_trade_size(signal.strategy_name)
        check = self.risk_manager.check_trade(signal, state, trade_size)

        if not check.approved:
            logger.warning(f"Signal BLOCKED: {signal} | Reason: {check.reason}")
            return None

        logger.info(f"Signal APPROVED: {signal} | Size: {check.adjusted_size_eur:.2f} EUR")

        # Log for dashboard / eToro manual execution
        self._signal_log.append({
            "timestamp": signal.timestamp,
            "strategy": signal.strategy_name,
            "ticker": signal.ticker,
            "action": signal.signal_type.value,
            "price": signal.price,
            "size_eur": check.adjusted_size_eur,
            "confidence": signal.confidence,
            "metadata": signal.metadata,
        })

        # Route to execution layer
        if self._executor:
            self._executor.execute(signal, check.adjusted_size_eur)

        # Update paper equity if in paper mode
        if self.settings.execution.mode == "paper":
            self._update_paper_position(signal, check.adjusted_size_eur)

        return signal

    def _update_paper_position(self, signal: Signal, size_eur: float) -> None:
        """Track virtual positions for paper trading P&L."""
        from strategies.base import SignalType
        strat = signal.strategy_name
        pos = self._paper_positions.get(strat, {})

        if signal.signal_type == SignalType.BUY and not pos:
            self._paper_positions[strat] = {
                "ticker": signal.ticker,
                "entry_price": signal.price,
                "size_eur": size_eur,
                "entry_time": signal.timestamp,
            }
            self._paper_equity[strat] = self._paper_equity.get(strat, 0) - size_eur

        elif signal.signal_type in (SignalType.SELL, SignalType.CLOSE) and pos:
            entry = pos.get("entry_price", signal.price)
            ret = (signal.price / entry - 1) if entry > 0 else 0
            pnl = pos.get("size_eur", 0) * ret
            self._paper_equity[strat] = self._paper_equity.get(strat, 0) + pos.get("size_eur", 0) + pnl
            self.allocator.update_strategy_capital(strat, self._paper_equity[strat])
            self._paper_positions.pop(strat, None)
            logger.info(f"Paper trade closed: {strat} | PnL={pnl:.2f} EUR ({ret:.2%})")

    def _update_equity_tracking(self) -> None:
        """Record equity + update drawdown tracker daily."""
        today = datetime.now()
        for name, eq in self._paper_equity.items():
            peak = self._peak_equity.get(name, eq)
            if eq > peak:
                self._peak_equity[name] = eq
            self.risk_manager.update_drawdown(name, eq, self._peak_equity[name])
            self.allocator.record_equity(name, today, eq)

    def _load_live_data(self, strat) -> pd.DataFrame:
        """Load live data appropriate for strategy type."""
        if strat.name == "cointegration_pairs_vecm":
            tickers = strat.get_required_tickers()
            return self.fetcher.get_multi_close(tickers)
        primary = strat.get_required_tickers()
        if not primary:
            return pd.DataFrame()
        return self.fetcher.get_live_ohlcv(
            primary[0], lookback_days=self.settings.data.live_lookback_days
        )

    def _build_executor(self) -> Optional[ExecutionBase]:
        from execution.signal_only import SignalOnlyExecutor
        mode = self.settings.execution.mode
        if mode == "signal_only":
            return SignalOnlyExecutor()
        elif mode == "etoro":
            try:
                from execution.etoro import EToroExecutor
                return EToroExecutor(self.settings.execution)
            except ImportError:
                logger.warning("eToro executor not available — falling back to signal_only")
                return SignalOnlyExecutor()
        return SignalOnlyExecutor()

    def _save_signals(self, signals: List[Signal]) -> None:
        """Append signals to CSV for record-keeping."""
        if not signals:
            return
        rows = [
            {
                "timestamp": s.timestamp,
                "strategy": s.strategy_name,
                "ticker": s.ticker,
                "action": s.signal_type.value,
                "price": s.price,
                "confidence": s.confidence,
            }
            for s in signals
        ]
        df = pd.DataFrame(rows)
        path = Path("reports") / "signals_log.csv"
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, mode="a", header=not path.exists(), index=False)

    @property
    def signal_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._signal_log) if self._signal_log else pd.DataFrame()
