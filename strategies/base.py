"""
Abstract base class for all trading strategies.
Every strategy must implement: generate_signals() and backtest().
Adding a new strategy is as simple as:
  1. Create a new file in strategies/
  2. Subclass Strategy
  3. Decorate with @registry.register("my_strategy")
The portfolio allocator auto-detects all registered strategies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


@dataclass
class Signal:
    """A single actionable signal from a strategy."""
    strategy_name: str
    ticker: str                        # primary instrument (or "PAIR:AAPL/MSFT")
    signal_type: SignalType
    timestamp: pd.Timestamp
    price: float                       # reference price at signal time
    confidence: float = 1.0           # 0–1, used for position sizing
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[{self.timestamp.date()}] {self.strategy_name} | "
            f"{self.signal_type.value} {self.ticker} @ {self.price:.4f} "
            f"(conf={self.confidence:.2f})"
        )


@dataclass
class StrategyResult:
    """Output of a backtest run for a single strategy."""
    strategy_name: str
    equity_curve: pd.Series            # daily portfolio value
    trades: pd.DataFrame               # columns: entry_date, exit_date, pnl, return_pct
    signals: List[Signal]
    metrics: Dict[str, float]          # Sharpe, CAGR, max_drawdown, win_rate, etc.
    params: Dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """
    Base class for all trading strategies.

    Subclasses must implement:
        generate_signals(data) -> List[Signal]
        backtest(data, initial_capital) -> StrategyResult

    Subclasses should call super().__init__(name, config) in their __init__.
    """

    #: Set by @registry.register decorator or in subclass
    name: str = "unnamed_strategy"

    def __init__(self, config: Any = None):
        self.config = config
        self._last_signals: List[Signal] = []

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Given OHLCV data (and any pre-computed indicators), return a list of
        actionable signals for the current bar (last row = today).
        Must be free of look-ahead bias.
        """
        ...

    @abstractmethod
    def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float = 50.0,
        commission: float = 0.001,
        slippage: float = 0.001,
    ) -> StrategyResult:
        """
        Vectorized backtest over historical data.
        Returns a StrategyResult with equity curve, trades, and metrics.
        """
        ...

    def get_required_tickers(self) -> List[str]:
        """Return list of tickers this strategy needs data for."""
        return []

    def warmup_period(self) -> int:
        """Minimum number of bars required before signals are valid."""
        return 200

    @property
    def last_signals(self) -> List[Signal]:
        return self._last_signals
