"""
Triple RSI Strategy
====================
Source: Finance/Investment-Strategies/Triple-RSI.md (Obsidian vault)
Win Rate: 90% | Backtest: 1993–2026 | Asset: SPY daily | Profit Factor: 4

Entry (ALL 4 conditions must be true at close):
  1. RSI(5) < 30               (oversold)
  2. RSI(5) declining 3+ days  (momentum filter)
  3. RSI(5) was < 60 three days ago  (regime confirmation)
  4. Close > 200-day SMA       (long-term uptrend)

Exit:
  RSI(5) > 50 at close → SELL next open

Long-only. ~3 trades/year — very high signal quality over quantity.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from config.settings import TRIPLE_RSI_CONFIG, TripleRSIConfig
from indicators.technical import compute_rsi, compute_sma
from strategies.base import Signal, SignalType, Strategy, StrategyResult
from strategies.registry import registry
from utils.logger import get_logger

logger = get_logger("strategies.triple_rsi")


@registry.register("triple_rsi")
class TripleRSI(Strategy):
    """
    Triple-RSI mean-reversion on SPY.
    4 stacked conditions dramatically reduce false signals.
    """

    name = "triple_rsi"

    def __init__(self, config: TripleRSIConfig = None):
        super().__init__(config or TRIPLE_RSI_CONFIG)

    def get_required_tickers(self) -> List[str]:
        return [self.config.instrument]

    def warmup_period(self) -> int:
        return self.config.ma_period + 10

    # ──────────────────────────────────────────
    # SIGNAL GENERATION
    # ──────────────────────────────────────────

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        if len(data) < self.warmup_period():
            return []

        df = self._compute_indicators(data)
        last = df.iloc[-1]
        signals = []

        if last["entry"]:
            signals.append(Signal(
                strategy_name=self.name,
                ticker=self.config.instrument,
                signal_type=SignalType.BUY,
                timestamp=data.index[-1],
                price=float(last["close"]),
                confidence=self._entry_confidence(last),
                metadata={
                    "rsi5": round(float(last["rsi"]), 2),
                    "ma200": round(float(last[f"sma_{self.config.ma_period}"]), 2),
                    "conditions_met": self._debug_conditions(last),
                },
            ))
        elif last["exit"]:
            signals.append(Signal(
                strategy_name=self.name,
                ticker=self.config.instrument,
                signal_type=SignalType.SELL,
                timestamp=data.index[-1],
                price=float(last["close"]),
                confidence=1.0,
                metadata={"rsi5": round(float(last["rsi"]), 2)},
            ))

        self._last_signals = signals
        return signals

    # ──────────────────────────────────────────
    # BACKTEST
    # ──────────────────────────────────────────

    def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float = 50.0,
        commission: float = 0.001,
        slippage: float = 0.001,
    ) -> StrategyResult:
        df = self._compute_indicators(data).copy()
        df = df.iloc[self.warmup_period():]

        position = 0
        cash = initial_capital
        shares = 0.0
        equity = []
        trades = []
        entry_price = 0.0
        entry_date = None

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            exec_price = float(curr["open"]) * (1 + slippage)

            if position == 0 and prev["entry"]:
                cost_per_share = exec_price * (1 + commission)
                shares = cash / cost_per_share
                cash = 0.0
                position = 1
                entry_price = exec_price
                entry_date = curr.name

            elif position == 1 and prev["exit"]:
                proceeds = shares * exec_price * (1 - commission)
                pnl = proceeds - (shares * entry_price)
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": curr.name,
                    "entry_price": entry_price,
                    "exit_price": exec_price,
                    "pnl": pnl,
                    "return_pct": (exec_price / entry_price - 1) * 100,
                    "win": pnl > 0,
                })
                cash = proceeds
                shares = 0.0
                position = 0

            equity.append(cash + shares * float(curr["close"]))

        equity_series = pd.Series(equity, index=df.index[1:], name="equity")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["entry_date", "exit_date", "entry_price", "exit_price", "pnl", "return_pct", "win"]
        )

        from backtest.metrics import compute_metrics
        metrics = compute_metrics(equity_series, trades_df, initial_capital)
        signals = self._signals_from_df(df)

        logger.info(
            f"TripleRSI backtest | trades={len(trades_df)} | "
            f"win_rate={metrics.get('win_rate', 0):.1%} | "
            f"sharpe={metrics.get('sharpe', 0):.2f}"
        )

        return StrategyResult(
            strategy_name=self.name,
            equity_curve=equity_series,
            trades=trades_df,
            signals=signals,
            metrics=metrics,
            params={
                "rsi_period": self.config.rsi_period,
                "oversold_threshold": self.config.oversold_threshold,
                "exit_threshold": self.config.exit_threshold,
                "ma_period": self.config.ma_period,
                "instrument": self.config.instrument,
            },
        )

    # ──────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────

    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["rsi"] = compute_rsi(df["close"], period=self.config.rsi_period)
        ma_col = f"sma_{self.config.ma_period}"
        df[ma_col] = compute_sma(df["close"], self.config.ma_period)

        # Condition 1: RSI < 30
        c1 = df["rsi"] < self.config.oversold_threshold

        # Condition 2: RSI declining for 3 consecutive days
        # rsi[t] < rsi[t-1] < rsi[t-2] < rsi[t-3]
        c2 = (
            (df["rsi"] < df["rsi"].shift(1)) &
            (df["rsi"].shift(1) < df["rsi"].shift(2)) &
            (df["rsi"].shift(2) < df["rsi"].shift(3))
        )

        # Condition 3: RSI was below 60 three trading days ago
        c3 = df["rsi"].shift(3) < self.config.lookback_threshold

        # Condition 4: Price above 200-day MA (long-term uptrend)
        c4 = df["close"] > df[ma_col]

        df["entry"] = c1 & c2 & c3 & c4
        df["exit"] = df["rsi"] > self.config.exit_threshold

        # Store conditions for debug metadata
        df["c1"] = c1
        df["c2"] = c2
        df["c3"] = c3
        df["c4"] = c4
        return df

    def _entry_confidence(self, row) -> float:
        """All 4 conditions met = 1.0. Partial = fraction."""
        n_met = sum([bool(row.get("c1")), bool(row.get("c2")),
                     bool(row.get("c3")), bool(row.get("c4"))])
        return n_met / 4.0

    def _debug_conditions(self, row) -> dict:
        return {
            "rsi_below_30": bool(row.get("c1")),
            "rsi_declining_3d": bool(row.get("c2")),
            "rsi_below_60_3d_ago": bool(row.get("c3")),
            "price_above_200ma": bool(row.get("c4")),
        }

    def _signals_from_df(self, df: pd.DataFrame) -> List[Signal]:
        signals = []
        for idx, row in df.iterrows():
            if row.get("entry"):
                signals.append(Signal(
                    strategy_name=self.name,
                    ticker=self.config.instrument,
                    signal_type=SignalType.BUY,
                    timestamp=idx,
                    price=float(row["close"]),
                    confidence=self._entry_confidence(row),
                    metadata={"rsi5": round(float(row["rsi"]), 2)},
                ))
            elif row.get("exit"):
                signals.append(Signal(
                    strategy_name=self.name,
                    ticker=self.config.instrument,
                    signal_type=SignalType.SELL,
                    timestamp=idx,
                    price=float(row["close"]),
                    confidence=1.0,
                    metadata={"rsi5": round(float(row["rsi"]), 2)},
                ))
        return signals
