"""
RSI(2) Mean Reversion Strategy
================================
Source: Finance/Investment-Strategies/RSI2-Mean-Reversion.md (Obsidian vault)
Win Rate: 91% | Backtest: 1993–2020 | Asset: SPY daily

Entry:  RSI(2) < 15  → BUY
Exit:   RSI(2) > 85  → SELL

Optional trend filter: only trade when Close > 200-day SMA.
Long-only. Best on low-volatility broad-market ETFs (SPY, QQQ, IWM).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from config.settings import RSI2_CONFIG, RSI2Config
from indicators.technical import compute_rsi, compute_sma
from strategies.base import Signal, SignalType, Strategy, StrategyResult
from strategies.registry import registry
from utils.logger import get_logger

logger = get_logger("strategies.rsi2")


@registry.register("rsi2_mean_reversion")
class RSI2MeanReversion(Strategy):
    """RSI(2) long-only mean-reversion on broad-market ETFs."""

    name = "rsi2_mean_reversion"

    def __init__(self, config: RSI2Config = None):
        super().__init__(config or RSI2_CONFIG)

    def get_required_tickers(self) -> List[str]:
        return [self.config.instrument]

    def warmup_period(self) -> int:
        return max(self.config.rsi_period + 5, self.config.trend_ma_period + 1)

    # ──────────────────────────────────────────
    # SIGNAL GENERATION (live / forward-test)
    # ──────────────────────────────────────────

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Called with recent OHLCV data. Returns signals for the most recent bar.
        data must have columns: open, high, low, close, volume
        """
        if len(data) < self.warmup_period():
            logger.debug("Not enough data for warmup — skipping")
            return []

        df = self._compute_indicators(data)
        last = df.iloc[-1]
        signals = []

        in_position = self.config.trend_filter and (last["close"] <= last[f"sma_{self.config.trend_ma_period}"])

        if last["entry"] and not in_position:
            signals.append(Signal(
                strategy_name=self.name,
                ticker=self.config.instrument,
                signal_type=SignalType.BUY,
                timestamp=data.index[-1],
                price=float(last["close"]),
                confidence=self._entry_confidence(last["rsi"]),
                metadata={"rsi2": round(float(last["rsi"]), 2)},
            ))
        elif last["exit"]:
            signals.append(Signal(
                strategy_name=self.name,
                ticker=self.config.instrument,
                signal_type=SignalType.SELL,
                timestamp=data.index[-1],
                price=float(last["close"]),
                confidence=1.0,
                metadata={"rsi2": round(float(last["rsi"]), 2)},
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
        """
        Vectorized backtest — no look-ahead bias.
        Executes at next-day open after signal on previous close.
        """
        df = self._compute_indicators(data).copy()
        df = df.iloc[self.warmup_period():]

        # Apply trend filter if configured
        if self.config.trend_filter:
            df.loc[df["close"] <= df[f"sma_{self.config.trend_ma_period}"], "entry"] = False

        # Simulate positions
        position = 0          # 0 = flat, 1 = long
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
                # BUY at next open
                cost_per_share = exec_price * (1 + commission)
                shares = cash / cost_per_share
                cash = 0.0
                position = 1
                entry_price = exec_price
                entry_date = curr.name

            elif position == 1 and prev["exit"]:
                # SELL at next open
                proceeds = shares * exec_price * (1 - commission)
                pnl = proceeds - (shares * entry_price)
                return_pct = (exec_price / entry_price - 1) * 100
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": curr.name,
                    "entry_price": entry_price,
                    "exit_price": exec_price,
                    "pnl": pnl,
                    "return_pct": return_pct,
                    "win": pnl > 0,
                })
                cash = proceeds
                shares = 0.0
                position = 0

            equity_val = cash + shares * float(curr["close"])
            equity.append(equity_val)

        equity_series = pd.Series(equity, index=df.index[1:], name="equity")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["entry_date", "exit_date", "entry_price", "exit_price", "pnl", "return_pct", "win"]
        )

        from backtest.metrics import compute_metrics
        metrics = compute_metrics(equity_series, trades_df, initial_capital)
        signals = self._signals_from_df(df)

        logger.info(
            f"RSI2 backtest complete | trades={len(trades_df)} | "
            f"win_rate={metrics.get('win_rate', 0):.1%} | "
            f"sharpe={metrics.get('sharpe', 0):.2f} | "
            f"cagr={metrics.get('cagr', 0):.1%}"
        )

        return StrategyResult(
            strategy_name=self.name,
            equity_curve=equity_series,
            trades=trades_df,
            signals=signals,
            metrics=metrics,
            params={
                "rsi_period": self.config.rsi_period,
                "entry_threshold": self.config.entry_threshold,
                "exit_threshold": self.config.exit_threshold,
                "trend_filter": self.config.trend_filter,
                "instrument": self.config.instrument,
            },
        )

    # ──────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────

    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["rsi"] = compute_rsi(df["close"], period=self.config.rsi_period)
        df[f"sma_{self.config.trend_ma_period}"] = compute_sma(df["close"], self.config.trend_ma_period)
        df["entry"] = df["rsi"] < self.config.entry_threshold
        df["exit"] = df["rsi"] > self.config.exit_threshold
        return df

    def _entry_confidence(self, rsi_val: float) -> float:
        """Higher confidence when RSI is more deeply oversold."""
        if pd.isna(rsi_val) or rsi_val >= self.config.entry_threshold:
            return 0.0
        return min(1.0, (self.config.entry_threshold - rsi_val) / self.config.entry_threshold)

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
                    confidence=self._entry_confidence(float(row["rsi"])),
                    metadata={"rsi2": round(float(row["rsi"]), 2)},
                ))
            elif row.get("exit"):
                signals.append(Signal(
                    strategy_name=self.name,
                    ticker=self.config.instrument,
                    signal_type=SignalType.SELL,
                    timestamp=idx,
                    price=float(row["close"]),
                    confidence=1.0,
                    metadata={"rsi2": round(float(row["rsi"]), 2)},
                ))
        return signals
