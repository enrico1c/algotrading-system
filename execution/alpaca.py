"""
Alpaca Execution Layer
========================
Free broker with:
  - Instant account creation (alpaca.markets)
  - Paper trading (no real money) — perfect for forward-test validation
  - Live trading US stocks/ETFs (SPY, QQQ, IWM) — zero commission
  - No minimum order size
  - Full REST API with Python SDK

This is the recommended execution layer while waiting for eToro API approval.
Switch paper=True → paper=False when ready to go live.

Credentials stored in Windows Credential Manager (never in code).
Setup: python setup_secrets.py → choose option 1 (Alpaca)
"""
from __future__ import annotations

from typing import List, Optional

from execution.base import ExecutionBase
from strategies.base import Signal, SignalType
from utils.secrets import get_secret
from utils.logger import get_logger

logger = get_logger("execution.alpaca")

# Tickers that map directly to Alpaca (all US stocks/ETFs)
ALPACA_SUPPORTED = {
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "META", "NVDA",
    "AMD", "INTC", "QCOM", "JPM", "BAC", "WFC", "GS", "MS", "C",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "XOM", "CVX", "COP",
    "JNJ", "PFE", "MRK", "UNH", "ABBV", "XLF", "XLK", "XLE",
}


class AlpacaExecutor(ExecutionBase):
    """
    Alpaca paper + live trading executor.
    Handles fractional shares so small EUR amounts work fine.

    Mode is set by ALPACA_PAPER secret:
      "true"  → paper account (no real money)
      "false" → live account (real money)
    """

    def __init__(self):
        self._client = None
        self._trading_client = None
        self._is_paper = True
        self._open_positions: dict = {}
        self._connect()

    def _connect(self) -> None:
        api_key = get_secret("ALPACA_API_KEY")
        secret_key = get_secret("ALPACA_SECRET_KEY")
        paper_flag = get_secret("ALPACA_PAPER") or "true"
        self._is_paper = paper_flag.lower() != "false"

        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca credentials not found.\n"
                "Run: python setup_secrets.py → choose option 1"
            )

        try:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=self._is_paper,
            )
            account = self._client.get_account()
            mode = "PAPER" if self._is_paper else "LIVE"
            logger.info(
                f"Alpaca connected [{mode}] | "
                f"Balance: ${float(account.equity):.2f} | "
                f"Buying power: ${float(account.buying_power):.2f}"
            )
        except ImportError:
            raise ImportError(
                "alpaca-py not installed.\n"
                "Run: pip install alpaca-py"
            )

    def execute(self, signal: Signal, size_eur: float) -> dict:
        """
        Route signal to Alpaca order.
        size_eur is converted to USD (approximate — no live FX, use 1.08 rate).
        Uses notional (dollar) orders so fractional shares work with small capital.
        """
        from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        ticker = signal.ticker
        if ticker not in ALPACA_SUPPORTED:
            logger.warning(f"Ticker {ticker} not in Alpaca supported list — skipping")
            return {"status": "skipped", "reason": f"{ticker} not supported on Alpaca"}

        # EUR → USD conversion (approximate)
        EUR_USD_RATE = 1.08
        size_usd = round(size_eur * EUR_USD_RATE, 2)

        mode = "PAPER" if self._is_paper else "LIVE"

        if signal.signal_type == SignalType.BUY:
            try:
                order = MarketOrderRequest(
                    symbol=ticker,
                    notional=size_usd,          # dollar-based fractional order
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                result = self._client.submit_order(order)
                logger.info(
                    f"[{mode}] BUY {ticker} ${size_usd:.2f} → "
                    f"order_id={result.id} status={result.status}"
                )
                return {
                    "status": "submitted",
                    "order_id": str(result.id),
                    "ticker": ticker,
                    "side": "buy",
                    "notional_usd": size_usd,
                    "mode": mode,
                }
            except Exception as e:
                logger.error(f"Alpaca BUY failed for {ticker}: {e}")
                return {"status": "error", "reason": str(e)}

        elif signal.signal_type in (SignalType.SELL, SignalType.CLOSE):
            try:
                # Close entire position for this ticker
                result = self._client.close_position(ticker)
                logger.info(f"[{mode}] CLOSE {ticker} → {result}")
                return {"status": "closed", "ticker": ticker, "mode": mode}
            except Exception as e:
                # Position might not exist — not necessarily an error
                logger.warning(f"Close {ticker} failed (may not be open): {e}")
                return {"status": "skipped", "reason": str(e)}

        return {"status": "skipped", "reason": f"Unhandled signal type: {signal.signal_type}"}

    def execute_short(self, signal: Signal, size_eur: float) -> dict:
        """Short sell — for VECM pairs strategy (sell leg)."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        ticker = signal.ticker
        EUR_USD_RATE = 1.08
        size_usd = round(size_eur * EUR_USD_RATE, 2)
        mode = "PAPER" if self._is_paper else "LIVE"

        try:
            order = MarketOrderRequest(
                symbol=ticker,
                notional=size_usd,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            result = self._client.submit_order(order)
            logger.info(f"[{mode}] SHORT {ticker} ${size_usd:.2f} → order_id={result.id}")
            return {"status": "submitted", "order_id": str(result.id), "side": "short"}
        except Exception as e:
            logger.error(f"Alpaca SHORT failed for {ticker}: {e}")
            return {"status": "error", "reason": str(e)}

    def get_open_positions(self) -> list:
        try:
            return list(self._client.get_all_positions())
        except Exception as e:
            logger.error(f"Could not fetch positions: {e}")
            return []

    def get_account_balance(self) -> float:
        try:
            account = self._client.get_account()
            return float(account.equity) / 1.08  # USD → EUR approx
        except Exception as e:
            logger.error(f"Could not fetch balance: {e}")
            return 0.0

    def is_market_open(self) -> bool:
        try:
            clock = self._client.get_clock()
            return clock.is_open
        except Exception:
            return False
