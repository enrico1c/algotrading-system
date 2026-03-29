"""
eToro Execution Layer
======================
Source: Finance/Algo-Trading-Guide/05-eToro-Connection.md (Obsidian vault)

Requires:
  - eToro developer API key (apply at: https://api-portal.etoro.com)
  - Set environment variable: ETORO_API_KEY=your_key_here
  - Install: pip install git+https://github.com/mkhaled87/etoro-api.git

Until API key is obtained, the system runs in SIGNAL_ONLY mode automatically.
This module is a drop-in replacement once credentials are available.
"""
from __future__ import annotations

from typing import Optional

from config.settings import ExecutionConfig
from execution.base import ExecutionBase
from strategies.base import Signal, SignalType
from utils.logger import get_logger

logger = get_logger("execution.etoro")


class EToroExecutor(ExecutionBase):
    """
    eToro REST API execution via the etoro-api Python client.
    Vault reference: Finance/Algo-Trading-Guide/05-eToro-Connection.md

    IMPORTANT NOTES from vault:
    - eToro uses numeric instrument IDs (not ticker symbols)
    - CFD-based for most non-US assets — check fee structure
    - Rate limits apply — add sleep between rapid calls
    - Always test with paper account first
    - Set leverage=1 until strategy is validated
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self._client = None
        self._trading_api = None
        self._instrument_map: dict = {}
        self._connect()

    def _connect(self) -> None:
        """Initialize eToro API connection."""
        if not self.config.etoro_api_key:
            raise ValueError(
                "ETORO_API_KEY not set. Apply at https://api-portal.etoro.com\n"
                "Set environment variable: export ETORO_API_KEY=your_key"
            )
        try:
            from etoro_api import ApiClient, Configuration
            from etoro_api.api import trading_api, market_data_api

            cfg = Configuration(
                host=self.config.etoro_host,
                api_key={"x-token": self.config.etoro_api_key},
            )
            self._client = ApiClient(cfg)
            self._trading_api = trading_api.TradingApi(self._client)
            self._market_api = market_data_api.MarketDataApi(self._client)
            self._build_instrument_map()
            logger.info("eToro API connected successfully")

        except ImportError:
            raise ImportError(
                "etoro-api package not installed.\n"
                "Install: pip install git+https://github.com/mkhaled87/etoro-api.git"
            )

    def _build_instrument_map(self) -> None:
        """Fetch all instruments and build ticker → instrument_id mapping."""
        try:
            instruments = self._market_api.get_instruments()
            self._instrument_map = {
                inst.symbol_full: inst.instrument_id
                for inst in instruments
            }
            logger.info(f"Loaded {len(self._instrument_map)} eToro instruments")
        except Exception as e:
            logger.warning(f"Could not load instrument map: {e}")

    def execute(self, signal: Signal, size_eur: float) -> dict:
        """Route signal to eToro order."""
        from etoro_api.models import OpenTradeRequest

        instrument_id = self._instrument_map.get(signal.ticker)
        if instrument_id is None:
            logger.error(f"Instrument ID not found for {signal.ticker}")
            return {"status": "error", "reason": f"Unknown ticker: {signal.ticker}"}

        if signal.signal_type == SignalType.BUY:
            try:
                order = OpenTradeRequest(
                    instrument_id=instrument_id,
                    direction="buy",
                    amount=size_eur,   # in EUR; eToro converts to USD internally
                    leverage=1,         # always x1 until strategy validated
                    stop_loss_rate=None,
                    take_profit_rate=None,
                )
                response = self._trading_api.open_trade(open_trade_request=order)
                logger.info(f"eToro BUY order placed: {signal.ticker} {size_eur:.2f} EUR → {response}")
                return {"status": "filled", "order": str(response)}
            except Exception as e:
                logger.error(f"eToro BUY failed: {e}")
                return {"status": "error", "reason": str(e)}

        elif signal.signal_type in (SignalType.SELL, SignalType.CLOSE):
            try:
                # Close all open positions for this instrument
                positions = self._trading_api.get_open_trades()
                closed = 0
                for pos in positions:
                    if pos.instrument_id == instrument_id:
                        self._trading_api.close_trade(pos.position_id)
                        closed += 1
                logger.info(f"eToro CLOSE: {signal.ticker} — {closed} position(s) closed")
                return {"status": "closed", "positions_closed": closed}
            except Exception as e:
                logger.error(f"eToro CLOSE failed: {e}")
                return {"status": "error", "reason": str(e)}

        return {"status": "skipped", "reason": f"Signal type {signal.signal_type} not handled"}

    def get_open_positions(self) -> list:
        try:
            return list(self._trading_api.get_open_trades())
        except Exception as e:
            logger.error(f"Could not fetch open positions: {e}")
            return []

    def get_account_balance(self) -> float:
        try:
            account = self._trading_api.get_account()
            return float(account.balance)
        except Exception as e:
            logger.error(f"Could not fetch account balance: {e}")
            return 0.0
