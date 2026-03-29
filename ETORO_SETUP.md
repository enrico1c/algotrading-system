# eToro Connection Guide

## Current Status: Signal-Only Mode

By default the system runs in **signal-only mode**:
- Signals are printed to the console with step-by-step eToro instructions
- All signals are logged to `reports/signals_log.json`
- No automated trades are placed

This is the recommended mode until:
1. You have validated the system on real historical data (backtest)
2. You have paper-traded for at least 1–3 months (forward test)
3. You have obtained an eToro developer API key

---

## Step 1 — Apply for eToro Developer API Access

1. Create a verified eToro account at [etoro.com](https://etoro.com) (ID verification required)
2. Apply for API access at **[api-portal.etoro.com](https://api-portal.etoro.com)**
3. Wait for approval — eToro reviews applications manually (days to weeks)
4. You will receive: API key + secret (`x-token` / `x-csrf` format)

---

## Step 2 — Set Your API Key

**Never hardcode credentials in source files.**

Set as environment variable:

```bash
# Windows (PowerShell)
$env:ETORO_API_KEY = "your_api_key_here"

# Windows (Command Prompt)
set ETORO_API_KEY=your_api_key_here

# Linux / macOS
export ETORO_API_KEY=your_api_key_here
```

Or create a `.env` file in the project root (already in `.gitignore`):

```
ETORO_API_KEY=your_api_key_here
```

Then load it in Python before running:
```bash
pip install python-dotenv
```

And add to `main.py` (top of file):
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Step 3 — Install the eToro Python Client

```bash
pip install git+https://github.com/mkhaled87/etoro-api.git
```

---

## Step 4 — Enable eToro Execution Mode

In `config/settings.py`, change:

```python
@dataclass
class ExecutionConfig:
    mode: str = "etoro"   # was "signal_only"
```

Or pass it at runtime (add support in `main.py` if needed):
```bash
python main.py forward --mode etoro
```

---

## Step 5 — Test the Connection

```python
# Run this quick connection test:
import os
from etoro_api import ApiClient, Configuration
from etoro_api.api import market_data_api

config = Configuration(
    host="https://api.etoro.com",
    api_key={"x-token": os.environ["ETORO_API_KEY"]}
)

with ApiClient(config) as client:
    market = market_data_api.MarketDataApi(client)
    instruments = market.get_instruments()
    print(f"Connected! {len(instruments)} instruments available.")

    # Find SPY instrument ID
    spy = next((i for i in instruments if i.symbol_full == "SPY"), None)
    print(f"SPY instrument ID: {spy.instrument_id if spy else 'NOT FOUND'}")
```

---

## eToro-Specific Notes

| Note | Detail |
|---|---|
| **Instrument IDs** | eToro uses numeric IDs, not ticker symbols. The system auto-maps them via `_build_instrument_map()` |
| **CFDs** | Most non-US assets are CFD-based on eToro — check the fee structure before trading |
| **Leverage** | Always use `leverage=1` (no leverage) until strategies are fully validated |
| **Min order** | ~$50 USD minimum order size on eToro |
| **Rate limits** | Add `time.sleep(0.5)` between rapid API calls |
| **EUR → USD** | eToro may require amounts in USD — check your account currency settings |

---

## Order Flow (once connected)

```
yfinance (free data)
    ↓
Strategy.generate_signals()
    ↓
RiskManager.check_trade()
  → Size check (≤10% of strategy capital)
  → Drawdown check (kill-switch if DD > 20%)
  → Confidence check
    ↓
EToroExecutor.execute()
  → Looks up instrument ID
  → Places OpenTradeRequest via REST API
  → Logs result
    ↓
PortfolioAllocator.update_strategy_capital()
  → Compounds capital
  → Triggers rebalance if weights drift > 10%
```

---

## Troubleshooting

**"ETORO_API_KEY not set"**
→ Set the environment variable as shown in Step 2.

**"Instrument ID not found for SPY"**
→ Run the connection test above to see available instruments. SPY may be listed as `SPY` or `SPY.US`.

**"etoro-api package not installed"**
→ Run: `pip install git+https://github.com/mkhaled87/etoro-api.git`

**API returns 401 Unauthorized**
→ Your key may have expired or not been approved yet. Check api-portal.etoro.com.
