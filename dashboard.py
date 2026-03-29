"""
AlgoTrading Dashboard
=====================
Self-contained testing + dashboard system.
Run: python dashboard.py
Generates: reports/dashboard.html

Phases:
  1. Data download (SPY/QQQ/IWM/XLF/XLK/XLE via yfinance, cached)
  2. Backtests (in-sample: before 2023-01-01)
  3. Forward tests (out-of-sample: 2023-01-01 to today)
  4. Current signals (live last-252-day window)
  5. Portfolio state (150 EUR)
  6. HTML dashboard (all inline, charts as base64 PNG)
"""
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows console encoding for UTF-8 output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import base64
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ─── Project imports ─────────────────────────────────────────────────────────
from config.settings import (
    PORTFOLIO_CONFIG, RSI2Config, TripleRSIConfig, PairsVECMConfig,
    DATA_CACHE_DIR, REPORTS_DIR,
)
from data.fetcher import DataFetcher
from strategies.rsi2_mean_reversion import RSI2MeanReversion
from strategies.triple_rsi import TripleRSI
from strategies.cointegration_pairs_vecm import CointegrationPairsVECM
from backtest.metrics import compute_metrics

# ─── Constants ────────────────────────────────────────────────────────────────
TRAIN_END = "2023-01-01"
FULL_START = "1993-01-01"
VECM_START = "2010-01-01"
PAIRS_UNIVERSE = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLE"]
PAIRS_TO_TEST = [("SPY", "QQQ"), ("SPY", "IWM"), ("QQQ", "IWM")]
INITIAL_CAPITAL = 150.0
CAPITAL_PER_STRATEGY = INITIAL_CAPITAL / 3  # 50 EUR each

fetcher = DataFetcher()

# ─── PHASE 1: DATA ────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: Downloading market data...")
print("=" * 60)

# Full history for RSI strategies (SPY back to 1993)
spy_full = fetcher.get_ohlcv("SPY", start=FULL_START)
qqq_full = fetcher.get_ohlcv("QQQ", start=FULL_START)
iwm_full = fetcher.get_ohlcv("IWM", start="2000-05-26")  # IWM IPO date

# Multi-close for VECM pairs universe
pairs_full = fetcher.get_multi_close(PAIRS_UNIVERSE, start=VECM_START)

today_str = datetime.now().strftime("%Y-%m-%d")
print(f"SPY: {len(spy_full)} bars ({spy_full.index[0].date()} to {spy_full.index[-1].date()})")
print(f"QQQ: {len(qqq_full)} bars")
print(f"IWM: {len(iwm_full)} bars")
print(f"Pairs universe: {pairs_full.shape}")

# Split: train = before 2023, forward = 2023 onwards
spy_train = spy_full[spy_full.index < TRAIN_END]
spy_fwd   = spy_full[spy_full.index >= TRAIN_END]
qqq_train = qqq_full[qqq_full.index < TRAIN_END]
qqq_fwd   = qqq_full[qqq_full.index >= TRAIN_END]
iwm_train = iwm_full[iwm_full.index < TRAIN_END]
iwm_fwd   = iwm_full[iwm_full.index >= TRAIN_END]
pairs_train = pairs_full[pairs_full.index < TRAIN_END]
pairs_fwd   = pairs_full[pairs_full.index >= TRAIN_END]

print(f"\nTrain split: {len(spy_train)} bars (up to {TRAIN_END})")
print(f"Forward split: {len(spy_fwd)} bars ({TRAIN_END} to today)")

# ─── PHASE 2: BACKTESTS (IN-SAMPLE) ──────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2: Running in-sample backtests (before 2023)...")
print("=" * 60)


def safe_backtest(name, fn):
    """Run a backtest function, catching all errors. Returns StrategyResult or None."""
    try:
        print(f"  Running {name}...")
        result = fn()
        m = result.metrics
        print(f"  {name}: Sharpe={m.get('sharpe', 0):.2f}  CAGR={m.get('cagr', 0):.1%}  "
              f"MaxDD={m.get('max_drawdown', 0):.1%}  WinRate={m.get('win_rate', 0):.1%}  "
              f"Trades={int(m.get('n_trades', 0))}")
        return result
    except Exception as e:
        print(f"  {name}: FAILED — {e}")
        traceback.print_exc()
        return None


# In-sample
rsi2_bt = safe_backtest(
    "RSI2 (in-sample)",
    lambda: RSI2MeanReversion(RSI2Config(instrument="SPY")).backtest(
        spy_train, initial_capital=CAPITAL_PER_STRATEGY
    ),
)

triple_bt = safe_backtest(
    "TripleRSI (in-sample)",
    lambda: TripleRSI(TripleRSIConfig(instrument="SPY")).backtest(
        spy_train, initial_capital=CAPITAL_PER_STRATEGY
    ),
)

vecm_bt = safe_backtest(
    "VECM Pairs (in-sample)",
    lambda: CointegrationPairsVECM(
        PairsVECMConfig(
            formation_window=252,
            trading_window=126,
            max_active_pairs=3,
        )
    ).backtest(pairs_train[PAIRS_UNIVERSE], initial_capital=CAPITAL_PER_STRATEGY),
)

# ─── PHASE 3: FORWARD TESTS (OUT-OF-SAMPLE) ───────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: Running forward (out-of-sample) tests (2023 → today)...")
print("=" * 60)

rsi2_fwd_bt = safe_backtest(
    "RSI2 (forward)",
    lambda: RSI2MeanReversion(RSI2Config(instrument="SPY")).backtest(
        spy_fwd, initial_capital=CAPITAL_PER_STRATEGY
    ),
)

triple_fwd_bt = safe_backtest(
    "TripleRSI (forward)",
    lambda: TripleRSI(TripleRSIConfig(instrument="SPY")).backtest(
        spy_fwd, initial_capital=CAPITAL_PER_STRATEGY
    ),
)

# For VECM forward test: we need enough history to form pairs.
# Use the last 252 days of training data + full forward period for context,
# then only trade the forward period.
if len(pairs_full) > 252:
    # Provide formation context: last 252 bars of training + all forward bars
    formation_context = pairs_full[pairs_full.index < TRAIN_END].tail(252)
    pairs_fwd_with_context = pd.concat([formation_context, pairs_fwd])
    vecm_fwd_bt = safe_backtest(
        "VECM Pairs (forward)",
        lambda: CointegrationPairsVECM(
            PairsVECMConfig(
                formation_window=252,
                trading_window=126,
                max_active_pairs=3,
            )
        ).backtest(pairs_fwd_with_context[PAIRS_UNIVERSE], initial_capital=CAPITAL_PER_STRATEGY),
    )
else:
    vecm_fwd_bt = None
    print("  VECM Pairs (forward): SKIPPED — insufficient data")

# ─── PHASE 4: CURRENT SIGNALS ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 4: Generating current signals (last 252 days)...")
print("=" * 60)

current_signals = {}

# Use full data for live signals (last 252 bars as context)
def get_current_signal(name, strategy, data):
    try:
        sigs = strategy.generate_signals(data)
        if sigs:
            s = sigs[-1]
            return {
                "signal": s.signal_type.value,
                "price": s.price,
                "confidence": s.confidence,
                "metadata": s.metadata,
            }
        # No signal = HOLD, get last close + indicators
        return {"signal": "HOLD", "price": float(data["close"].iloc[-1]) if "close" in data.columns else None,
                "confidence": 0.0, "metadata": {}}
    except Exception as e:
        print(f"  Signal error for {name}: {e}")
        return {"signal": "N/A", "price": None, "confidence": 0.0, "metadata": {}}


# RSI2 signal
rsi2_live = RSI2MeanReversion(RSI2Config(instrument="SPY"))
spy_live = spy_full.tail(300)  # enough warmup + recent
current_signals["RSI2"] = get_current_signal("RSI2", rsi2_live, spy_live)

# TripleRSI signal
triple_live = TripleRSI(TripleRSIConfig(instrument="SPY"))
current_signals["TripleRSI"] = get_current_signal("TripleRSI", triple_live, spy_live)

# VECM Pairs signal
try:
    vecm_live = CointegrationPairsVECM(PairsVECMConfig())
    pairs_live = pairs_full[PAIRS_UNIVERSE].tail(400)
    vecm_sigs = vecm_live.generate_signals(pairs_live)
    if vecm_sigs:
        sig_summary = []
        for s in vecm_sigs[:3]:
            sig_summary.append(f"{s.ticker}: {s.signal_type.value} (z={s.metadata.get('zscore', '?')})")
        current_signals["VECM Pairs"] = {
            "signal": vecm_sigs[0].signal_type.value,
            "price": vecm_sigs[0].price,
            "confidence": vecm_sigs[0].confidence,
            "metadata": {"pairs_signals": "; ".join(sig_summary)},
        }
    else:
        current_signals["VECM Pairs"] = {
            "signal": "HOLD",
            "price": None,
            "confidence": 0.0,
            "metadata": {"note": "No cointegrated pairs triggering entry"},
        }
except Exception as e:
    current_signals["VECM Pairs"] = {"signal": "N/A", "price": None, "confidence": 0.0, "metadata": {"error": str(e)}}

for name, sig in current_signals.items():
    print(f"  {name}: {sig['signal']}  conf={sig['confidence']:.2f}")

# ─── PHASE 5: PORTFOLIO STATE ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 5: Portfolio state (150 EUR)...")
print("=" * 60)

n_strategies = 3
equal_weight = 1.0 / n_strategies
max_weight = PORTFOLIO_CONFIG.max_weight_for_n(n_strategies)
max_trade_pct = PORTFOLIO_CONFIG.max_trade_size_pct

portfolio_rows = [
    {
        "strategy": "RSI2 Mean Reversion",
        "instrument": "SPY",
        "weight": equal_weight,
        "eur_allocated": INITIAL_CAPITAL * equal_weight,
        "max_trade_eur": INITIAL_CAPITAL * equal_weight * max_trade_pct,
        "max_weight": max_weight,
    },
    {
        "strategy": "Triple RSI",
        "instrument": "SPY",
        "weight": equal_weight,
        "eur_allocated": INITIAL_CAPITAL * equal_weight,
        "max_trade_eur": INITIAL_CAPITAL * equal_weight * max_trade_pct,
        "max_weight": max_weight,
    },
    {
        "strategy": "VECM Pairs",
        "instrument": "SPY/QQQ/IWM",
        "weight": equal_weight,
        "eur_allocated": INITIAL_CAPITAL * equal_weight,
        "max_trade_eur": INITIAL_CAPITAL * equal_weight * max_trade_pct,
        "max_weight": max_weight,
    },
]
for r in portfolio_rows:
    print(f"  {r['strategy']}: {r['weight']:.1%} = {r['eur_allocated']:.2f} EUR "
          f"(max trade: {r['max_trade_eur']:.2f} EUR)")

# ─── PHASE 6: HTML DASHBOARD ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6: Generating HTML dashboard...")
print("=" * 60)


# ── Helper: metrics dict → display row ───────────────────────────────────────
def fmt_metrics(result):
    if result is None:
        return {k: "N/A" for k in ["sharpe", "cagr", "max_drawdown", "win_rate", "profit_factor", "n_trades"]}
    m = result.metrics
    return {
        "sharpe": f"{m.get('sharpe', 0):.2f}",
        "cagr": f"{m.get('cagr', 0):.1%}",
        "max_drawdown": f"{m.get('max_drawdown', 0):.1%}",
        "win_rate": f"{m.get('win_rate', 0):.1%}",
        "profit_factor": f"{m.get('profit_factor', 0):.2f}" if m.get('profit_factor', 0) != float("inf") else "∞",
        "n_trades": str(int(m.get("n_trades", 0))),
    }


def metric_color(key, value_str):
    """Return a CSS class based on metric pass/fail targets."""
    try:
        v = float(value_str.replace("%", "").replace("∞", "9999"))
    except (ValueError, AttributeError):
        return "metric-na"
    thresholds = {
        "sharpe": (1.0, 0.5),      # (green, orange)
        "cagr": (0.10, 0.05),
        "max_drawdown": None,       # special: negative, lower is worse
        "win_rate": (0.55, 0.45),
        "profit_factor": (1.5, 1.0),
    }
    if key == "max_drawdown":
        if v < -25:
            return "metric-bad"
        elif v < -15:
            return "metric-warn"
        return "metric-good"
    if key in thresholds and thresholds[key]:
        hi, lo = thresholds[key]
        if v >= hi * 100 if "%" not in value_str else v >= hi * 100:
            return "metric-good"
        elif v >= lo * 100 if "%" not in value_str else v >= lo * 100:
            return "metric-warn"
        return "metric-bad"
    return "metric-na"


def metric_badge(key, value_str):
    css = metric_color(key, value_str)
    return f'<span class="{css}">{value_str}</span>'


# ── Charts: equity curves ─────────────────────────────────────────────────────
def make_equity_chart():
    """6 equity subplots: 3 in-sample + 3 forward."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    strategy_data = [
        ("RSI2", rsi2_bt, rsi2_fwd_bt),
        ("Triple RSI", triple_bt, triple_fwd_bt),
        ("VECM Pairs", vecm_bt, vecm_fwd_bt),
    ]

    for col, (label, bt_res, fwd_res) in enumerate(strategy_data):
        for row, (res, period_label, color) in enumerate([
            (bt_res, "In-Sample (1993-2022)", "#4caf50"),
            (fwd_res, "Forward (2023-Now)", "#2196f3"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="#aaa", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#333")
            ax.set_title(f"{label}\n{period_label}", color="#eee", fontsize=8, pad=4)

            if res is not None and not res.equity_curve.empty and len(res.equity_curve) > 1:
                eq = res.equity_curve
                normalized = (eq / eq.iloc[0]) * 100
                ax.plot(eq.index, normalized.values, color=color, linewidth=1.2)
                ax.axhline(100, color="#555", linewidth=0.7, linestyle="--")
                ax.set_ylabel("NAV (rebased 100)", color="#aaa", fontsize=7)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="#888", fontsize=9)

    fig.suptitle("Equity Curves — In-Sample vs Forward Test", color="white", fontsize=12, y=1.01)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_drawdown_chart():
    """Drawdown chart for all strategies, in-sample."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor("#0d1117")

    strategy_data = [
        ("RSI2 (in-sample)", rsi2_bt, "#4caf50"),
        ("Triple RSI (in-sample)", triple_bt, "#ff9800"),
        ("VECM Pairs (in-sample)", vecm_bt, "#e91e63"),
    ]

    for ax, (label, res, color) in zip(axes, strategy_data):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.set_title(label, color="#eee", fontsize=9)

        if res is not None and not res.equity_curve.empty and len(res.equity_curve) > 1:
            eq = res.equity_curve
            peak = eq.cummax()
            dd = (eq - peak) / peak * 100
            ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.5)
            ax.plot(dd.index, dd.values, color=color, linewidth=0.8)
            ax.set_ylabel("Drawdown %", color="#aaa", fontsize=7)
            ax.axhline(-25, color="red", linewidth=0.7, linestyle="--", alpha=0.7)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="#888", fontsize=9)

    fig.suptitle("Drawdown Profiles (In-Sample)", color="white", fontsize=11)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


print("  Rendering equity chart...")
equity_chart_b64 = make_equity_chart()
print("  Rendering drawdown chart...")
drawdown_chart_b64 = make_drawdown_chart()

# ── Signals HTML table ────────────────────────────────────────────────────────
def signal_badge(sig):
    if sig == "BUY":
        return '<span class="sig-buy">BUY</span>'
    elif sig == "SELL":
        return '<span class="sig-sell">SELL</span>'
    elif sig == "HOLD":
        return '<span class="sig-hold">HOLD</span>'
    return f'<span class="sig-na">{sig}</span>'


def build_signals_table():
    rows = ""
    signal_labels = {
        "RSI2": ("RSI2 Mean Reversion", "SPY", "RSI(2)"),
        "TripleRSI": ("Triple RSI", "SPY", "RSI(5)+MA200"),
        "VECM Pairs": ("VECM Pairs", "SPY/QQQ/IWM", "z-score"),
    }
    for key, (full_name, instrument, indicator) in signal_labels.items():
        sig_data = current_signals.get(key, {})
        signal = sig_data.get("signal", "N/A")
        price = sig_data.get("price")
        conf = sig_data.get("confidence", 0.0)
        meta = sig_data.get("metadata", {})
        price_str = f"${price:.2f}" if price else "—"
        conf_str = f"{conf:.0%}"
        # Extract first meaningful indicator value from metadata
        ind_val = "—"
        for k, v in meta.items():
            if k not in ("error", "note") and v:
                ind_val = f"{k}={v}" if not isinstance(v, dict) else str(list(meta.values())[0])[:40]
                break
        rows += f"""
        <tr>
          <td><strong>{full_name}</strong></td>
          <td>{instrument}</td>
          <td>{signal_badge(signal)}</td>
          <td>{price_str}</td>
          <td>{conf_str}</td>
          <td class="meta-cell">{ind_val[:60]}</td>
        </tr>"""
    return rows


def build_metrics_table(results_dict, phase_label):
    """Build an HTML metrics table from a dict of {label: StrategyResult}."""
    rows = ""
    for label, res in results_dict.items():
        m = fmt_metrics(res)
        rows += f"""
        <tr>
          <td><strong>{label}</strong></td>
          <td>{metric_badge('sharpe', m['sharpe'])}</td>
          <td>{metric_badge('cagr', m['cagr'])}</td>
          <td>{metric_badge('max_drawdown', m['max_drawdown'])}</td>
          <td>{metric_badge('win_rate', m['win_rate'])}</td>
          <td>{metric_badge('profit_factor', m['profit_factor'])}</td>
          <td>{m['n_trades']}</td>
        </tr>"""
    return rows


def build_portfolio_table():
    rows = ""
    for r in portfolio_rows:
        rows += f"""
        <tr>
          <td><strong>{r['strategy']}</strong></td>
          <td>{r['instrument']}</td>
          <td>{r['weight']:.1%}</td>
          <td>{r['eur_allocated']:.2f} EUR</td>
          <td>{r['max_trade_eur']:.2f} EUR</td>
          <td>{r['max_weight']:.1%}</td>
        </tr>"""
    return rows


def build_etoro_section():
    """Show actionable eToro instructions for BUY signals."""
    buy_signals = {
        name: sig for name, sig in current_signals.items()
        if sig.get("signal") == "BUY"
    }
    if not buy_signals:
        return """
        <div class="etoro-none">
          <p>No active BUY signals today. Monitor daily — signals trigger 3-5x per year per strategy.</p>
        </div>"""

    html = ""
    spy_price = current_signals.get("RSI2", {}).get("price") or current_signals.get("TripleRSI", {}).get("price")
    for name, sig in buy_signals.items():
        alloc = INITIAL_CAPITAL / n_strategies
        max_trade = alloc * max_trade_pct
        price = sig.get("price")
        price_str = f"${price:.2f}" if price else "market price"
        html += f"""
        <div class="etoro-action">
          <h4>ACTION: {name} → BUY Signal</h4>
          <ul>
            <li><strong>Instrument:</strong> {portfolio_rows[list(current_signals.keys()).index(name)]['instrument']}</li>
            <li><strong>Allocated capital:</strong> {alloc:.2f} EUR ({INITIAL_CAPITAL:.0f} EUR × {1/n_strategies:.0%})</li>
            <li><strong>Max trade size:</strong> {max_trade:.2f} EUR (10% of allocated)</li>
            <li><strong>Current price:</strong> {price_str}</li>
            <li><strong>Confidence:</strong> {sig.get('confidence', 0):.0%}</li>
            <li><strong>Steps on eToro:</strong>
              <ol>
                <li>Go to eToro.com → Search for the instrument</li>
                <li>Click <em>Trade</em> → Set amount to <strong>{max_trade:.0f} EUR</strong></li>
                <li>Select <em>Market Order</em> (execute at open next day)</li>
                <li>Set Stop Loss at -10% from entry (risk management)</li>
                <li>Confirm order</li>
              </ol>
            </li>
          </ul>
        </div>"""
    return html


# ── Assemble HTML ─────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

in_sample_results = {
    "RSI2 (1993-2022)": rsi2_bt,
    "Triple RSI (1993-2022)": triple_bt,
    "VECM Pairs (2010-2022)": vecm_bt,
}
fwd_results = {
    "RSI2 (2023-Now)": rsi2_fwd_bt,
    "Triple RSI (2023-Now)": triple_fwd_bt,
    "VECM Pairs (2023-Now)": vecm_fwd_bt,
}

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AlgoTrading Dashboard — {timestamp}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f0f2f5; color: #333; }}

  /* Header */
  .header {{ background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
             color: white; padding: 24px 32px; display: flex;
             justify-content: space-between; align-items: center; }}
  .header h1 {{ font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }}
  .header .meta {{ text-align: right; font-size: 0.85rem; opacity: 0.85; }}
  .portfolio-value {{ font-size: 1.3rem; font-weight: 600; color: #a5d6a7; }}

  /* Layout */
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px 24px; }}
  .section {{ background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.08);
              margin-bottom: 24px; overflow: hidden; }}
  .section-header {{ background: #1a237e; color: white; padding: 12px 20px;
                     font-weight: 600; font-size: 0.95rem; display: flex;
                     align-items: center; gap: 8px; }}
  .section-body {{ padding: 20px; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th {{ background: #f5f6fa; color: #555; font-weight: 600; text-align: left;
        padding: 10px 12px; border-bottom: 2px solid #e0e0e0; white-space: nowrap; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover {{ background: #fafafa; }}

  /* Metric badges */
  .metric-good {{ background: #e8f5e9; color: #2e7d32; padding: 3px 8px;
                  border-radius: 12px; font-weight: 600; font-size: 0.82rem; white-space: nowrap; }}
  .metric-warn {{ background: #fff3e0; color: #e65100; padding: 3px 8px;
                  border-radius: 12px; font-weight: 600; font-size: 0.82rem; white-space: nowrap; }}
  .metric-bad  {{ background: #ffebee; color: #c62828; padding: 3px 8px;
                  border-radius: 12px; font-weight: 600; font-size: 0.82rem; white-space: nowrap; }}
  .metric-na   {{ background: #f5f5f5; color: #888; padding: 3px 8px;
                  border-radius: 12px; font-size: 0.82rem; white-space: nowrap; }}

  /* Signal badges */
  .sig-buy  {{ background: #4caf50; color: white; padding: 4px 12px;
               border-radius: 14px; font-weight: 700; font-size: 0.82rem; }}
  .sig-sell {{ background: #f44336; color: white; padding: 4px 12px;
               border-radius: 14px; font-weight: 700; font-size: 0.82rem; }}
  .sig-hold {{ background: #9e9e9e; color: white; padding: 4px 12px;
               border-radius: 14px; font-weight: 700; font-size: 0.82rem; }}
  .sig-na   {{ background: #bdbdbd; color: white; padding: 4px 12px;
               border-radius: 14px; font-size: 0.82rem; }}

  /* Meta cell */
  .meta-cell {{ font-size: 0.78rem; color: #666; max-width: 220px; overflow: hidden;
                text-overflow: ellipsis; white-space: nowrap; }}

  /* Charts */
  .chart-container {{ text-align: center; padding: 8px 0; }}
  .chart-container img {{ max-width: 100%; border-radius: 6px; }}

  /* Walk-forward note */
  .wf-note {{ background: #e3f2fd; border-left: 4px solid #1976d2;
              padding: 14px 18px; border-radius: 0 6px 6px 0; margin: 0; }}
  .wf-note h4 {{ color: #1565c0; margin-bottom: 8px; font-size: 0.95rem; }}
  .wf-note p {{ color: #444; font-size: 0.87rem; line-height: 1.6; margin-bottom: 6px; }}

  /* eToro section */
  .etoro-action {{ background: #f1f8e9; border: 1px solid #aed581;
                   border-radius: 8px; padding: 16px 20px; margin-bottom: 14px; }}
  .etoro-action h4 {{ color: #33691e; margin-bottom: 10px; }}
  .etoro-action ul {{ padding-left: 18px; }}
  .etoro-action li {{ margin-bottom: 5px; font-size: 0.88rem; line-height: 1.5; }}
  .etoro-action ol {{ padding-left: 22px; margin-top: 4px; }}
  .etoro-none {{ color: #666; font-size: 0.9rem; padding: 10px 0; }}

  /* Legend */
  .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 12px; font-size: 0.8rem; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}

  /* Footer */
  .footer {{ text-align: center; color: #999; font-size: 0.78rem; padding: 20px;
             margin-top: 8px; }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div>
    <h1>AlgoTrading Dashboard</h1>
    <div style="margin-top:6px;font-size:0.82rem;opacity:0.8;">
      3 Strategies &bull; In-Sample + Out-of-Sample Validation &bull; 150 EUR Portfolio
    </div>
  </div>
  <div class="meta">
    <div>Generated: {timestamp}</div>
    <div class="portfolio-value">Portfolio: {INITIAL_CAPITAL:.0f} EUR</div>
    <div style="margin-top:4px;font-size:0.78rem;">Forward test: {TRAIN_END} → {today_str}</div>
  </div>
</div>

<div class="container">

  <!-- SECTION 1: CURRENT SIGNALS -->
  <div class="section">
    <div class="section-header">
      <span>&#128308;</span> Current Signals (as of {timestamp[:10]})
    </div>
    <div class="section-body">
      <table>
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Instrument</th>
            <th>Signal</th>
            <th>Price</th>
            <th>Confidence</th>
            <th>Indicator Values</th>
          </tr>
        </thead>
        <tbody>
          {build_signals_table()}
        </tbody>
      </table>
    </div>
  </div>

  <!-- SECTION 2: PORTFOLIO ALLOCATION -->
  <div class="section">
    <div class="section-header">
      <span>&#128200;</span> Portfolio Allocation (150 EUR)
    </div>
    <div class="section-body">
      <table>
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Instrument</th>
            <th>Weight</th>
            <th>EUR Allocated</th>
            <th>Max Trade Size</th>
            <th>Max Weight Cap</th>
          </tr>
        </thead>
        <tbody>
          {build_portfolio_table()}
        </tbody>
      </table>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:#1a237e"></div>
          Equal weight: {equal_weight:.1%} per strategy ({INITIAL_CAPITAL * equal_weight:.2f} EUR)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#e65100"></div>
          Max trade per signal: 10% of allocation = {INITIAL_CAPITAL * equal_weight * max_trade_pct:.2f} EUR</div>
        <div class="legend-item"><div class="legend-dot" style="background:#2e7d32"></div>
          Max weight cap (N=3): {max_weight:.1%}</div>
      </div>
    </div>
  </div>

  <!-- SECTION 3: WALK-FORWARD NOTE -->
  <div class="section">
    <div class="section-header">
      <span>&#128202;</span> Walk-Forward Validation Methodology
    </div>
    <div class="section-body">
      <div class="wf-note">
        <h4>In-Sample vs Out-of-Sample Split</h4>
        <p><strong>In-Sample (Training):</strong> All data before {TRAIN_END}. Strategy parameters
           were derived from this period. RSI2 and Triple RSI use 1993-2022 (30 years).
           VECM Pairs uses 2010-2022 (enough history for VECM fitting).</p>
        <p><strong>Out-of-Sample (Forward Test):</strong> All data from {TRAIN_END} to today
           ({today_str}). This is data the strategies have <em>never seen</em> — the most honest
           measure of real-world performance. No parameter changes were made after seeing this data.</p>
        <p><strong>Why this matters:</strong> In-sample metrics measure how well a strategy fits
           historical data. Forward test metrics measure whether the statistical edge persists.
           Significant degradation in the forward test indicates overfitting.</p>
        <p><strong>Targets:</strong> Sharpe &gt; 1.0, CAGR &gt; 10%, Max Drawdown &lt; 25%,
           Win Rate &gt; 55%, Profit Factor &gt; 1.5</p>
      </div>
    </div>
  </div>

  <!-- SECTION 4: IN-SAMPLE BACKTEST RESULTS -->
  <div class="section">
    <div class="section-header">
      <span>&#128293;</span> In-Sample Backtest Results (before {TRAIN_END})
    </div>
    <div class="section-body">
      <table>
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Sharpe</th>
            <th>CAGR</th>
            <th>Max Drawdown</th>
            <th>Win Rate</th>
            <th>Profit Factor</th>
            <th>Trades</th>
          </tr>
        </thead>
        <tbody>
          {build_metrics_table(in_sample_results, 'In-Sample')}
        </tbody>
      </table>
      <p style="font-size:0.78rem;color:#888;margin-top:10px;">
        Green = meets target &bull; Orange = marginal &bull; Red = below target
      </p>
    </div>
  </div>

  <!-- SECTION 5: FORWARD TEST RESULTS -->
  <div class="section">
    <div class="section-header">
      <span>&#127919;</span> Forward Test Results — OUT-OF-SAMPLE ({TRAIN_END} to {today_str})
    </div>
    <div class="section-body">
      <table>
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Sharpe</th>
            <th>CAGR</th>
            <th>Max Drawdown</th>
            <th>Win Rate</th>
            <th>Profit Factor</th>
            <th>Trades</th>
          </tr>
        </thead>
        <tbody>
          {build_metrics_table(fwd_results, 'Forward')}
        </tbody>
      </table>
      <p style="font-size:0.78rem;color:#555;margin-top:10px;">
        <strong>This is the most important table.</strong> Forward test results show performance
        on data the strategies never saw during parameter development.
        Compare with in-sample to check for overfitting.
      </p>
    </div>
  </div>

  <!-- SECTION 6: EQUITY CURVES -->
  <div class="section">
    <div class="section-header">
      <span>&#128200;</span> Equity Curves — In-Sample (top) vs Forward Test (bottom)
    </div>
    <div class="section-body">
      <div class="chart-container">
        <img src="data:image/png;base64,{equity_chart_b64}" alt="Equity Curves">
      </div>
    </div>
  </div>

  <!-- SECTION 7: DRAWDOWN CHART -->
  <div class="section">
    <div class="section-header">
      <span>&#128308;</span> Drawdown Profiles (In-Sample) — Red dashed = -25% target
    </div>
    <div class="section-body">
      <div class="chart-container">
        <img src="data:image/png;base64,{drawdown_chart_b64}" alt="Drawdown Chart">
      </div>
    </div>
  </div>

  <!-- SECTION 8: ETORO ACTIONS -->
  <div class="section">
    <div class="section-header">
      <span>&#128176;</span> eToro Actions — Based on Current Signals
    </div>
    <div class="section-body">
      {build_etoro_section()}
    </div>
  </div>

</div><!-- /container -->

<div class="footer">
  AlgoTrading Dashboard &bull; Generated {timestamp} &bull;
  Data: yfinance (free, no API key) &bull;
  Strategies: RSI2, Triple RSI, VECM Pairs &bull;
  Capital: {INITIAL_CAPITAL:.0f} EUR
</div>

</body>
</html>"""

# ─── Save HTML ────────────────────────────────────────────────────────────────
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
output_path = REPORTS_DIR / "dashboard.html"
output_path.write_text(HTML, encoding="utf-8")

print(f"\n{'=' * 60}")
print(f"Dashboard saved: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

# ─── Console summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — IN-SAMPLE METRICS")
print("=" * 60)
header = f"{'Strategy':<30} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7}"
print(header)
print("-" * len(header))
for label, res in in_sample_results.items():
    m = fmt_metrics(res)
    print(f"{label:<30} {m['sharpe']:>8} {m['cagr']:>8} {m['max_drawdown']:>8} "
          f"{m['win_rate']:>8} {m['n_trades']:>7}")

print("\n" + "=" * 60)
print("SUMMARY — FORWARD TEST METRICS (out-of-sample)")
print("=" * 60)
print(header)
print("-" * len(header))
for label, res in fwd_results.items():
    m = fmt_metrics(res)
    print(f"{label:<30} {m['sharpe']:>8} {m['cagr']:>8} {m['max_drawdown']:>8} "
          f"{m['win_rate']:>8} {m['n_trades']:>7}")

print("\n" + "=" * 60)
print("CURRENT SIGNALS")
print("=" * 60)
for name, sig in current_signals.items():
    print(f"  {name:<20} {sig['signal']:<6}  conf={sig['confidence']:.0%}")

print("\nDone. Open reports/dashboard.html in your browser.")
