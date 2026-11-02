"""
Strategy backtester for ICICI Bank (NSE) using the local ICICIBANK.csv file.
Strategies:
  1. Moving Average Crossover (trend-following)
  2. RSI Mean-Reversion
  3. "both" -> runs both and compares them with Buy & Hold

Usage:
    python backtest.py

Edit the CONFIG block to change file path, dates, strategy and parameters.
Requires: pandas, numpy, matplotlib
    pip install pandas numpy matplotlib

Data notes (handled automatically below):
  * The CSV has UNADJUSTED prices. ICICI Bank had a 5:1 stock split (4-Dec-2014)
    and a 1:10 bonus issue (20-Jun-2017). Without adjusting, the backtest would
    see a fake -80% crash and a fake -9% drop. Prices before those dates are
    back-adjusted.
  * Data runs 2000-01-03 to 2021-04-30 (the old 2022-2025 dates had no data).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== CONFIG ==============================
CSV_PATH = "ICICIBANK.csv"
TICKER = "ICICIBANK"
START = "2000-01-01"
END = "2021-04-30"

STRATEGY = "both"          # "ma_crossover", "rsi_meanrev" or "both"

# MA crossover params
FAST_MA = 20
SLOW_MA = 50

# RSI mean-reversion params
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 55        # exit level (mean-reversion target)

# Trading costs
COST_PER_TRADE = 0.0007    # 0.07% per side (slippage + brokerage), on position changes
RISK_FREE_RATE = 0.065     # annual, for Sharpe

TRADING_DAYS = 252

# Corporate actions: (ex-date, price divisor applied to all earlier prices)
CORPORATE_ACTIONS = [
    ("2014-12-04", 5.0),        # 5:1 stock split (face value Rs10 -> Rs2)
    ("2017-06-20", 11 / 10),    # 1:10 bonus issue
]
# ======================================================================


def load_data(path, start, end):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[cols].apply(pd.to_numeric, errors="coerce").astype(float).dropna(subset=["Close"])

    # Back-adjust prices for splits/bonus; scale volume the opposite way
    for ex_date, factor in CORPORATE_ACTIONS:
        mask = df.index < pd.Timestamp(ex_date)
        df.loc[mask, ["Open", "High", "Low", "Close"]] /= factor
        df.loc[mask, "Volume"] *= factor

    # Sanity check: flag any remaining suspicious single-day moves
    big = df["Close"].pct_change().abs()
    big = big[big > 0.25]
    if len(big):
        print("WARNING: daily moves >25% remain (check for unadjusted corporate actions):")
        print(big.round(3).to_string())

    df = df.loc[start:end]
    if df.empty:
        raise ValueError("No data in the selected date range.")
    return df


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def generate_signal_ma_crossover(df, fast, slow):
    df = df.copy()
    df["fast_ma"] = df["Close"].rolling(fast).mean()
    df["slow_ma"] = df["Close"].rolling(slow).mean()
    df["signal"] = (df["fast_ma"] > df["slow_ma"]).astype(int)
    return df


def generate_signal_rsi_meanrev(df, period, oversold, overbought):
    df = df.copy()
    df["rsi"] = compute_rsi(df["Close"], period)
    signal = np.zeros(len(df))
    position = 0
    rsi_vals = df["rsi"].values
    for i in range(len(df)):
        if not np.isnan(rsi_vals[i]):
            if position == 0 and rsi_vals[i] < oversold:
                position = 1
            elif position == 1 and rsi_vals[i] > overbought:
                position = 0
        signal[i] = position
    df["signal"] = signal
    return df


def run_backtest(df, cost_per_trade):
    df = df.copy()
    # Signal at close of day t is traded on day t+1 (no lookahead)
    df["position"] = df["signal"].shift(1).fillna(0)
    df["daily_return"] = df["Close"].pct_change().fillna(0)
    df["trade"] = df["position"].diff().abs()
    df.iloc[0, df.columns.get_loc("trade")] = abs(df["position"].iloc[0])
    df["cost"] = df["trade"] * cost_per_trade
    df["strategy_return"] = df["position"] * df["daily_return"] - df["cost"]
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    df["buy_hold_equity"] = (1 + df["daily_return"]).cumprod()
    return df


def _metrics(ret, equity, position, risk_free_rate, trading_days):
    years = len(ret) / trading_days
    total_return = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1
    excess = ret - risk_free_rate / trading_days
    sharpe = excess.mean() / excess.std() * np.sqrt(trading_days) if excess.std() else np.nan
    max_dd = (equity / equity.cummax() - 1).min()

    out = {"Total Return": total_return, "CAGR": cagr, "Sharpe": sharpe, "Max Drawdown": max_dd}

    if position is not None:
        run_id = (position != position.shift(1)).cumsum()
        in_mkt = position == 1
        trade_rets = (1 + ret[in_mkt]).groupby(run_id[in_mkt]).prod() - 1
        out["Win Rate"] = (trade_rets > 0).mean() if len(trade_rets) else np.nan
        out["Trades"] = len(trade_rets)
        out["Time in Market"] = in_mkt.mean()
    return out


def compute_metrics(df, risk_free_rate, trading_days):
    strat = _metrics(df["strategy_return"], df["strategy_equity"], df["position"],
                     risk_free_rate, trading_days)
    bh = _metrics(df["daily_return"], df["buy_hold_equity"], None,
                  risk_free_rate, trading_days)
    return strat, bh


def fmt(k, v):
    if isinstance(v, (int, np.integer)):
        return str(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.2f}" if k == "Sharpe" else f"{v:.2%}"


def print_results(name, df, strat, bh):
    print(f"\n=== {name} on {TICKER} ===")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}  ({len(df)} trading days)")
    print(f"{'Metric':>16} | {'Strategy':>10} | {'Buy & Hold':>10}")
    for k, v in strat.items():
        print(f"{k:>16} | {fmt(k, v):>10} | {fmt(k, bh.get(k, np.nan)) if k in bh else '':>10}")


def plot_equity_curves(results, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, df in results.items():
        ax.plot(df.index, df["strategy_equity"], label=name, linewidth=1.6)
    first = next(iter(results.values()))
    ax.plot(first.index, first["buy_hold_equity"], label="Buy & Hold", linewidth=1.4,
            alpha=0.75, color="black")
    ax.set_yscale("log")
    ax.set_title(f"Equity Curves on {TICKER} (split/bonus adjusted)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of Rs 1 (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved equity curve chart to {out_path}")


def main():
    print(f"Loading {CSV_PATH} ...")
    raw = load_data(CSV_PATH, START, END)

    strategies = {}
    if STRATEGY in ("ma_crossover", "both"):
        strategies[f"MA Crossover ({FAST_MA}/{SLOW_MA})"] = generate_signal_ma_crossover(
            raw, FAST_MA, SLOW_MA)
    if STRATEGY in ("rsi_meanrev", "both"):
        strategies[f"RSI Mean-Rev ({RSI_PERIOD}, {RSI_OVERSOLD}/{RSI_OVERBOUGHT})"] = \
            generate_signal_rsi_meanrev(raw, RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT)
    if not strategies:
        raise ValueError("STRATEGY must be 'ma_crossover', 'rsi_meanrev' or 'both'")

    results = {}
    for name, sig_df in strategies.items():
        res = run_backtest(sig_df, COST_PER_TRADE)
        strat, bh = compute_metrics(res, RISK_FREE_RATE, TRADING_DAYS)
        print_results(name, res, strat, bh)
        results[name] = res
        safe = name.split(" (")[0].replace(" ", "_").replace("-", "").lower()
        res.to_csv(f"backtest_results_{safe}.csv")

    plot_equity_curves(results, "equity_curve.png")


if __name__ == "__main__":
    main()
