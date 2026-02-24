"""
=============================================================================
STEP 1: DATA PIPELINE — Fetching Stock Data via Alpaca Market API
=============================================================================



I used the Alpaca Market Data API to pull REAL historical stock data:
  - Daily OHLCV bars (Open, High, Low, Close, Volume)
  - From January 2022 to present (~3+ years of data)
  - For the 6 most popular stocks

Then we "engineer" features — turning raw prices into meaningful
patterns that the ML model can learn from.

ALPACA SETUP:
  1. Sign up at https://alpaca.markets (free account)
  2. Go to Dashboard → Paper Trading → API Keys → Generate
  3. Set environment variables:
       export ALPACA_API_KEY="your-api-key-id"
       export ALPACA_SECRET_KEY="your-secret-key"
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# The 6 most popular stocks by trading volume & market cap
STOCKS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
}

# Date range for historical data
DATA_START = "2022-01-01"  # Pull from January 2022
DATA_END = None  # None = up to today


# ═══════════════════════════════════════════════════════════════
# ALPACA DATA FETCHING
# ═══════════════════════════════════════════════════════════════


def get_alpaca_client(api_key=None, secret_key=None):
    """
    🎓 CREATE AN ALPACA API CLIENT

    The Alpaca SDK uses a client object to communicate with their servers.
    You authenticate with an API key + secret key (free account).

    Free tier includes:
      - 5+ years of daily historical bar data
      - 200 requests per minute
      - 15-min delayed real-time quotes
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
    except ImportError as e:
        print(f"   alpaca-py import failed: {e}")
        print("      Try: pip install alpaca-py")
        return None
    except Exception as e:
        print(f"   alpaca-py error: {e}")
        return None

    key = api_key or os.environ.get("ALPACA_API_KEY")
    secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")

    if not key or not secret:
        print("   No Alpaca API keys found.")
        print("      Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars,")
        print("      or sign up free at https://alpaca.markets")
        print("      → Falling back to sample data for demo...\n")
        return None

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    print("  Alpaca client authenticated successfully")
    return client


def fetch_stock_data_alpaca(client, ticker, start_date=DATA_START, end_date=DATA_END):
    """
    🎓 FETCH HISTORICAL BAR DATA FROM ALPACA

    "Bars" (a.k.a. candles) are OHLCV data for a time period:
      O = Open   — price when the market opened (9:30 AM ET)
      H = High   — highest price during the day
      L = Low    — lowest price during the day
      C = Close  — price when the market closed (4:00 PM ET)
      V = Volume — total number of shares traded

    We request daily (1Day) bars from 2022-01-01 to present.
    This gives us ~750+ data points per stock — plenty for ML training.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    print(
        f"  Fetching {ticker} from Alpaca API "
        f"({start_date} → {'today' if not end_date else end_date})..."
    )

    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day,
            start=datetime.strptime(start_date, "%Y-%m-%d"),
            end=datetime.strptime(end_date, "%Y-%m-%d") if end_date else None,
        )

        bars = client.get_stock_bars(request_params)
        df = bars.df  # Multi-index DataFrame (symbol, timestamp)

        if df.empty:
            print(f"  No data returned for {ticker}")
            return None

        # Select this ticker from the multi-index
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[ticker]

        # Rename Alpaca columns → our standard names
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Clean up the datetime index
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.dropna().sort_index()

        print(
            f"  ✅ Got {len(df)} trading days for {ticker} "
            f"({df.index[0].strftime('%Y-%m-%d')} → "
            f"{df.index[-1].strftime('%Y-%m-%d')})"
        )
        return df

    except Exception as e:
        print(f"  ❌ Error fetching {ticker}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# SAMPLE DATA FALLBACK (when Alpaca API is unavailable)
# ═══════════════════════════════════════════════════════════════

# Realistic price profiles loosely based on actual 2022-2026 trajectories
STOCK_PROFILES = {
    "AAPL": {"start": 182, "end": 245, "vol": 0.017, "avg_vol": 55_000_000},
    "MSFT": {"start": 336, "end": 420, "vol": 0.016, "avg_vol": 22_000_000},
    "GOOGL": {"start": 145, "end": 192, "vol": 0.019, "avg_vol": 25_000_000},
    "AMZN": {"start": 170, "end": 230, "vol": 0.021, "avg_vol": 45_000_000},
    "NVDA": {"start": 75, "end": 138, "vol": 0.032, "avg_vol": 40_000_000},
    "TSLA": {"start": 350, "end": 360, "vol": 0.035, "avg_vol": 95_000_000},
}


def generate_sample_data(ticker, start_date=DATA_START):
    """
    🎓 GEOMETRIC BROWNIAN MOTION (GBM)
    When the Alpaca API is unavailable we generate realistic sample data
    using the same math behind the famous Black-Scholes options pricing
    model:
        dS = μ·S·dt + σ·S·dW
    Where:
      S  = stock price
      μ  = drift (long-term average return)
      σ  = volatility (how much the price jumps around)
      dW = random Brownian motion (daily shock)
    """
    profile = STOCK_PROFILES.get(ticker)
    if not profile:
        return None

    np.random.seed(abs(hash(ticker)) % (2**31))
    dates = pd.bdate_range(start=start_date, end=datetime.now() - timedelta(days=1))
    n = len(dates)

    total_return = np.log(profile["end"] / profile["start"])
    drift = total_return / n
    returns = np.random.normal(drift, profile["vol"], n)

    # Add slight autocorrelation (momentum) — 
    for i in range(1, n):
        returns[i] += 0.03 * returns[i - 1]

    log_prices = np.cumsum(returns) + np.log(profile["start"])
    close = np.exp(log_prices)

    rng = np.abs(np.random.normal(0, profile["vol"] * 0.6, n))
    high = close * (1 + rng)
    low = close * (1 - rng)
    opn = np.clip(
        close * (1 + np.random.normal(0, profile["vol"] * 0.2, n)), low, high
    )
    vol = np.random.lognormal(np.log(profile["avg_vol"]), 0.3, n).astype(int)

    df = pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates[:n],
    )
    df.index.name = "timestamp"
    print(
        f"  📊 Generated {n} sample days for {ticker} "
        f"(${close[0]:.2f} → ${close[-1]:.2f})"
    )
    return df


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

# These are the 13 columns our model uses as INPUT
FEATURE_COLUMNS = [
    "daily_return",
    "price_vs_ma5",
    "price_vs_ma10",
    "price_vs_ma20",
    "volatility_5",
    "volatility_20",
    "daily_range",
    "volume_ratio",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "rsi_14",
]


def engineer_features(df):
    """
    🎓 FEATURE ENGINEERING — The Most Important Step in ML!

    We transform raw OHLCV data into 13 meaningful features:

    MOMENTUM (4 features):
      daily_return     — % change from yesterday
      price_vs_ma5     — price position relative to 5-day moving average
      price_vs_ma10    — price position relative to 10-day moving average
      price_vs_ma20    — price position relative to 20-day moving average

    VOLATILITY (3 features):
      volatility_5     — 5-day rolling standard deviation of returns
      volatility_20    — 20-day rolling standard deviation of returns
      daily_range      — (High − Low) / Close for each day

    VOLUME (1 feature):
      volume_ratio     — today's volume divided by 5-day average volume

    LAGGED RETURNS (4 features):
      return_lag_1/2/3/5  — returns from 1, 2, 3, and 5 days ago

    OSCILLATOR (1 feature):
      rsi_14           — Relative Strength Index (14-day window)

    TARGET:
      next_return      — tomorrow's percentage return (what we predict)
    """
    df = df.copy()

    # ── Momentum ───────────────────────────────────────────────
    df["daily_return"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["price_vs_ma5"] = df["Close"] / df["ma_5"] - 1
    df["price_vs_ma10"] = df["Close"] / df["ma_10"] - 1
    df["price_vs_ma20"] = df["Close"] / df["ma_20"] - 1

    # ── Volatility ─────────────────────────────────────────────
    df["volatility_5"] = df["daily_return"].rolling(5).std()
    df["volatility_20"] = df["daily_return"].rolling(20).std()
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]

    # ── Volume ─────────────────────────────────────────────────
    df["volume_ma5"] = df["Volume"].rolling(5).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma5"]

    # ── Lagged Returns ─────────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)

    # ── RSI (14-day) ───────────────────────────────────────────
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / loss))

    # ── Target: tomorrow's RETURN ─────────────────────────────
    # 🎓 WHY PREDICT RETURNS INSTEAD OF RAW PRICES?
    #
    # Raw prices trend over time (NVDA went $30 → $190 in 4 years).
    # Our 13 features are all normalised (returns, ratios, %).
    # They can't extrapolate a trending price — but they CAN learn
    # patterns in daily returns like +0.5% or −1.2%.
    #
    # This is exactly how professional quant models work.
    # We convert back to dollars during inference:
    #   predicted_price = current_price × (1 + predicted_return)
    df["next_close"] = df["Close"].shift(-1)
    df["next_return"] = (df["next_close"] - df["Close"]) / df["Close"]
    df["price_direction"] = (df["next_return"] > 0).astype(int)

    df = df.dropna()
    return df


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════


def prepare_all_stocks(api_key=None, secret_key=None):
    """
    Run the full data pipeline for all 6 stocks.
    Tries Alpaca first; falls back to sample data if keys are missing.
    """
    print("=" * 60)
    print("STOCK DATA PIPELINE (Alpaca Market API)")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)
    client = get_alpaca_client(api_key, secret_key)
    use_alpaca = client is not None
    source_label = "Alpaca Market API" if use_alpaca else "Sample data (GBM)"

    print(f"\n Data source: {source_label}")
    print(f" Date range: {DATA_START} → today\n")

    all_data = {}
    summary = {}

    for ticker, name in STOCKS.items():
        print(f"{'─' * 45}")
        print(f"  {name} ({ticker})")

        raw = (
            fetch_stock_data_alpaca(client, ticker)
            if use_alpaca
            else generate_sample_data(ticker)
        )
        if raw is None:
            continue

        featured = engineer_features(raw)
        featured.to_csv(f"data/{ticker}_featured.csv")
        all_data[ticker] = featured

        latest = featured.iloc[-1]
        summary[ticker] = {
            "name": name,
            "latest_close": round(float(latest["Close"]), 2),
            "total_samples": len(featured),
            "date_range": (
                f"{featured.index[0].strftime('%Y-%m-%d')} → "
                f"{featured.index[-1].strftime('%Y-%m-%d')}"
            ),
            "avg_daily_return": round(
                float(featured["daily_return"].mean() * 100), 4
            ),
            "volatility": round(float(featured["volatility_20"].iloc[-1] * 100), 2),
            "data_source": source_label,
        }
        print(
            f"   {len(featured)} samples | Latest: ${latest['Close']:.2f} | "
            f"13 features engineered\n"
        )

    with open("data/data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'=' * 60}")
    print(f" Pipeline complete! {len(all_data)} stocks | Source: {source_label}")
    print(f"{'=' * 60}")
    return all_data, summary


if __name__ == "__main__":
    all_data, summary = prepare_all_stocks()
    print("\n SUMMARY:")
    for t, s in summary.items():
        print(
            f"  {t}: ${s['latest_close']} | {s['total_samples']} samples | "
            f"{s['data_source']}"
        )