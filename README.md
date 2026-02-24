# 📊 TemieStockSage AI — ML-Powered Stock Price Predictions

A complete machine learning web application that predicts daily closing prices for the 6 most popular stocks using **real market data from the Alpaca Market API** (January 2022 → present), and sends email alerts with predictions every day after market close.

---

## Quick Start

### 1. Get Free Alpaca API Keys

1. Sign up at [https://alpaca.markets](https://alpaca.markets) — completely free
2. Go to **Dashboard → Paper Trading → API Keys → Generate**
3. Copy your **API Key ID** and **Secret Key**

### 2. Set Environment Variables

```bash
export ALPACA_API_KEY="your-api-key-id"
export ALPACA_SECRET_KEY="your-secret-key"
```

### 3. Install & Run

```bash
cd stock-predictor
pip install -r requirements.txt
python run.py
```

Open **http://localhost:5000** in your browser — done!

The pipeline automatically:

1. Pulls ~750+ days of real daily OHLCV data per stock from Alpaca
2. Engineers 13 technical features from the raw data
3. Trains 3 ML models per stock and selects the best performer
4. Generates tomorrow's price predictions
5. Launches an interactive web dashboard

> **No API keys?** The app automatically falls back to realistic sample data so you can explore everything immediately.

---

## Alpaca Market Data API

This project uses Alpaca's **free** Market Data API:

| Detail             | Value                                                 |
|--------------------|-------------------------------------------------------|
| **SDK**            | `alpaca-py` — official Python SDK                     |
| **Endpoint**       | `StockHistoricalDataClient.get_stock_bars()`          |
| **Timeframe**      | 1-Day bars                                            |
| **Date Range**     | 2022-01-01 → today                                    |
| **Data Per Bar**   | Open, High, Low, Close, Volume                        |
| **Stocks**         | AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA                  |
| **Free Tier**      | 200 req/min, 15-min delayed quotes, 5+ years history  |

### Why Alpaca?

- Completely **free** — no credit card needed
- **3+ years** of daily historical data on the free plan
- Clean Python SDK with native **pandas DataFrame** output
- Used by **100,000+ developers** worldwide

---

## Machine Learning Concepts Explained

### The Problem: Supervised Regression

We're predicting a **continuous number** (tomorrow's closing price) from **labeled examples** (historical data where we know the actual outcomes). This is called supervised regression.

### Feature Engineering (13 Features)

Raw prices aren't useful alone — the patterns matter more. We compute:

| Category    | Features                         | What It Captures                   |
|-------------|----------------------------------|------------------------------------|
| Momentum    | daily_return, price_vs_ma5/10/20 | Trend direction and strength       |
| Volatility  | volatility_5/20, daily_range     | Risk level and price stability     |
| Volume      | volume_ratio                     | Unusual trading activity           |
| Lag         | return_lag_1/2/3/5               | Recent price momentum patterns     |
| Oscillator  | rsi_14                           | Overbought / oversold conditions   |

### Three Competing Algorithms

| Algorithm           | How It Works                                        | Strengths                    |
|---------------------|-----------------------------------------------------|------------------------------|
| Linear Regression   | Fits a straight line through the data               | Fast, simple, interpretable  |
| Random Forest       | 100 decision trees averaged together                | Handles complex patterns     |
| Gradient Boosting   | Trees built sequentially, each fixing prior errors  | Often the most accurate      |

### Evaluation Metrics

- **MAE** — Mean Absolute Error: average $ we're off by
- **RMSE** — Root Mean Squared Error: penalises big misses more heavily
- **R²** — Coefficient of determination: 1.0 = perfect, 0.0 = random guessing
- **MAPE** — Mean Absolute Percentage Error: error as a % of the price

---

## Email Alerts Setup

### Gmail App Password

1. Go to [Google Account → Security](https://myaccount.google.com/security)
2. Enable **2-Step Verification** (required for App Passwords)
3. Go to **App Passwords** → Generate one for "Mail"
4. Copy the 16-character code and set:

```bash
export SENDER_EMAIL="you@gmail.com"
export SENDER_PASSWORD="abcd efgh ijkl mnop"
```

Emails are sent automatically at **6:30 PM ET, Mon–Fri** after market close.

Subscribe via the web dashboard or add emails directly to `data/subscribers.json`.

---

## Project Structure

```
stock-predictor/
├── run.py                    # Main entry point — orchestrates the full pipeline
├── startup.py                # Runs during deployment (fetches data + trains)
├── data_pipeline.py          # Alpaca data fetching + feature engineering
├── train_models.py           # Model training, evaluation, comparison
├── predict_and_notify.py     # Prediction engine + email notifications
├── app.py                    # Flask web server with scheduler
├── requirements.txt          # Python dependencies
├── render.yaml               # One-click Render deployment config
├── .gitignore                # Keeps secrets & generated files out of git
├── templates/
│   └── index.html            # Dashboard UI (Instrument Sans font)
├── models/                   # Saved trained models (generated at runtime)
└── data/                     # Processed CSV data + predictions (generated at runtime)
```

---

## Run Modes

```bash
python run.py              # Full pipeline → data → train → predict → web app
python run.py --train-only # Fetch data + train models only
python run.py --predict    # Generate predictions only (models must exist)
python run.py --app-only   # Launch web app only (models must exist)
```

---

## Deploy Publicly (Render — Free)

### 1. Push to GitHub

```bash
git add .
git commit -m "StockSage AI — ML stock predictor"
git push origin master
```

### 2. Deploy on Render

1. Go to [https://render.com](https://render.com) and sign in with GitHub
2. Click **New → Web Service**
3. Connect your `stocks-predictor` repository
4. Render will auto-detect `render.yaml`, or configure manually:
   - **Build Command:** `pip install -r requirements.txt && python startup.py`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Add **Environment Variables** (Dashboard → Environment):
   - `ALPACA_API_KEY` → your Alpaca key
   - `ALPACA_SECRET_KEY` → your Alpaca secret
   - `SENDER_EMAIL` → your Gmail address (optional, for email alerts)
   - `SENDER_PASSWORD` → your Gmail App Password (optional, I didn't have to do this)
6. Click **Deploy** — your app will be live at `https://stocksage-ai.onrender.com`



---

## Customisation Ideas

- **Add more stocks** — edit the `STOCKS` dict in `data_pipeline.py`
- **Try new features** — add indicators in `engineer_features()`
- **Use deep learning** — swap scikit-learn models for LSTM / Transformers
- **Real-time updates** — use Alpaca's WebSocket streaming API
- **Deploy publicly** — use Railway, Render, or Heroku

---

## Disclaimer

Please This project is for **educational purposes only**. Stock markets are influenced by countless factors beyond historical price patterns (earnings, news, geopolitics, sentiment). Please ML models can find statistical patterns but **cannot guarantee future prices** especially mine. Never invest real money based solely on model predictions. Always consult a qualified financial advisor.
