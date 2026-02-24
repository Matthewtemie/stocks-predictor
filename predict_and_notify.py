"""
=============================================================================
STEP 3: PREDICTION ENGINE & EMAIL NOTIFICATIONS
=============================================================================

🎓 "Inference" — After training, the model makes predictions on new data.
  1. Fetch TODAY's stock features (from Alpaca or sample data)
  2. Run through the saved model
  3. Get tomorrow's predicted closing price
  4. Email the results to all subscribers

⚠️ DISCLAIMER: Stock markets are inherently unpredictable. These
predictions are for educational purposes only. Never invest based
solely on model output.
=============================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

from data_pipeline import (
    STOCKS,
    FEATURE_COLUMNS,
    get_alpaca_client,
    fetch_stock_data_alpaca,
    generate_sample_data,
    engineer_features,
)


def load_model(ticker):
    """Load a previously saved model and its scaler from disk."""
    mp = f"models/{ticker}_model.pkl"
    sp = f"models/{ticker}_scaler.pkl"
    if not os.path.exists(mp):
        return None, None
    return joblib.load(mp), joblib.load(sp)


def predict_single_stock(ticker, client=None):
    """
    🎓 PREDICTION WORKFLOW
    1. Load saved model from disk
    2. Fetch latest stock data (Alpaca or sample)
    3. Engineer the exact same 13 features used during training
    4. Scale features → feed into model → predicted return!
    5. Convert: predicted_price = today's_close × (1 + predicted_return)
    """
    model, scaler = load_model(ticker)
    if model is None:
        return None

    # Get recent data (need ~30 days for rolling window calculations)
    if client:
        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        df_raw = fetch_stock_data_alpaca(client, ticker, start_date=start)
    else:
        df_raw = generate_sample_data(ticker)

    if df_raw is None:
        return None

    # 🎓 THE FIX: grab today's REAL closing price BEFORE feature engineering.
    # engineer_features() creates next_return = shift(-1), which makes the
    # very last row NaN. Then dropna() removes it — so after engineering,
    # the "latest" row is actually YESTERDAY. By saving the real close
    # first, we use the correct current price.
    actual_current_price = float(df_raw.iloc[-1]["Close"])

    df = engineer_features(df_raw)
    latest = df.iloc[-1]  # this is yesterday (last row with complete features)

    features = pd.DataFrame([latest[FEATURE_COLUMNS]])
    features_scaled = scaler.transform(features)
    predicted_return = model.predict(features_scaled)[0]

    # Convert return → dollar price using TODAY's real close
    predicted_price = actual_current_price * (1 + predicted_return)
    change = predicted_price - actual_current_price

    return {
        "ticker": ticker,
        "stock_name": STOCKS[ticker],
        "current_price": round(actual_current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "predicted_change": round(change, 2),
        "predicted_change_pct": round(predicted_return * 100, 2),
        "direction": "UP" if predicted_return > 0 else "DOWN",
        "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rsi": round(float(latest["rsi_14"]), 1),
        "volatility": round(float(latest["volatility_20"] * 100), 2),
    }


def predict_all_stocks():
    """Generate predictions for all 6 stocks."""
    print("=" * 60)
    print("🔮 GENERATING PREDICTIONS")
    print("=" * 60)

    client = get_alpaca_client()
    predictions = {}

    for ticker in STOCKS:
        print(f"\n  Predicting {ticker}...")
        result = predict_single_stock(ticker, client)
        if result:
            predictions[ticker] = result
            print(
                f"  {ticker}: ${result['current_price']} → "
                f"${result['predicted_price']} ({result['direction']} "
                f"{result['predicted_change_pct']:+.2f}%)"
            )

    os.makedirs("data", exist_ok=True)
    with open("data/latest_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    print(f"\n {len(predictions)} predictions saved")
    return predictions


# ═══════════════════════════════════════════════════════════════
# EMAIL NOTIFICATION SYSTEM
# ═══════════════════════════════════════════════════════════════


def build_email_html(predictions):
    """Build a styled HTML email with all predictions."""
    date_str = datetime.now().strftime("%B %d, %Y")

    rows = ""
    for _, p in predictions.items():
        color = "#10b981" if p["predicted_change"] > 0 else "#ef4444"
        arrow = "▲" if p["predicted_change"] > 0 else "▼"
        rows += f"""
        <tr>
          <td style="padding:12px 16px;border-bottom:1px solid #1e293b">
            <strong style="color:#f8fafc">{p['ticker']}</strong><br>
            <span style="color:#94a3b8;font-size:13px">{p['stock_name']}</span>
          </td>
          <td style="padding:12px 16px;border-bottom:1px solid #1e293b;
              color:#f8fafc;font-family:monospace">${p['current_price']:.2f}</td>
          <td style="padding:12px 16px;border-bottom:1px solid #1e293b">
            <strong style="color:{color};font-family:monospace">
              ${p['predicted_price']:.2f}
            </strong>
          </td>
          <td style="padding:12px 16px;border-bottom:1px solid #1e293b">
            <span style="color:{color};font-weight:bold;font-family:monospace">
              {arrow} {p['predicted_change_pct']:+.2f}%
            </span>
          </td>
          <td style="padding:12px 16px;border-bottom:1px solid #1e293b;
              color:#94a3b8;font-family:monospace">{p['rsi']:.0f}</td>
        </tr>"""

    return f"""
    <html>
    <body style="margin:0;padding:0;background:#0f172a;
                 font-family:'Instrument Sans',Arial,sans-serif">
    <div style="max-width:640px;margin:0 auto;padding:24px">
      <div style="text-align:center;padding:32px 0">
        <h1 style="color:#f8fafc;margin:0;font-size:28px">
           StockSage AI</h1>
        <p style="color:#64748b;margin:8px 0 0">
          Daily Price Predictions — {date_str}</p>
      </div>
      <div style="background:#1e293b;border-radius:12px;overflow:hidden">
        <table style="width:100%;border-collapse:collapse;font-size:14px">
          <thead><tr style="background:#334155">
            <th style="padding:12px 16px;text-align:left;color:#94a3b8">Stock</th>
            <th style="padding:12px 16px;text-align:left;color:#94a3b8">Current</th>
            <th style="padding:12px 16px;text-align:left;color:#94a3b8">Predicted</th>
            <th style="padding:12px 16px;text-align:left;color:#94a3b8">Change</th>
            <th style="padding:12px 16px;text-align:left;color:#94a3b8">RSI</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
      <div style="margin-top:24px;padding:16px;background:#1e293b;
                  border-radius:8px;border-left:4px solid #f59e0b">
        <p style="color:#f59e0b;margin:0 0 4px;font-weight:bold;
                  font-size:13px"> Disclaimer</p>
        <p style="color:#94a3b8;margin:0;font-size:12px;line-height:1.5">
          These predictions are for educational purposes only. Stock markets
          are inherently unpredictable. Do NOT make investment decisions
          based solely on these predictions.</p>
      </div>
      <p style="text-align:center;color:#475569;font-size:12px;
                margin-top:24px">
        StockSage AI — Powered by Alpaca Market Data &amp; scikit-learn
      </p>
    </div></body></html>"""


def send_email(
    predictions, recipient_email, sender_email=None, sender_password=None
):
    """
    🎓 SEND EMAIL VIA GMAIL SMTP

    Setup:
      1. Enable 2-Step Verification on your Google Account
      2. Generate an App Password
         (Google Account → Security → App Passwords)
      3. Set env vars:
           export SENDER_EMAIL="you@gmail.com"
           export SENDER_PASSWORD="abcd efgh ijkl mnop"
    """
    sender = sender_email or os.environ.get("SENDER_EMAIL")
    password = sender_password or os.environ.get("SENDER_PASSWORD")

    if not sender or not password:
        print(
            "   Email not configured. "
            "Set SENDER_EMAIL & SENDER_PASSWORD env vars."
        )
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = (
            f" StockSage: Daily Predictions — "
            f"{datetime.now().strftime('%b %d, %Y')}"
        )
        msg["From"] = sender
        msg["To"] = recipient_email
        msg.attach(MIMEText(build_email_html(predictions), "html"))

        # Try port 587 (STARTTLS) first — works on most networks.
        # Falls back to port 465 (SSL) if 587 fails.
        try:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender, password)
                server.send_message(msg)
        except Exception:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
                server.login(sender, password)
                server.send_message(msg)

        print(f"  ✅ Email sent to {recipient_email}")
        return True
    except Exception as e:
        print(f"   Email failed: {e}")
        return False


def daily_job(recipient_email=None):
    """Run daily: predict all stocks + email subscribers."""
    print(f"\n DAILY JOB — {datetime.now()}")
    preds = predict_all_stocks()
    if preds and recipient_email:
        send_email(preds, recipient_email)
    return preds


if __name__ == "__main__":
    predictions = predict_all_stocks()
    print("\n RESULTS:")
    for _, p in predictions.items():
        print(
            f"  {p['direction']} {p['ticker']}: "
            f"${p['current_price']} → ${p['predicted_price']} "
            f"({p['predicted_change_pct']:+.2f}%)"
        )