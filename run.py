"""
🚀 StockSage AI — Main Runner

Usage:
  python run.py              → Full pipeline (data → train → predict → web app)
  python run.py --train-only → Just fetch data + train models
  python run.py --predict    → Just generate predictions
  python run.py --app-only   → Just launch web app (models must already exist)

Environment variables:
  ALPACA_API_KEY     → Your Alpaca API Key ID   (free at alpaca.markets)
  ALPACA_SECRET_KEY  → Your Alpaca Secret Key
  SENDER_EMAIL       → Gmail address for sending daily alerts
  SENDER_PASSWORD    → Gmail App Password (NOT your regular password)
"""

import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--full"

    print(
        """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║   📊  S T O C K S A G E   A I                        ║
    ║   ─────────────────────────────                       ║
    ║   ML-Powered Stock Price Predictions                  ║
    ║   Data: Alpaca Market API (2022 → today)              ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    )

    if mode in ("--full", "--train-only"):
        print("STEP 1: Fetching data & engineering features...")
        from data_pipeline import prepare_all_stocks

        prepare_all_stocks()

        print("\n STEP 2: Training ML models...")
        from train_models import train_all_models

        train_all_models()

        if mode == "--train-only":
            print(
                "\n Done! Run `python run.py --app-only` to start the web app."
            )
            return

    if mode in ("--full", "--predict"):
        print("\n STEP 3: Generating predictions...")
        from predict_and_notify import predict_all_stocks

        predict_all_stocks()
        if mode == "--predict":
            return

    if mode in ("--full", "--app-only"):
        print("\n STEP 4: Launching web app at http://localhost:5000\n")
        from app import app

        app.run(debug=False, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
