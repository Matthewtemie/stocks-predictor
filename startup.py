"""
Startup script — runs during deployment build step.
Fetches data from Alpaca and trains models so the app
is ready to serve predictions immediately on boot.
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("TemieStockSage startup: building models...")

from data_pipeline import prepare_all_stocks
from train_models import train_all_models
from predict_and_notify import predict_all_stocks

print("\n Step 1: Fetching data...")
prepare_all_stocks()

print("\n Step 2: Training models...")
train_all_models()

print("\n Step 3: Generating initial predictions...")
predict_all_stocks()

print("\n Startup complete — app is ready to serve!")