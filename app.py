"""
=============================================================================
STEP 4: FLASK WEB APPLICATION
=============================================================================

🎓 WEB APP CONCEPT:
Flask is a Python "web framework" that lets you build websites with Python.

Routes (URL addresses):
  /               → Dashboard with predictions + model stats + subscribe form
  /predict        → Trigger fresh predictions (POST)
  /subscribe      → Sign up for daily email alerts (POST)
  /api/predictions → JSON API endpoint for other apps
  /api/model-report → JSON model training metrics
=============================================================================
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import os
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

from data_pipeline import STOCKS
from predict_and_notify import predict_all_stocks, send_email

# Get the directory where THIS file lives — ensures Flask finds templates/
# even when run.py does os.chdir() or the user runs from a different folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
SUBSCRIBERS_FILE = "data/subscribers.json"


def load_json(path, default=None):
    """Safely load a JSON file; return default if missing."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}


def save_json(path, data):
    """Save data to a JSON file, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════


@app.route("/")
def home():
    """Dashboard — loads predictions, model report, and data summary."""
    predictions = load_json("data/latest_predictions.json")
    report = load_json("models/training_report.json")
    summary = load_json("data/data_summary.json")
    return render_template(
        "index.html",
        predictions=predictions,
        report=report,
        summary=summary,
        stocks=STOCKS,
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


@app.route("/predict", methods=["POST"])
def run_predictions():
    """Trigger fresh predictions when the user clicks the button."""
    try:
        predict_all_stocks()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return redirect(url_for("home"))


@app.route("/subscribe", methods=["POST"])
def subscribe():
    """Add an email address to the daily alert subscriber list."""
    email = request.form.get("email", "").strip()
    if email and "@" in email:
        subs = load_json(SUBSCRIBERS_FILE, [])
        if email not in subs:
            subs.append(email)
            save_json(SUBSCRIBERS_FILE, subs)
    return redirect(url_for("home"))


@app.route("/api/predictions")
def api_predictions():
    """Return latest predictions as raw JSON."""
    return jsonify(load_json("data/latest_predictions.json"))


@app.route("/api/model-report")
def api_report():
    """Return model training report as raw JSON."""
    return jsonify(load_json("models/training_report.json"))


# ══════════════════════════════════════════════════════════════
# DAILY SCHEDULER — runs in both local and production mode
# ══════════════════════════════════════════════════════════════


def scheduled_daily_prediction():
    """
    Runs automatically at 6:30 PM ET every weekday
    (after the stock market closes at 4 PM).
    Generates predictions and emails all subscribers.
    """
    preds = predict_all_stocks()
    if preds:
        for email in load_json(SUBSCRIBERS_FILE, []):
            send_email(preds, email)


# Start scheduler when the module is loaded (works with gunicorn too)
try:
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scheduled_daily_prediction,
        "cron",
        day_of_week="mon-fri",
        hour=18,
        minute=30,
        timezone="US/Eastern",
    )
    scheduler.start()
    print("📅 Scheduler active — predictions at 6:30 PM ET, Mon–Fri")
except Exception as e:
    print(f"⚠️ Scheduler not started: {e}")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT (local development)
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Generate initial predictions if none exist
    if not os.path.exists("data/latest_predictions.json"):
        print("🔮 Generating initial predictions...")
        predict_all_stocks()

    print("\n🚀 StockSage AI running at http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)