# src/api/main.py
import os
import hashlib
import sqlite3
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import joblib
import numpy as np
import uvicorn

# --- paths (normalized) ---
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "models", "nlp_model")
TIMESERIES_MODEL = os.path.join(ROOT_DIR, "models", "timeseries_model.pkl")
DB_PATH = os.path.join(ROOT_DIR, "data", "requests_log.db")

# ensure project directory exists
os.makedirs(ROOT_DIR, exist_ok=True)

print("DB PATH:", DB_PATH)

# --- Initialize SQLite DB ---
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_id TEXT,
        journal_hash TEXT,
        pred_emotion TEXT,
        pred_productivity REAL
    )
    """)
    con.commit()
    con.close()

def write_log_sqlite(row):
    """Insert one record into the SQLite log."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO requests (timestamp, user_id, journal_hash, pred_emotion, pred_productivity)
        VALUES (?, ?, ?, ?, ?)
    """, (
        row['timestamp'],
        row['user_id'],
        row['journal_hash'],
        row['pred_emotion'],
        row['pred_productivity']
    ))
    con.commit()
    con.close()

# initialize DB on startup
init_db()

# --- FastAPI app ---
app = FastAPI(title="AI Habit Coach - Inference")

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Habit Coach API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# --- Pydantic models ---
class Activity(BaseModel):
    sleep_hours: float
    steps: int
    screen_hours: float
    sleep_roll_3: float | None = None
    steps_roll_7: float | None = None
    screen_roll_3: float | None = None
    sleep_lag_1: float | None = None
    dow: int = 0

class PredictRequest(BaseModel):
    user_id: str = "user_1"
    date: str | None = None
    journal: str = ""
    activity: Activity

# --- Load models at startup ---
print("Loading models...")
nlp_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
nlp_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
# top_k=None returns scores for all labels (instead of deprecated return_all_scores=True)
nlp_classifier = pipeline(
    "text-classification",
    model=nlp_model,
    tokenizer=nlp_tokenizer,
    top_k=None,
    device=-1  # CPU
)
ts_model = joblib.load(TIMESERIES_MODEL)

# read labels
with open(os.path.join(MODEL_DIR, "labels.txt"), encoding="utf8") as f:
    labels = [l.strip() for l in f.readlines()]

# --- helpers ---
def _label_to_index(label_str, labels_list):
    """Convert pipeline label ('LABEL_2' or 'productive') to index in labels list."""
    if isinstance(label_str, str) and label_str.startswith("LABEL_"):
        try:
            return int(label_str.split("_")[-1])
        except Exception:
            pass
    try:
        return labels_list.index(label_str)
    except ValueError:
        try:
            return int(label_str)
        except Exception:
            return None

def _hash_text(s: str):
    return hashlib.sha256((s or "").encode("utf8")).hexdigest()

# --- Predict endpoint ---
@app.post("/predict")
def predict(req: PredictRequest):
    # NLP inference
    text = (req.journal or "").strip()[:512]
    try:
        nlp_out = nlp_classifier(text)  # returns list-of-lists when top_k=None
        item_scores = nlp_out[0] if isinstance(nlp_out, list) and len(nlp_out) > 0 else nlp_out
        scores = {}
        for d in item_scores:
            lbl = d.get("label")
            score = float(d.get("score", 0.0))
            idx = _label_to_index(lbl, labels)
            key = labels[idx] if (idx is not None and 0 <= idx < len(labels)) else str(lbl)
            scores[key] = score
        best = max(item_scores, key=lambda x: x.get("score", 0.0))
        best_idx = _label_to_index(best.get("label"), labels)
        top_emotion = labels[best_idx] if best_idx is not None and 0 <= best_idx < len(labels) else best.get("label")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP inference failed: {e}")

    # Time-series features
    a = req.activity
    try:
        x = [
            float(a.sleep_hours),
            int(a.steps),
            float(a.screen_hours),
            float(a.sleep_roll_3) if (a.sleep_roll_3 is not None) else float(a.sleep_hours),
            float(a.steps_roll_7) if (a.steps_roll_7 is not None) else int(a.steps),
            float(a.screen_roll_3) if (a.screen_roll_3 is not None) else float(a.screen_hours),
            float(a.sleep_lag_1) if (a.sleep_lag_1 is not None) else float(a.sleep_hours),
            int(a.dow if a.dow is not None else 0)
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid activity features: {e}")

    try:
        pred_prod = float(ts_model.predict([x])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timeseries model failed: {e}")

    # --- Logging to SQLite ---
    row = {
        "timestamp": datetime.now().isoformat(),
        "user_id": req.user_id,
        "journal_hash": _hash_text(req.journal),
        "pred_emotion": top_emotion,
        "pred_productivity": pred_prod
    }
    try:
        write_log_sqlite(row)
        print("DB: log entry saved")
    except Exception as e:
        print("Warning: failed to write SQLite log:", e)

    # --- Rule-based insights ---
    insights = []
    if top_emotion in ["tired", "anxious", "sad"]:
        insights.append("Your journal shows negative / low-energy emotions — consider improving sleep hygiene.")
    if pred_prod > 3.5:
        insights.append("Model predicts high productivity today — great!")
    else:
        insights.append("Model predicts lower productivity — focus on short wins & reduce screen time before bed.")

    return {
        "nlp": {"predicted_emotion": top_emotion, "scores": scores},
        "timeseries": {"predicted_productivity": pred_prod},
        "insights": insights
    }

# --- Run directly ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
