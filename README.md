📘 AI Habit Coach — End-to-End ML System

An end-to-end AI/ML project that combines NLP emotion detection, time-series productivity forecasting, and a FastAPI backend with a Streamlit user interface.
This project demonstrates full MLOps workflow: data simulation → model training → API serving → lightweight app UI.

🚀 Project Overview

AI Habit Coach analyzes:

your journal text → predicts emotions

your daily activity metrics (sleep, steps, screen time) → predicts productivity

…and returns personalized insights to help improve habits.

This is a practical, deploy-ready system built with:

Transformers (DistilBERT) for NLP

RandomForestRegressor for time-series signals

FastAPI backend for real-time inference

SQLite for logging predictions

Streamlit UI for end users

✨ Features
🔹 1. NLP Emotion Classifier

Fine-tuned DistilBERT predicts 7 emotional categories:

happy

productive

neutral

anxious

sad

tired

angry

🔹 2. Productivity Time-Series Model

RandomForestRegressor trained on simulated behavioral data predicts:

daily productivity score (0–5)

🔹 3. FastAPI Backend

/predict endpoint combines both models

logs requests into SQLite: requests_log.db

model-agnostic (load from local or Hugging Face)

🔹 4. Streamlit Frontend

Interactive input form

Calls FastAPI in real-time

Displays predictions + suggested habits

🔹 5. Clean MLOps Project Structure

modular code

reproducible training scripts

.gitignore for large weights

ready for cloud deployment

🏗️ System Architecture
                        +----------------------+
                        |  User (Streamlit UI) |
                        +---------+------------+
                                  |
                                  v
                      HTTP POST /predict (JSON)
                                  |
                     +------------+-------------+
                     |        FastAPI Server    |
                     +------------+-------------+
                                  |
       +--------------------------+-----------------------------+
       |                                                        |
       v                                                        v
+--------------+                                     +----------------------+
|  NLP Model   |   DistilBERT fine-tuned → emotion   | Time-Series Model    |
| (Transformers)|----------------------------------->| RandomForestRegressor|
+--------------+                                     +----------------------+
                                  |
                                  v
                           JSON Response
                                  |
                                  v
                       +----------------------+
                       | Insights & Dashboard |
                       +----------------------+

📂 Repository Structure
ai-habit-coach/
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_nlp_finetune.ipynb
│   ├── 03_timeseries_baseline.ipynb
│   └── streamlit_app.py
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── requirements_api.txt
│   ├── data_generation.py
│   ├── nlp_train.py
│   ├── nlp_eval.py
│   └── timeseries_train.py
│
├── requirements.txt
├── .gitignore
└── terminal_guide.txt

⚙️ Local Setup
1. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt

🧠 Train the Models
Train NLP Model
python src/nlp_train.py


Result stored in:

src/models/nlp_model/

Train Time-Series Model
python src/timeseries_train.py


Result stored in:

src/models/timeseries_model.pkl


⚠️ These trained model files should be uploaded to Hugging Face — not committed to GitHub.

🔥 Run the FastAPI Backend

Navigate to API directory:

cd src/api
uvicorn main:app --reload --port 8000


Test endpoint:

curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"journal\":\"Feeling good today!\",\"activity\":{\"sleep_hours\":7,\"steps\":5000,\"screen_hours\":3,\"dow\":2}}"

🖥️ Run Streamlit UI
streamlit run notebooks/streamlit_app.py

📦 Model Deployment (Recommended)

Upload trained models to Hugging Face:

huggingface-cli login
git clone https://huggingface.co/<username>/ai-habit-coach-model


Your FastAPI can then load models like:

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("HtetLwinKyaw/ai-habit-coach-model")

📊 SQLite Logging

All prediction requests are logged into:

src/data/requests_log.db


Query logs (example):

SELECT * FROM logs LIMIT 20;

🛠️ Roadmap

 Deploy API to Render / Railway

 Deploy Streamlit to Streamlit Cloud

 Convert models to ONNX for faster inference

 Add user authentication

 Add daily habit recommendations (LLM-based)

 Add mobile app (Flutter / React Native)
