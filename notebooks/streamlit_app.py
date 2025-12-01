# notebooks/streamlit_app.py (safer requests)
import streamlit as st
import requests
import json
from requests.exceptions import RequestException

API_URL = st.sidebar.text_input("API URL", "http://127.0.0.1:8000/predict")

st.title("🧠 AI Habit Coach")

journal = st.text_area("📝 Today's Journal", "I felt productive after a good sleep.")
sleep_hours = st.number_input("😴 Sleep Hours", 0.0, 24.0, 7.5)
steps = st.number_input("🚶 Steps", 0, 50000, 8000)
screen_hours = st.number_input("📱 Screen Time (hrs)", 0.0, 24.0, 3.0)
dow = st.selectbox("📅 Day of Week", list(range(7)),
                   format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

if st.button("Get Insight"):
    payload = {
        "user_id": "user_1",
        "journal": journal,
        "activity": {
            "sleep_hours": sleep_hours,
            "steps": steps,
            "screen_hours": screen_hours,
            "dow": dow
        }
    }
    try:
        with st.spinner("Contacting API..."):
            r = requests.post(API_URL, json=payload, timeout=10)
            r.raise_for_status()
            res = r.json()
        st.success(f"**Emotion:** {res['nlp']['predicted_emotion']}")
        st.info(f"**Predicted Productivity:** {res['timeseries']['predicted_productivity']:.2f}")
        st.write("💡 Insights:")
        for tip in res.get("insights", []):
            st.write(f"- {tip}")
    except RequestException as e:
        st.error("Failed to contact API. Make sure the FastAPI server is running.")
        st.write("Error details:", str(e))
        st.write("Try opening the API docs at http://127.0.0.1:8000/docs")
