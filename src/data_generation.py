# src/data_generation.py
import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

EMOTIONS = ["happy","sad","angry","anxious","neutral","tired","productive"]

journal_templates = {
    "happy": [
        "Felt great today. I had energy and completed my tasks early.",
        "Good day! I had a productive morning and spent time with friends."
    ],
    "sad": [
        "I felt low today. Hard to concentrate on work.",
        "Had a rough day, stayed in bed, small motivation."
    ],
    "anxious": [
        "Was nervous all day. Couldn't stop worrying about upcoming deadlines.",
        "My mind kept racing, focus was poor."
    ],
    "tired": [
        "Extremely tired — slept poorly and felt like napping at work.",
        "Sleep deprived today; coffee helped but I was sluggish."
    ],
    "productive": [
        "Super productive — deep work session in the afternoon led to big progress.",
        "Completed my goals and felt very efficient today."
    ],
    "neutral": [
        "An ordinary day. Nothing special.",
        "Average day, chores done, nothing unusual."
    ],
    "angry": [
        "Felt irritated — small things triggered me.",
        "Snappy and impatient today, not my best mood."
    ],
}

def generate_journals(n_users=50, days=60, start_date="2024-01-01"):
    rows = []
    start = datetime.fromisoformat(start_date)
    for user_id in range(1, n_users+1):
        for d in range(days):
            date = (start + timedelta(days=d)).date().isoformat()
            # pick base emotion with some probability skew
            emo = random.choices(EMOTIONS, weights=[15,10,6,8,25,12,24], k=1)[0]
            text = random.choice(journal_templates[emo])
            # add small personalization noise
            if random.random() < 0.2:
                text += " I went for a walk."
            rows.append({
                "user_id": f"user_{user_id}",
                "date": date,
                "journal": text,
                "emotion": emo
            })
    return pd.DataFrame(rows)

def generate_activity_logs(n_users=50, days=60, start_date="2024-01-01"):
    rows = []
    start = datetime.fromisoformat(start_date)
    for user_id in range(1, n_users+1):
        baseline_sleep = random.uniform(6.5,8.0)
        baseline_steps = random.randint(3000,9000)
        for d in range(days):
            date = (start + timedelta(days=d)).date().isoformat()
            # small random variation
            sleep = max(3.0, baseline_sleep + np.random.normal(0, 1.0))
            steps = max(0, baseline_steps + int(np.random.normal(0, 2500)))
            screen_hours = max(0.5, np.random.normal(4,1.5))
            # synthetic productivity influenced by sleep & steps & random
            productivity = 0.3*sleep + 0.0001*steps + np.random.normal(0,0.5)
            productivity = max(0.0, min(5.0, productivity*2))  # scale to 0-5
            rows.append({
                "user_id": f"user_{user_id}",
                "date": date,
                "sleep_hours": round(sleep,2),
                "steps": int(steps),
                "screen_hours": round(screen_hours,2),
                "productivity_score": round(productivity,2)
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    journals = generate_journals()
    activities = generate_activity_logs()
    journals.to_csv("data/simulated/journals.csv", index=False)
    activities.to_csv("data/simulated/activity_logs.csv", index=False)
    print("Saved simulated data to data/simulated/")
