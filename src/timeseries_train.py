# src/timeseries_train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

DATA_PATH = "data/simulated/activity_logs.csv"
MODEL_PATH = "src/models/timeseries_model.pkl"

def build_features(df):
    df = df.sort_values(['user_id','date']).copy()
    # rolling features per user
    df['sleep_roll_3'] = df.groupby('user_id')['sleep_hours'].rolling(3, min_periods=1).mean().reset_index(0,drop=True)
    df['steps_roll_7'] = df.groupby('user_id')['steps'].rolling(7, min_periods=1).mean().reset_index(0,drop=True)
    df['screen_roll_3'] = df.groupby('user_id')['screen_hours'].rolling(3, min_periods=1).mean().reset_index(0,drop=True)
    # lag features
    df['sleep_lag_1'] = df.groupby('user_id')['sleep_hours'].shift(1).fillna(df['sleep_hours'].mean())
    # day of week
    df['dow'] = pd.to_datetime(df['date']).dt.weekday
    return df.fillna(0)

def main():
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)
    features = ['sleep_hours','steps','screen_hours','sleep_roll_3','steps_roll_7','screen_roll_3','sleep_lag_1','dow']
    X = df[features]
    y = df['productivity_score']
    # simple user-stratified split
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print("RMSE:", rmse)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Saved timeseries model to", MODEL_PATH)

if __name__ == "__main__":
    main()
