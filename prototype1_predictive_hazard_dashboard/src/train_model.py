import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_navigation_timeseries.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

def make_features(df, lags=[1,3,6,12], rolls=[3,6,12]):
    df = df.copy()
    df = df.sort_values("timestamp")
    for col in ["wind_speed","wave_height","precip","visibility","traffic_density","pressure","temperature","hazard_index"]:
        for l in lags:
            df[f"{{col}}_lag{{l}}"] = df[col].shift(l)
        for r in rolls:
            df[f"{{col}}_roll{{r}}h_mean"] = df[col].rolling(r).mean()
    # Drop rows with NaNs due to lag/roll
    df = df.dropna().reset_index(drop=True)
    return df

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = make_features(df)

    # Train/val/test split by time
    n = len(df)
    train_end = int(n*0.7)
    val_end = int(n*0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    target = "hazard_index"
    drop_cols = ["timestamp", target]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train, y_train = train[feature_cols], train[target]
    X_val, y_val = val[feature_cols], val[target]
    X_test, y_test = test[feature_cols], test[target]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    def eval_split(name, X, y):
        pred = model.predict(X)
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        return {"split": name, "mae": mae, "r2": r2}

    report = [
        eval_split("train", X_train, y_train),
        eval_split("val", X_val, y_val),
        eval_split("test", X_test, y_test),
    ]

    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)

    pd.DataFrame(report).to_csv(ARTIFACTS_DIR / "report.csv", index=False)
    # Save the last known window for forecasting demo
    last_window = df.iloc[-24:][["timestamp"] + feature_cols + [target]]
    last_window.to_csv(ARTIFACTS_DIR / "last_window.csv", index=False)

if __name__ == "__main__":
    main()
