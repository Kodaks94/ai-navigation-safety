import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DATA = ROOT / "data" / "synthetic_navigation_timeseries.csv"

st.set_page_config(page_title="Predictive Hazard Dashboard", layout="wide")
st.title("Predictive Hazard Dashboard")

# Load artifacts
model_path = ART / "model.pkl"
last_window_path = ART / "last_window.csv"

if not model_path.exists() or not last_window_path.exists():
    st.error("Artifacts not found. Please run:  python src/train_model.py")
    st.stop()

with open(model_path, "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]
feature_cols = bundle["feature_cols"]

last_window = pd.read_csv(last_window_path, parse_dates=["timestamp"])
st.sidebar.header("Forecast Settings")
horizon = st.sidebar.slider("Forecast horizon (hours)", min_value=6, max_value=48, value=24, step=6)

# Simple naive feature projection for demo (hold last values or small drift)
proj = last_window.copy()
for step in range(horizon):
    row = proj.iloc[-1:].copy()
    # Introduce small random walk to simulate incoming conditions
    rng = np.random.default_rng(123 + step)
    for base_col in ["wind_speed","wave_height","precip","visibility","traffic_density","pressure","temperature"]:
        if base_col in row.columns:
            noise = rng.normal(0, 0.2)
            if base_col in ["visibility","pressure","temperature"]:
                row[base_col] = max(0.1, row[base_col].values[0] + noise)
            else:
                row[base_col] = max(0.0, row[base_col].values[0] + noise)
    row["timestamp"] = row["timestamp"].values[0] + pd.Timedelta(hours=1)
    proj = pd.concat([proj, row], ignore_index=True)

# Rebuild lag/rolling features on the fly (same logic as training)
def make_features(df, lags=[1,3,6,12], rolls=[3,6,12]):
    df = df.copy().sort_values("timestamp")
    for col in ["wind_speed","wave_height","precip","visibility","traffic_density","pressure","temperature","hazard_index"]:
        for l in lags:
            df[f"{{col}}_lag{{l}}"] = df[col].shift(l)
        for r in rolls:
            df[f"{{col}}_roll{{r}}h_mean"] = df[col].rolling(r).mean()
    return df

proj_feats = make_features(proj)
proj_feats = proj_feats.dropna().reset_index(drop=True)
future = proj_feats.iloc[-horizon:].copy()

# Predict hazard for future horizon
X = future[feature_cols]
future["pred_hazard"] = model.predict(X)

# Display chart
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(future["timestamp"], future["pred_hazard"], label="Predicted Hazard")
ax.axhspan(0.6, 1.0, alpha=0.2, label="High-risk zone")
ax.set_ylabel("Hazard Index (0-1)")
ax.set_xlabel("Time")
ax.legend()
st.pyplot(fig)

st.markdown("**High-risk periods (predicted hazard ≥ 0.6):**")
high = future[future["pred_hazard"] >= 0.6][["timestamp","pred_hazard"]]
if len(high) == 0:
    st.success("No high-risk periods in the selected horizon.")
else:
    st.dataframe(high.reset_index(drop=True))

st.caption("Demo data and model for interview prototype. Train with synthetic time-series; predict short-term hazard.")
