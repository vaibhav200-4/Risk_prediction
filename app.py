import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import io
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Failure Risk Predictor", layout="wide")

st.title("🚆 KMRL — Failure Risk Prediction (XGBoost)")
st.markdown("""
This app loads a pre-trained XGBoost model (`xgb_failure_risk.json`)  
and predicts **Failure Risk (0–1)** for trains.
""")

# -------------------------
# DEFAULT FEATURES
# -------------------------
FEATURE_COLS = [
    "mileage_total",
    "days_since_last_maintenance",
    "days_since_FC_validation",
    "open_jobcard_count",
    "high_priority_jobcard_count",
    "fc_expired_flag"
]


# -------------------------
# Prepare features
# -------------------------
def prepare_features(df):
    df = df.copy()

    # FC flag
    if "fitness_certificate_status" in df.columns:
        df["fc_expired_flag"] = (df["fitness_certificate_status"].str.lower() == "expired").astype(int)
    else:
        df["fc_expired_flag"] = df.get("fc_expired_flag", 0)

    today = pd.to_datetime("2025-11-30")

    if "days_since_last_maintenance" not in df or df["days_since_last_maintenance"].isnull().any():
        if "last_maintenance_date" in df.columns:
            df["days_since_last_maintenance"] = (today - pd.to_datetime(df["last_maintenance_date"])).dt.days
        else:
            df["days_since_last_maintenance"] = 0

    if "days_since_FC_validation" not in df or df["days_since_FC_validation"].isnull().any():
        if "validation_date_of_FC" in df.columns:
            df["days_since_FC_validation"] = (today - pd.to_datetime(df["validation_date_of_FC"])).dt.days
        else:
            df["days_since_FC_validation"] = 0

    # Numeric cast + fill
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

    return df


# -------------------------
# Load XGBoost model
# -------------------------
@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model("xgb_failure_risk.json")
    return booster


try:
    model = load_model()
    st.success("Model loaded successfully ✔")
except:
    st.error("❌ Could not load model file `xgb_failure_risk.json`.")
    st.stop()


def predict(df):
    dmat = xgb.DMatrix(df[FEATURE_COLS], feature_names=FEATURE_COLS)
    return model.predict(dmat)


# ======================================================
# 1️⃣  BATCH PREDICTION
# ======================================================

st.header("📂 Batch Prediction from CSV")

csv_file = st.file_uploader("Upload CSV with required fields", type=["csv"])

if csv_file:
    df = pd.read_csv(csv_file)
    df_prep = prepare_features(df)
    preds = predict(df_prep)

    df_out = df.copy()
    df_out["pred_failure_risk"] = np.round(preds, 3)

    st.subheader("Preview")
    st.dataframe(df_out.head(20))

    # download
    buffer = io.StringIO()
    df_out.to_csv(buffer, index=False)
    st.download_button("Download Predictions CSV", buffer.getvalue(),
                       file_name="predictions.csv", mime="text/csv")


# ======================================================
# 2️⃣  SINGLE ROW PREDICTION
# ======================================================

st.header("🧮 Single Input Prediction")

with st.form("single_form"):
    mileage_total = st.number_input("Mileage Total", value=50000)
    days_since_last_maintenance = st.number_input("Days Since Last Maintenance", value=30)
    days_since_FC_validation = st.number_input("Days Since FC Validation", value=10)
    open_jobcard_count = st.number_input("Open Jobcards", value=1, min_value=0)
    high_priority_jobcard_count = st.number_input("High Priority Jobcards", value=0, min_value=0)
    fc_status = st.selectbox("Fitness Certificate Status", ["valid", "expired"])

    submit = st.form_submit_button("Predict")

    if submit:
        row = pd.DataFrame([{
            "mileage_total": mileage_total,
            "days_since_last_maintenance": days_since_last_maintenance,
            "days_since_FC_validation": days_since_FC_validation,
            "open_jobcard_count": open_jobcard_count,
            "high_priority_jobcard_count": high_priority_jobcard_count,
            "fc_expired_flag": 1 if fc_status == "expired" else 0
        }])

        pred = float(predict(row)[0])

        st.metric("Predicted Failure Risk", f"{pred:.3f}")
        st.info("0–0.33 Low | 0.33–0.66 Medium | >0.66 High")


# ======================================================
# 3️⃣ FEATURE IMPORTANCE
# ======================================================

st.header("📊 Feature Importance (Gain)")
try:
    score = model.get_score(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "gain": [score.get(f, 0) for f in FEATURE_COLS]
    }).sort_values("gain", ascending=False)

    fig = px.bar(imp_df, x="feature", y="gain", text="gain")
    st.plotly_chart(fig, use_container_width=True)
except:
    st.warning("Feature importance unavailable.")

# Footer
st.markdown("---")
st.write("Last updated:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
