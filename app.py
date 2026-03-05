import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

st.set_page_config(page_title="JazzMusafir — Fare Intelligence", layout="wide")

@st.cache_data
def load_data():
    rd = pd.read_csv("route_day_pricing.csv")
    rd["date"] = pd.to_datetime(rd["date"], errors="coerce")
    rs = pd.read_csv("route_summary.csv")
    df = pd.read_csv("route_defaults.csv")
    return rd, rs, df

@st.cache_resource
def load_model():
    return joblib.load("fare_model_xgb.joblib")

route_day, route_summary, route_defaults = load_data()
model = load_model()

st.title("✈️ JazzMusafir — Fare Intelligence & Best Time to Buy")
st.caption("Portfolio project inspired by OTAs (JazzMusafir-like). Data: Kaggle Flight Prices 2M. Model: XGBoost (log target).")

# ---------- Sidebar controls ----------
st.sidebar.header("Search")
top_routes = route_summary.sort_values("total_offers", ascending=False)["route"].head(5000).tolist()
route = st.sidebar.selectbox("Route", top_routes)

flight_dt = st.sidebar.date_input(
    "Flight date",
    value=route_day["date"].max().date() if route_day["date"].notna().any() else date.today()
)

lead_time_days = st.sidebar.slider("Lead time (days before departure)", min_value=0, max_value=180, value=21, step=1)

# get default features for selected route
defaults_row = route_defaults[route_defaults["route"] == route]
if defaults_row.empty:
    # fallback defaults if route not found
    defaults = {
        "default_airline": "DL",
        "default_cabin": "coach",
        "med_duration": 180.0,
        "med_stops": 0.0,
        "med_distance": np.nan,
        "med_seats": 5.0,
        "pct_basic": 0.0,
        "pct_refund": 0.0,
        "pct_nonstop": 1.0
    }
else:
    r = defaults_row.iloc[0].to_dict()
    defaults = {
        "default_airline": r.get("default_airline", "DL"),
        "default_cabin": r.get("default_cabin", "coach"),
        "med_duration": float(r.get("med_duration", 180.0)),
        "med_stops": float(r.get("med_stops", 0.0)),
        "med_distance": float(r.get("med_distance", np.nan)),
        "med_seats": float(r.get("med_seats", 5.0)),
        "pct_basic": float(r.get("pct_basic", 0.0)),
        "pct_refund": float(r.get("pct_refund", 0.0)),
        "pct_nonstop": float(r.get("pct_nonstop", 1.0)),
    }

# user-tunable inputs (realistic)
airline_code = st.sidebar.text_input("Airline code (e.g., DL, AA, UA)", value=str(defaults["default_airline"]))
cabin_code = st.sidebar.text_input("Cabin (e.g., coach, business)", value=str(defaults["default_cabin"]))

duration_min = st.sidebar.slider("Duration (minutes)", min_value=30, max_value=1500, value=int(np.clip(defaults["med_duration"], 30, 1500)))
n_stops = st.sidebar.slider("Stops", min_value=0, max_value=3, value=int(np.clip(round(defaults["med_stops"]), 0, 3)))
seats_remaining = st.sidebar.slider("Seats remaining", min_value=0, max_value=30, value=int(np.clip(defaults["med_seats"], 0, 30)))

is_basic = st.sidebar.checkbox("Basic economy", value=(defaults["pct_basic"] >= 0.5))
is_refund = st.sidebar.checkbox("Refundable", value=(defaults["pct_refund"] >= 0.5))
is_nonstop = st.sidebar.checkbox("Non-stop", value=(defaults["pct_nonstop"] >= 0.5 and n_stops == 0))

# distance is optional (many rows have it; if missing, keep NaN)
use_distance = st.sidebar.checkbox("Use distance feature (if available)", value=not np.isnan(defaults["med_distance"]))
total_distance = defaults["med_distance"] if use_distance else np.nan

# ---------- Main layout ----------
colA, colB = st.columns([2, 1])

with colA:
    st.subheader("Route overview")
    rs = route_summary[route_summary["route"] == route]
    if not rs.empty:
        rs = rs.iloc[0]
        st.markdown(
            f"""
            **Route:** `{route}`  
            **Avg fare:** `${rs['avg_fare']:.2f}` | **Min:** `${rs['min_fare']:.2f}` | **Max:** `${rs['max_fare']:.2f}`  
            **Total offers (proxy demand):** `{int(rs['total_offers'])}` | **Volatility:** `{rs['volatility']:.3f}`
            """
        )
    else:
        st.info("No route summary found for this route in your aggregated table.")

    tmp = route_day[route_day["route"] == route].copy()
    tmp = tmp.dropna(subset=["date"]).sort_values("date")
    if not tmp.empty:
        st.markdown("**Price trend (avg_fare)**")
        st.line_chart(tmp.set_index("date")[["avg_fare"]])
    else:
        st.warning("No time-series data for this route in route_day_pricing.csv.")

with colB:
    st.subheader("Best time to buy")
    if not tmp.empty:
        cheapest = tmp.sort_values("avg_fare").head(10)[["date","avg_fare","min_fare","max_fare","offers"]].copy()
        cheapest["date"] = cheapest["date"].dt.date
        st.dataframe(cheapest, use_container_width=True, height=340)
    else:
        st.write("No cheapest-days list available.")

# ---------- Prediction ----------
st.subheader("ML fare prediction (XGBoost)")
flight_ts = pd.to_datetime(flight_dt)
dow = int(flight_ts.dayofweek)
month = int(flight_ts.month)

X_pred = pd.DataFrame([{
    "route": route,
    "airline_code": airline_code.strip().upper(),
    "cabin_code": cabin_code.strip().lower(),
    "lead_time_days": int(lead_time_days),
    "duration_min": float(duration_min),
    "n_stops": int(n_stops),
    "seatsRemaining": float(seats_remaining),
    "totalTravelDistance": float(total_distance) if not pd.isna(total_distance) else np.nan,
    "dow": dow,
    "month": month,
    "isBasicEconomy": int(is_basic),
    "isRefundable": int(is_refund),
    "isNonStop": int(is_nonstop),
}])

try:
    pred_log = model.predict(X_pred)
    pred = float(np.expm1(pred_log)[0])  # model trained on log1p
    st.success(f"Estimated fare for **{route}** on **{flight_dt}**:  **${pred:,.2f}**")
except Exception as e:
    st.error("Prediction failed. This usually happens if the model was trained with a different set of columns.")
    st.code(str(e))

with st.expander("Show model input row"):
    st.dataframe(X_pred, use_container_width=True)

st.caption("Tip: For best accuracy, keep airline/cabin/duration/stops realistic for the selected route.")
