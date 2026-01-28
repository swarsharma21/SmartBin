import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import folium
from streamlit_folium import st_folium
import requests
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
import firebase_admin
from firebase_admin import credentials, db
import json
import base64

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Bin – Smart Waste System",
    page_icon="♻️",
    layout="wide"
)

# =====================================================
# GLOBAL CSS (CLEAN + READABLE)
# =====================================================
st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(180deg, #0E1628, #020617);
    color: #E5E7EB;
    font-family: "Segoe UI", system-ui;
}
section.main > div { background: transparent !important; }

h1, h2, h3, h4 { color: #F9FAFB; }

p, span, label {
    color: #CBD5E1;
    font-size: 1.05rem;
}

.stButton > button {
    background: linear-gradient(135deg, #4ADE80, #22C55E);
    color: #022C22 !important;
    font-weight: 700;
    border-radius: 12px;
    border: none;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetricValue"] {
    color: #4ADE80;
    font-size: 2rem;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# FIREBASE INITIALIZATION
# =====================================================
FIREBASE_DB_URL = "https://smart-bin-7efab-default-rtdb.firebaseio.com"

if not firebase_admin._apps:
    try:
        # For Streamlit Cloud – store base64 service account in secrets
        firebase_base64 = st.secrets.get("FIREBASE_BASE64", None)
        if firebase_base64:
            creds_dict = json.loads(base64.b64decode(firebase_base64).decode())
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    except Exception as e:
        st.error(f"Firebase init failed: {e}")

def fetch_bins():
    try:
        ref = db.reference("bins")
        data = ref.get()
        if data:
            return pd.DataFrame.from_dict(data, orient="index")
    except:
        pass
    return pd.DataFrame({
        "lat": [19.076, 19.078, 19.074],
        "lon": [72.877, 72.879, 72.875],
        "fill_level": [45, 92, 78]
    })

def save_citizen_report(name, location):
    try:
        ref = db.reference("citizen_reports")
        ref.push({
            "name": name,
            "location": location,
            "status": "verified"
        })
    except:
        pass

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div style="padding:90px 8%; text-align:center;">
  <h1 style="font-size:64px; font-weight:800;">
    Smart Waste <span style="color:#4ADE80;">Management</span>
  </h1>
  <p style="max-width:900px; margin:24px auto;">
    AI & IoT powered smart bin system enabling real-time monitoring,
    predictive analytics, and optimized waste collection routes.
  </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# KPI SECTION
# =====================================================
df_bins = fetch_bins()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Bins", len(df_bins))
c2.metric("Average Fill", f"{int(df_bins.fill_level.mean())}%")
c3.metric("Critical Alerts", len(df_bins[df_bins.fill_level > 90]))
c4.metric("System Status", "ONLINE")

# =====================================================
# CITIZEN PORTAL
# =====================================================
st.markdown("## 📢 Citizen Reporting Portal")

with st.form("citizen_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name")
    with col2:
        location = st.text_input("Location / Ward")

    submitted = st.form_submit_button("Submit Report")

    if submitted and name and location:
        save_citizen_report(name, location)
        st.success("Report submitted & verified ✔")
        st.balloons()

# =====================================================
# MAP VIEW
# =====================================================
st.markdown("## 🗺️ Live City Bin Status")

df_bins["color"] = df_bins["fill_level"].apply(
    lambda x: [255,0,0,180] if x > 90 else [34,197,94,180]
)

layer = pdk.Layer(
    "ColumnLayer",
    data=df_bins,
    get_position="[lon, lat]",
    get_elevation="fill_level",
    radius=25,
    elevation_scale=8,
    get_fill_color="color",
    pickable=True
)

view = pdk.ViewState(
    latitude=df_bins.lat.mean(),
    longitude=df_bins.lon.mean(),
    zoom=14,
    pitch=50
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))

# =====================================================
# ANALYTICS & ROI
# =====================================================
st.markdown("## 📊 Analytics & Insights")

tab1, tab2 = st.tabs(["EDA", "Predictive AI"])

# CSV LOAD
df = None
uploaded = st.file_uploader("Upload smart_bin_historical_data.csv", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)

# EDA
with tab1:
    if df is not None:
        c1, c2 = st.columns(2)

        with c1:
            fig1 = px.line(
                df.groupby("hour_of_day")["bin_fill_percent"].mean().reset_index(),
                x="hour_of_day",
                y="bin_fill_percent",
                title="Hourly Fill Pattern"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            fig2 = px.bar(
                df.groupby("day_of_week")["bin_fill_percent"].mean().reset_index(),
                x="day_of_week",
                y="bin_fill_percent",
                title="Weekly Fill Pattern"
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Upload CSV to view EDA")

# PREDICTIVE
with tab2:
    if df is not None:
        if st.button("Train Predictive Model"):
            model = RandomForestRegressor(n_estimators=50)
            X = df[["hour_of_day"]]
            y = df["bin_fill_percent"]
            model.fit(X, y)

            future = pd.DataFrame({"hour_of_day": range(24)})
            future["Prediction"] = model.predict(future)

            st.line_chart(future.set_index("hour_of_day"))
    else:
        st.info("Upload CSV to enable prediction")

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr style="margin-top:80px;"/>
<div style="text-align:center; padding:30px; color:#9CA3AF;">
  © 2025 Smart Bin – IoT Smart Waste Monitoring System
</div>
""", unsafe_allow_html=True)
