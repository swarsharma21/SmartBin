import streamlit as st
import time
import pandas as pd
import pydeck as pdk
import plotly.express as px
import requests
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="EcoSort Infinity", page_icon="♻️", layout="wide")

# =====================================================
# LANDINGUI STYLE (LIGHT GREEN)
# =====================================================
st.markdown("""
<style>

/* GLOBAL */
.stApp {
    background-color: #f7fbf7;
    font-family: 'Segoe UI', sans-serif;
    color: #1f2937;
}

/* REMOVE SIDEBAR COMPLETELY */
section[data-testid="stSidebar"] {
    display: none;
}

/* HEADINGS */
h1 { font-size: 3rem; color: #064e3b; font-weight: 800; }
h2 { font-size: 2.2rem; color: #065f46; font-weight: 700; }
h3 { color: #047857; font-weight: 600; }

/* SECTION CARDS */
.block-container > div {
    background: white;
    padding: 32px;
    border-radius: 22px;
    border: 1px solid #d1fae5;
    box-shadow: 0 16px 40px rgba(0,0,0,0.04);
    margin-bottom: 3rem;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #34d399, #10b981);
    color: white;
    border-radius: 999px;
    padding: 0.7rem 2rem;
    font-weight: 600;
    border: none;
}

/* METRICS */
div[data-testid="stMetricValue"] {
    color: #059669;
    font-size: 2rem;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# CONFIG
# =====================================================
FIREBASE_URL = "https://smart-bin-7efab-default-rtdb.firebaseio.com"

# =====================================================
# HELPERS
# =====================================================
def fetch_live_data():
    try:
        r = requests.get(f"{FIREBASE_URL}/bins.json")
        return r.json() if r.json() else {}
    except:
        return {}

def solve_route(df):
    full_bins = df[df.fill_level > 80]
    if full_bins.empty:
        return None

    depot = (19.0760, 72.8777)
    path = [depot] + list(zip(full_bins.lat, full_bins.lon)) + [depot]
    return path

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<h1>EcoSort Infinity</h1>
<p style="font-size:1.3rem; max-width:900px;">
AI & IoT powered smart waste management platform for predictive collection,
optimized routing, and cleaner smart cities.
</p>
""", unsafe_allow_html=True)

# =====================================================
# COMMAND CENTER SECTION
# =====================================================
st.markdown("## 🏙️ Urban Command Center")

data = fetch_live_data()

if data:
    df = pd.DataFrame.from_dict(data, orient="index")

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Sensors", len(df))
    c2.metric("Average Fill", f"{int(df.fill_level.mean())}%")
    c3.metric("Critical Bins", len(df[df.fill_level > 90]))

    st.subheader("📍 Live City Map")

    df["color"] = df.fill_level.apply(
        lambda x: [255,0,0,200] if x > 90 else [0,255,0,200]
    )

    layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position="[lon, lat]",
        get_elevation="fill_level",
        radius=20,
        elevation_scale=10,
        get_fill_color="color"
    )

    view = pdk.ViewState(
        latitude=19.0760,
        longitude=72.8777,
        zoom=14,
        pitch=55
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))

    st.subheader("🚛 Optimized Collection Route")
    if st.button("Generate Route"):
        path = solve_route(df)
        if path:
            m = folium.Map(location=[19.0760, 72.8777], zoom_start=14)
            folium.PolyLine(path, color="green", weight=5).add_to(m)
            st_folium(m, height=400)
else:
    st.info("Waiting for live data from bins…")

# =====================================================
# ANALYTICS & ROI SECTION (NO SLIDERS)
# =====================================================
st.markdown("## 📊 Analytics & Impact")

# Fixed assumptions
num_trucks = 5
dist_old = 1500
dist_new = 900
fuel_price = 104
efficiency = 4

cost_old = (dist_old * num_trucks / efficiency) * fuel_price
cost_new = (dist_new * num_trucks / efficiency) * fuel_price
savings = cost_old - cost_new

c1, c2, c3 = st.columns(3)
c1.metric("Monthly Savings", f"₹{int(savings):,}")
c2.metric("Efficiency Gain", f"{int((savings/cost_old)*100)}%")
c3.metric("Fleet Size", num_trucks)

fig = px.bar(
    pd.DataFrame({
        "Source": ["Operational Savings", "Recycling Revenue"],
        "Amount": [savings, 90000]
    }),
    x="Source",
    y="Amount"
)
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr>
<p style="text-align:center; color:#6b7280;">
© 2025 EcoSort Infinity — Smart Waste for Smart Cities
</p>
""", unsafe_allow_html=True)
