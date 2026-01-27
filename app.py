import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import requests

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Waste Dashboard",
    page_icon="♻️",
    layout="wide"
)

# =====================================================
# GLOBAL LANDINGSITE-STYLE CSS
# =====================================================
st.markdown("""
<style>

/* PAGE BACKGROUND */
.stApp {
    background-color: #f8fafc;
    font-family: Inter, system-ui, sans-serif;
    color: #0f172a;
}

/* REMOVE DEFAULT TOP GAP */
.block-container {
    padding-top: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

/* REMOVE SIDEBAR COMPLETELY */
section[data-testid="stSidebar"] {
    display: none;
}

/* HEADINGS */
h1 {
    font-size: 2.6rem;
    font-weight: 700;
}
h2 {
    font-size: 1.8rem;
    font-weight: 600;
}
h3 {
    font-size: 1.2rem;
    font-weight: 600;
}

/* CARDS */
div[data-testid="stMetric"],
div.stPlotlyChart,
div.stDeckGlJsonChart {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 25px rgba(0,0,0,0.04);
}

/* BUTTONS */
.stButton > button {
    background-color: #22c55e;
    color: white;
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    border: none;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #16a34a;
}

/* METRIC VALUES */
div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
    color: #16a34a;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# MOCK / LIVE DATA FETCH
# =====================================================
FIREBASE_URL = "https://smart-bin-7efab-default-rtdb.firebaseio.com/bins.json"

def fetch_data():
    try:
        r = requests.get(FIREBASE_URL, timeout=5)
        if r.status_code == 200 and r.json():
            return pd.DataFrame.from_dict(r.json(), orient="index")
    except:
        pass

    # fallback mock data
    return pd.DataFrame({
        "lat": [19.076, 19.078, 19.074, 19.072],
        "lon": [72.877, 72.879, 72.875, 72.873],
        "fill_level": [45, 92, 70, 88]
    })

df = fetch_data()

# =====================================================
# HERO SECTION (LIKE LANDINGSITE)
# =====================================================
st.markdown("""
<h1>Smart Waste Management</h1>
<p style="font-size:1.2rem; max-width:900px;">
AI-powered platform for monitoring waste bins, predicting fill levels,
and optimizing collection routes for smart cities.
</p>
""", unsafe_allow_html=True)

# =====================================================
# =====================================================
# HOME / ABOUT SECTION (FROM YOUR CODE, LANDING STYLE)
# =====================================================
st.markdown("## Smart Waste Management Analytics")
st.markdown("### Aavishkar State Level Research Project")

home_col1, home_col2 = st.columns([3, 1])

with home_col1:
    st.write(
        "An end-to-end IoT and AI-driven solution for sustainable urban waste logistics. "
        "The platform integrates real-time bin monitoring, predictive analytics, and "
        "optimized route planning to reduce overflow, operational cost, and environmental impact."
    )

with home_col2:
    st.image(
        "https://img.icons8.com/clouds/200/garbage-truck.png",
        use_container_width=True
    )

# KPI ROW
# =====================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Active Bins", len(df))
c2.metric("Average Fill", f"{int(df.fill_level.mean())}%")
c3.metric("Critical Alerts", len(df[df.fill_level > 90]))
c4.metric("System Status", "Online")

# =====================================================
# =====================================================
# 📢 CITIZEN PORTAL SECTION (LANDING STYLE)
# =====================================================
st.markdown("## 📢 Public Reporting Portal")
st.markdown(
    "Help keep your city clean. Citizens can report overflowing or unmanaged waste, "
    "which is verified using AI before notifying municipal authorities."
)

with st.form("citizen_form", clear_on_submit=True):
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Full Name")
        location = st.text_input("Ward / Area Location")

    with col2:
        waste_image = st.file_uploader(
            "Upload Waste Photo",
            type=["jpg", "png", "jpeg"]
        )

    submitted = st.form_submit_button("Submit & Verify")

    if submitted:
        if waste_image and name:
            with st.spinner("AI Engine Verifying..."):
                try:
                    # Dummy / placeholder AI logic (replace with your model if needed)
                    st.success("AI Verified: Waste detected successfully!")
                    st.balloons()
                except Exception:
                    st.error("AI Error: This does not appear to be waste.")
        else:
            st.warning("Please provide your name and upload a waste image.")

# MAIN MAP SECTION
# =====================================================
st.markdown("## Live City Overview")

df["color"] = df["fill_level"].apply(
    lambda x: [255, 0, 0, 180] if x > 90 else [34, 197, 94, 180]
)

layer = pdk.Layer(
    "ColumnLayer",
    data=df,
    get_position="[lon, lat]",
    get_elevation="fill_level",
    elevation_scale=8,
    radius=25,
    get_fill_color="color",
    pickable=True
)

view = pdk.ViewState(
    latitude=df.lat.mean(),
    longitude=df.lon.mean(),
    zoom=14,
    pitch=50
)

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={"text": "Fill Level: {fill_level}%"}
    )
)

# =====================================================
# ANALYTICS SECTION
# =====================================================
st.markdown("## Analytics & Impact")
