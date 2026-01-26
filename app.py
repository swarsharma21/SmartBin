import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import firebase_admin
from firebase_admin import credentials, db
import base64
import json
from github import Github
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
REPO_NAME = st.secrets.get("REPO_NAME", "krishajain2405/smart-infinity")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
POWER_BI_URL = "https://app.powerbi.com/links/PQ2P41cZAi?ctid=c290ab75-f93e-4848-8b0b-550dd7acfc33&pbi_source=linkShare"
FIREBASE_DB_URL = "https://smart-waste-management-52e19-default-rtdb.firebaseio.com/"

st.set_page_config(
    page_title="EcoSmart Waste Management",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🎨 CLEAN DARK UI (FIXED – NO CONFLICTS)
# ==========================================
st.markdown("""
<style>

/* Global App */
.stApp {
    background-color: #020617;
    color: #E5E7EB;
}

/* Hide Streamlit UI */
#MainMenu, footer, header {
    visibility: hidden;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
}
[data-testid="stSidebar"] * {
    color: #E5E7EB !important;
}

/* Cards / Containers */
div.stElementContainer, div.stBlock, .stForm, [data-testid="stExpander"] {
    background: rgba(17, 24, 39, 0.92);
    backdrop-filter: blur(12px);
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 40px rgba(0,0,0,0.6);
    margin-bottom: 20px;
}

/* Headings */
h1 {
    color: #34D399;
    font-size: 44px;
    font-weight: 800;
}
h2, h3 {
    color: #A7F3D0;
    font-weight: 700;
}

/* Text */
p, span, label, li {
    color: #E5E7EB;
    font-weight: 500;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #10B981;
    font-size: 36px;
    font-weight: 800;
}
[data-testid="stMetricLabel"] {
    color: #9CA3AF;
}

/* Buttons */
button {
    background: linear-gradient(135deg, #10B981, #059669);
    color: white;
    border-radius: 10px;
    font-weight: 700;
    border: none;
}
button:hover {
    transform: scale(1.03);
    box-shadow: 0 10px 25px rgba(16,185,129,0.4);
}

/* Inputs */
input, textarea {
    background-color: #020617;
    color: #E5E7EB;
    border: 1px solid #10B981;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #9CA3AF;
    font-weight: 700;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #34D399;
    border-bottom: 3px solid #34D399;
}

/* Plotly */
.js-plotly-plot {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 🔐 FIREBASE INIT
# ==========================================
if not firebase_admin._apps:
    try:
        if "FIREBASE_BASE64" in st.secrets:
            creds_dict = json.loads(base64.b64decode(st.secrets["FIREBASE_BASE64"]).decode("utf-8"))
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
    except Exception as e:
        st.error(f"Firebase failed: {e}")

# ==========================================
# 🧠 AI ENGINE
# ==========================================
@st.cache_resource
def load_ai():
    return pipeline("image-classification", model="yangy50/garbage-classification")

# ==========================================
# 🧭 NAVIGATION
# ==========================================
page = st.sidebar.radio(
    "EcoSmart Navigation",
    ["Home", "Citizen Portal", "Financial Model", "Municipality Command", "Driver Portal"]
)

# ==========================================
# 🏠 HOME
# ==========================================
if page == "Home":
    st.title("Smart Waste Management Analytics")
    st.markdown("### Aavishkar State Level Research Project")
    st.write("An end-to-end IoT and AI-driven solution for sustainable urban waste logistics.")
    st.image("https://img.icons8.com/clouds/200/garbage-truck.png")

# ==========================================
# 📢 CITIZEN PORTAL
# ==========================================
elif page == "Citizen Portal":
    st.title("📢 Public Reporting Portal")
    with st.form("citizen_form", clear_on_submit=True):
        n = st.text_input("Full Name")
        l = st.text_input("Ward Location")
        f = st.file_uploader("Upload Waste Photo", type=['jpg','png','jpeg'])
        if st.form_submit_button("Submit & Verify"):
            if f and n:
                with st.spinner("AI Engine Verifying..."):
                    res = load_ai()(Image.open(f))
                    if res[0]['label'] in ['glass', 'metal', 'paper', 'plastic', 'trash']:
                        st.success(f"AI Verified: {res[0]['label']} detected!")
                        st.balloons()
                    else:
                        st.error("AI Error: This does not appear to be waste.")

# ==========================================
# 💎 FINANCIAL MODEL
# ==========================================
elif page == "Financial Model":
    st.title("💎 Smart Economics & Sustainability")
    ev_mode = st.toggle("⚡ Activate EV Fleet Simulation", value=True)

    capex = 200000
    dist_saved = 600
    fuel_cost = 10.0 if ev_mode else 104.0
    opex_savings = (dist_saved * fuel_cost) + 25000
    total_benefit = opex_savings + 80000

    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly Savings", f"₹{total_benefit:,.0f}")
    c2.metric("ROI Payback", f"{capex/total_benefit:.1f} Months")
    c3.metric("Carbon Credits", "14.2 Credits/Mo")

    fig = px.bar(
        x=["OPEX", "Recycling", "Penalty Avoidance"],
        y=[opex_savings, 45000, 35000],
        color_discrete_sequence=['#10B981']
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 🛡️ MUNICIPALITY COMMAND
# ==========================================
elif page == "Municipality Command":
    st.title("🛡️ Admin Control Center")
    t1, t2 = st.tabs(["Route Optimization", "Strategic Dashboard"])

    with t1:
        st.subheader("🌍 Fleet-Wide Optimized Routes")
        st_folium(folium.Map(location=[19.0760, 72.8777], zoom_start=12), width=1000, height=500)

    with t2:
        st.subheader("📊 Executive Analytics View")
        st.link_button("🔗 Launch Full Interactive Dashboard", POWER_BI_URL)

# ==========================================
# 🚚 DRIVER PORTAL
# ==========================================
elif page == "Driver Portal":
    st.title("🚚 Driver Logistics Center")

    if not st.session_state.get('d_auth', False):
        st.subheader("Driver Authentication")
        u = st.text_input("Driver ID")
        p = st.text_input("Pin", type="password")
        if st.button("Login") and u == "driver01":
            st.session_state.d_auth = True
            st.rerun()
    else:
        st.success("Driver 01: Secure Session Active")
        st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"d_auth": False, "scanned": False}))

        if st.button("📷 Scan QR at Waste Depot"):
            st.session_state.scanned = True

        if st.session_state.get('scanned'):
            st.markdown("### 📋 Your Tasks for Today")
            with st.expander("Task 1: Bin B104 - Ward A (92% Full)", expanded=True):
                st.write("Current Status: Critical - Pickup Required")
                st.link_button(
                    "🚀 Start Navigation",
                    "https://www.google.com/maps/dir/?api=1&destination=19.0760,72.8777"
                )
                st.code("📲 SMS STATUS: Pickup Request Sent to Admin.", language="text")
        else:
            st.info("Please scan the depot QR code to receive your daily route.")
