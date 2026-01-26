import streamlit as st
import time
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import requests
import folium
from streamlit_folium import st_folium
from PIL import Image
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# 1. PAGE CONFIG
# =====================================================
st.set_page_config(page_title="EcoSort Infinity", page_icon="♻️", layout="wide")

# =====================================================
# 2. LANDINGUI-STYLE LIGHT GREEN THEME (ONLY CSS)
# =====================================================
st.markdown("""
<style>

/* GLOBAL */
.stApp {
    background-color: #f7fbf7;
    color: #1f2937;
    font-family: 'Segoe UI', sans-serif;
}

/* HEADINGS */
h1, h2, h3 {
    color: #065f46;
    font-weight: 700;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #ecfdf5;
    border-right: 1px solid #d1fae5;
}
section[data-testid="stSidebar"] label {
    color: #065f46;
    font-weight: 600;
}

/* CARDS */
div.stBlock {
    background: white;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #d1fae5;
    box-shadow: 0 12px 30px rgba(0,0,0,0.05);
}

/* METRICS */
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #d1fae5;
}
div[data-testid="stMetricValue"] {
    color: #059669;
    font-size: 1.8rem;
    font-weight: 700;
}

/* BUTTONS */
.stButton>button,
a[role="button"] {
    background: linear-gradient(135deg, #34d399, #10b981);
    color: white !important;
    border-radius: 999px;
    border: none;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #10b981, #059669);
}

/* TABS */
button[data-baseweb="tab"] {
    font-weight: 600;
    color: #065f46;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #10b981;
    color: #059669;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. CONFIG / KEYS
# =====================================================
FIREBASE_URL = "https://smart-bin-7efab-default-rtdb.firebaseio.com"
HF_API_KEY = "YOUR_HF_KEY"
AI_MODEL_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-large-patch14"

# =====================================================
# 4. HELPER FUNCTIONS
# =====================================================
def verify_image(image_bytes):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        r = requests.post(AI_MODEL_URL, headers=headers, data=image_bytes)
        data = r.json()
        return data['labels'][0], data['scores'][0]
    except:
        return "Error", 0.0

def fetch_live_data():
    try:
        r = requests.get(f"{FIREBASE_URL}/bins.json")
        return r.json() if r.json() else {}
    except:
        return {}

def solve_route(df):
    full_bins = df[df['fill_level'] > 80]
    if full_bins.empty:
        return None, None

    depot = pd.DataFrame([{'lat': 19.0760, 'lon': 72.8777, 'fill_level': 0}])
    route_data = pd.concat([depot, full_bins]).reset_index(drop=True)

    locations = list(zip(route_data['lat'], route_data['lon']))
    manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(i, j):
        a, b = locations[manager.IndexToNode(i)], locations[manager.IndexToNode(j)]
        return int(abs(a[0]-b[0])*10000 + abs(a[1]-b[1])*10000)

    transit = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(params)
    if not solution:
        return None, None

    path = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        path.append(locations[manager.IndexToNode(idx)])
        idx = solution.Value(routing.NextVar(idx))
    path.append(locations[manager.IndexToNode(idx)])

    return path, full_bins

# =====================================================
# 5. NAVIGATION
# =====================================================
st.sidebar.title("♻️ EcoSort Infinity")
menu = st.sidebar.radio("Modules", [
    "Command Center",
    "Citizen AI Portal",
    "Driver Ops",
    "Analytics & ROI"
])

# =====================================================
# COMMAND CENTER
# ============================
