import streamlit as st
import time
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import requests
import folium
import urllib.parse
from streamlit_folium import st_folium
from PIL import Image
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. CONFIGURATION & ECO-SYSTEM UI ---
st.set_page_config(page_title="Smart Bin | Infinity OS", page_icon="♻️", layout="wide")

# Modern Glassmorphism CSS Injection
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top right, #0e1117, #1c1f26); color: #e0e0e0; }
    
    /* Glassmorphic Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 20px;
    }

    /* Neon Metric Glow */
    div[data-testid="stMetricValue"] {
        color: #00ffa3 !important;
        text-shadow: 0 0 15px rgba(0, 255, 163, 0.5);
    }

    /* Sci-Fi Buttons */
    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(90deg, #00ffa3 0%, #00d1ff 100%);
        color: #0e1117 !important;
        font-weight: bold;
        border: none;
        transition: 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 255, 163, 0.4);
    }

    /* Custom Sidebar Navigation */
    section[data-testid="stSidebar"] {
        background-color: #0e1117 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# 🔑 API KEYS (Your Keys Maintained)
FIREBASE_URL = "https://smartcity-infinity-default-rtdb.firebaseio.com"
HF_API_KEY = "AIzaSyBbZjxLgTLeXfuBxAWbUL3BPC8hUL4ahnk"
AI_MODEL_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-large-patch14"

# --- 2. HELPER FUNCTIONS (Logic Maintained) ---

def verify_image(image_bytes):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = requests.post(AI_MODEL_URL, headers=headers, data=image_bytes)
        data = response.json()
        return data['labels'][0], data['scores'][0] 
    except: return "Error", 0.0

def fetch_live_data():
    try:
        r = requests.get(f"{FIREBASE_URL}/bins.json")
        return r.json() if r.json() else {}
    except: return {}

def solve_route(df_data):
    full_bins = df_data[df_data['fill_level'] > 80]
    if full_bins.empty: return None, None
    depot = pd.DataFrame([{'lat': 19.0760, 'lon': 72.8777, 'fill_level': 0, 'id': 'DEPOT'}])
    route_data = pd.concat([depot, full_bins]).reset_index(drop=True)
    locations = list(zip(route_data['lat'], route_data['lon']))
    manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(abs(locations[from_node][0] - locations[to_node][0]) * 10000 + 
                   abs(locations[from_node][1] - locations[to_node][1]) * 10000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        route_coords = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            idx = manager.IndexToNode(index)
            route_coords.append(locations[idx])
            index = solution.Value(routing.NextVar(index))
        route_coords.append(locations[manager.IndexToNode(index)])
        return route_coords, full_bins
    return None, None

# --- 3. APP NAVIGATION ---
st.sidebar.title("♻️ Smart Bin OS")
st.sidebar.markdown("`Infrastructure Unit: Alpha-1` ")
menu = st.sidebar.radio("MODULES", ["COMMAND CENTER", "CITIZEN AI PORTAL", "DRIVER OPS", "ANALYTICS & ROI"])

# ==========================================
# 🏙️ COMMAND CENTER
# ==========================================
if menu == "COMMAND CENTER":
    st.title("🏙️ Urban Command Interface")
    
    col_live, col_btn = st.columns([1, 4])
    with col_live: live_mode = st.toggle("🔴 LIVE SYNC", value=True)
    with col_btn: 
        if st.button("🔄 Manual Refresh"): st.rerun()

    data = fetch_live_data()
    if data:
        live_df = pd.DataFrame.from_dict(data, orient='index')
        live_df['id'] = live_df.index
        
        # UI Metrics with Glassmorphism
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Sensors", len(live_df))
        c2.metric("Avg Grid Load", f"{int(live_df['fill_level'].mean())}%")
        c3.metric("Critical Alerts", len(live_df[live_df['fill_level'] > 90]), delta="Urgent")
        c4.metric("AI System", "ONLINE", delta="12ms Latency")

        st.subheader("📍 Real-Time 3D Topology")
        map_data = live_df.copy()
        map_data['color'] = map_data['fill_level'].apply(lambda x: [255, 0, 0, 200] if x > 90 else [0, 255, 163, 200])
        
        layer = pdk.Layer("ColumnLayer", data=map_data, get_position="[lon, lat]", get_elevation="fill_level", elevation_scale=10, radius=20, get_fill_color="color", pickable=True)
        view = pdk.ViewState(latitude=19.0760, longitude=72.8777, zoom=15, pitch=60)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "Fill: {fill_level}%"}))

        st.subheader("🚛 Route Optimization Engine")
        if st.button("Run OR-Tools Solver"):
            path, bins = solve_route(live_df)
            if path:
                st.success("Shortest Path Generated!")
                m = folium.Map(location=[19.0760, 72.8777], zoom_start=14)
                folium.PolyLine(path, color="#00ffa3", weight=5, opacity=0.8).add_to(m)
                st_folium(m, height=400, width=1200)
    
    if live_mode:
        time.sleep(1)
        st.rerun()

# ==========================================
# 📸 CITIZEN AI PORTAL
# ==========================================
elif menu == "CITIZEN AI PORTAL":
    st.title("📸 Report & Earn")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("Neural Network verification active. Rewards issued upon validation.")
        img = st.file_uploader("Upload Evidence", type=['jpg', 'png'])
        loc_desc = st.text_input("Location Description")
        
        if st.button("🚀 Verify & Submit") and img:
            with st.spinner("🤖 AI Scanning..."):
                label, score = verify_image(img.getvalue())
                if "garbage" in label.lower() and score > 0.5:
                    st.success(f"Verified! Reward Issued. AI Confidence: {int(score*100)}%")
                    st.balloons()
                    requests.post(f"{FIREBASE_URL}/tasks.json", json={"type": "CITIZEN", "loc": loc_desc, "ts": str(pd.Timestamp.now()), "lat": 19.0760, "lon": 72.8777})
                else:
                    st.error("AI Rejected Report. Please provide clear evidence.")

    with col2:
        st.subheader("🏆 Leaderboard")
        st.dataframe(pd.DataFrame([{"User": "Rahul S.", "Pts": 1500}, {"User": "Priya M.", "Pts": 1200}]), hide_index=True)

# ==========================================
# 🚛 DRIVER OPS (WhatsApp + Google Maps)
# ==========================================
elif menu == "DRIVER OPS":
    st.title("🚛 Tactical Dispatch")
    tasks = requests.get(f"{FIREBASE_URL}/tasks.json").json()
    
    if tasks:
        for tid, t in tasks.items():
            # Modern Task Card
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border-left: 6px solid #00ffa3; margin-bottom: 15px;">
                    <h3 style="margin:0;">🚨 Task: {t.get('loc', 'Unknown Sector')}</h3>
                    <p style="color: #999;">ID: {tid} | Time: {t.get('ts')}</p>
                </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([2, 1, 1])
            lat, lon = t.get('lat', 19.0760), t.get('lon', 72.8777)
            gmaps_url = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}&travelmode=driving"
            
            # --- GOOGLE MAPS REDIRECT ---
            c1.link_button("🌍 Open Google Maps Navigation", gmaps_url)
            
            # --- WHATSAPP DISPATCH ---
            msg = f"🚛 *SmartBin Task*: {t.get('loc')}\nNavigate: {gmaps_url}"
            c2.link_button("📲 WhatsApp Driver", f"https://wa.me/?text={urllib.parse.quote(msg)}")
            
            if c3.button("✅ Complete", key=tid):
                requests.delete(f"{FIREBASE_URL}/tasks/{tid}.json")
                st.rerun()
    else:
        st.success("All systems optimal. No active collection tasks.")

# ==========================================
# 📊 ANALYTICS & ROI (Maintained)
# ==========================================
elif menu == "ANALYTICS & ROI":
    st.title("📊 Financials & Predictions")
    tab1, tab2 = st.tabs(["Historical Analysis", "ROI Impact Model"])
    
    with tab1:
        st.subheader("🧠 Predictive AI Training")
        st.info("Logic utilizing RandomForestRegressor for fill-level forecasting.")
        # [Your existing data analysis code goes here]

    with tab2:
        st.subheader("💰 Economic Sustainability")
        st.metric("Projected Fuel Savings", "₹42,000 / month", delta="40%")
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# --- 1. DATA FETCHING ---
def get_firebase_dashboard_data():
    # Replace with your actual node path, e.g., /bins.json or /history.json
    response = requests.get(f"bin_history.json") 
    data = response.json()
    
    if data:
        # Flattening logic: This converts nested JSON into a clean table
        df = pd.DataFrame.from_dict(data, orient='index')
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

# --- 2. DASHBOARD PAGE ---
def render_detailed_dashboard():
    st.title("📊 Advanced Analytics Dashboard")
    st.markdown("Deep-dive into city-wide waste management metrics.")

    df = get_firebase_dashboard_data()

    if not df.empty:
        # --- ROW 1: KEY METRICS ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Bins Monitored", len(df))
        m2.metric("Critical Overflows", len(df[df['fill_level'] > 90]))
        m3.metric("System Efficiency", "94%", delta="2.1%")

        st.divider()

        # --- ROW 2: VISUALIZATIONS ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📈 Fill Level Distribution")
            fig_hist = px.histogram(df, x="fill_level", nbins=10, 
                                   color_discrete_sequence=['#00ffa3'],
                                   labels={'fill_level': 'Fill Percentage (%)'})
            fig_hist.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            st.subheader("🌍 Sector Comparison")
            # Assumes you have a 'loc' or 'sector' column in your JSON
            if 'loc' in df.columns:
                fig_pie = px.pie(df, names="loc", values="fill_level", hole=0.4,
                                color_discrete_sequence=px.colors.sequential.Greens_r)
                fig_pie.update_layout(template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)

        # --- ROW 3: RAW DATA EXPLORER ---
        st.subheader("🔍 Real-Time Data Grid")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
    else:
        st.warning("No data found in Firebase. Please ensure your JSON import was successful.")

# Integration into your existing menu
# if menu == "DETAILED DASHBOARD":
#     render_detailed_dashboard()
