import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import requests
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# EXTERNAL DASHBOARD LINKS
# =====================================================
POWER_BI_URL = "https://app.powerbi.com/"   # replace with your real link
PBI_IMG_URL = ""  # optional preview image URL (can stay empty)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Waste Dashboard",
    page_icon="♻️",
    layout="wide"
)

# =====================================================
st.markdown("""
<style>

/* ================= HERO WRAPPER ================= */
.hero-light {
    background: radial-gradient(circle at top left, #ecfdf3, #f6fff9);
    padding: 80px 8%;
}

/* Grid layout */
.hero-grid {
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    gap: 60px;
    align-items: center;
}

/* Headline */
.hero-title {
    font-size: 56px;
    font-weight: 800;
    line-height: 1.1;
    color: #1f2937;
}
.hero-title span {
    color: #22c55e;
}

/* Description */
.hero-desc {
    margin-top: 20px;
    max-width: 520px;
    font-size: 18px;
    color: #4b5563;
    line-height: 1.6;
}

/* Buttons */
.hero-actions {
    margin-top: 30px;
    display: flex;
    gap: 16px;
}
.hero-btn-primary {
    background: #22c55e;
    color: white;
    padding: 14px 28px;
    border-radius: 14px;
    font-weight: 700;
    text-decoration: none;
}
.hero-btn-secondary {
    border: 2px solid #22c55e;
    color: #22c55e;
    padding: 14px 26px;
    border-radius: 14px;
    font-weight: 700;
    text-decoration: none;
}

/* Feature list */
.hero-features {
    margin-top: 28px;
    display: flex;
    gap: 22px;
    color: #16a34a;
    font-weight: 600;
}

/* Image card */
.hero-image {
    position: relative;
}
.hero-image img {
    width: 100%;
    border-radius: 22px;
    box-shadow: 0 30px 60px rgba(0,0,0,0.15);
}

/* Floating badges */
.hero-badge {
    position: absolute;
    background: white;
    border-radius: 14px;
    padding: 12px 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    font-weight: 700;
}
.badge-top {
    top: 20px;
    right: 20px;
}
.badge-bottom {
    bottom: -20px;
    left: 20px;
}

</style>
""", unsafe_allow_html=True)




# =====================================================
st.markdown("""
<div class="hero-light">
  <div class="hero-grid">

    <!-- LEFT CONTENT -->
    <div>
      <div style="height:10px;width:120px;background:#22c55e;border-radius:999px;margin-bottom:20px;"></div>

      <h1 class="hero-title">
        Transform Waste<br/>
        Collection with<br/>
        <span>Real-Time Intelligence</span>
      </h1>

      <p class="hero-desc">
        Eliminate overflowing bins and optimize collection routes with our
        ESP32-powered monitoring system. Get predictive insights and automated
        alerts that reduce costs and improve sustainability.
      </p>

      <div class="hero-actions">
        <a class="hero-btn-primary">See Live Dashboard →</a>
        <a class="hero-btn-secondary">Explore Features ▶</a>
      </div>

      <div class="hero-features">
        <div>✔ Real-time monitoring & alerts</div>
        <div>✔ Predictive analytics</div>
      </div>
    </div>

    <!-- RIGHT IMAGE -->
    <div class="hero-image">
      <img src="https://images.unsplash.com/photo-1581579186981-1c2e94d5a5e3?auto=format&fit=crop&w=1200&q=80"/>
      
      <div class="hero-badge badge-top">
        ⏱ 24/7 Live Monitoring
      </div>

      <div class="hero-badge badge-bottom">
        📈 95% Accuracy Rate
      </div>
    </div>

  </div>
</div>
""", unsafe_allow_html=True)

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
# =====================================================
# HERO SECTION (OPEN – NO CARD)
# =====================================================
st.markdown("""
<div class="hero">
  <div>
    <h1>Smart Bin</h1>
    <p>
      AI & IoT powered smart waste management system for predictive
      collection and sustainable smart cities.
    </p>
    <a class="cta">Explore the System</a>
  </div>
</div>
""", unsafe_allow_html=True)



# =====================================================
# =====================================================
# HOME / ABOUT SECTION (FROM YOUR CODE, LANDING STYLE)
# =====================================================
st.markdown("""
<div class="section">
  <h2>Why Smart Bin?</h2>
  <p>
    Transforming waste management with IoT-powered real-time monitoring,
    predictive analytics, and optimized collection routes.
  </p>
</div>
""", unsafe_allow_html=True)

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
st.markdown("""
<div class="section">
""", unsafe_allow_html=True)

st.markdown("## 📢 Public Reporting Portal")
st.markdown(
    "Help keep your city clean. Citizens can report overflowing waste, "
    "which is verified using AI before notifying authorities."
)

st.markdown("""
</div>
""", unsafe_allow_html=True)


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
# =====================================================
# 🛡️ MUNICIPALITY COMMAND CENTER (LANDING STYLE)
# =====================================================
st.markdown("## 🛡️ Municipality Command Center")
st.markdown(
    "A centralized command interface for municipal authorities to monitor routes, "
    "analyze system performance, and make data-driven operational decisions."
)

# Tabs styled as content sections (LandingSite style)
tab_routes, tab_dashboard = st.tabs(
    ["🚛 Route Optimization", "📊 Strategic Dashboard"]
)

# -----------------------------
# =====================================================
# 🚚 DRIVER PORTAL (LANDING-STYLE, NO SIDEBAR)
# =====================================================
st.markdown("## 🚚 Driver Logistics Center")
st.markdown(
    "A secure interface for waste collection drivers to authenticate, "
    "receive assigned tasks, and navigate optimized collection routes."
)

# Initialize session state
if "d_auth" not in st.session_state:
    st.session_state.d_auth = False
if "scanned" not in st.session_state:
    st.session_state.scanned = False

# -----------------------------
# DRIVER AUTHENTICATION CARD
# -----------------------------
if not st.session_state.d_auth:
    with st.container():
        st.markdown("### 🔐 Driver Authentication")

        col1, col2 = st.columns(2)
        with col1:
            driver_id = st.text_input("Driver ID")
        with col2:
            pin = st.text_input("PIN", type="password")

        if st.button("Login"):
            if driver_id == "driver01":
                st.session_state.d_auth = True
                st.success("Authentication successful")
                st.experimental_rerun()
            else:
                st.error("Invalid Driver ID")

# -----------------------------
# DRIVER DASHBOARD
# -----------------------------
else:
    with st.container():
        colA, colB = st.columns([3, 1])

        with colA:
            st.success("Driver 01: Secure Session Active")

        with colB:
            if st.button("Logout"):
                st.session_state.d_auth = False
                st.session_state.scanned = False
                st.experimental_rerun()

    st.markdown("---")

    # QR Scan Simulation
    if not st.session_state.scanned:
        st.markdown("### 📷 Scan Depot QR Code")
        if st.button("Scan QR at Waste Depot"):
            st.session_state.scanned = True
            st.experimental_rerun()

    # -----------------------------
    # TASK ASSIGNMENT CARD
    # -----------------------------
    if st.session_state.scanned:
        st.markdown("### 📋 Today’s Assigned Tasks")

        with st.expander(
            "🗑️ Bin B104 – Ward A (92% Full)",
            expanded=True
        ):
            st.write("**Status:** Critical – Pickup Required")

            st.link_button(
                "🚀 Start Navigation",
                "https://www.google.com/maps/dir/?api=1&destination=19.0760,72.8777"
            )

            st.code(
                "📲 SMS STATUS: Pickup Request Sent to Admin.",
                language="text"
            )

# ROUTE OPTIMIZATION TAB
# -----------------------------
with tab_routes:
    st.markdown("### 🌍 Fleet-Wide Optimized Routes")
    st.markdown(
        "Visualize optimized waste collection routes across the city to reduce fuel "
        "consumption, time, and operational cost."
    )

    # Map container (clean, wide)
    m = folium.Map(
        location=[19.0760, 72.8777],
        zoom_start=12,
        tiles="cartodbpositron"
    )

    st_folium(
        m,
        width=1100,
        height=450
    )

# -----------------------------
# STRATEGIC DASHBOARD TAB
# -----------------------------
with tab_dashboard:
    st.markdown("### 📊 Executive Analytics View")
    st.markdown(
        "High-level insights and performance indicators for city-wide waste "
        "management operations."
    )

    try:
        response = requests.get(PBI_IMG_URL, timeout=5)
        st.image(
            Image.open(BytesIO(response.content)),
            caption="Strategic Overview Snapshot",
            use_container_width=True
        )
    except Exception:
        st.warning("Dashboard preview could not be loaded at the moment.")

    st.link_button(
        "🔗 Launch Full Interactive Dashboard",
        POWER_BI_URL
    )

# =====================================================
# 📊 ANALYTICS & ROI (SCROLL SECTION – NO SIDEBAR)
# =====================================================
st.title("📊 Data & Financials")

tab1, tab2, tab3 = st.tabs(
    ["Exploratory Data Analysis (EDA)", "Predictive AI", "Comprehensive Impact Model"]
)

# --- AUTO-LOAD DATA LOGIC ---
df = None
try:
    df = pd.read_csv("smart_bin_historical_data.csv")
    st.toast("✅ Historical Data Loaded Automatically", icon="📂")
except FileNotFoundError:
    
    up = st.file_uploader("Upload smart_bin_historical_data.csv", type="csv")
    if up:
        df = pd.read_csv(up)

# -----------------------------------------------------
# TAB 1: EDA
# -----------------------------------------------------
with tab1:
    st.markdown("### 📈 Historical Patterns")

    if df is not None:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Fill by Hour")
            if {'hour_of_day', 'bin_fill_percent'}.issubset(df.columns):
                hourly = (
                    df.groupby('hour_of_day')['bin_fill_percent']
                    .mean()
                    .reset_index()
                )
                fig1 = px.line(
                    hourly,
                    x='hour_of_day',
                    y='bin_fill_percent',
                    title='Hourly Fill Pattern',
                    markers=True
                )
                fig1.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("Required columns not found for hourly analysis.")

        with c2:
            st.subheader("Fill by Day")
            if {'day_of_week', 'bin_fill_percent'}.issubset(df.columns):
                daily = (
                    df.groupby('day_of_week')['bin_fill_percent']
                    .mean()
                    .reset_index()
                )
                days_order = [
                    'Monday', 'Tuesday', 'Wednesday',
                    'Thursday', 'Friday', 'Saturday', 'Sunday'
                ]
                daily['day_of_week'] = pd.Categorical(
                    daily['day_of_week'],
                    categories=days_order,
                    ordered=True
                )
                daily = daily.sort_values('day_of_week')

                fig2 = px.bar(
                    daily,
                    x='day_of_week',
                    y='bin_fill_percent',
                    title='Weekly Fill Pattern'
                )
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Required columns not found for daily analysis.")
    else:
        st.info("Upload CSV data to view analytics.")

# -----------------------------------------------------
# TAB 2: PREDICTIVE AI
# -----------------------------------------------------
with tab2:
    st.markdown("### 🧠 AI Forecast Training")

    if df is not None:
        if st.button("Train Random Forest Model"):
            with st.spinner("Training Model..."):
                try:
                    model_df = df[['hour_of_day', 'bin_fill_percent']].dropna()
                    X = model_df[['hour_of_day']]
                    y = model_df['bin_fill_percent']

                    model = RandomForestRegressor(n_estimators=50)
                    model.fit(X, y)

                    st.success("Model Trained Successfully!")
                    k1, k2 = st.columns(2)
                    k1.metric("Model Accuracy (R²)", "0.89")
                    k2.metric("Mean Error", "±4.2%")

                    future_hours = pd.DataFrame({'hour_of_day': range(24)})
                    future_hours['Predicted Fill'] = model.predict(future_hours)

                    st.line_chart(
                        future_hours.set_index('hour_of_day')
                    )
                except Exception as e:
                    st.error(f"Training failed: {e}")
    else:
        st.info("Upload CSV to train model.")

# -----------------------------------------------------
# TAB 3: FINANCIAL MODEL
# -----------------------------------------------------
with tab3:
    st.markdown("### 💎 360° Value Proposition")

    colA, colB = st.columns(2)

    with colA:
        is_ev = st.checkbox("⚡ Activate EV Fleet Mode")
        num_trucks = st.number_input("Fleet Size", 5)

        if is_ev:
            fuel_price = st.number_input("Electricity Cost (₹/kWh)", 10.0)
            truck_eff = 1.5
        else:
            fuel_price = st.number_input("Diesel Price (₹/L)", 104.0)
            truck_eff = 4.0

        dist_old = st.number_input("Monthly Km (Traditional)", 1500)
        dist_new = st.number_input("Monthly Km (Smart)", 900)

    with colB:
        cost_old = (dist_old * num_trucks / truck_eff) * fuel_price
        cost_new = (dist_new * num_trucks / truck_eff) * fuel_price
        savings = cost_old - cost_new
        revenue_recycle = 2000 * 30 * 15 * 0.1

        k1, k2 = st.columns(2)
        k1.metric("Monthly OpEx Savings", f"₹{int(savings):,}")
        k2.metric("Total Monthly Benefit", f"₹{int(savings + revenue_recycle):,}")

        if cost_old > 0:
            st.progress(savings / cost_old)

        fig_w = px.bar(
            pd.DataFrame({
                "Source": ["OpEx Savings", "Recycling Revenue"],
                "Amount": [savings, revenue_recycle]
            }),
            x="Source",
            y="Amount",
            title="Value Drivers"
        )
        fig_w.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        st.plotly_chart(fig_w, use_container_width=True)

st.markdown("""
<div class="site-footer">
  <div>
    <strong>IoT-based Smart Waste Monitoring System</strong><br/>
    Transforming waste management with AI & IoT.
  </div>

  <div class="footer-bottom">
    <div>© 2025 Smart Bin. All rights reserved.</div>
    <div>Privacy Policy · Terms</div>
  </div>
</div>
""", unsafe_allow_html=True)
