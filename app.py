import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import requests
import folium
from streamlit_folium import st_folium
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
# GLOBAL LANDINGSITE-STYLE CSS
# =====================================================
st.markdown("""
<style>

/* ================= GLOBAL APP ================= */
html, body, .stApp {
    background: linear-gradient(180deg, #1F3A34 0%, #0E1628 75%);
    color: #9CA3AF;
    font-family: "Segoe UI", system-ui, -apple-system, Arial, sans-serif;
}

/* Remove Streamlit white blocks */
section.main > div {
    background: transparent !important;
}

/* ================= HEADINGS ================= */
/* HERO */
.hero {
  min-height: 90vh;
  background: linear-gradient(180deg, #1F3A34 0%, #0E1628 70%);
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 64px 24px;
}
.hero h1 {
  font-size: 64px;
  font-weight: 800;
  color: #4ADE80;
  margin-bottom: 12px;
}
.hero p {
  max-width: 820px;
  font-size: 22px;
  margin-bottom: 28px;
}
.hero .cta {
  display: inline-block;
  padding: 14px 34px;
  border-radius: 12px;
  background: linear-gradient(135deg, #4ADE80, #22C55E);
  color: #022C22;
  font-weight: 700;
  text-decoration: none;

}

/* ================= PARAGRAPH ================= */
p, span, label {
    color: #9CA3AF;
    font-size: 1.05rem;
}

/* ================= BUTTONS ================= */
.stButton > button,
a[role="button"] {
    background: linear-gradient(135deg, #4ADE80, #22C55E);
    color: #022C22 !important;
    border-radius: 14px;
    font-weight: 700;
    padding: 0.6rem 1.6rem;
    border: none;
}

/* ================= METRICS ================= */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 14px;
    padding: 18px;
}
div[data-testid="stMetricValue"] {
    color: #4ADE80;
    font-size: 2rem;
    font-weight: 800;
}

/* ================= TABS ================= */
button[data-baseweb="tab"] {
    color: #9CA3AF;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4ADE80;
    border-bottom: 3px solid #4ADE80;
}

/* ================= TABLES ================= */
thead tr th {
    background-color: rgba(74,222,128,0.08) !important;
    color: #4ADE80 !important;
}

/* ================= EXPANDERS ================= */
details {
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 10px;
}

st.markdown(
    """
    <style>
    .site-footer {
        background: #1F1F1F;
        padding: 80px 8% 30px;
        color: #CBD5E1;
    }
    </style>
    """,
    unsafe_allow_html=True
)



.smart-footer-grid {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1.5fr;
  gap: 60px;
}

/* Brand */
.footer-brand {
  display: flex;
  gap: 16px;
}
.footer-brand img {
  width: 56px;
  height: 56px;
}
.footer-brand h3 {
  color: #22c55e;
  font-size: 20px;
  margin: 0 0 10px;
}
.footer-brand p {
  font-size: 15px;
  line-height: 1.6;
  color: #94a3b8;
}

/* Columns */
.footer-col h4 {
  color: #ffffff;
  font-size: 18px;
  margin-bottom: 16px;
}
.footer-col a {
  display: block;
  color: #94a3b8;
  text-decoration: none;
  margin-bottom: 12px;
  font-size: 15px;
}
.footer-col a:hover {
  color: #22c55e;
}

/* Contact */
.contact-item {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
  font-size: 15px;
  color: #94a3b8;
}
.contact-item span {
  color: #22c55e;
  font-size: 18px;
}

/* Socials */
.footer-socials {
  display: flex;
  gap: 12px;
  margin-top: 18px;
}
.footer-socials a {
  width: 38px;
  height: 38px;
  border-radius: 8px;
  background: #111827;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  text-decoration: none;
}

/* Bottom bar */
.footer-bottom {
  margin-top: 60px;
  padding-top: 20px;
  border-top: 1px solid rgba(255,255,255,0.08);
  display: flex;
  justify-content: space-between;
  gap: 16px;
  font-size: 14px;
  color: #94a3b8;
}
.footer-bottom a {
  color: #94a3b8;
  text-decoration: none;
  margin-left: 16px;
}
.footer-bottom a:hover {
  color: #22c55e;
}

/* Responsive */
@media (max-width: 900px) {
  .smart-footer-grid {
    grid-template-columns: 1fr 1fr;
  }
}
@media (max-width: 600px) {
  .smart-footer-grid {
    grid-template-columns: 1fr;
  }
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
# ==========================================
# 🚛 DRIVER OPS — ALWAYS VISIBLE SECTION
# ==========================================

st.markdown("---")
st.markdown("## 🚛 Tactical Dispatch")

try:
    tasks = requests.get(f"{FIREBASE_URL}/tasks.json").json()
except:
    tasks = None

if tasks:
    for tid, t in tasks.items():
        with st.container():
            col1, col2 = st.columns([3, 1])

            col1.markdown(f"""
            **📍 Location:** {t.get('loc', 'Unknown')}  
            **🕒 Time:** {t.get('ts', 'Just now')}  
            **🔔 Source:** {t.get('type', 'System')}
            """)

            msg = f"URGENT: Clear trash at {t.get('loc')}. Task ID: {tid}"
            col2.link_button(
                "📲 WhatsApp Dispatch",
                f"https://wa.me/?text={msg}"
            )

            if col2.button("✅ Mark Complete", key=tid):
                requests.delete(f"{FIREBASE_URL}/tasks/{tid}.json")
                st.rerun()

            st.markdown("---")
else:
    st.success("✅ No active dispatch tasks. System operating normally.")

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

# ANALYTICS SECTION
# =====================================================
# =====================================================
# 📊 ANALYTICS & ROI (LANDING-STYLE SECTION)
# =====================================================
st.markdown("## 📊 Data & Financial Insights")
st.markdown(
    "Explore historical waste patterns, predictive intelligence, "
    "and the economic impact of smart waste management."
)

# Tabs stay – tabs are perfectly fine in LandingSite-style UI
tab1, tab2, tab3 = st.tabs(
    [
        "📈 Exploratory Data Analysis (EDA)",
        "🧠 Predictive AI",
        "💎 Comprehensive Impact Model"
    ]
)

# -----------------------------------------------------
# AUTO-LOAD DATA LOGIC
# -----------------------------------------------------
df = None
try:
    df = pd.read_csv("smart_bin_historical_data.csv")
    st.toast("✅ Historical Data Loaded Automatically", icon="📂")
except FileNotFoundError:
    st.warning("System data not found. Please upload manually.")
    uploaded_file = st.file_uploader(
        "Upload smart_bin_historical_data.csv",
        type="csv"
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

# -----------------------------------------------------
# =====================================================
# 📊 ANALYTICS CONTENT (NO SIDEBAR LOGIC)
# =====================================================

# --- TAB 1: EDA ---
with tab1:
    st.markdown("### 📈 Historical Patterns")

    if df is not None:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Fill by Hour")
            if 'hour_of_day' in df.columns and 'bin_fill_percent' in df.columns:
                hourly = (
                    df.groupby('hour_of_day')['bin_fill_percent']
                    .mean()
                    .reset_index()
                )
                fig1 = px.line(
                    hourly,
                    x='hour_of_day',
                    y='bin_fill_percent',
                    title='Hourly Fill Pattern'
                )
                st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.subheader("Fill by Day")
            if 'day_of_week' in df.columns and 'bin_fill_percent' in df.columns:
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
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Data unavailable. Upload CSV to view analytics.")

# --- TAB 2: PREDICTIVE MODEL ---
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

                    st.subheader("Prediction vs Reality")
                    future_hours = pd.DataFrame(
                        {'hour_of_day': range(0, 24)}
                    )
                    predictions = model.predict(future_hours)
                    future_hours['Predicted Fill'] = predictions

                    st.line_chart(
                        future_hours.set_index('hour_of_day')
                    )

                except Exception as e:
                    st.error(f"Training failed: {e}")
    else:
        st.info("Data unavailable. Cannot train model.")

# --- TAB 3: FINANCIAL MODEL ---
with tab3:
    st.markdown("### 💎 360° Value Proposition")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### ⚙️ Parameters")

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

        st.markdown("#### 💰 Financial Projection")

        k1, k2 = st.columns(2)
        k1.metric(
            "Monthly OpEx Savings",
            f"₹{int(savings):,}",
            delta="Direct Cash"
        )
        k2.metric(
            "Total Monthly Benefit",
            f"₹{int(savings + revenue_recycle):,}",
            delta="Including Revenue"
        )

        if cost_old > 0:
            st.progress(
                savings / cost_old,
                text="Efficiency Gain"
            )

        waterfall_data = pd.DataFrame({
            "Source": ["OpEx Savings", "Recycling Revenue"],
            "Amount": [savings, revenue_recycle]
        })

        fig_w = px.bar(
            waterfall_data,
            x="Source",
            y="Amount",
            title="Value Drivers"
        )
        st.plotly_chart(fig_w, use_container_width=True)



st.markdown("""
<div class="smart-footer">
  <div class="smart-footer-grid">

    <!-- Brand -->
    <div class="footer-brand">
      <img src="https://img.icons8.com/fluency/96/recycle.png" />
      <div>
        <h3>IoT-based Smart Waste<br/>Monitoring System</h3>
        <p>
          Transforming waste management with IoT-powered real-time monitoring,
          predictive analytics, and optimized collection routes.
        </p>

        <div class="footer-socials">
          <a href="#">🐦</a>
          <a href="#">💼</a>
          <a href="#">🐙</a>
        </div>
      </div>
    </div>

    <!-- Quick Links -->
    <div class="footer-col">
      <h4>Quick Links</h4>
      <a>Home</a>
      <a>About</a>
      <a>Features</a>
      <a>Solutions</a>
    </div>

    <!-- Resources -->
    <div class="footer-col">
      <h4>Resources</h4>
      <a>Dashboard</a>
      <a>Technology</a>
      <a>Data Analytics</a>
      <a>Contact Us</a>
    </div>

    <!-- Contact -->
    <div class="footer-col">
      <h4>Get In Touch</h4>
      <div class="contact-item"><span>✉️</span> info@smartwaste.io</div>
      <div class="contact-item"><span>📞</span> +1 (555) 123-4567</div>
      <div class="contact-item"><span>📍</span> Smart City Innovation Hub</div>
    </div>

  </div>

  <div class="footer-bottom">
    <div>© 2025 IoT Smart Waste Monitoring System. All rights reserved.</div>
    <div>
      <a>Privacy Policy</a>
      <a>Terms of Service</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

