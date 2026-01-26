import streamlit as st

st.set_page_config(
    page_title="Smart Bin – Smart Waste Management",
    layout="wide"
)

# ---------------- LANDING PAGE HTML ----------------
landing_page = """
<style>
/* GLOBAL RESET */
html, body {
    margin: 0;
    padding: 0;
    font-family: 'Segoe UI', sans-serif;
    background-color: #020617;
}

/* HERO SECTION */
.hero {
    min-height: 100vh;
    background: radial-gradient(circle at top, #064e3b, #020617);
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: white;
    padding: 40px;
}

.hero h1 {
    font-size: 64px;
    font-weight: 800;
    color: #34D399;
}

.hero p {
    font-size: 22px;
    max-width: 800px;
    margin: 20px auto;
    color: #E5E7EB;
}

.hero button {
    margin-top: 30px;
    padding: 16px 36px;
    font-size: 18px;
    background: linear-gradient(135deg, #10B981, #059669);
    border: none;
    border-radius: 12px;
    color: white;
    font-weight: 700;
    cursor: pointer;
}

/* SECTION */
.section {
    padding: 80px 10%;
    background-color: #020617;
    color: white;
}

.section h2 {
    font-size: 42px;
    color: #34D399;
    margin-bottom: 20px;
}

.section p {
    font-size: 18px;
    max-width: 900px;
    color: #D1D5DB;
}

/* CARDS */
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 30px;
    margin-top: 50px;
}

.card {
    background: rgba(17, 24, 39, 0.9);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
}

.card h3 {
    color: #A7F3D0;
    font-size: 22px;
    margin-bottom: 10px;
}

.card p {
    color: #9CA3AF;
    font-size: 16px;
}

/* FOOTER */
.footer {
    padding: 40px;
    text-align: center;
    color: #9CA3AF;
    background: #020617;
}
</style>

<!-- HERO -->
<div class="hero">
    <div>
        <h1>Smart Bin</h1>
        <p>
            An AI & IoT powered smart waste management system that enables
            real-time monitoring, predictive waste collection, and sustainable
            urban planning for smart cities.
        </p>
        <button>Explore the System</button>
    </div>
</div>

<!-- ABOUT -->
<div class="section">
    <h2>Why Smart Bin?</h2>
    <p>
        Traditional waste collection is reactive, inefficient, and costly.
        Smart Bin transforms waste management using real-time IoT data,
        cloud analytics, and AI-driven predictions to eliminate overflow,
        optimize routes, and reduce environmental impact.
    </p>
</div>

<!-- FEATURES -->
<div class="section">
    <h2>Key Features</h2>
    <div class="cards">
        <div class="card">
            <h3>IoT-Based Monitoring</h3>
            <p>Ultrasonic and load sensors track fill level and weight in real time.</p>
        </div>
        <div class="card">
            <h3>Real-Time Alerts</h3>
            <p>Automated notifications to authorities before bins overflow.</p>
        </div>
        <div class="card">
            <h3>AI Waste Verification</h3>
            <p>AI verifies waste images submitted by citizens for accuracy.</p>
        </div>
        <div class="card">
            <h3>Predictive Analytics</h3>
            <p>Forecasts bin fill times and enables proactive route planning.</p>
        </div>
        <div class="card">
            <h3>Smart Routing</h3>
            <p>Optimized collection routes reduce fuel usage and emissions.</p>
        </div>
        <div class="card">
            <h3>Citizen & Driver Portals</h3>
            <p>Dedicated portals for public reporting and driver task management.</p>
        </div>
    </div>
</div>

<!-- FOOTER -->
<div class="footer">
    Smart Bin • IoT • AI • Smart Cities • Sustainability
</div>
"""

st.markdown(landing_page, unsafe_allow_html=True)
