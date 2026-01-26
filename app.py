import streamlit as st

st.set_page_config(
    page_title="Smart Bin – Smart Waste Management",
    layout="wide"
)

landing_page = """
<style>
/* GLOBAL */
html, body {
    margin: 0;
    padding: 0;
    font-family: 'Segoe UI', sans-serif;
    background-color: #020617;
}

/* HERO */
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
    padding: 90px 10%;
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

/* CONTACT */
.contact {
    background: #020617;
    padding: 90px 10%;
}
.contact-box {
    max-width: 700px;
    margin: auto;
    background: rgba(17, 24, 39, 0.95);
    padding: 40px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}
.contact-box h2 {
    text-align: center;
    color: #34D399;
    margin-bottom: 20px;
}
.contact-box p {
    text-align: center;
    color: #9CA3AF;
    margin-bottom: 30px;
}
.contact-box input,
.contact-box textarea {
    width: 100%;
    padding: 14px;
    margin-bottom: 18px;
    border-radius: 10px;
    border: none;
    background: #020617;
    color: #E5E7EB;
}
.contact-box button {
    width: 100%;
    padding: 14px;
    background: linear-gradient(135deg, #10B981, #059669);
    border: none;
    border-radius: 10px;
    color: white;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
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
            An AI & IoT powered smart waste management system enabling
            real-time monitoring, predictive collection, and sustainable
            smart city operations.
        </p>
        <button>Explore the System</button>
    </div>
</div>

<!-- ABOUT -->
<div class="section">
    <h2>Why Smart Bin?</h2>
    <p>
        Smart Bin transforms traditional waste collection into a data-driven,
        proactive system using IoT sensors, cloud analytics, and AI-based
        predictions to eliminate overflow and optimize municipal operations.
    </p>
</div>

<!-- FEATURES -->
<div class="section">
    <h2>Key Features</h2>
    <div class="cards">
        <div class="card">
            <h3>IoT Monitoring</h3>
            <p>Real-time fill-level and weight sensing using smart sensors.</p>
        </div>
        <div class="card">
            <h3>Automated Alerts</h3>
            <p>Authorities are notified before bins overflow.</p>
        </div>
        <div class="card">
            <h3>AI Verification</h3>
            <p>AI validates waste images submitted by citizens.</p>
        </div>
        <div class="card">
            <h3>Predictive Analytics</h3>
            <p>Forecasts fill times for proactive route planning.</p>
        </div>
        <div class="card">
            <h3>Route Optimization</h3>
            <p>Optimized routes reduce fuel cost and emissions.</p>
        </div>
        <div class="card">
            <h3>Citizen & Driver Portals</h3>
            <p>Dedicated portals for reporting and logistics.</p>
        </div>
    </div>
</div>

<!-- CONTACT -->
<div class="contact">
    <div class="contact-box">
        <h2>Contact Us</h2>
        <p>
            Interested in Smart Bin? Want to collaborate or deploy it in your city?
            Reach out to us.
        </p>
        <input type="text" placeholder="Your Name">
        <input type="email" placeholder="Your Email">
        <textarea rows="4" placeholder="Your Message"></textarea>
        <button>Send Message</button>
    </div>
</div>

<!-- FOOTER -->
<div class="footer">
    © 2026 Smart Bin • AI • IoT • Smart Cities • Sustainability
</div>
"""

st.markdown(landing_page, unsafe_allow_html=True)
