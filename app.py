import streamlit as st

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Bin – Smart Waste Management",
    layout="wide"
)

# -------------------------------------------------
# FULL PAGE HTML + CSS
# -------------------------------------------------
page_html = """
<style>
/* ===== GLOBAL ===== */
html, body {
    margin: 0;
    padding: 0;
    font-family: 'Segoe UI', sans-serif;
    background-color: #0E1628;
}

/* ===== HERO ===== */
.hero {
    min-height: 100vh;
    background: linear-gradient(180deg, #1F3A34, #0E1628);
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 40px;
}
.hero h1 {
    font-size: 64px;
    font-weight: 800;
    color: #4ADE80;
}
.hero p {
    font-size: 22px;
    max-width: 800px;
    margin: 20px auto;
    color: #9CA3AF;
}
.hero button {
    margin-top: 30px;
    padding: 16px 36px;
    font-size: 18px;
    background: linear-gradient(135deg, #4ADE80, #22C55E);
    border: none;
    border-radius: 12px;
    color: #022C22;
    font-weight: 700;
}

/* ===== SECTION ===== */
.section {
    padding: 90px 10%;
    background-color: #111B2E;
}
.section h2 {
    font-size: 42px;
    color: #E5E7EB;
}
.section p {
    font-size: 18px;
    max-width: 900px;
    color: #9CA3AF;
}

/* ===== FOOTER ===== */
.site-footer {
    background-color: #1f1f1f;
    padding: 80px 8% 30px 8%;
    color: #cbd5e1;
}

.footer-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1.5fr;
    gap: 60px;
}

.footer-brand {
    display: flex;
    gap: 15px;
}

.footer-brand img {
    width: 48px;
}

.footer-brand h3 {
    color: #22c55e;
    margin-bottom: 10px;
    font-size: 20px;
}

.footer-brand p {
    font-size: 15px;
    line-height: 1.6;
    color: #94a3b8;
}

.footer-col h4 {
    color: #ffffff;
    margin-bottom: 18px;
    font-size: 18px;
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

.socials {
    display: flex;
    gap: 12px;
    margin-top: 20px;
}
.socials a {
    width: 38px;
    height: 38px;
    border-radius: 8px;
    background-color: #111827;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    text-decoration: none;
}

.footer-bottom {
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.08);
    display: flex;
    justify-content: space-between;
    font-size: 14px;
    color: #94a3b8;
}
.footer-bottom a {
    color: #94a3b8;
    margin-left: 20px;
    text-decoration: none;
}
.footer-bottom a:hover {
    color: #22c55e;
}
</style>

<!-- HERO -->
<div class="hero">
    <div>
        <h1>Smart Bin</h1>
        <p>
            AI & IoT powered smart waste management system for predictive
            collection and sustainable smart cities.
        </p>
        <button>Explore the System</button>
    </div>
</div>

<!-- ABOUT -->
<div class="section">
    <h2>Why Smart Bin?</h2>
    <p>
        Smart Bin replaces reactive waste collection with real-time monitoring,
        predictive analytics, and optimized routing.
    </p>
</div>

<!-- FOOTER -->
<footer class="site-footer">
    <div class="footer-grid">

        <div class="footer-brand">
            <img src="https://img.icons8.com/fluency/96/recycle.png" />
            <div>
                <h3>IoT-based Smart Waste<br>Monitoring System</h3>
                <p>
                    Transforming waste management with IoT-powered real-time
                    monitoring, predictive analytics, and optimized routes.
                </p>
                <div class="socials">
                    <a href="#">🐦</a>
                    <a href="#">💼</a>
                    <a href="#">🐙</a>
                </div>
            </div>
        </div>

        <div class="footer-col">
            <h4>Quick Links</h4>
            <a href="#">Home</a>
            <a href="#">About</a>
            <a href="#">Features</a>
            <a href="#">Solutions</a>
        </div>

        <div class="footer-col">
            <h4>Resources</h4>
            <a href="#">Dashboard</a>
            <a href="#">Technology</a>
            <a href="#">Data Analytics</a>
            <a href="#">Contact Us</a>
        </div>

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
            <a href="#">Privacy Policy</a>
            <a href="#">Terms of Service</a>
        </div>
    </div>
</footer>
"""

# -------------------------------------------------
# RENDER HTML (THIS LINE IS CRITICAL)
# -------------------------------------------------
st.markdown(page_html, unsafe_allow_html=True)
