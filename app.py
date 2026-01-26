import streamlit as st

st.set_page_config(
    page_title="Smart Bin – Smart Waste Management",
    layout="wide"
)

# 🔥 NUCLEAR OVERRIDE — STREAMLIT CANNOT IGNORE THIS
st.markdown("""
<style>
/* Kill Streamlit default theme completely */
:root, html, body, .stApp {
    background-color: #0B1220 !important;
    color-scheme: dark !important;
}

/* Main container */
section.main > div {
    background-color: #0B1220 !important;
}

/* Remove white padding containers */
.block-container {
    padding: 0 !important;
    margin: 0 !important;
    background-color: #0B1220 !important;
}

/* Hide Streamlit UI */
#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# 🔹 LANDING PAGE HTML (COLOR-CORRECT)
st.markdown("""
<style>
.hero {
    min-height: 100vh;
    background: linear-gradient(180deg, #0B1220, #0F172A);
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 40px;
}
.hero h1 {
    font-size: 64px;
    font-weight: 800;
    color: #22C55E;
}
.hero p {
    font-size: 22px;
    max-width: 800px;
    margin: 20px auto;
    color: #9CA3AF;
}
.hero button {
    padding: 16px 36px;
    background: linear-gradient(135deg, #22C55E, #16A34A);
    border: none;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 700;
    cursor: pointer;
    color: #022C22;
}
.section {
    padding: 90px 10%;
    background-color: #0F172A;
}
.section h2 {
    color: #E5E7EB;
    font-size: 42px;
}
.section p {
    color: #9CA3AF;
    font-size: 18px;
    max-width: 900px;
}
</style>

<div class="hero">
    <div>
        <h1>Smart Bin</h1>
        <p>
            AI & IoT powered smart waste management system for
            predictive collection and sustainable smart cities.
        </p>
        <button>Explore the System</button>
    </div>
</div>

<div class="section">
    <h2>Why Smart Bin?</h2>
    <p>
        A premium, data-driven solution that replaces reactive waste
        collection with intelligent monitoring and planning.
    </p>
</div>
""", unsafe_allow_html=True)
