import streamlit as st

st.set_page_config(page_title="Smart Bin", layout="wide")

# EVERYTHING HTML + CSS MUST BE INSIDE THIS STRING
footer_html = """
<style>
.site-footer {
    background: #1f1f1f;
    padding: 80px 8% 30px 8%;
    color: #cbd5e1;
}

.footer-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1.5fr;
    gap: 60px;
}

.footer-col h4 {
    color: #ffffff;
}
</style>

<footer class="site-footer">
    <div class="footer-grid">
        <div class="footer-col">
            <h4>Test Footer</h4>
            <p>This is a test</p>
        </div>
    </div>
</footer>
"""

st.markdown(footer_html, unsafe_allow_html=True)
