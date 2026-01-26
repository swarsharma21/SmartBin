<style>
/* FOOTER WRAPPER */
.site-footer {
    background: #1f1f1f;
    padding: 80px 8% 30px 8%;
    color: #cbd5e1;
}

/* FOOTER GRID */
.footer-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1.5fr;
    gap: 60px;
}

/* BRAND */
.footer-brand {
    display: flex;
    gap: 15px;
    align-items: flex-start;
}
.footer-brand img {
    width: 48px;
}
.footer-brand h3 {
    color: #22c55e;
    margin-bottom: 10px;
}
.footer-brand p {
    font-size: 15px;
    line-height: 1.6;
    color: #94a3b8;
}

/* HEADINGS */
.footer-col h4 {
    color: #ffffff;
    margin-bottom: 18px;
    font-size: 18px;
}

/* LINKS */
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

/* CONTACT INFO */
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

/* SOCIAL ICONS */
.socials {
    display: flex;
    gap: 12px;
    margin-top: 20px;
}
.socials a {
    width: 38px;
    height: 38px;
    border-radius: 8px;
    background: #111827;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    text-decoration: none;
}

/* BOTTOM BAR */
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

<footer class="site-footer">
    <div class="footer-grid">

        <!-- BRAND -->
        <div class="footer-brand">
            <img src="https://img.icons8.com/fluency/96/recycle.png" />
            <div>
                <h3>IoT-based Smart Waste<br>Monitoring System</h3>
                <p>
                    Transforming waste management with IoT-powered real-time
                    monitoring, predictive analytics, and optimized collection routes.
                </p>
                <div class="socials">
                    <a href="#">🐦</a>
                    <a href="#">💼</a>
                    <a href="#">🐙</a>
                </div>
            </div>
        </div>

        <!-- QUICK LINKS -->
        <div class="footer-col">
            <h4>Quick Links</h4>
            <a href="#">Home</a>
            <a href="#">About</a>
            <a href="#">Features</a>
            <a href="#">Solutions</a>
        </div>

        <!-- RESOURCES -->
        <div class="footer-col">
            <h4>Resources</h4>
            <a href="#">Dashboard</a>
            <a href="#">Technology</a>
            <a href="#">Data Analytics</a>
            <a href="#">Contact Us</a>
        </div>

        <!-- CONTACT -->
        <div class="footer-col">
            <h4>Get In Touch</h4>
            <div class="contact-item"><span>✉️</span> info@smartwaste.io</div>
            <div class="contact-item"><span>📞</span> +1 (555) 123-4567</div>
            <div class="contact-item"><span>📍</span> Smart City Innovation Hub</div>
        </div>

    </div>

    <!-- BOTTOM -->
    <div class="footer-bottom">
        <div>© 2025 IoT Smart Waste Monitoring System. All rights reserved.</div>
        <div>
            <a href="#">Privacy Policy</a>
            <a href="#">Terms of Service</a>
        </div>
    </div>
</footer>
