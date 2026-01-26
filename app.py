import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import folium
from streamlit_folium import st_folium

# Custom CSS for Dashboard UI (mimicking modern landing page/dashboard)
st.markdown("""
<style>
    body { background-color: #0f0f23; color: white; font-family: 'Arial', sans-serif; }
    .sidebar .sidebar-content { background-color: #1a1a2e; border-radius: 10px; padding: 10px; }
    .main { background-color: #16213e; padding: 20px; border-radius: 10px; }
    .header { text-align: center; font-size: 2.5em; color: #e94560; margin-bottom: 20px; font-weight: bold; }
    .card { background: linear-gradient(135deg, #0f3460, #1a1a2e); padding: 20px; border-radius: 15px; margin: 10px; box-shadow: 0 8px 16px rgba(0,0,0,0.4); text-align: center; }
    .metric { font-size: 2em; color: #e94560; font-weight: bold; }
    .subtext { font-size: 0.9em; color: #bbb; }
    .button { background-color: #e94560; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 1em; }
    .button:hover { background-color: #d6336c; }
    .map { height: 400px; border-radius: 10px; }
    .chart { margin: 20px 0; border-radius: 10px; }
    .form { background-color: #0f3460; padding: 15px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Session State for Login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

# Sample Data for Bins
bins_data = [
    {"bin_id": "Bin1", "fill_level": 75, "location": (40.7128, -74.0060)},
    {"bin_id": "Bin2", "fill_level": 50, "location": (34.0522, -118.2437)},
    {"bin_id": "Bin3", "fill_level": 90, "location": (41.8781, -87.6298)},
]

def fetch_bins_data():
    return pd.DataFrame(bins_data)

def train_predictive_model(data):
    data['time'] = np.arange(len(data))
    X = data[['time']]
    y = data['fill_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions

def optimize_route(locations):
    return {"distance": "Approx 2000 miles", "path": locations}

def send_whatsapp(message, to_number):
    print(f"Mock WhatsApp sent to {to_number}: {message}")

# Main App
def main():
    st.sidebar.markdown("## 🗂️ Navigation")
    page = st.sidebar.radio("", ["🏠 Dashboard", "📊 Predictive Modeling", "🚗 Route Optimization", "📱 WhatsApp Dispatch", "🔐 Driver Login", "🛤️ Shortest Path"])

    if page == "🏠 Dashboard":
        st.markdown('<div class="header">Smart Bin Dashboard</div>', unsafe_allow_html=True)
        
        # Metric Cards
        bins_df = fetch_bins_data()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card"><div class="metric">{}</div><div class="subtext">Total Bins</div></div>'.format(len(bins_df)), unsafe_allow_html=True)
        with col2:
            avg_fill = int(bins_df['fill_level'].mean())
            st.markdown('<div class="card"><div class="metric">{}%</div><div class="subtext">Avg Fill Level</div></div>'.format(avg_fill), unsafe_allow_html=True)
        with col3:
            full_bins = len(bins_df[bins_df['fill_level'] > 80])
            st.markdown('<div class="card"><div class="metric">{}</div><div class="subtext">Bins >80% Full</div></div>'.format(full_bins), unsafe_allow_html=True)
        
        # Chart
        st.markdown("### Bin Fill Levels Chart")
        fig = px.bar(bins_df, x='bin_id', y='fill_level', color='fill_level', title="")
        st.plotly_chart(fig, use_container_width=True)
        
        # Map
        st.markdown("### Bin Locations Map")
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=5)
        for _, row in bins_df.iterrows():
            folium.Marker(location=row['location'], popup=f"{row['bin_id']} - {row['fill_level']}%").add_to(m)
        st_folium(m, width=700, height=400)

    elif page == "📊 Predictive Modeling":
        st.markdown('<div class="header">Predictive Modeling</div>', unsafe_allow_html=True)
        bins_df = fetch_bins_data()
        model, predictions = train_predictive_model(bins_df)
        st.markdown("### Predictions Chart")
        fig = px.line(x=np.arange(len(predictions)), y=predictions, title="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Model trained on historical data.")

    elif page == "🚗 Route Optimization":
        st.markdown('<div class="header">Route Optimization</div>', unsafe_allow_html=True)
        bins_df = fetch_bins_data()
        locations = [row['location'] for _, row in bins_df.iterrows()]
        if len(locations) > 1:
            route = optimize_route(locations)
            st.markdown(f"**Optimized Distance:** {route['distance']}")
            m = folium.Map(location=locations[0], zoom_start=5)
            folium.PolyLine(locations=route['path'], color="blue").add_to(m)
            st_folium(m, width=700, height=400)
        else:
            st.write("Need at least 2 bins.")

    elif page == "📱 WhatsApp Dispatch":
        st.markdown('<div class="header">WhatsApp Dispatch</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="form">', unsafe_allow_html=True)
            message = st.text_area("Message")
            to_number = st.text_input("Driver Number (e.g., +1234567890)")
            if st.button("Send Dispatch", key="dispatch"):
                send_whatsapp(message, to_number)
                st.success("Message sent!")
            st.markdown('</div>', unsafe_allow_html=True)

    elif page == "🔐 Driver Login":
        st.markdown('<div class="header">Driver Login</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="form">', unsafe_allow_html=True)
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if username == "driver" and password == "pass":
                    st.session_state.logged_in = True
                    st.success("Logged in!")
                else:
                    st.error("Invalid credentials")
            st.markdown('</div>', unsafe_allow_html=True)

    elif page == "🛤️ Shortest Path":
        st.markdown('<div class="header">Shortest Path</div>', unsafe_allow_html=True)
        st.markdown("Feature disabled (networkx not installed).")

if __name__ == "__main__":
    main()
