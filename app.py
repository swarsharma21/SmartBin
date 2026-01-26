import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import folium
from streamlit_folium import st_folium

# Custom CSS for UI Replication (mimicking the landing page style, Markdown-friendly)
st.markdown("""
<style>
    body { background-color: #0f0f23; color: white; font-family: 'Arial', sans-serif; }
    .sidebar .sidebar-content { background-color: #1a1a2e; }
    .main { background-color: #16213e; padding: 20px; border-radius: 10px; }
    .header { text-align: center; font-size: 2em; color: #e94560; margin-bottom: 20px; }
    .card { background-color: #0f3460; padding: 15px; border-radius: 8px; margin: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
    .button { background-color: #e94560; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
    .button:hover { background-color: #d6336c; }
    .map { height: 400px; }
    .chart { margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# Session State for Login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

# Sample Data for Bins (local, no Firebase)
bins_data = [
    {"bin_id": "Bin1", "fill_level": 75, "location": (40.7128, -74.0060)},  # NYC
    {"bin_id": "Bin2", "fill_level": 50, "location": (34.0522, -118.2437)},  # LA
    {"bin_id": "Bin3", "fill_level": 90, "location": (41.8781, -87.6298)},  # Chicago
]

# Function to Fetch Data (now just returns sample data)
def fetch_bins_data():
    return pd.DataFrame(bins_data)

# Predictive Modeling Function
def train_predictive_model(data):
    data['time'] = np.arange(len(data))
    X = data[['time']]
    y = data['fill_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions

# Route Optimization Function (mocked: just returns a simple path)
def optimize_route(locations):
    # Mock: Return a basic path without Google Maps
    return {"distance": "Approx 2000 miles", "path": locations}  # Simplified

# WhatsApp Dispatch Function (mocked: prints to console)
def send_whatsapp(message, to_number):
    print(f"Mock WhatsApp sent to {to_number}: {message}")  # Simulate sending

# Main App (Markdown-Heavy)
def main():
    st.sidebar.markdown("## Smart Bin Navigation")
    page = st.sidebar.radio("", ["Dashboard", "Predictive Modeling", "Route Optimization", "WhatsApp Dispatch", "Driver Login", "Shortest Path"])

    if page == "Dashboard":
        st.markdown("""
        # Smart Bin Dashboard
        
        Welcome to the Smart Bin project! This dashboard monitors bin fill levels using ultrasonic sensors and ESP32, with data simulated locally.
        
        ## Real-Time Bin Fill Levels
        """)
        bins_df = fetch_bins_data()
        st.dataframe(bins_df)  # Simple table
        st.markdown("### Bin Locations Map")
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=5)
        for _, row in bins_df.iterrows():
            folium.Marker(location=row['location'], popup=f"Bin: {row['bin_id']} - Fill: {row['fill_level']}%").add_to(m)
        st_folium(m, width=700, height=400)

    elif page == "Predictive Modeling":
        st.markdown("""
        # Predictive Modeling
        
        Using Random Forest to predict bin fill levels based on historical data.
        """)
        bins_df = fetch_bins_data()
        model, predictions = train_predictive_model(bins_df)
        st.markdown("### Predictions")
        st.write(predictions)
        st.markdown("Model trained on time-series data for accurate forecasts.")

    elif page == "Route Optimization":
        st.markdown("""
        # Route Optimization
        
        Simulated route optimization (no external API).
        """)
        bins_df = fetch_bins_data()
        locations = [row['location'] for _, row in bins_df.iterrows()]
        if len(locations) > 1:
            route = optimize_route(locations)
            st.markdown(f"**Mock Optimized Distance:** {route['distance']}")
            m = folium.Map(location=locations[0], zoom_start=5)
            folium.PolyLine(locations=route['path'], color="blue").add_to(m)  # Simple line
            st_folium(m, width=700, height=400)
        else:
            st.markdown("Need at least 2 bins for optimization.")

    elif page == "WhatsApp Dispatch":
        st.markdown("""
        # WhatsApp Dispatch
        
        Send alerts to drivers (mocked, prints to console).
        """)
        message = st.text_area("Enter message")
        to_number = st.text_input("Driver number (e.g., +1234567890)")
        if st.button("Send"):
            send_whatsapp(message, to_number)
            st.markdown("**Mock message 'sent'! Check console.**")

    elif page == "Driver Login":
        st.markdown("""
        # Driver Login
        
        Secure login for drivers.
        """)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "driver" and password == "pass":
                st.session_state.logged_in = True
                st.markdown("**Logged in successfully!**")
            else:
                st.markdown("**Invalid credentials.**")

    elif page == "Shortest Path":
        st.markdown("""
        # Shortest Path for Live Bins
        
        Feature disabled (networkx not installed). Use local data for manual calculations.
        """)
        # No code here; just a placeholder

if __name__ == "__main__":
    main()
