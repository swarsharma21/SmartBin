import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Bin AI", layout="wide")

# --- 2. DATA LOADING (The Ironclad Version) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('smart_bin_data.csv')
        # Clean column names (lowercase and strip spaces)
        df.columns = df.columns.str.strip().str.lower()
        
        # TRANSLATION LAYER: This prevents the 'fill_level' KeyError
        # We find WHATEVER column the user has and rename it to 'bin_fill_percent'
        possible_fill_names = ['fill_level', 'level', 'fill', 'bin_fill_percent', 'bin_fill']
        for name in possible_fill_names:
            if name in df.columns:
                df = df.rename(columns={name: 'bin_fill_percent'})
                break # Stop once we find a match
        
        # Ensure lat/lon are also mapped
        df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Waiting for CSV: {e}")
        return pd.DataFrame()

df = load_data()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🚛 Control Panel")
    if not df.empty:
        # Use our standardized name 'bin_fill_percent'
        selected_bins = st.multiselect("Select Bins", options=df['bin_id'].unique(), default=df['bin_id'].unique())
        fill_range = st.slider("Fill Level Range (%)", 0, 100, (0, 100))
        
        # FILTERING: Always use the standardized 'bin_fill_percent'
        filtered_df = df[
            (df['bin_id'].isin(selected_bins)) & 
            (df['bin_fill_percent'] >= fill_range[0]) & 
            (df['bin_fill_percent'] <= fill_range[1])
        ]
    else:
        st.stop()

# --- 4. MAIN INTERFACE ---
st.title("📊 Smart City Waste Intelligence")

if not filtered_df.empty:
    m1, m2, m3 = st.columns(3)
    m1.metric("Active Bins", len(filtered_df))
    # We use .get() or standardized names to be safe
    m2.metric("Avg Fill", f"{filtered_df['bin_fill_percent'].mean():.1f}%")
    m3.metric("Critical", len(filtered_df[filtered_df['bin_fill_percent'] >= 80]))

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📍 Bin Locations")
        avg_lat = filtered_df['lat'].mean()
        avg_lon = filtered_df['lon'].mean()
        
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles="cartodbpositron")
        
        for _, row in filtered_df.iterrows():
            # Standardized column name used here too!
            color = 'red' if row['bin_fill_percent'] >= 80 else 'green'
            folium.Marker(
                [row['lat'], row['lon']], 
                popup=f"ID: {row['bin_id']} | Fill: {row['bin_fill_percent']}%",
                icon=folium.Icon(color=color, icon='trash')
            ).add_to(m)
        st_folium(m, width="100%", height=500)

    with col_right:
        st.subheader("📊 Fill Levels")
        st.bar_chart(filtered_df.set_index('bin_id')['bin_fill_percent'])
else:
    st.warning("No data matches your filters.")
