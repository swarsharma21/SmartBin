import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Bin AI Dashboard", layout="wide")

# Custom CSS to make Streamlit look like a high-end BI tool
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING ---
@st.cache_data
def load_csv():
    try:
        df = pd.read_csv('data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except:
        return pd.DataFrame() # Returns empty if file not found

df = load_csv()

# --- 3. THE SIDEBAR (Slicers) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299935.png", width=100)
    st.title("Control Panel")
    st.markdown("---")
    
    if not df.empty:
        # Filter by Bin ID
        selected_bins = st.multiselect("Select Bins", options=df['bin_id'].unique(), default=df['bin_id'].unique())
        # Filter by Fill Level
        fill_range = st.slider("Fill Level Range (%)", 0, 100, (0, 100))
        
        filtered_df = df[(df['bin_id'].isin(selected_bins)) & 
                         (df['bin_fill_percent'] >= fill_range[0]) & 
                         (df['bin_fill_percent'] <= fill_range[1])]
    else:
        st.error("No CSV data found on GitHub.")

# --- 4. MAIN INTERFACE ---
st.title("🚛 Smart City: Waste Management Intelligence")

# Use Tabs to separate Python Logic and Power BI
tab1, tab2 = st.tabs(["📊 Live IoT Analytics", "📈 Detailed Power BI Report"])

with tab1:
    if not df.empty:
        # --- TOP ROW: KPI CARDS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Bins", len(filtered_df))
        m2.metric("Avg Fill", f"{filtered_df['fill_level'].mean():.1f}%")
        m3.metric("Critical (80%+)", len(filtered_df[filtered_df['bin_fill_percent'] >= 80]))
        m4.metric("Last Sync", df['timestamp'].max().strftime('%H:%M'))

        st.markdown("---")

        # --- MIDDLE ROW: MAP AND CHART ---
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("📍 Real-time GPS Tracking")
            # Map centered on data
            avg_lat, avg_lon = filtered_df['lat'].mean(), filtered_df['lon'].mean()
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles="cartodbpositron")
            
            for _, row in filtered_df.iterrows():
                color = 'red' if row['fill_level'] >= 80 else 'green'
                folium.Marker(
                    [row['lat'], row['lon']],
                    popup=f"ID: {row['bin_id']} | Fill: {row['bin_fill_percent']}%",
                    icon=folium.Icon(color=color, icon='trash', prefix='fa')
                ).add_to(m)
            
            st_folium(m, width="100%", height=500)

        with col_right:
            st.subheader("📊 Fill Level by Bin")
            st.bar_chart(filtered_df.set_index('bin_id')['bin_fill_percent'])
            st.subheader("📋 Priority List")
            st.dataframe(filtered_df[['bin_id', 'bin_fill_percent']].sort_values('bin_fill_percent', ascending=False), hide_index=True)

with tab2:
    st.subheader("External Power BI Integration")
    # Replace the link below with your "Publish to Web" Power BI link
    pbi_url = "https://app.powerbi.com/links/PQ2P41cZAi?ctid=c290ab75-f93e-4848-8b0b-550dd7acfc33&pbi_source=linkShare "
    
    if "YOUR_PBI_EMBED_LINK" in pbi_url:
        st.info("💡 To show your Power BI report here, paste your 'Publish to Web' link into the code.")
    
    st.components.v1.iframe(pbi_url, height=800, scrolling=True)
