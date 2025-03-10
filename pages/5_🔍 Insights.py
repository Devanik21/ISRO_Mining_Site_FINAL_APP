import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import folium
from streamlit_folium import st_folium
import base64

# Custom CSS for a futuristic, sleek look
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #1e1e2f 0%, #3b3b5e 100%);}
    .stTitle {color: #00d4ff; font-family: 'Orbitron', sans-serif;}
    .stMarkdown {color: #e0e0e0;}
    .stButton>button {background-color: #00d4ff; color: #1e1e2f; border-radius: 10px;}
    .stDataFrame {border: 2px solid #00d4ff; border-radius: 10px; background-color: #2a2a3d;}
    </style>
    """, unsafe_allow_html=True)

# Simulate a futuristic loading animation
with st.spinner("🚀 Initializing Space Mining Analytics..."):
    # Load dataset (assuming it exists; replace with actual data loading logic)
    try:
        df = pd.read_csv("space_mining_dataset.csv")
    except FileNotFoundError:
        # Dummy data for demonstration
        np.random.seed(42)
        df = pd.DataFrame({
            'Celestial Body': np.random.choice(['Moon', 'Mars', 'Asteroid X', 'Europa'], 100),
            'iron': np.random.uniform(5, 50, 100),
            'nickel': np.random.uniform(2, 40, 100),
            'water_ice': np.random.uniform(0, 30, 100),
            'Estimated Value (B USD)': np.random.uniform(10, 500, 100),
            'sustainability_index': np.random.uniform(0, 1, 100),
            'efficiency_index': np.random.uniform(0, 1, 100),
            'distance_from_earth': np.random.uniform(0.38, 400, 100),  # in million km
            'lat': np.random.uniform(-90, 90, 100),  # Simulated coordinates
            'lon': np.random.uniform(-180, 180, 100)
        })

def show_insights_page():
    # Header with futuristic flair
    st.title("🌌 Interstellar Mining Dashboard", anchor="top")
    st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')} | **Powered by xAI**", unsafe_allow_html=True)
    st.write("Explore actionable insights from cosmic mining sites with cutting-edge analytics.")

    # Sidebar for controls
    with st.sidebar:
        st.header("🛠️ Control Panel")
        celestial_body_selected = st.multiselect(
            "🌕 Filter by Celestial Body", 
            options=df['Celestial Body'].unique(), 
            default=df['Celestial Body'].unique(),
            help="Select one or more celestial bodies to analyze."
        )
        
        value_range = st.slider(
            "💰 Estimated Value Range (B USD)", 
            min_value=float(df['Estimated Value (B USD)'].min()), 
            max_value=float(df['Estimated Value (B USD)'].max()), 
            value=(float(df['Estimated Value (B USD)'].min()), float(df['Estimated Value (B USD)'].max())),
            step=1.0
        )
        
        distance_range = st.slider(
            "🌍 Distance from Earth (M km)", 
            min_value=float(df['distance_from_earth'].min()), 
            max_value=float(df['distance_from_earth'].max()), 
            value=(float(df['distance_from_earth'].min()), float(df['distance_from_earth'].max())),
            step=0.1
        )
        
        sustainability_threshold = st.slider(
            "🌱 Sustainability Index", 0.0, 1.0, (0.0, 1.0), step=0.05
        )

    # Filter the dataset dynamically
    df_filtered = df[
        (df['Celestial Body'].isin(celestial_body_selected)) &
        (df['Estimated Value (B USD)'].between(value_range[0], value_range[1])) &
        (df['distance_from_earth'].between(distance_range[0], distance_range[1])) &
        (df['sustainability_index'].between(sustainability_threshold[0], sustainability_threshold[1]))
    ]

    # General Insights with Metrics
    st.write("## 🌍 Galactic Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Mining Sites", f"{df_filtered.shape[0]}", "🚀")
    col2.metric("Unique Celestial Bodies", f"{df_filtered['Celestial Body'].nunique()}", "🌕")
    col3.metric("Avg Value (B USD)", f"{df_filtered['Estimated Value (B USD)'].mean():.2f}", "💰")

    # Interactive Map
    st.write("## 🗺️ Cosmic Map")
    m = folium.Map(location=[0, 0], zoom_start=2, tiles="CartoDB dark_matter")
    for _, row in df_filtered.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=row['Estimated Value (B USD)'] / 50,
            popup=f"{row['Celestial Body']}: ${row['Estimated Value (B USD)']}B",
            color="#00d4ff",
            fill=True,
            fill_color="#00d4ff"
        ).add_to(m)
    st_folium(m, width=700, height=400)

    # Detailed Insights by Celestial Body
    st.write("## 🌟 Celestial Body Breakdown")
    celestial_summary = df_filtered.groupby('Celestial Body').agg({
        'iron': 'mean', 'nickel': 'mean', 'water_ice': 'mean',
        'Estimated Value (B USD)': 'mean', 'sustainability_index': 'mean',
        'efficiency_index': 'mean', 'distance_from_earth': 'mean'
    }).reset_index()
    fig = px.bar(celestial_summary, x='Celestial Body', y='Estimated Value (B USD)', 
                 title="Average Value by Celestial Body", color='sustainability_index',
                 color_continuous_scale='Viridis', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # High vs Low Value Sites with Pie Chart
    st.write("## 💰 Value Distribution")
    median_value = df_filtered['Estimated Value (B USD)'].median()
    high_value_sites = df_filtered[df_filtered['Estimated Value (B USD)'] > median_value]
    low_value_sites = df_filtered[df_filtered['Estimated Value (B USD)'] <= median_value]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(f"💎 High-Value Sites: {high_value_sites.shape[0]}")
        st.write(f"📉 Low-Value Sites: {low_value_sites.shape[0]}")
    with col2:
        pie_data = pd.DataFrame({'Category': ['High-Value', 'Low-Value'], 
                                 'Count': [high_value_sites.shape[0], low_value_sites.shape[0]]})
        fig_pie = px.pie(pie_data, values='Count', names='Category', title="Site Value Distribution",
                         color_discrete_sequence=['#00d4ff', '#ff4d4d'], height=300)
        st.plotly_chart(fig_pie)

    # Resource Composition Scatter Plot
    st.write("## 🔬 Resource Composition")
    fig_scatter = px.scatter(df_filtered, x='iron', y='nickel', size='water_ice', 
                             color='Estimated Value (B USD)', hover_data=['Celestial Body'],
                             title="Iron vs Nickel vs Water Ice", color_continuous_scale='Plasma',
                             height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Recommendations with Expander
    with st.expander("🔍 Strategic Recommendations", expanded=True):
        st.markdown("""
        - **High-Value Targets:** Prioritize sites with elevated iron (>30%) and nickel (>25%) concentrations.
        - **Sustainability Focus:** Select sites with sustainability indices above 0.7 for long-term viability.
        - **Logistical Efficiency:** Opt for sites within 100M km of Earth to minimize transport costs.
        - **Exploration Potential:** Investigate Asteroid X for untapped high-value opportunities.
        """)

    # Downloadable Report
    st.write("## 📥 Export Insights")
    csv = df_filtered.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="space_mining_insights.csv">Download CSV Report</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.success("🌠 Analytics Complete - Ready for Interstellar Deployment!")

if __name__ == "__main__":
    show_insights_page()
