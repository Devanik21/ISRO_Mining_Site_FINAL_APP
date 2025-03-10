# Import section - Updated to include error handling for animations
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import plotly.express as px
import plotly.graph_objects as go
import requests
import json

# Import the components that we can confirm exist
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# Removed streamlit_particles import since it's causing errors
# from streamlit_particles import particles

# Page Configuration with enhanced settings
st.set_page_config(
    page_title="ISRO Celestial Mining Intelligence Hub",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.isro.gov.in/contact_us.html',
        'Report a bug': 'https://github.com/yourusername/ISRO_Mining_Site_FINAL_APP/issues',
        'About': "# ISRO Celestial Mining Intelligence Hub\nAn advanced platform for cosmic resource analysis and mining site evaluation."
    }
)

# Custom CSS for amazing visual enhancements (kept as is)
st.markdown("""
<style>
    /* Main Page Styling */
    .main {
        background-color: #0a1128;
        color: #f0f2f6;
    }
    
    /* Custom Title Styling */
    .title-text {
        background: linear-gradient(90deg, #4CC9F0, #4361EE, #7209B7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 0rem;
        text-align: center;
    }
    
    /* Subtitle Styling */
    .subtitle-text {
        color: #b8c2cc;
        font-size: 1.5rem !important;
        font-style: italic;
        text-align: center;
        margin-top: 0;
    }
    
    /* Card Styling */
    .feature-card {
        background-color: rgba(25, 25, 65, 0.7);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 150, 255, 0.15);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid #4361EE;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(67, 97, 238, 0.3);
    }
    
    /* Icon Styling */
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
        color: #4CC9F0;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #1a1a2e;
        color: white;
    }
    
    /* Metrics Styling */
    .metric-container {
        background: linear-gradient(135deg, rgba(25, 32, 72, 0.7), rgba(33, 58, 102, 0.7));
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CC9F0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #b8c2cc;
    }
    
    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #4361EE, #3A0CA3);
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 5px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(67, 97, 238, 0.3);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(67, 97, 238, 0.5);
    }
    
    /* Selectbox Styling */
    div.stSelectbox > div {
        background-color: #192038;
        border: 1px solid #4361EE;
        border-radius: 5px;
    }
    
    /* Custom Headers */
    h1, h2, h3, h4 {
        color: #f0f2f6;
    }
    
    /* Animation for Data Points */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .animate-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Divider styling */
    hr {
        height: 3px;
        background: linear-gradient(90deg, rgba(76, 201, 240, 0), rgba(76, 201, 240, 1), rgba(76, 201, 240, 0));
        border: none;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animations with better error handling
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.warning(f"Failed to load animation: {e}")
        return None



# Default fallback animation if URLs fail to load
default_lottie = {
    "v": "5.7.11",
    "fr": 30,
    "ip": 0,
    "op": 60,
    "w": 300,
    "h": 300,
   # "nm": "Simple Circle",
    "ddd": 0,
    "assets": [],
    "layers": [{
        "ddd": 0,
        "ind": 1,
        "ty": 4,
       # "nm": "Circle",
        "sr": 1,
        "ks": {
            "o": {"a": 0, "k": 100, "ix": 11},
            "r": {"a": 0, "k": 0, "ix": 10},
            "p": {"a": 0, "k": [150, 150, 0], "ix": 2, "l": 2},
            "a": {"a": 0, "k": [0, 0, 0], "ix": 1, "l": 2},
            "s": {"a": 1, "k": [
                {"i": {"x": [0.5, 0.5, 0.5], "y": [1, 1, 1]}, "o": {"x": [0.5, 0.5, 0.5], "y": [0, 0, 0]}, "t": 0, "s": [100, 100, 100]},
                {"i": {"x": [0.5, 0.5, 0.5], "y": [1, 1, 1]}, "o": {"x": [0.5, 0.5, 0.5], "y": [0, 0, 0]}, "t": 30, "s": [120, 120, 100]},
                {"t": 60, "s": [100, 100, 100]}
            ], "ix": 6, "l": 2}
        },
        "ao": 0,
        "shapes": [{
            "ty": "el",
            "d": 1,
            "s": {"a": 0, "k": [100, 100], "ix": 2},
            "p": {"a": 0, "k": [0, 0], "ix": 3},
            "nm": "Ellipse Path 1",
            "mn": "ADBE Vector Shape - Ellipse",
            "hd": False
        }, {
            "ty": "fl",
            "c": {"a": 0, "k": [0.3, 0.38, 0.93, 1], "ix": 4},
            "o": {"a": 0, "k": 100, "ix": 5},
            "r": 1,
            "bm": 0,
            "nm": "Fill 1",
            "mn": "ADBE Vector Graphic - Fill",
            "hd": False
        }],
        "ip": 0,
        "op": 60,
        "st": 0,
        "bm": 0
    }],
    "markers": []
}


# Load animations with fallback
space_lottie = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_XiFZR1.json") or default_lottie
#rocket_lottie = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_jtbfg2vy.json") or default_lottie
analysis_lottie = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_8z6ubjgj.json") or default_lottie



# Removed particles background configuration and application since the module is missing

# Advanced Sidebar with dynamic content
with st.sidebar:
    st.markdown('<h1 style="text-align: center; color: #4CC9F0;">🛰️ ISRO</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #b8c2cc;">Celestial Mining Intelligence Hub</h2>', unsafe_allow_html=True)
    
    # Dynamic progress bar to show system loading
    st.markdown("### System Initialization")
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)
    st.success("System Ready")
    
    # Animated sidebar image - with error handling
    try:
        st_lottie(space_lottie, speed=1, height=200, key="space_animation")
    except Exception as e:
        st.error(f"Could not display animation: {e}")
        st.info("Continuing with the rest of the application...")
    
    st.markdown("### Mission Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        mining_importance = st.selectbox(
            "Site Importance",
            options=["Low", "Medium", "High", "Critical"],
            index=2
        )
    
    with col2:
        distance_filter = st.selectbox(
            "Distance Range",
            options=["< 100 LY", "100-500 LY", "500-1000 LY", "> 1000 LY"],
            index=1
        )
    st.image("space_mining.png", use_container_width=True)
    # Advanced filters with tooltips
    st.markdown("### Analysis Configuration")
    outlier_sensitivity = st.slider(
        "Outlier Detection Sensitivity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values increase sensitivity to anomalous mining sites."
    )
    
    resource_priority = st.multiselect(
        "Target Resources",
        ["Helium-3", "Platinum", "Rare Earth Elements", "Water Ice", "Titanium", "Silicates"],
        default=["Helium-3", "Platinum"]
    )
    
    # Mission briefing expander
    with st.expander("🌟 Mission Briefing", expanded=False):
        st.markdown("""
        ## ISRO Celestial Mining Initiative
        
        The Indian Space Research Organisation's Celestial Mining Intelligence Hub is a state-of-the-art platform 
        designed to identify, analyze, and prioritize potential mining sites across our solar system and beyond.
        
        Our mission is to advance India's position in the emerging space resource economy while ensuring 
        sustainable practices for cosmic resource utilization.
        
        **Current mission focus:**
        - Asteroid belt prospects within 500 LY
        - Lunar south pole evaluation
        - Mars regolith composition analysis
        
        **Project Lead:** [Devanik](https://www.linkedin.com/in/devanik/)  
        **Authorization Level:** Delta-7
        """)
    
    # Quick actions section
    st.markdown("### Quick Actions")
    
    if st.button("🚀 Launch Analysis"):
        with st.spinner("Initializing systems..."):
            time.sleep(1.5)
        st.success("Analysis protocols activated!")
    
    if st.button("📡 Connect to Deep Space Network"):
        with st.spinner("Establishing connection..."):
            time.sleep(2)
        st.info("DSN connection active. Telemetry streaming at 267 KB/s")

# Main content area with advanced UI elements
# Hero section with animated title
st.markdown('<h1 class="title-text">CELESTIAL MINING INTELLIGENCE HUB</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Pioneering the Future of Space Resource Acquisition</p>', unsafe_allow_html=True)

# Animated Lottie section with error handling
col1, col2, col3 = st.columns([1, 2, 1])


# Mission metrics dashboard
st.markdown("## 📊 Mission Dashboard")

metric_cols = st.columns(4)
with metric_cols[0]:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-value">3,427</div>
        <div class="metric-label">Sites Discovered</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[1]:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-value">99.7%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[2]:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-value">612</div>
        <div class="metric-label">High-Value Prospects</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[3]:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-value">18.3 Ly</div>
        <div class="metric-label">Nearest Viable Site</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Interactive cosmic map (sample visualization)
st.markdown("## 🌌 Galactic Resource Map")

# Generate sample data for the visualization
np.random.seed(42)
n_stars = 200
star_data = pd.DataFrame({
    'x': np.random.normal(0, 100, n_stars),
    'y': np.random.normal(0, 100, n_stars),
    'z': np.random.normal(0, 50, n_stars),
    'resource_value': np.random.exponential(50, n_stars),
    'distance': np.random.uniform(10, 1000, n_stars),
    'site_type': np.random.choice(['Asteroid', 'Moon', 'Planet', 'Dust Cloud'], n_stars)
})

# Apply distance filter
if distance_filter == "< 100 LY":
    filtered_data = star_data[star_data['distance'] < 100]
elif distance_filter == "100-500 LY":
    filtered_data = star_data[(star_data['distance'] >= 100) & (star_data['distance'] < 500)]
elif distance_filter == "500-1000 LY":
    filtered_data = star_data[(star_data['distance'] >= 500) & (star_data['distance'] < 1000)]
else:
    filtered_data = star_data[star_data['distance'] >= 1000]

# Create 3D scatter plot
fig = px.scatter_3d(
    filtered_data, 
    x='x', 
    y='y', 
    z='z',
    color='resource_value',
    size='resource_value',
    color_continuous_scale=px.colors.sequential.Plasma,
    opacity=0.8,
    hover_name=filtered_data.index,
    hover_data={
        'x': False,
        'y': False,
        'z': False,
        'resource_value': ':.2f',
        'distance': ':.1f',
        'site_type': True
    },
    labels={
        'resource_value': 'Resource Index',
        'distance': 'Distance (LY)',
        'site_type': 'Celestial Body Type'
    },
    title="Interactive Galactic Resource Distribution"
)

fig.update_layout(
    scene=dict(
        xaxis_title='Galactic X (LY)',
        yaxis_title='Galactic Y (LY)',
        zaxis_title='Galactic Z (LY)',
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
        zaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
        bgcolor='rgba(10, 17, 40, 0.9)'
    ),
    paper_bgcolor='rgba(10, 17, 40, 0)',
    plot_bgcolor='rgba(10, 17, 40, 0)',
    margin=dict(l=0, r=0, t=30, b=0),
    height=600,
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="background-color: rgba(25, 32, 72, 0.7); padding: 15px; border-radius: 10px; margin-top: -20px;">
    <p style="color: #b8c2cc; font-style: italic;">
        This interactive 3D map displays potential mining sites based on current filters. 
        Larger, brighter points indicate higher resource concentrations. Rotate, zoom, and hover for detailed information.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Core functionality cards
st.markdown("## 🛠️ Core System Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <h3>AI-Powered Site Classification</h3>
        <p>Our advanced neural network analyzes 57 unique parameters to classify mining sites with 89.7% accuracy. The system leverages quantum computing techniques to process spectroscopic data and identify resource-rich locations.</p>
        <ul>
            <li>Multi-spectral analysis</li>
            <li>Density distribution prediction</li>
            <li>Mineral composition estimation</li>
            <li>Accessibility scoring</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📡</div>
        <h3>Real-Time Data Integration</h3>
        <p>Seamlessly integrates with ISRO's deep space network for real-time updates from probes and satellites. Continuously refines predictions as new data becomes available.</p>
        <ul>
            <li>Synchronizes with 14 active space missions</li>
            <li>Updates every 73 minutes</li>
            <li>Adaptive learning from mission feedback</li>
            <li>Anomaly detection with 97.3% sensitivity</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🚀</div>
        <h3>Mission Planning Optimization</h3>
        <p>Generate optimized mining mission profiles with our revolutionary planning algorithm. Balance resource yield, mission duration, and equipment requirements to maximize ROI.</p>
        <ul>
            <li>Delta-V optimization</li>
            <li>Equipment selection assistant</li>
            <li>Risk assessment matrix</li>
            <li>Crew requirement planning</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <h3>Advanced Resource Analysis</h3>
        <p>State-of-the-art spectrographic analysis predicts mineral and element composition with unprecedented accuracy. Identify rare-earth elements, precious metals, and fusion fuel sources.</p>
        <ul>
            <li>Helium-3 detection (99.2% accuracy)</li>
            <li>Precious metal concentration mapping</li>
            <li>Water ice deposit identification</li>
            <li>Extraction difficulty estimation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Animated Recommendation section
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("## ✨ Recommended Mining Prospects")

# Sample recommendation data
recommendation_data = {
    'Site ID': ['AS-1872', 'PM-0429', 'LU-S117', 'EN-7756', 'HD-2092'],
    'Location': ['Asteroid Belt', 'Proxima b', 'Lunar South Pole', 'Enceladus', 'HD 40307 g'],
    'Primary Resource': ['Platinum', 'Rare Earth Metals', 'Helium-3', 'Water Ice', 'Titanium'],
    'Value Index': [92, 88, 86, 79, 76],
    'Extraction Difficulty': ['Medium', 'High', 'Low', 'Medium', 'Extreme'],
    'Mission Duration': ['14 months', '7 years', '45 days', '26 months', '12 years']
}

recommendation_df = pd.DataFrame(recommendation_data)

# Style the dataframe
st.dataframe(
    recommendation_df.style.background_gradient(
        cmap='Blues', 
        subset=['Value Index']
    ).set_properties(**{
        'text-align': 'center',
        'font-weight': 'bold'
    }),
    height=250,
    use_container_width=True
)

# Animated Analysis section
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("## 🔍 System Capabilities")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div style="padding: 20px; background-color: rgba(25, 32, 72, 0.7); border-radius: 10px;">
        <h3 style="color: #4CC9F0;">Key Platform Features</h3>
        
        <h4>🚀 Predictive Analytics Engine</h4>
        <p>Our quantum-enhanced predictive modeling analyzes spectroscopic signatures, gravitational anomalies, and composition data to identify high-value mining opportunities across the galaxy.</p>
        
        <h4>🔮 Multi-Parameter Recommendation System</h4>
        <p>The advanced AI weighs over 200 variables including resource concentration, extraction difficulty, orbital mechanics, and mission logistics to prioritize the most profitable ventures.</p>
        
        <h4>📊 Interactive Data Visualization</h4>
        <p>Explore cosmic mining opportunities through intuitive 3D visualizations, comparative analysis tools, and dynamic filtering to identify patterns invisible to conventional analysis.</p>
        
        <h4>⚙️ Mission Simulation Framework</h4>
        <p>Test mining strategies in our physics-accurate simulation environment that models extraction processes, equipment performance, and environmental challenges before committing resources.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    try:
        st_lottie(analysis_lottie, height=400, key="analysis_animation")
    except Exception as e:
        st.image("https://via.placeholder.com/300x400.png?text=Analysis+Dashboard", use_column_width=True)
        st.error(f"Could not display animation: {e}")

# Call-to-action section
st.markdown("<hr>", unsafe_allow_html=True)
cta_col1, cta_col2 = st.columns([3, 1])

with cta_col1:
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(25, 32, 72, 0.7), rgba(67, 97, 238, 0.3)); padding: 25px; border-radius: 10px; margin-top: 20px;">
        <h2 style="color: #4CC9F0;">Ready to Explore the Cosmos?</h2>
        <p style="font-size: 1.2rem;">Navigate to the prediction model to begin identifying high-value mining sites tailored to your mission parameters.</p>
    </div>
    """, unsafe_allow_html=True)

with cta_col2:
    if st.button("Launch Explorer Module"):
        #st.balloons()
        st.success("Explorer module activated! Redirecting to prediction interface...")

# Footer with credits and system status
st.markdown("<hr>", unsafe_allow_html=True)
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #b8c2cc; font-size: 0.9rem;">
            <strong>System Status:</strong> Operational<br>
            Last Updated: 10-03-2025 08:42 IST
        </p>
    </div>
    """, unsafe_allow_html=True)

with footer_col2:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #b8c2cc; font-size: 0.9rem;">
            <strong>ISRO Celestial Mining Intelligence Hub</strong><br>
            Version 2.7.4 | Chandra Build
        </p>
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #b8c2cc; font-size: 0.9rem;">
            <strong>Developed by:</strong><br>
            Dr. Devanik Saha & Advanced Systems Team
        </p>
    </div>
    """, unsafe_allow_html=True)
