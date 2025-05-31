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
    page_title="🪐 CelestAI Nexus - Cosmic Intelligence Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.isro.gov.in/contact_us.html',
        'Report a bug': 'https://github.com/yourusername/ISRO_Mining_Site_FINAL_APP/issues',
        'About': "# 🪐 CelestAI Nexus\nAn advanced AI-driven platform for cosmic resource analysis, mining site evaluation, and celestial exploration."
    }
)

# Custom CSS for amazing visual enhancements (kept as is)
st.markdown("""
<style>
    /* Main Page Styling */
    body, .main {
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
    hr.enhanced-hr {
        height: 3px;
        background: linear-gradient(90deg, rgba(76, 201, 240, 0), rgba(76, 201, 240, 1), rgba(76, 201, 240, 0));
        border: none;
        margin: 2rem 0;
    }

    /* Frosted Glass Card */
    .frosted-glass-card {
        background: rgba(40, 50, 100, 0.55); /* Semi-transparent background */
        backdrop-filter: blur(12px); /* Blur effect for the background */
        -webkit-backdrop-filter: blur(12px); /* For Safari */
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 25px;
        transition: transform 0.4s ease, box-shadow 0.4s ease;
    }

    .frosted-glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 45px rgba(76, 201, 240, 0.4);
    }

    .frosted-glass-card h3 {
        color: #4CC9F0; /* Light blue for headers inside frosted cards */
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(76, 201, 240, 0.5);
        padding-bottom: 8px;
        font-weight: 700;
    }

    .frosted-glass-card p, .frosted-glass-card li {
        color: #e0e0e0; /* Lighter text for readability on blurred background */
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Section Header Box */
    .section-header-box {
        background: linear-gradient(135deg, rgba(67, 97, 238, 0.3), rgba(114, 9, 183, 0.3));
        padding: 15px 25px;
        border-radius: 10px;
        margin-bottom: 25px;
        text-align: center;
        border-left: 6px solid #7209B7;
        border-right: 6px solid #4CC9F0;
    }
    .section-header-box h2 {
        color: #f0f2f6; margin: 0; font-size: 2rem; font-weight: 700; text-shadow: 0 0 10px rgba(76, 201, 240, 0.5);
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
platform_lottie = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_8z6ubjgj.json") or default_lottie
hero_lottie = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_4p3fDM.json") or default_lottie # Galaxy animation


# New Lottie animations for advanced features
quantum_comm_lottie = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_p1qiuahe.json") or default_lottie
subsurface_scanner_lottie = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_ofa3xwo7.json") or default_lottie
terraforming_lottie = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_p2u571tg.json") or default_lottie
risk_matrix_lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_x1gjdldd.json") or default_lottie
ai_advisor_lottie = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_vPnn3K.json") or default_lottie
data_insights_lottie = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_vS022M.json") or default_lottie # For new data insights section
planet_lottie = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_R0gD5C.json") or default_lottie # Generic planet
news_lottie = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_zcdjf40p.json") or default_lottie # News/Intel animation

# Removed particles background configuration and application since the module is missing

# Advanced Sidebar with dynamic content
with st.sidebar:
    st.image("space_mining.png", use_container_width=True)
    st.markdown('<h1 style="text-align: center; color: #4CC9F0;">🪐 CelestAI Nexus</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #b8c2cc; font-size: 1.1rem;">Cosmic Intelligence Platform</h2>', unsafe_allow_html=True)
    
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
        ## CelestAI Nexus Initiative
        
        CelestAI Nexus is a state-of-the-art AI platform designed to identify, analyze, and prioritize potential 
        celestial mining sites and exploration targets across the cosmos.
        
        Our mission is to pioneer humanity's expansion into space resource utilization while ensuring 
        sustainable practices for cosmic resource utilization.
        
        **Current mission focus:**
        - Asteroid belt prospects within 500 LY
        - Exo-lunar resource evaluation
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
st.markdown('<h1 class="title-text">🪐 CELESTAI NEXUS</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Navigating the Cosmos with Artificial Intelligence & Advanced Analytics</p>', unsafe_allow_html=True)

# Animated Lottie section with error handling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        st_lottie(hero_lottie, speed=1, height=300, key="hero_animation")
    except Exception as e:
        st.warning(f"Could not display hero animation: {e}")

st.markdown("<hr class='enhanced-hr'>", unsafe_allow_html=True)
# Mission metrics dashboard
st.markdown("## 📊 Mission Dashboard")

dashboard_metric_cols = st.columns(4)
dashboard_icons = ["🌌", "🎯", "💎", "🛰️"]
dashboard_labels = ["Sites Discovered", "Model Accuracy", "High-Value Prospects", "Nearest Viable Site"]
dashboard_values = ["3,427", "99.7%", "612", "18.3 Ly"]

for i, col in enumerate(dashboard_metric_cols):
    with col:
        st.markdown(f"""
        <div class="metric-container animate-pulse">
            <div style="font-size: 2.5rem; color: #4CC9F0; margin-bottom: 5px;">{dashboard_icons[i]}</div>
            <div class="metric-value" style="font-size: 2rem;">{dashboard_values[i]}</div>
            <div class="metric-label" style="font-size: 0.9rem;">{dashboard_labels[i]}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="metric-container" style="margin-top: 20px; background: linear-gradient(135deg, rgba(114, 9, 183, 0.3), rgba(67, 97, 238, 0.3)); padding: 10px;">
    <span class="metric-label" style="font-size: 1.1rem;">System Health:</span> <span class="metric-value" style="font-size: 1.3rem; color: #28a745;">OPTIMAL</span> | 
    <span class="metric-label" style="font-size: 1.1rem;">Threat Level:</span> <span class="metric-value" style="font-size: 1.3rem; color: #28a745;">LOW</span> | 
    <span class="metric-label" style="font-size: 1.1rem;">Active Probes:</span> <span class="metric-value" style="font-size: 1.3rem; color: #4CC9F0;">17</span>
</div>
""", unsafe_allow_html=True)


st.markdown("<hr class='enhanced-hr'>", unsafe_allow_html=True)

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

st.markdown("<hr class='enhanced-hr'>", unsafe_allow_html=True)

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

# New Advanced Celestial Intelligence Modules Section
st.markdown("<hr class='enhanced-hr'>", unsafe_allow_html=True)
st.markdown('<div class="section-header-box"><h2><span style="text-shadow: 0 0 8px #4CC9F0;">🌌 Advanced Celestial Intelligence Modules</span></h2></div>', unsafe_allow_html=True)

feature_cols1 = st.columns(2)
with feature_cols1[0]:
    st.markdown("""
    <div class="frosted-glass-card">
        <h3>🔗 Quantum Entanglement Comms Link</h3>
        <p>Establish ultra-secure, faster-than-light communication channels using quantum entanglement for instantaneous data transfer across vast cosmic distances.</p>
    </div>
    """, unsafe_allow_html=True)
    if quantum_comm_lottie:
        st_lottie(quantum_comm_lottie, height=150, key="quantum_comm")
    st.radio("Link Status:", ("Active", "Standby", "Initializing"), horizontal=True, key="q_com_status")
    st.caption("Current Throughput: 7.2 ZB/s (Zettabytes per second)")
    st.caption("Encryption Level: Quantum Grade XII")

with feature_cols1[1]:
    st.markdown("""
    <div class="frosted-glass-card">
        <h3>⛏️ AI-Driven Subsurface Scanner</h3>
        <p>Deploy AI algorithms to analyze gravimetric and seismic data, revealing deep subsurface compositions and potential resource deposits on target celestial bodies.</p>
    </div>
    """, unsafe_allow_html=True)
    if subsurface_scanner_lottie:
        st_lottie(subsurface_scanner_lottie, height=150, key="subsurface_scan")
    
    scan_target = st.selectbox("Target Body for Deep Scan:", ["Europa", "Titan", "Ceres", "Ganymede"], key="scan_target")
    scan_depth = st.slider("Scan Depth (km):", 1, 500, 50, key="scan_depth")
    if st.button("Initiate Deep Scan", key="deep_scan_button"):
        st.info(f"Initiating scan of {scan_target} to {scan_depth} km... Analyzing quantum resonance signatures...")
        # Placeholder for scan result
        time.sleep(1)
        st.success(f"Preliminary Scan Complete: High probability of water ice and silicate deposits detected at {scan_depth*0.7:.1f} km.")

feature_cols2 = st.columns(2)
with feature_cols2[0]:
    st.markdown("""
    <div class="frosted-glass-card">
        <h3>🌍 Terraforming Suitability Index (TSI)</h3>
        <p>Assess celestial bodies for their potential to be terraformed. The TSI combines atmospheric, geological, and radiation data to provide a comprehensive suitability score.</p>
    </div>
    """, unsafe_allow_html=True)
    if terraforming_lottie:
        st_lottie(terraforming_lottie, height=150, key="terraforming_idx")
    
    tsi_body = st.selectbox("Evaluate Body for Terraforming:", ["Mars", "Luna", "Exoplanet Kepler-186f"], key="tsi_body")
    # Mock TSI data based on selection
    tsi_score = np.random.randint(20, 85) if tsi_body == "Mars" else np.random.randint(5, 40)
    st.metric(label=f"Overall TSI Score for {tsi_body}", value=f"{tsi_score}%")
    st.progress(tsi_score)
    st.caption("Parameters: Atmospheric Pressure (25%), Liquid Water Potential (70%), Gravitational Stability (90%), Temp. Regulation (40%), Radiation Shielding (30%)")

with feature_cols2[1]:
    st.markdown("""
    <div class="frosted-glass-card">
        <h3>⚠️ Dynamic Risk Assessment Matrix</h3>
        <p>Proactively identify and mitigate mission risks. This module uses predictive AI to forecast potential hazards across various mission phases.</p>
    </div>
    """, unsafe_allow_html=True)
    if risk_matrix_lottie:
        st_lottie(risk_matrix_lottie, height=150, key="risk_matrix_anim")

    risk_phases = st.multiselect("Select Mission Phases for Risk Analysis:", ["Launch", "Interplanetary Cruise", "Orbital Insertion", "Surface Operations", "Return Journey"], default=["Launch", "Surface Operations"], key="risk_phases")
    if risk_phases:
        risk_data = {
            'Risk Factor': ['Equipment Malfunction', 'Solar Flare Event', 'Micrometeoroid Impact', 'Navigation Error', 'Resource Depletion'],
        }
        for phase in risk_phases:
            risk_data[phase] = np.random.choice(['Low', 'Medium', 'High', 'Critical'], 5)
        
        risk_df = pd.DataFrame(risk_data)
        def style_risk(val):
            color = 'lightgreen'
            if val == 'Medium': color = 'orange'
            elif val == 'High': color = 'red'
            elif val == 'Critical': color = 'darkred'
            return f'color: {color}; font-weight: bold;'
        st.dataframe(risk_df.style.applymap(style_risk, subset=risk_phases), use_container_width=True)

# Animated Recommendation section
st.markdown("<hr class='enhanced-hr'>", unsafe_allow_html=True)
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

# New Interactive Data Analysis & AI Insights Section
st.markdown("<hr class='enhanced-hr'>", unsafe_allow_html=True)
st.markdown('<div class="section-header-box"><h2><span style="text-shadow: 0 0 8px #4CC9F0;">📊 Interactive Data Analysis & AI Insights</span></h2></div>', unsafe_allow_html=True)

data_col1, data_col2 = st.columns([1,2])

with data_col1:
    if data_insights_lottie:
        st_lottie(data_insights_lottie, height=250, key="data_insights_anim")
    
    dataset_choice = st.selectbox(
        "Select Dataset for Analysis:",
        options=["Recommended Mining Prospects", "Galactic Resource Map Data (Current Filter)", "Overall Star Catalog (Unfiltered)"],
        key="dataset_analysis_choice"
    )

with data_col2:
    st.markdown(f"### AI Insights for: _{dataset_choice}_")
    
    if dataset_choice == "Recommended Mining Prospects":
        st.dataframe(recommendation_df.head(), use_container_width=True)
        st.markdown("""
        <div class="frosted-glass-card" style="font-size: 0.9rem;">
            <p><strong>Total Prospects:</strong> {}</p>
            <p><strong>Highest Value Prospect:</strong> Site {} (Value Index: {})</p>
            <p><strong>Most Common Primary Resource:</strong> {}</p>
            <p><strong>Average Mission Duration (approx):</strong> {} (Note: Qualitative data)</p>
        </div>
        """.format(
            len(recommendation_df),
            recommendation_df.loc[recommendation_df['Value Index'].idxmax()]['Site ID'],
            recommendation_df['Value Index'].max(),
            recommendation_df['Primary Resource'].mode()[0] if not recommendation_df['Primary Resource'].mode().empty else "N/A",
            recommendation_df['Mission Duration'].mode()[0] if not recommendation_df['Mission Duration'].mode().empty else "N/A" # Simplistic avg
        ), unsafe_allow_html=True)

    elif dataset_choice == "Galactic Resource Map Data (Current Filter)":
        # Use the globally available filtered_data from the map section
        st.dataframe(filtered_data.head(), use_container_width=True)
        if not filtered_data.empty:
            st.markdown("""
            <div class="frosted-glass-card" style="font-size: 0.9rem;">
                <p><strong>Sites in Current View ({}):</strong> {}</p>
                <p><strong>Average Resource Index:</strong> {:.2f}</p>
                <p><strong>Predominant Site Type:</strong> {}</p>
                <p><strong>Distance Range Covered:</strong> {:.1f} LY to {:.1f} LY</p>
            </div>
            """.format(
                distance_filter, len(filtered_data),
                filtered_data['resource_value'].mean(),
                filtered_data['site_type'].mode()[0] if not filtered_data['site_type'].mode().empty else "N/A",
                filtered_data['distance'].min(), filtered_data['distance'].max()
            ), unsafe_allow_html=True)
        else:
            st.info(f"No data available for the current filter: {distance_filter}")

    elif dataset_choice == "Overall Star Catalog (Unfiltered)":
        st.dataframe(star_data.head(), use_container_width=True)
        st.markdown("""
        <div class="frosted-glass-card" style="font-size: 0.9rem;">
            <p><strong>Total Cataloged Celestial Bodies:</strong> {}</p>
            <p><strong>Overall Average Resource Index:</strong> {:.2f}</p>
            <p><strong>Site Type Distribution:</strong> {} ...</p>
        </div>
        """.format(
            len(star_data),
            star_data['resource_value'].mean(),
            ", ".join([f"{idx}: {val}" for idx, val in star_data['site_type'].value_counts().nlargest(3).items()])
        ), unsafe_allow_html=True)

st.markdown("<hr class='enhanced-hr'>", unsafe_allow_html=True)
st.markdown('<div class="section-header-box"><h2><span style="text-shadow: 0 0 8px #4CC9F0;">🌠 Featured Celestial Body of the Week</span></h2></div>', unsafe_allow_html=True)

featured_body_col1, featured_body_col2 = st.columns([1, 2])

with featured_body_col1:
    if planet_lottie:
        st_lottie(planet_lottie, height=250, key="featured_planet_anim")
    # Placeholder for an image, ideally dynamic
    # st.image("https://science.nasa.gov/wp-content/uploads/2023/09/PIA25822-HR-web.jpg?w=4096&format=jpeg", caption="Artist's concept of TRAPPIST-1e", use_column_width=True)

with featured_body_col2:
    st.markdown("""
    <div class="frosted-glass-card">
        <h3>TRAPPIST-1e: Prime Exoplanet Candidate</h3>
        <p><strong>Type:</strong> Earth-sized Exoplanet</p>
        <p><strong>Constellation:</strong> Aquarius</p>
        <p><strong>Distance:</strong> ~40 Light-years</p>
        <p><strong>Key Resources Potential:</strong> Liquid Water, Silicates, Iron Core</p>
        <p><strong>Orbital Period:</strong> 6.1 Earth days</p>
        <p><strong>Discovery:</strong> 2017, TRAPPIST Telescope</p>
        <p><strong>CelestAI Insight:</strong> High probability of atmospheric biosignatures. Recommended for advanced spectroscopic follow-up. Potential for subsurface liquid water oceans makes it a high-priority target for astrobiological research and future long-range probe missions.</p>
        <p><strong>Terraforming Suitability Index (TSI):</strong> 68% (Provisional)</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='enhanced-hr'>", unsafe_allow_html=True)
st.markdown('<div class="section-header-box"><h2><span style="text-shadow: 0 0 8px #4CC9F0;">🛰️ Mission Success Predictor (Simulation)</span></h2></div>', unsafe_allow_html=True)

predictor_col1, predictor_col2 = st.columns([1,1])

with predictor_col1:
    st.markdown("""
    <div class="frosted-glass-card">
        <p>Configure hypothetical mission parameters to receive an AI-driven success probability assessment. This tool leverages historical data and complex event modeling.</p>
    </div>
    """, unsafe_allow_html=True)
    if ai_advisor_lottie: # Re-using AI advisor lottie for this specific predictive feature
        st_lottie(ai_advisor_lottie, height=200, key="mission_predictor_anim")

with predictor_col2:
    st.markdown("#### Configure Mission Parameters:")
    mission_target_type = st.selectbox(
        "Select Target Type:",
        ["Asteroid - C-Type (Carbonaceous)", "Lunar South Pole Crater", "Exoplanet - Super-Earth", "Gas Giant Moon - Icy Crust", "Deep Space Anomaly"],
        key="mission_target_sim"
    )
    mission_complexity = st.slider(
        "Mission Complexity Score (1-10):", 1, 10, 5,
        help="Higher scores indicate more challenging missions (e.g., extreme environments, new technologies).",
        key="mission_complexity_sim"
    )
    mission_resource_prio = st.multiselect(
        "Key Resource Priorities:",
        ["Water Ice", "Helium-3", "Platinum Group Metals", "Exotic Isotopes", "Organic Compounds"],
        default=["Water Ice", "Helium-3"],
        key="mission_resource_sim"
    )

    if st.button("🔮 Predict Mission Outcome", key="predict_outcome_button"):
        with st.spinner("Calculating quantum probabilities..."):
            time.sleep(2)
            # Mock prediction logic
            base_success = 85 - (mission_complexity * 3) + (len(mission_resource_prio) * 2)
            if "Exotic Isotopes" in mission_resource_prio or "Exoplanet" in mission_target_type:
                base_success -= 10
            if "Deep Space Anomaly" in mission_target_type:
                base_success = np.random.randint(30,70) # More unpredictable
            
            predicted_success_rate = np.clip(base_success + np.random.randint(-5, 5), 20, 95)
            
            st.markdown(f"#### Predicted Success Rate: **{predicted_success_rate}%**")
            st.progress(predicted_success_rate)

            insight_message = f"CelestAI projects a {predicted_success_rate}% success probability for a mission targeting a {mission_target_type} with complexity {mission_complexity}, prioritizing {', '.join(mission_resource_prio)}. "
            if predicted_success_rate > 75:
                insight_message += "Favorable conditions indicated. Key factors: robust resource profile and manageable complexity."
                st.success(insight_message)
            elif predicted_success_rate > 50:
                insight_message += "Moderate probability. Recommend further risk assessment on identified variables (e.g., equipment stress under target conditions)."
                st.warning(insight_message)
            else:
                insight_message += "Challenging outlook. Significant risk factors identified. Consider alternative approaches or enhanced contingency planning."
                st.error(insight_message)

st.markdown("<hr class='enhanced-hr'>", unsafe_allow_html=True)
st.markdown('<div class="section-header-box"><h2><span style="text-shadow: 0 0 8px #4CC9F0;">📡 Cosmic Event Horizon - Latest Intel</span></h2></div>', unsafe_allow_html=True)

intel_col1, intel_col2, intel_col3 = st.columns(3)
intel_items = [
    {"icon": "🌊", "title": "Subsurface Ocean Confirmed on Kepler-452b", "source": "Deep Space Probe 'Odysseus'", "time": "2 min ago", "lottie": news_lottie},
    {"icon": "✨", "title": "Unusual Energy Signature Detected - Sector Gamma-7", "source": "DSN Array 3", "time": "15 min ago", "lottie": news_lottie},
    {"icon": "⛏️", "title": "New High-Yield Titanium Deposits on Asteroid Bennu", "source": "Mining Drone Swarm Alpha", "time": "45 min ago", "lottie": news_lottie}
]

for i, col in enumerate([intel_col1, intel_col2, intel_col3]):
    item = intel_items[i]
    with col:
        st.markdown(f"""
        <div class="frosted-glass-card" style="height: 280px;">
            <h4 style="color: #4CC9F0;">{item['icon']} {item['title']}</h4>
            <p style="font-size: 0.85rem;"><strong>Source:</strong> {item['source']}<br><strong>Reported:</strong> {item['time']}</p>
        </div>
        """, unsafe_allow_html=True)
        if item.get("lottie"):
             st_lottie(item["lottie"], height=100, key=f"intel_anim_{i}", speed=0.7)

# Animated Analysis section
st.markdown("<hr class='enhanced-hr'>", unsafe_allow_html=True)
st.markdown("## ✨ Platform Highlights")

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
        st_lottie(platform_lottie, height=400, key="platform_animation")
    except Exception as e:
        st.image("https://via.placeholder.com/300x400.png?text=Analysis+Dashboard", use_column_width=True)
        st.error(f"Could not display animation: {e}")

# Call-to-action section
st.markdown("<hr class='enhanced-hr'>", unsafe_allow_html=True)
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
st.markdown("<hr class='enhanced-hr'>", unsafe_allow_html=True)
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #b8c2cc; font-size: 0.9rem;">
            <strong>System Status:</strong> <span style="color: #28a745;">Optimal</span><br>
            Last Sync: {time.strftime("%d-%m-%Y %H:%M:%S")} IST
        </p>
    </div>
    """, unsafe_allow_html=True)

with footer_col2:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #b8c2cc; font-size: 0.9rem;">
            <strong>🪐 CelestAI Nexus</strong><br>
            Version 3.0.0 | Orionis Build
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
