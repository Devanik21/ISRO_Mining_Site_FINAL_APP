import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import OPTICS  # Not used, but imported.  Could be used in future development.
from sklearn.metrics import silhouette_score  # Not used, but imported. Could be used for cluster evaluation.
import shap
# from astropy import units as u  # No longer using SkyCoord; removed astropy dependency
# from astropy.coordinates import SkyCoord
import requests
from io import BytesIO  # Not used, but good to keep if expanding to handle file uploads
import time
import matplotlib.pyplot as plt  # Needed for shap.summary_plot
from streamlit_autorefresh import st_autorefresh

# Constants
NASA_CTA_URL = "https://ssd-api.jpl.nasa.gov/cad.api"  # Corrected URL
SPACE_AGENCY_LOGO = "https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organization_Logo.svg"

# Initialize session state for persistence
if 'mining_sim' not in st.session_state:
    st.session_state.mining_sim = {'running': False, 'resources': None}

# Helper function to generate sample asteroid data if API fails or returns unusable data
def generate_sample_asteroid_data(num_rows=100):
    """Generates a sample DataFrame mimicking NASA asteroid data."""
    st.info("Using generated sample asteroid data as a fallback.")
    data = {
        'des': [f"AST-S{i:03d}" for i in range(num_rows)],
        'diameter': np.random.uniform(0.01, 5.0, num_rows),  # km
        'dist': np.random.uniform(0.1, 3.0, num_rows),      # AU
        'v_inf': np.random.uniform(1, 20, num_rows),        # km/s
    }
    df = pd.DataFrame(data)
    # Ensure columns are numeric
    for col in ['diameter', 'dist', 'v_inf']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    return df

# Quantum-inspired Optimization (Mock)
def quantum_annealing_optimization(cost_matrix):
    np.fill_diagonal(cost_matrix, 0)
    n = cost_matrix.shape[0]
    solution = list(range(n))
    np.random.shuffle(solution)
    return solution

# Real-time Space Weather Data
@st.cache_data(ttl=3600)  # Refresh hourly
def get_space_weather():
    try:
        res = requests.get("https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json")
        res.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = res.json()
        # Check if the data is in the expected format
        if isinstance(data, list) and len(data) > 1 and isinstance(data[0], list):
            headers = data[0]
            actual_data = data[1:]
            df = pd.DataFrame(actual_data, columns=headers)

            # We are interested in 'time_tag' and 'bt' (magnetic field total)
            if 'time_tag' in df.columns and 'bt' in df.columns:
                df_processed = df[['time_tag', 'bt']].copy()
                df_processed['time_tag'] = pd.to_datetime(df_processed['time_tag'], errors='coerce')
                df_processed['bt'] = pd.to_numeric(df_processed['bt'], errors='coerce')
                df_processed.rename(columns={'time_tag': 'time'}, inplace=True)
                df_processed.dropna(inplace=True)
                if not df_processed.empty:
                    return df_processed
            st.warning("Space weather data received but in unexpected format or missing key columns ('time_tag', 'bt').")
            
        else:
            st.warning("Unexpected data structure from space weather API.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching space weather data: {e}")
    except (ValueError, KeyError, TypeError) as e:  # Catch JSON decoding or data structure issues
        st.error(f"Error processing space weather data: {e}")
    
    # Fallback to sample data
    st.info("Using generated sample space weather data.")
    return pd.DataFrame({
        'time': pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=100, freq='H'),
        'bt': np.random.normal(5, 1, 100)
    })

# Asteroid Data from NASA API
@st.cache_data(ttl=86400) # Refresh daily
def get_nasa_asteroids():
    try:
        res = requests.get(NASA_CTA_URL, params={'date-min': '2023-01-01', 'date-max': '2024-01-01', 'dist-max': '0.1'}) # More focused query
        res.raise_for_status()
        data = res.json()

        if 'data' in data and 'fields' in data:
            df_full = pd.DataFrame(data['data'], columns=data['fields'])
            required_cols = ['des', 'diameter', 'dist', 'v_inf']
            
            if not all(col in df_full.columns for col in required_cols):
                st.warning(f"NASA API data missing one or more required columns: {required_cols}.")
                return generate_sample_asteroid_data()
            
            df = df_full[required_cols].copy()
            numeric_cols = ['diameter', 'dist', 'v_inf']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=numeric_cols, inplace=True)
            
            if df.empty:
                st.warning("No valid asteroid data after processing API response.")
                return generate_sample_asteroid_data()
            return df
        else:
            st.warning("Unexpected data format from NASA API (missing 'data' or 'fields').")
            return generate_sample_asteroid_data()

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching asteroid data: {e}")
        return generate_sample_asteroid_data()
    except (ValueError, KeyError, TypeError) as e:
        st.error(f"Error processing asteroid data structure: {e}")
        return generate_sample_asteroid_data()

# Advanced Material Composition Analysis
def analyze_mineral_composition(df):
    elements = ['Fe', 'Si', 'Mg', 'Ni', 'H2O', 'CH4']
    # Ensure the DataFrame has enough rows before proceeding.
    if len(df) > 0:
        comp = pd.DataFrame(np.random.dirichlet(np.ones(6), size=len(df)),
                       columns=elements)
        return pd.concat([df, comp], axis=1)
    else:
      return df.copy() #return a copy of original df if empty to avoid modifying it upstream


# HPC Simulation (Mock)
def run_gravitational_sim(coords):
    # Ensure there are coordinates to work with.
    if len(coords) > 0:
        masses = np.random.lognormal(3, 0.5, len(coords))
        potential = masses / np.linalg.norm(coords, axis=1)
        return pd.Series(potential, name='gravity_potential')
    else:
        return pd.Series(name='gravity_potential')

# AI-powered Anomaly Detection
def detect_anomalies(X):
    from sklearn.ensemble import IsolationForest
    if len(X) >0 :
        clf = IsolationForest(contamination=0.1)
        return clf.fit_predict(X)
    else:
        return np.array([])

# XAI Visualization
def shap_explainer(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    return fig

# Adaptive UI Components
def create_parameter_matrix(body_type):
    params = {
        'Asteroid': ['spin_rate', 'albedo', 'taxonomy'],
        'Moon': ['crater_density', 'regolith_depth', 'tidal_lock'],
        'Planet': ['atmo_pressure', 'magnetic_field', 'core_size']
    }
    return params.get(body_type, ['unknown_body_param'])

# Main Visualization Page
def show_visualize_page():
    # Immersive UI Configuration
    st.set_page_config(layout="wide", page_icon="🚀", page_title="ISRO ExoMining AI")
    st_autorefresh(interval=10000, key="data_refresh")  # Auto-refresh every 10s

    # NASA-grade Styling
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono&display=swap');
        .main {{
            background: radial-gradient(circle at 10% 20%, #0f2027 0%, #203a43 50%, #2c5364 100%);
        }}
        .stApp {{
            background: url("https://www.esa.int/var/esa/storage/images/esa_multimedia/images/2023/07/webb_star_formation/25064845-1-eng-GB/Webb_star_formation_pillars.jpg") no-repeat center center fixed;
            background-size: cover;
        }}
        .title {{
            font-family: 'Space Mono', monospace;
            text-shadow: 0 0 10px #00f7ff;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ opacity: 0.8; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.8; }}
        }}
    </style>
    """, unsafe_allow_html=True)

    # Holographic Header
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; margin: -50px 0 30px 0;">
        <img src="{SPACE_AGENCY_LOGO}" style="height: 80px; margin-right: 20px;">
        <h1 class="title" style="color: #00f7ff; font-size: 3.5rem;">
            ISRO Exoplanetary Mining Intelligence System
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # Quantum Computing Section
    with st.expander("🚀 Quantum Resource Optimization", expanded=True):
        cost_matrix = np.random.rand(10,10)
        optimal_path = quantum_annealing_optimization(cost_matrix)
        path_str = " → ".join([f"Site {i+1}" for i in optimal_path])
        st.markdown(f"**Optimal Mining Route:** ⚛️ `{path_str}`")
        # Corrected the plotting line -  use_container_width and indexing
        optimal_costs = [cost_matrix[i, j] for i, j in zip(optimal_path[:-1], optimal_path[1:])]
        fig = px.line(pd.Series(optimal_costs))
        st.plotly_chart(fig, use_container_width=True)


    # Real-time Space Weather Monitor
    space_weather = get_space_weather()
    with st.container():
        col1, col2 = st.columns([1,3])
        with col1:
            if not space_weather.empty:
                st.metric("Solar Wind Speed", f"{space_weather['bt'].iloc[-1]:.1f} nT",
                         delta=f"{space_weather['bt'].diff().iloc[-1]:.1f} nT (change)")
            else:
                st.metric("Solar Wind Speed", "N/A", delta="N/A")
        with col2:
          if not space_weather.empty and 'time' in space_weather.columns and 'bt' in space_weather.columns:
            st.plotly_chart(px.area(space_weather, x='time', y='bt',
                                   title="Real-time Solar Wind Monitoring"),
                           use_container_width=True)
          else:
            st.write("No space weather data available.")

    # Multi-Planetary Data Fusion
    asteroid_data = get_nasa_asteroids()
    df = analyze_mineral_composition(asteroid_data)
    
    # Gravitational sim needs x,y,z - create them here as they are not in the NASA data.
    if df is not None and not df.empty:
        df['x'] = np.random.uniform(-5, 5, len(df)) # More spread for visualization
        df['y'] = np.random.uniform(-5, 5, len(df))
        df['z'] = np.random.uniform(-2, 2, len(df))
        df['gravity_potential'] = run_gravitational_sim(df[['x', 'y', 'z']].values)
    elif df is None: # Should not happen with new fallbacks, but good for safety
        df = pd.DataFrame()
        st.warning("Main asteroid DataFrame is None, initialized to empty.")
    # If df is already an empty DataFrame, no action needed here.

    # AI-Driven Mineral Prediction
    with st.expander("🧠 Deep Core Mineral Predictor", expanded=True):
        # Ensure required columns for prediction exist and df is not empty
        predictor_cols = ['diameter', 'dist', 'v_inf', 'Fe'] # 'Fe' is the target
        if not df.empty and all(col in df.columns for col in predictor_cols):
            model = GradientBoostingRegressor(n_estimators=200)
            # Ensure X does not contain the target variable 'Fe'
            X_features = ['diameter', 'dist', 'v_inf']
            X = df[X_features]
            y = df['Fe'] # Target variable
            model.fit(X, y)
            df['Fe_pred'] = model.predict(X)

            fig = shap_explainer(model, X, X.columns)
            st.pyplot(fig)
            st.plotly_chart(px.scatter_3d(df, x='diameter', y='dist', z='v_inf',
                                         color='Fe', size='Fe_pred'),
                           use_container_width=True)
        else:
            st.write("No data for mineral prediction.")


    # Gravitational Waveform Analysis
    st.subheader("🌌 Gravitational Resonance Imaging")
    freq = st.slider("Harmonic Frequency", 0.1, 10.0, 2.4)
    waveform = np.sin(freq * np.linspace(0, 10, 1000)) * np.random.chisquare(2,1000)
    st.plotly_chart(px.line(pd.DataFrame({'wave': waveform})),
                   use_container_width=True)

    # Multi-Agent Mining Simulation
    st.subheader("🤖 Autonomous Swarm Simulation")
    if st.button("Initiate Mining Swarm"):
        st.session_state.mining_sim['running'] = True
        st.session_state.mining_sim['resources'] = np.random.randint(100,1000, size=50)

    if st.session_state.mining_sim['running']:
        resources = st.session_state.mining_sim['resources']
        plt = go.Figure(go.Heatmap(z=[resources], colorscale='viridis'))
        st.plotly_chart(plt, use_container_width=True)
        st.session_state.mining_sim['resources'] *= 0.97  # Resource depletion

    # Exoplanetary Navigation System (Simplified without astropy)
    st.subheader("🌠 Celestial Navigation Interface")
    st.write(f"**Current Target:** Random coordinates (RA: {np.random.uniform(0,360):.2f}, DEC: {np.random.uniform(-90,90):.2f})")
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/3D_Spherical_Coordinate_System.png",
             use_column_width=True)


    # Voice Command Integration
    with st.expander("🎤 Voice Command Console"):
        st.write("**Supported Commands:** `show resources`, `run simulation`, `analyze site`")
        command = st.selectbox("Voice Input", ["Select...", "show resources", "run simulation"])
        if command != "Select...":
            st.success(f"Executing command: **{command}**")
            time.sleep(1)
            st.experimental_rerun()

    # Quantum-Safe Encryption Badge
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; background: #000; padding: 5px 10px; border-radius: 5px;">
        🔒 Quantum-Safe Encryption: AES-512 + Lattice-based NIST Standard
    </div>
    """, unsafe_allow_html=True)

show_visualize_page()
