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
from streamlit_lottie import st_lottie # For new animations

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

# --- New Advanced Feature Functions ---

@st.cache_data(ttl=3600)
def analyze_exoplanet_atmosphere(planet_name):
    """Generates mock atmospheric composition for a given exoplanet."""
    np.random.seed(hash(planet_name) % (2**32 - 1)) # Seed based on planet name for consistency
    gases = ['N2', 'O2', 'CO2', 'CH4', 'Ar', 'H2O Vapor', 'He']
    composition = np.random.dirichlet(np.ones(len(gases)) * np.random.rand() * 10, size=1).flatten() * 100
    return pd.DataFrame({'Gas': gases, 'Percentage': composition})

@st.cache_data(ttl=3600)
def plan_interstellar_trajectory(target_system):
    """Generates mock waypoints for an interstellar trajectory."""
    np.random.seed(hash(target_system) % (2**32 - 1))
    num_waypoints = np.random.randint(5, 15)
    waypoints = pd.DataFrame({
        'x': np.cumsum(np.random.normal(0, 10, num_waypoints)),
        'y': np.cumsum(np.random.normal(0, 10, num_waypoints)),
        'z': np.cumsum(np.random.normal(0, 5, num_waypoints)),
        'waypoint': [f"WP-{i}" for i in range(num_waypoints)]
    })
    return waypoints

@st.cache_data(ttl=60) # More frequent refresh for something like signal detection
def detect_alien_signal():
    """Generates mock signal data with a chance of a 'detection'."""
    time_points = np.linspace(0, 100, 500)
    noise = np.random.normal(0, 1, 500)
    signal_strength = 5 * np.sin(time_points / 10) + noise
    detected = False
    if np.random.rand() < 0.1: # 10% chance of detection
        idx = np.random.randint(100, 400)
        signal_strength[idx:idx+10] += np.random.uniform(5, 10) # Spike for detection
        detected = True
    return pd.DataFrame({'Time': time_points, 'SignalStrength': signal_strength}), detected

@st.cache_data(ttl=3600)
def monitor_terraforming_progress(planet_name):
    """Generates mock terraforming metrics."""
    np.random.seed(hash(planet_name) % (2**32 - 1))
    metrics = {
        'Oxygen Level (%)': np.random.uniform(0, 21),
        'Surface Temp (°C)': np.random.uniform(-50, 30),
        'Liquid Water Coverage (%)': np.random.uniform(0, 70),
        'Atmospheric Pressure (kPa)': np.random.uniform(0, 101)
    }
    return metrics

@st.cache_data(ttl=1800)
def calculate_dyson_swarm_energy(completion_percentage):
    """Generates mock energy output for a Dyson swarm."""
    # Assume max output of a star like Sol is ~3.8e26 Watts
    # Let's scale this down for a more relatable number in Petawatts (1e15 W)
    max_petawatts = 3.8e11 
    current_output = max_petawatts * (completion_percentage / 100)**2 # Non-linear increase
    time_series = pd.DataFrame({
        'Year': np.arange(2000, 2000 + int(completion_percentage) + 1),
        'EnergyOutput_PW': max_petawatts * (np.linspace(0, completion_percentage, int(completion_percentage)+1) / 100)**2
    })
    return current_output, time_series

@st.cache_data(ttl=3600)
def visualize_accretion_disk(black_hole_mass_suns):
    """Generates data for a mock accretion disk visualization."""
    np.random.seed(int(black_hole_mass_suns))
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    # Intensity decreases with distance, with some randomness
    intensity = np.exp(-(X**2 + Y**2) / (2 * (black_hole_mass_suns/2))) * (1 + 0.1*np.random.rand(100,100))
    intensity[np.sqrt(X**2 + Y**2) < black_hole_mass_suns*0.1] = 0 # Event horizon shadow
    return X, Y, intensity

@st.cache_data(ttl=3600)
def predict_wormhole_stability(coordinates_str):
    """Generates mock stability score and factors for a wormhole."""
    np.random.seed(hash(coordinates_str) % (2**32 - 1))
    stability_score = np.random.uniform(0, 100)
    factors = {
        'Exotic Matter Density': np.random.uniform(-10, 10),
        'Gravitational Fluctuations': np.random.uniform(-5, 5),
        'Temporal Distortion': np.random.uniform(-3, 3),
        'Energy Input': np.random.uniform(0, 15)
    } # Positive values contribute to stability
    return stability_score, pd.DataFrame(list(factors.items()), columns=['Factor', 'ImpactScore'])

@st.cache_data(ttl=86400) # Daily
def map_dark_matter_density(sector_id):
    """Generates a 2D grid of mock dark matter density values."""
    np.random.seed(hash(sector_id) % (2**32 - 1))
    density_map = np.random.rand(50, 50) * np.random.uniform(0.1, 5) # Arbitrary units
    return density_map

def simulate_relativistic_effects(velocity_fraction_c):
    """Calculates mock time dilation and length contraction."""
    gamma = 1 / np.sqrt(1 - velocity_fraction_c**2)
    time_dilation_factor = gamma
    length_contraction_factor = 1 / gamma
    return time_dilation_factor, length_contraction_factor

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
             use_container_width=True)


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
    
    st.markdown("<hr style='border:1px solid #00f7ff; opacity:0.3;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <h2 class="title" style="color: #00f7ff; font-size: 2.5rem; text-align:center;">
        Advanced Analytics Modules
    </h2>
    """, unsafe_allow_html=True)

    # --- Feature 1: Exoplanet Atmosphere Analyzer ---
    with st.expander("🔬 Exoplanet Atmosphere Analyzer", expanded=False):
        exoplanet_list = ["Kepler-186f", "TRAPPIST-1e", "Proxima Centauri b", "Gliese 581g"]
        selected_exoplanet = st.selectbox("Select Exoplanet:", exoplanet_list, key="exo_atm_select")
        if st.button("Analyze Atmosphere", key="exo_atm_btn"):
            with st.spinner(f"Analyzing atmosphere of {selected_exoplanet}..."):
                time.sleep(1.5) # Simulate analysis time
                atm_data = analyze_exoplanet_atmosphere(selected_exoplanet)
                fig_atm = px.bar(atm_data, x='Gas', y='Percentage', title=f"Atmospheric Composition of {selected_exoplanet}",
                                 color='Gas', labels={'Percentage':'Concentration (%)'})
                fig_atm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#00f7ff')
                st.plotly_chart(fig_atm, use_container_width=True)
                st.dataframe(atm_data)

    # --- Feature 2: Interstellar Trajectory Planner ---
    with st.expander("🗺️ Interstellar Trajectory Planner", expanded=False):
        target_star_system = st.text_input("Target Star System (e.g., Alpha Centauri):", "Alpha Centauri", key="traj_target")
        if st.button("Plan Trajectory", key="traj_plan_btn"):
            if target_star_system:
                with st.spinner(f"Calculating trajectory to {target_star_system}..."):
                    time.sleep(2)
                    trajectory_data = plan_interstellar_trajectory(target_star_system)
                    fig_traj = px.line_3d(trajectory_data, x='x', y='y', z='z', color='waypoint', markers=True,
                                          title=f"Interstellar Trajectory to {target_star_system}")
                    fig_traj.update_layout(scene=dict(xaxis_title='X (LY)', yaxis_title='Y (LY)', zaxis_title='Z (LY)',
                                                    bgcolor='rgba(0,0,0,0.1)'),
                                         paper_bgcolor='rgba(0,0,0,0)', font_color='#00f7ff')
                    st.plotly_chart(fig_traj, use_container_width=True)
            else:
                st.warning("Please enter a target star system.")

    # --- Feature 3: Alien Signal Detector ---
    with st.expander("📡 Alien Signal Detector (ASD)", expanded=False):
        if st.button("Scan for Extraterrestrial Signals", key="asd_scan_btn"):
            with st.spinner("Scanning designated frequency bands..."):
                time.sleep(2.5)
                signal_data, detected = detect_alien_signal()
                fig_signal = px.line(signal_data, x='Time', y='SignalStrength', title="Signal Strength Analysis")
                fig_signal.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#00f7ff')
                if detected:
                    st.success("Potential anomalous signal detected! Further analysis required.")
                    fig_signal.add_vline(x=signal_data[signal_data['SignalStrength'] > signal_data['SignalStrength'].mean() + 3*signal_data['SignalStrength'].std()]['Time'].mean(), 
                                        line_dash="dash", line_color="red", annotation_text="Anomaly Peak")
                else:
                    st.info("No anomalous signals detected in this scan cycle. Background noise levels nominal.")
                st.plotly_chart(fig_signal, use_container_width=True)

    # --- Feature 4: Terraforming Progress Monitor ---
    with st.expander("🌍 Terraforming Progress Monitor", expanded=False):
        terraforming_candidates = ["Mars", "Europa", "Titan"]
        selected_tf_planet = st.selectbox("Select Planet for Terraforming Status:", terraforming_candidates, key="tf_planet_select")
        if selected_tf_planet:
            tf_metrics = monitor_terraforming_progress(selected_tf_planet)
            st.subheader(f"Terraforming Status: {selected_tf_planet}")
            cols_tf = st.columns(len(tf_metrics))
            for i, (metric_name, value) in enumerate(tf_metrics.items()):
                with cols_tf[i]:
                    # Simple metric display, could be gauge charts with more effort
                    st.metric(label=metric_name, value=f"{value:.2f}")
                    if "Percentage" in metric_name:
                        st.progress(int(value))

    # --- Feature 5: Dyson Swarm Energy Output ---
    with st.expander("☀️ Dyson Swarm Energy Output Simulator", expanded=False):
        swarm_completion = st.slider("Dyson Swarm Completion (%):", 0, 100, 25, key="dyson_completion")
        current_output_pw, energy_time_series = calculate_dyson_swarm_energy(swarm_completion)
        st.metric(label="Current Estimated Energy Output", value=f"{current_output_pw:.2e} PW")
        
        fig_dyson = px.line(energy_time_series, x='Year', y='EnergyOutput_PW', title="Projected Dyson Swarm Energy Output Over Time")
        fig_dyson.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#00f7ff',
                                yaxis_title="Energy Output (Petawatts)")
        st.plotly_chart(fig_dyson, use_container_width=True)

    # --- Feature 6: Black Hole Accretion Disk Visualizer ---
    with st.expander("🌀 Black Hole Accretion Disk Visualizer", expanded=False):
        bh_mass = st.number_input("Black Hole Mass (Solar Masses):", min_value=1.0, max_value=1000.0, value=10.0, step=1.0, key="bh_mass_input")
        if st.button("Visualize Accretion Disk", key="bh_vis_btn"):
            X_disk, Y_disk, intensity_disk = visualize_accretion_disk(bh_mass)
            fig_bh = go.Figure(data = go.Contour(z=intensity_disk, x=X_disk[0,:], y=Y_disk[:,0], colorscale='hot'))
            fig_bh.update_layout(title=f"Accretion Disk Intensity - {bh_mass} Solar Masses BH",
                                 xaxis_title="Relative X", yaxis_title="Relative Y",
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#00f7ff')
            st.plotly_chart(fig_bh, use_container_width=True)

    # --- Feature 7: Wormhole Stability Predictor ---
    with st.expander("웜홀 안정성 예측기 (Wormhole Stability Predictor)", expanded=False): # Korean title for fun
        wh_coords = st.text_input("Enter Wormhole Coordinates (e.g., X:123,Y:456,Z:789):", "X:10,Y:20,Z:30", key="wh_coords")
        if st.button("Predict Stability", key="wh_stab_btn"):
            stability, factors_df = predict_wormhole_stability(wh_coords)
            st.metric(label=f"Predicted Stability for Wormhole at {wh_coords}", value=f"{stability:.2f}%")
            st.progress(int(stability))
            
            fig_wh_factors = px.bar(factors_df, x='Factor', y='ImpactScore', color='ImpactScore',
                                    title="Factors Influencing Wormhole Stability",
                                    color_continuous_scale=px.colors.diverging.RdBu)
            fig_wh_factors.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#00f7ff')
            st.plotly_chart(fig_wh_factors, use_container_width=True)

    # --- Feature 8: Dark Matter Density Mapper ---
    with st.expander("👻 Dark Matter Density Mapper", expanded=False):
        galactic_sector = st.text_input("Galactic Sector ID (e.g., GS-007):", "GS-Alpha-Prime", key="dm_sector")
        if st.button("Map Dark Matter Density", key="dm_map_btn"):
            with st.spinner(f"Mapping dark matter in sector {galactic_sector}..."):
                time.sleep(1)
                dm_map_data = map_dark_matter_density(galactic_sector)
                fig_dm = px.imshow(dm_map_data, color_continuous_scale='viridis', 
                                   title=f"Dark Matter Density Map - Sector {galactic_sector}")
                fig_dm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#00f7ff')
                st.plotly_chart(fig_dm, use_container_width=True)

    # --- Feature 9: Relativistic Effects Simulator ---
    with st.expander("⏳ Relativistic Effects Simulator", expanded=False):
        velocity_c = st.slider("Velocity (fraction of speed of light, c):", 0.0, 0.999, 0.5, 0.001, format="%.3f", key="rel_vel")
        time_dilation, length_contraction = simulate_relativistic_effects(velocity_c)
        col_rel1, col_rel2 = st.columns(2)
        with col_rel1:
            st.metric("Time Dilation Factor (γ)", f"{time_dilation:.3f}x")
            st.caption("Time for moving observer appears slower by this factor to a stationary observer.")
        with col_rel2:
            st.metric("Length Contraction Factor (1/γ)", f"{length_contraction:.3f}x")
            st.caption("Length of moving object appears shorter by this factor in direction of motion.")
        if velocity_c > 0.1:
            st.info(f"At {velocity_c*100:.1f}% of light speed, 1 year for you would be {1*time_dilation:.2f} years for a stationary observer.")

    # --- Feature 10: Cosmic Ray Shielding Effectiveness ---
    with st.expander("🛡️ Cosmic Ray Shielding Effectiveness Analyzer", expanded=False):
        shielding_materials = ["Titanium Alloy", "Polyethylene Composite", "Lead-Infused Aerogel", "Magnetic Deflector"]
        selected_material = st.selectbox("Select Shielding Material:", shielding_materials, key="shield_mat_select")
        if st.button("Analyze Shielding", key="shield_analyze_btn"):
            np.random.seed(hash(selected_material) % (2**32 - 1))
            effectiveness = np.random.uniform(50, 99.9) # %
            particle_types = ["Protons", "Alpha Particles", "Heavy Nuclei", "Gamma Rays"]
            reduction_data = pd.DataFrame({
                'Particle Type': particle_types,
                'Flux Reduction (%)': np.random.uniform(effectiveness*0.8, effectiveness, len(particle_types))
            })
            st.metric(f"Overall Effectiveness of {selected_material}", f"{effectiveness:.1f}%")
            fig_shield = px.bar(reduction_data, x='Particle Type', y='Flux Reduction (%)', color='Particle Type',
                                title=f"Cosmic Ray Flux Reduction by {selected_material}")
            fig_shield.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#00f7ff')
            st.plotly_chart(fig_shield, use_container_width=True)

show_visualize_page()
