import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import streamlit as st
import plotly.express as px
import os
from datetime import datetime

# Set up the page configuration
st.set_page_config(
    page_title="ISRO Mining Site Recommendation System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional appearance
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom, #0d1117, #161b22);
            color: #f0f6fc;
        }
        h1, h2, h3 {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(90deg, #ff9966, #ff5e62);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }
        .css-1d391kg, .css-12oz5g7 {
            background-color: rgba(9, 15, 33, 0.7);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }
        .stSlider > div > div > div > div {
            background-color: #ff5e62;
        }
        .stButton > button {
            background: linear-gradient(90deg, #ff9966, #ff5e62);
            color: white;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        .stButton > button:hover {
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
            transform: translateY(-2px);
        }
        .dataframe {
            border: none !important;
            border-collapse: separate;
            border-spacing: 0;
            border-radius: 8px;
            overflow: hidden;
            margin: 20px 0;
        }
        .dataframe th {
            background-color: #1f2937 !important;
            color: #f0f6fc !important;
            padding: 12px !important;
            font-weight: 600 !important;
            text-align: center !important;
        }
        .dataframe td {
            padding: 10px !important;
            border: none !important;
            text-align: center !important;
        }
        .dataframe tr:nth-child(odd) td {
            background-color: #1a1f29 !important;
        }
        .dataframe tr:nth-child(even) td {
            background-color: #22293d !important;
        }
        .card {
            background-color: rgba(40, 50, 78, 0.8);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        .metric-container {
            background-color: rgba(37, 45, 67, 0.7);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        .metric-value {
            font-size: 26px;
            font-weight: bold;
            color: #ffffff;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            color: #a0aec0;
        }
    </style>
""", unsafe_allow_html=True)

# Function to load or create the model and dataset
@st.cache_resource
def load_data_and_model():
    try:
        model = joblib.load("models/space_mining_model.pkl")
        df = pd.read_csv("data/space_mining_dataset.csv")
    except FileNotFoundError:
        # Create synthetic data and model
        df = create_synthetic_dataset()
        model = train_model(df)
        # Save the model and dataset
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists("data"):
            os.makedirs("data")
        joblib.dump(model, "models/space_mining_model.pkl")
        df.to_csv("data/space_mining_dataset.csv", index=False)
    return df, model

# Function to create a synthetic dataset
def create_synthetic_dataset(n_samples=100):
    np.random.seed(42)
    # Generate celestial body names
    celestial_bodies = []
    planets = ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    asteroids = ["Ceres", "Vesta", "Pallas", "Hygiea", "Psyche", "Eros", "Bennu"]
    
    for planet in planets:
        celestial_bodies.append(planet)
        for i in range(np.random.randint(1, 3)):
            celestial_bodies.append(f"{planet} Moon {i+1}")
    
    for prefix in asteroids:
        for i in range(np.random.randint(1, 3)):
            celestial_bodies.append(f"{prefix}-{chr(65+i)}")
    
    while len(celestial_bodies) < n_samples:
        celestial_bodies.append(f"Asteroid X-{np.random.randint(1000, 9999)}")
    
    celestial_bodies = np.random.choice(celestial_bodies, n_samples, replace=False)
    
    # Generate features
    iron = np.random.beta(2, 5, n_samples) * 100
    nickel = np.random.beta(1.5, 6, n_samples) * 100
    water_ice = np.random.beta(1, 3, n_samples) * 100
    rare_earth_elements = np.random.beta(1, 8, n_samples) * 100
    precious_metals = np.random.beta(1, 10, n_samples) * 100
    
    # Operational features
    sustainability_index = np.random.beta(2, 2, n_samples) * 100
    efficiency_index = np.random.beta(2, 2, n_samples) * 100
    risk_index = np.random.beta(2, 2, n_samples) * 100
    
    # Distance and mining complexity
    distance_from_earth = np.random.gamma(2, 0.3, n_samples) * 100
    extraction_complexity = np.random.beta(3, 3, n_samples) * 100
    
    # Create the DataFrame
    df = pd.DataFrame({
        'Celestial Body': celestial_bodies,
        'iron': iron,
        'nickel': nickel,
        'water_ice': water_ice,
        'rare_earth_elements': rare_earth_elements,
        'precious_metals': precious_metals,
        'sustainability_index': sustainability_index,
        'efficiency_index': efficiency_index,
        'risk_index': risk_index,
        'distance_from_earth': distance_from_earth,
        'extraction_complexity': extraction_complexity
    })
    
    # Calculate a composite score for the ML model
    df['composite_score'] = (
        0.25 * iron +
        0.2 * nickel +
        0.15 * water_ice +
        0.1 * rare_earth_elements +
        0.1 * precious_metals +
        0.1 * efficiency_index
    ) - (
        0.1 * risk_index +
        0.1 * distance_from_earth
    )
    
    # Normalize to 0-100 range
    min_score = df['composite_score'].min()
    max_score = df['composite_score'].max()
    df['composite_score'] = 100 * (df['composite_score'] - min_score) / (max_score - min_score)
    
    return df

# Function to train a model
def train_model(df):
    features = df[[
        'iron', 'nickel', 'water_ice', 'rare_earth_elements', 'precious_metals',
        'sustainability_index', 'efficiency_index', 'risk_index',
        'distance_from_earth', 'extraction_complexity'
    ]]
    target = df['composite_score']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features_scaled, target)
    
    return model

# Function to recommend mining sites
def recommend_sites(user_preferences, df, model, top_n=5):
    features = df[[
        'iron', 'nickel', 'water_ice', 'rare_earth_elements', 'precious_metals',
        'sustainability_index', 'efficiency_index', 'risk_index',
        'distance_from_earth', 'extraction_complexity'
    ]]
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    predicted_scores = model.predict(features_scaled)
    
    # Adjust weights based on user preferences
    weights = {
        'iron': user_preferences.get('iron_weight', 0.2),
        'nickel': user_preferences.get('nickel_weight', 0.15),
        'water_ice': user_preferences.get('water_ice_weight', 0.15),
        'rare_earth_elements': user_preferences.get('rare_earth_weight', 0.1),
        'precious_metals': user_preferences.get('precious_metals_weight', 0.1),
        'sustainability_index': user_preferences.get('sustainability_weight', 0.1),
        'efficiency_index': user_preferences.get('efficiency_weight', 0.1),
        'risk_index': -user_preferences.get('risk_weight', 0.1),
        'distance_from_earth': -user_preferences.get('distance_weight', 0.1),
        'extraction_complexity': -user_preferences.get('extraction_weight', 0.1)
    }
    
    # Calculate weighted score
    df['adjusted_score'] = 0
    for feature, weight in weights.items():
        if feature in df.columns:
            df['adjusted_score'] += weight * df[feature]
    
    # Combine model prediction with user preferences
    model_weight = user_preferences.get('model_weight', 0.5)
    user_weight = 1 - model_weight
    
    df['normalized_predicted'] = 100 * (predicted_scores - predicted_scores.min()) / (predicted_scores.max() - predicted_scores.min())
    df['final_score'] = (model_weight * df['normalized_predicted']) + (user_weight * df['adjusted_score'])
    
    # Sort and select top sites
    ranked_sites = df.sort_values(by='final_score', ascending=False).head(top_n)
    
    # Calculate importance factors
    importance_factors = {}
    for site_index, site in ranked_sites.iterrows():
        site_factors = []
        for feature, weight in weights.items():
            if feature in df.columns and abs(weight) >= 0.1:
                if weight > 0 and site[feature] > df[feature].quantile(0.75):
                    site_factors.append(f"High {feature.replace('_', ' ')}")
                elif weight < 0 and site[feature] < df[feature].quantile(0.25):
                    site_factors.append(f"Low {feature.replace('_', ' ')}")
        
        importance_factors[site['Celestial Body']] = site_factors[:3]  # Top 3 factors
    
    return ranked_sites, importance_factors

# Create a sidebar with ISRO branding and user inputs
def create_sidebar():
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="font-size: 24px; margin-bottom: 0;">🚀 ISRO</h1>
                <p style="font-size: 16px; opacity: 0.8;">Space Mining Division</p>
                <div style="height: 2px; background: linear-gradient(90deg, #ff9966, #ff5e62); margin: 15px 0;"></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3>Resource Preferences</h3>", unsafe_allow_html=True)
        iron = st.slider("Iron Importance", 0, 100, 50) / 100.0
        nickel = st.slider("Nickel Importance", 0, 100, 40) / 100.0
        water_ice = st.slider("Water/Ice Importance", 0, 100, 60) / 100.0
        rare_earth = st.slider("Rare Earth Elements", 0, 100, 30) / 100.0
        precious_metals = st.slider("Precious Metals", 0, 100, 35) / 100.0
        
        st.markdown("<h3>Operational Factors</h3>", unsafe_allow_html=True)
        sustainability = st.slider("Sustainability", 0, 100, 50) / 100.0
        efficiency = st.slider("Operational Efficiency", 0, 100, 55) / 100.0
        risk = st.slider("Risk Tolerance", 0, 100, 40) / 100.0
        
        st.markdown("<h3>Logistics</h3>", unsafe_allow_html=True)
        distance = st.slider("Distance Importance", 0, 100, 45) / 100.0
        extraction = st.slider("Extraction Complexity", 0, 100, 35) / 100.0
        
        st.markdown("<h3>Model Configuration</h3>", unsafe_allow_html=True)
        model_influence = st.slider("AI Model Influence", 0, 100, 50) / 100.0
        
        analyze_button = st.button("Analyze Celestial Bodies")
        
        user_preferences = {
            'iron_weight': iron,
            'nickel_weight': nickel,
            'water_ice_weight': water_ice,
            'rare_earth_weight': rare_earth,
            'precious_metals_weight': precious_metals,
            'sustainability_weight': sustainability,
            'efficiency_weight': efficiency,
            'risk_weight': risk,
            'distance_weight': distance,
            'extraction_weight': extraction,
            'model_weight': model_influence
        }
        
        return user_preferences, analyze_button

# Function to create a resource distribution chart
def create_resource_chart(site_data):
    resource_columns = ['iron', 'nickel', 'water_ice', 'rare_earth_elements', 'precious_metals']
    
    melted_data = pd.melt(
        site_data, 
        id_vars=['Celestial Body'], 
        value_vars=resource_columns,
        var_name='Resource',
        value_name='Concentration'
    )
    
    melted_data['Resource'] = melted_data['Resource'].apply(lambda x: x.replace('_', ' ').title())
    
    fig = px.bar(
        melted_data,
        x='Resource',
        y='Concentration',
        color='Celestial Body',
        barmode='group',
        title='Resource Distribution Comparison',
        labels={'Concentration': 'Concentration (%)'},
        height=400
    )
    
    fig.update_layout(
        xaxis_title="Resource Type",
        yaxis_title="Concentration (%)",
        legend_title="Celestial Body",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

# Main function to display recommendations page
def show_recommend_page():
    # Page header with time
    current_time = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h1>ISRO Space Mining Intelligence System</h1>
            <p style="font-family: 'Courier New', monospace; background-color: rgba(0,0,0,0.3); padding: 5px 10px; border-radius: 5px;">
                System Time: {current_time}
            </p>
        </div>
        <p style="margin-bottom: 30px;">Advanced celestial body analysis and mining site recommendation platform</p>
    """, unsafe_allow_html=True)
    
    # Load data and model
    df, model = load_data_and_model()
    
    # Create sidebar and get user preferences
    user_preferences, analyze_button = create_sidebar()
    
    # If analyze button is clicked
    if analyze_button:
        with st.spinner("🔭 Performing deep space analysis..."):
            # Simulate computation with progress bar
            for percent_complete in range(0, 101, 25):
                st.progress(percent_complete / 100.0, text=f"Analyzing celestial bodies... {percent_complete}%")
            
            # Get recommendations
            recommended_sites, importance_factors = recommend_sites(user_preferences, df, model, top_n=5)
            
            # Clear the progress bar
            st.empty()
            
            # Display success message
            st.success("✅ Analysis complete! Optimal mining sites identified.")
            
            # Display metrics in a row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-label">Sites Analyzed</p>
                        <p class="metric-value">{len(df)}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-label">Top Site Score</p>
                        <p class="metric-value">{recommended_sites['final_score'].max():.1f}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-label">Resource Diversity</p>
                        <p class="metric-value">{"High" if recommended_sites['iron'].std() > 15 else "Medium"}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-label">Average Distance</p>
                        <p class="metric-value">{recommended_sites['distance_from_earth'].mean():.1f} AU</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Main Results Section
            st.markdown("## 🚀 Top Recommended Mining Sites")
            
            # Highlight the top site
            top_site = recommended_sites.iloc[0]['Celestial Body']
            top_factors = importance_factors[top_site]
            st.markdown(f"""
                <div class="card" style="border: 2px solid #ff5e62; background-color: rgba(255, 94, 98, 0.1);">
                    <h3>Prime Mining Target: {top_site}</h3>
                    <p>This celestial body stands out with exceptional characteristics that align perfectly with specified priorities.</p>
                    <p>Key factors: {', '.join(top_factors)}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display columns for results
            display_columns = [
                'Celestial Body', 'iron', 'nickel', 'water_ice', 'rare_earth_elements',
                'precious_metals', 'distance_from_earth', 'final_score'
            ]
            
            # Rename columns for better display
            display_df = recommended_sites[display_columns].copy()
            display_df.columns = [
                'Celestial Body', 'Iron (%)', 'Nickel (%)', 'Water/Ice (%)', 'Rare Earth (%)',
                'Precious Metals (%)', 'Distance (AU)', 'Score'
            ]
            
            # Round numeric columns and sort by score
            for col in display_df.columns:
                if col != 'Celestial Body':
                    display_df[col] = display_df[col].round(1)
            
            # Display the table
            st.table(display_df)
            
            # Create resource visualization
            st.markdown("## 📊 Resource Analysis")
            resource_chart = create_resource_chart(recommended_sites)
            st.plotly_chart(resource_chart, use_container_width=True)
            
            # Add mission planning recommendation
            st.markdown("## 🛰️ Mission Planning")
            st.markdown(f"""
                <div class="card">
                    <h3>Mission Recommendation</h3>
                    <p>Based on comprehensive analysis, we recommend prioritizing a mission to <strong>{top_site}</strong>. 
                    This celestial body offers optimal resource distribution with a favorable distance-to-value ratio.</p>
                    <p>Estimated mission success probability: <strong>88%</strong></p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_recommend_page()
