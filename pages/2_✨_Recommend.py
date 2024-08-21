import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import streamlit as st

st.set_page_config(page_title="Mining Site Recommendation", page_icon="🌌")

def recommend_site(user_preferences, top_n=5):
    model = joblib.load("space_mining_model.pkl")
    df = pd.read_csv("space_mining_dataset.csv")
    features = df[['iron', 'nickel', 'water_ice', 'other_minerals', 'sustainability_index', 'efficiency_index', 'distance_from_earth']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Adjust the weights based on user input
    adjusted_weights = {
        'iron': user_preferences.get('iron_weight', 0.3),
        'nickel': user_preferences.get('nickel_weight', 0.2),
        'water_ice': user_preferences.get('water_ice_weight', 0.2),
        'other_minerals': user_preferences.get('other_minerals_weight', 0.1),
        'sustainability_index': user_preferences.get('sustainability_weight', 0.1),
        'efficiency_index': user_preferences.get('sustainability_weight', 0.1),
        'distance_from_earth': -user_preferences.get('distance_weight', 0.1)
    }

    # Recalculate the composite score based on user preferences
    df['adjusted_score'] = (
        adjusted_weights['iron'] * df['iron'] +
        adjusted_weights['nickel'] * df['nickel'] +
        adjusted_weights['water_ice'] * df['water_ice'] +
        adjusted_weights['other_minerals'] * df['other_minerals'] +
        adjusted_weights['efficiency_index'] * df['efficiency_index'] +
        adjusted_weights['sustainability_index'] * df['sustainability_index'] +
        adjusted_weights['distance_from_earth'] * df['distance_from_earth']
    )

    predicted_scores = model.predict(features_scaled)
    df['final_score'] = predicted_scores + df['adjusted_score']

    # Sort the DataFrame by the final score in descending order and select the top N sites
    ranked_sites = df.sort_values(by='final_score', ascending=False).head(top_n)
    
    # Select specific columns to display
    columns_to_display = ['Celestial Body', 'iron', 'nickel', 'water_ice', 'distance_from_earth', 'final_score']
    ranked_sites_df = ranked_sites[columns_to_display]

    return ranked_sites_df

def show_recommend_page():
    st.title("🚀 Mining Site Recommendation")
    st.write("Set your preferences in the sidebar, and our model will recommend the most suitable mining sites for your needs!")

    iron = st.sidebar.slider("Iron (%)", 0, 100, 50)/100.0
    nickel = st.sidebar.slider("Nickel (%)", 0, 100, 50)/100.0
    water_ice = st.sidebar.slider("Water/Ice (%)", 0, 100, 50)/100.0
    other_minerals = st.sidebar.slider("Other Minerals (%)", 0, 100, 50)/100.0
    sustainability_efficiency = st.sidebar.slider("Sustainability/Efficiency (%)", 0, 100, 50)/100.0
    distance_from_earth = st.sidebar.slider("Distance from Earth (%)", 0, 100, 50)/100.0

    user_preferences = {
        'iron_weight': iron,
        'nickel_weight': nickel,
        'water_ice_weight': water_ice,
        'other_minerals_weight': other_minerals,
        'sustainability_weight': sustainability_efficiency,
        'distance_weight': distance_from_earth
    }
    
    ok = st.button("Recommend")
    if ok:
        with st.spinner("Scanning the Cosmos for Prime Mining Sites..."):
            recommended_site = recommend_site(user_preferences)
        st.markdown("### 🚀 Top 5 Mining Sites for Your Preferences:")
        st.table(recommended_site)
        st.markdown("<div style='text-align: center;'>✨ The search for new worlds is complete! ✨</div>", unsafe_allow_html=True)

show_recommend_page()
