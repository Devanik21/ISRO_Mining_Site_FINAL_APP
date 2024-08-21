import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mining Site Visualization", page_icon="🔍")

def load_data():
    # Load the dataset
    data = pd.read_csv("space_mining_dataset.csv")
    return data

def show_visualize_page():
    st.title("Mining Site Visualization")
    st.write("Explore different visualizations to understand the dataset and the impact of user preferences.")

    data = load_data()

    # Visualization 1: Distribution of Features
    st.subheader("Distribution of Features")
    feature = st.selectbox("Select Feature to Visualize", data.columns[1:])  # Exclude non-numeric columns if necessary
    fig, ax = plt.subplots()
    sns.histplot(data[feature], bins=20, kde=True, ax=ax)
    ax.set_xlabel(feature)
    st.pyplot(fig)

    # Visualization 2: Pairplot of Selected Features
    st.subheader("Pairplot of Selected Features")
    features = st.multiselect("Select Features for Pairplot", data.columns[1:])  # Exclude non-numeric columns if necessary
    if len(features) > 1:
        fig, ax = plt.subplots()
        sns.pairplot(data[features + ['final_score']], diag_kind='kde', hue='final_score')
        st.pyplot(fig)
    else:
        st.write("Please select more than one feature.")

    # Visualization 3: Impact of Weights on Recommendations
    st.subheader("Impact of Weights on Recommendations")
    st.write("Adjust the weights to see how the recommendations change.")
    
    iron_weight = st.slider("Iron Weight", 0.0, 1.0, 0.3)
    nickel_weight = st.slider("Nickel Weight", 0.0, 1.0, 0.2)
    water_ice_weight = st.slider("Water Ice Weight", 0.0, 1.0, 0.2)
    other_minerals_weight = st.slider("Other Minerals Weight", 0.0, 1.0, 0.1)
    sustainability_weight = st.slider("Sustainability Weight", 0.0, 1.0, 0.1)
    distance_weight = st.slider("Distance Weight", -1.0, 0.0, -0.1)

    # Calculate and display adjusted scores
    adjusted_scores = data.copy()
    adjusted_scores['adjusted_score'] = (
        iron_weight * adjusted_scores['iron'] +
        nickel_weight * adjusted_scores['nickel'] +
        water_ice_weight * adjusted_scores['water_ice'] +
        other_minerals_weight * adjusted_scores['other_minerals'] +
        sustainability_weight * adjusted_scores['sustainability_index'] +
        distance_weight * adjusted_scores['distance_from_earth']
    )

    # Display top N sites based on adjusted scores
    top_n = st.slider("Number of Top Sites to Display", 1, 10, 5)
    top_sites = adjusted_scores.sort_values(by='adjusted_score', ascending=False).head(top_n)
    st.subheader(f"Top {top_n} Sites Based on Adjusted Scores")
    st.write(top_sites[['Celestial Body', 'iron', 'nickel', 'water_ice', 'distance_from_earth', 'adjusted_score']])

show_visualize_page()
