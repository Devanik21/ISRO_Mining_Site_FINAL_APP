import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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

    # Visualization 4: Feature Importance from Model
    st.subheader("Feature Importance from Model")
    model = joblib.load("space_mining_model.pkl")
    feature_importances = model.feature_importances_
    features = data[['iron', 'nickel', 'water_ice', 'other_minerals', 'sustainability_index', 'efficiency_index', 'distance_from_earth']].columns
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    ax.set_xlabel('Importance')
    st.pyplot(fig)

    # Visualization 5: Correlation of Features with Final Score
    st.subheader("Correlation of Features with Final Score")
    corr_matrix = data[['iron', 'nickel', 'water_ice', 'other_minerals', 'sustainability_index', 'efficiency_index', 'distance_from_earth', 'final_score']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Visualization 6: Scatter Plot of Final Score vs. Distance from Earth
    st.subheader("Scatter Plot of Final Score vs. Distance from Earth")
    fig, ax = plt.subplots()
    sns.scatterplot(x='distance_from_earth', y='final_score', data=data, ax=ax)
    ax.set_xlabel('Distance from Earth (M km)')
    ax.set_ylabel('Final Score')
    st.pyplot(fig)

    # Visualization 7: Box Plot of Final Scores by Feature Ranges
    st.subheader("Box Plot of Final Scores by Feature Ranges")
    feature_range = st.selectbox("Select Feature Range", ['iron', 'nickel', 'water_ice', 'other_minerals', 'sustainability_index', 'efficiency_index', 'distance_from_earth'])
    fig, ax = plt.subplots()
    sns.boxplot(x=feature_range, y='final_score', data=data, ax=ax)
    st.pyplot(fig)

show_visualize_page()
