import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the title and icon of the Streamlit page
st.set_page_config(page_title="Mining Site Visualization", page_icon="🔍")

def load_data():
    #Load the dataset from CSV and handle errors if the file is not found.
    try:
       data = pd.read_csv("space_mining_dataset.csv")
       return data
    except FileNotFoundError:
        st.error("Dataset file not found. Please upload the correct file.")
        return pd.DataFrame() 
    
def show_visualize_page():
    #Main function to show the visualization page.
    st.title("Mining Site Visualization")
    st.write("Explore different visualizations to understand the dataset and the impact of user preferences.")

    data = load_data()
    if data.empty:
        return  

    # Check available columns in the dataset
    st.write("Available Columns:", data.columns)

    required_columns = ['iron', 'nickel', 'water_ice', 'other_minerals', 'sustainability_index', 'distance_from_earth']
    if not all(col in data.columns for col in required_columns):
        st.error(f"Dataset must contain the following columns: {', '.join(required_columns)}")
        return

    # If 'final_score' does not exist, calculate it based on other features
    if 'final_score' not in data.columns:
        st.write("The 'final_score' column does not exist, calculating it based on weights.")
        
        # Assuming the columns 'iron', 'nickel', 'water_ice', 'other_minerals', 'sustainability_index', and 'distance_from_earth' exist.
        iron_weight = 0.3
        nickel_weight = 0.2
        water_ice_weight = 0.2
        other_minerals_weight = 0.1
        sustainability_weight = 0.1
        distance_weight = -0.1
        
        # Calculate the final score
        data['final_score'] = (
            iron_weight * data['iron'] +
            nickel_weight * data['nickel'] +
            water_ice_weight * data['water_ice'] +
            other_minerals_weight * data['other_minerals'] +
            sustainability_weight * data['sustainability_index'] +
            distance_weight * data['distance_from_earth']
        )
    
    # Check again if final_score is now available
    st.write("Updated Columns:", data.columns)

    # Visualization 1: Distribution of Features
    st.subheader("Distribution of Features")
    feature = st.selectbox("Select Feature to Visualize", data.columns[1:])  # Exclude non-numeric columns if necessary
    fig, ax = plt.subplots()
    
    # Use a more colorful palette for the histogram
    sns.histplot(data[feature], bins=20, kde=True, ax=ax, color='teal')
    ax.set_xlabel(feature)
    ax.set_title(f"Distribution of {feature}")
    st.pyplot(fig)

    # Visualization 2: Pairplot of Selected Features
    st.subheader("Pairplot of Selected Features")
    features = st.multiselect("Select Features for Pairplot", data.columns[1:])  # Exclude non-numeric columns if necessary
    if len(features) > 1:
        if len(features) > 4:
            st.warning("Select up to 4 features for pairplot for better performance.")   
        else:
            fig, ax = plt.subplots()
        
        # Customizing pairplot with a color palette
        spairplot_fig = sns.pairplot(data[features + ['final_score']], diag_kind='kde', hue='final_score', palette="coolwarm")
        st.pyplot(pairplot_fig.fig)
    else:
        st.write("Please select more than one feature.")

    # Visualization 3: Correlation Heatmap (with only numeric columns)
    st.subheader("Correlation Heatmap")
    # Select only numeric columns for correlation calculation
    numeric_data = data.select_dtypes(include='number')
    corr_matrix = numeric_data.corr()
    
    # Displaying the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap", fontsize=16)
    st.pyplot(fig)

    # Visualization 4: Boxplot of Feature Distribution by Category
    st.subheader("Boxplot of Feature Distribution by Final Score")
    box_feature = st.selectbox("Select Feature for Boxplot", data.columns[1:])
    
    # Create a boxplot based on the 'final_score'
    fig, ax = plt.subplots()
    sns.boxplot(x='final_score', y=box_feature, data=data, palette="Set2", ax=ax)
    ax.set_title(f"Boxplot of {box_feature} by Final Score")
    st.pyplot(fig)

    # Visualization 5: Barplot for Aggregate Feature Insights
    st.subheader("Barplot for Aggregate Insights by Celestial Body")
    aggregate_feature = st.selectbox("Select Feature for Aggregate Barplot", data.columns[1:])
    
    # Create a barplot of average feature values by celestial body
    fig, ax = plt.subplots()
    sns.barplot(x='Celestial Body', y=aggregate_feature, data=data, palette="coolwarm", ax=ax)
    ax.set_title(f"Average {aggregate_feature} by Celestial Body")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels for clarity
    st.pyplot(fig)

    # Visualization 6: Impact of Weights on Recommendations
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
    
    # Customizing the table display with color for better insight
    st.subheader(f"Top {top_n} Sites Based on Adjusted Scores")
    top_sites_display = top_sites[['Celestial Body', 'iron', 'nickel', 'water_ice', 'distance_from_earth', 'adjusted_score']]
    
    # Use a color gradient for the 'adjusted_score' column for better visual appeal
    st.write(top_sites_display.style.background_gradient(subset=['adjusted_score'], cmap='coolwarm'))

show_visualize_page()
