import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration with wider layout
st.set_page_config(page_title="Mining Site Visualization", page_icon="🔍", layout="wide")

@st.cache_data
def load_data():
    # Load and cache the dataset for faster performance
    data = pd.read_csv("space_mining_dataset.csv")
    return data

def show_visualize_page():
    st.title("🚀 Mining Site Visualization")
    st.write("Explore different visualizations to understand the dataset and the impact of user preferences.")

    # Loading data with a progress bar
    with st.spinner("Loading dataset..."):
        data = load_data()

    # Add expander for dataset preview
    with st.expander("Preview Dataset"):
        st.write(data.head())

    # Add a sidebar for feature selection and additional options
    st.sidebar.header("Feature Settings")
    
    # Visualization 1: Distribution of Features with dynamic bins option
    st.subheader("Distribution of Features")
    st.write("Visualize the distribution of different features with a histogram and density plot.")
    
    feature = st.sidebar.selectbox("Select Feature to Visualize", data.columns[1:], help="Choose a feature to visualize.")
    bins = st.sidebar.slider("Number of Bins", 10, 50, 20, help="Set the number of bins for the histogram.")
    
    # Displaying the histogram and density plot
    fig, ax = plt.subplots()
    sns.histplot(data[feature], bins=bins, kde=True, ax=ax, color="skyblue")
    ax.set_xlabel(feature)
    st.pyplot(fig)

    # Visualization 2: Pairplot of Selected Features with hue options
    st.subheader("Pairplot of Selected Features")
    features = st.multiselect("Select Features for Pairplot", data.columns[1:], help="Select two or more features to visualize relationships.")
    
    if len(features) > 1:
        hue_feature = st.selectbox("Select Feature for Color Encoding", ['final_score'] + features, help="Select a feature to color the pairplot.")
        fig = sns.pairplot(data[features + ['final_score']], diag_kind='kde', hue=hue_feature, palette='coolwarm')
        st.pyplot(fig)
    else:
        st.warning("Please select more than one feature for pairplot.")

    # Visualization 3: Impact of Weights on Recommendations
    st.subheader("Impact of Weights on Recommendations")
    st.write("Adjust the weights below to see how the recommendations for mining sites change based on different criteria.")

    st.sidebar.header("Weights Settings")
    
    # Dynamically adjusting sliders for weights
    iron_weight = st.sidebar.slider("Iron Weight", 0.0, 1.0, 0.3, help="Set the weight for iron.")
    nickel_weight = st.sidebar.slider("Nickel Weight", 0.0, 1.0, 0.2, help="Set the weight for nickel.")
    water_ice_weight = st.sidebar.slider("Water Ice Weight", 0.0, 1.0, 0.2, help="Set the weight for water ice.")
    other_minerals_weight = st.sidebar.slider("Other Minerals Weight", 0.0, 1.0, 0.1, help="Set the weight for other minerals.")
    sustainability_weight = st.sidebar.slider("Sustainability Weight", 0.0, 1.0, 0.1, help="Set the weight for sustainability.")
    distance_weight = st.sidebar.slider("Distance Weight", -1.0, 0.0, -0.1, help="Set the weight for distance from Earth (negative weight for minimizing distance).")

    # Display a progress bar for score calculation
    with st.spinner("Calculating adjusted scores..."):
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
    top_n = st.sidebar.slider("Number of Top Sites to Display", 1, 10, 5)
    top_sites = adjusted_scores.sort_values(by='adjusted_score', ascending=False).head(top_n)

    st.subheader(f"Top {top_n} Sites Based on Adjusted Scores")
    st.write("The table below shows the top mining sites based on the weighted scores.")
    st.dataframe(top_sites[['Celestial Body', 'iron', 'nickel', 'water_ice', 'distance_from_earth', 'adjusted_score']])

    # Add bar chart to visualize top sites by adjusted scores
    st.subheader("Bar Chart of Adjusted Scores for Top Sites")
    st.bar_chart(top_sites[['Celestial Body', 'adjusted_score']].set_index('Celestial Body'))

    # Add a download button for users to download top sites as CSV
    csv = top_sites.to_csv(index=False)
    st.download_button(label="Download Top Sites as CSV", data=csv, file_name='top_mining_sites.csv', mime='text/csv')

show_visualize_page()
