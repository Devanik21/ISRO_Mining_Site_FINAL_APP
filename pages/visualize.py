import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mining Site Analysis", page_icon="📊")

def load_data():
    # Load your dataset here
    # For demonstration, assuming a CSV file with the same columns
    # Replace 'data.csv' with your actual data file
    data = pd.read_csv('data.csv')
    return data

def show_analysis_page():
    st.title("Mining Site Analysis")
    st.write("Explore key insights and visualizations from the mining site dataset")

    data = load_data()

    # Display data preview
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Visualization 1: Distribution of Estimated Value
    st.subheader("Distribution of Estimated Value (B USD)")
    fig, ax = plt.subplots()
    sns.histplot(data['Estimated Value (B USD)'], bins=20, kde=True, ax=ax)
    ax.set_xlabel('Estimated Value (B USD)')
    st.pyplot(fig)

    # Visualization 2: Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots()
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Visualization 3: Feature vs. Prediction
    st.subheader("Feature vs. Prediction")
    feature = st.selectbox('Select Feature for Analysis', data.columns[:-1])  # Exclude target column
    fig, ax = plt.subplots()
    sns.boxplot(x=data['Potential Mining Site'], y=data[feature], ax=ax)
    ax.set_xlabel('Potential Mining Site')
    ax.set_ylabel(feature)
    st.pyplot(fig)

show_analysis_page()
