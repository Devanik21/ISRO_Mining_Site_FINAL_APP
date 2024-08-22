import streamlit as st
import pandas as pd

def show_explore_page():
    st.title("Data Exploration")
    st.write("Explore and filter the space mining dataset to understand the data better.")

    # Load dataset
    df = pd.read_csv("space_mining_dataset.csv")
    
    # Display the first few rows of the dataset
    st.write("### Dataset Overview")
    st.write(df.head())
    
    # Filter by Celestial Body
    celestial_bodies = st.multiselect("Select Celestial Bodies", options=df['Celestial Body'].unique(), default=df['Celestial Body'].unique())
    filtered_df = df[df['Celestial Body'].isin(celestial_bodies)]
    
    # Display filtered data
    st.write(f"### Filtered Data ({len(filtered_df)} rows)")
    st.write(filtered_df)
    
    # Provide summary statistics for the filtered data
    st.write("### Summary Statistics")
    st.write(filtered_df.describe())
