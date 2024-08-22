import streamlit as st

def show_about_page():
    st.title("About This Project")
    st.write("""
    ## Stellar Minesite

    **Stellar Minesite** is an advanced tool designed to assist in the exploration and evaluation of potential mining sites across different celestial bodies.

    ### Project Objectives
    - To predict and recommend potential mining sites based on key characteristics.
    - To provide in-depth analysis and visualizations of mining site data.
    - To offer actionable insights for making informed decisions about space mining opportunities.

    ### Features
    - **Prediction**: Determine whether a site is a potential mining candidate.
    - **Recommendation**: Suggest top mining sites based on user preferences.
    - **Analysis**: Explore data characteristics, detect clusters, and identify outliers.
    - **Visualization**: Create detailed charts and graphs to understand data distributions and correlations.
    - **Insights**: Provide actionable recommendations based on the data analysis.

    ### Contact
    If you have any questions or feedback, please reach out to the development team.
    """)

show_about_page()
