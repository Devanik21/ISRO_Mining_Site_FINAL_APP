import streamlit as st

st.set_page_config(
    page_title="Galactic Mining Hub",
    page_icon="🌌",
)

with st.sidebar:
    st.success("Choose a model to explore its features!")
    st.title("🪐 Galactic Mining Hub")
    st.subheader("Exploring Cosmic Treasures, diving deep into the cosmic sea.")
    st.markdown(
        """
        Galactic Mining Hub is an advanced ML-based platform designed to
        analyze mining site data and offer personalized recommendations
        based on your inputs.
        """
    )
    with st.expander("Project Overview"):
        st.markdown(
            """
            ## Galactic Mining Hub

            This project, **Galactic Mining Hub**, is designed to showcase the capabilities of Machine Learning in space mining exploration. 
            It was developed to create an advanced web platform for analyzing and recommending mining sites based on user preferences.

            Developed by: **Devanik**
            """
        )
    st.markdown(
        """
        ### 🚀 Prediction Model
        - Predict potential mining sites based on various features.
        - Generated insights and visualizations from `1_🚀_Predict.py`.

        ### ✨ Recommendation Model
        - Evaluate mining sites using custom feature weights and a trained ML model.
        - Processed recommendations from `2_✨_Recommend.py`.

        ### 📊 Analysis
        - In-depth analysis and visualizations from `Analyze.py`.

        ### 📚 About
        - Information and details from `about.py`.

        ### 🔍 Insights
        - Detailed insights and additional information from `insights.py`.

        ### 📈 Visualizations
        - Advanced visualizations and charts from `visualize.py`.
        """
    )

st.markdown(
    """
    <div align="center">

    ## 🛰️ Galactic Mining Hub

    Explore, Analyze, and Recommend Cosmic Mining Sites.

    </div>
    """,
    unsafe_allow_html=True
)
st.divider()
st.markdown("""
    Galactic Mining Hub is a state-of-the-art Machine Learning platform 
    for exploring and analyzing data on mining sites and generating 
    tailored recommendations based on your preferences. Dive into 
    Machine Learning and Data Science projects.

    **🛩️ Select a model from the sidebar** to see its capabilities!
    ### 🚀 Prediction Model
    - Provides insights into potential mining sites based on various features.
    - Generates predictions about site suitability.
    - Visualizes key metrics and predictions.
    - Offers an interactive experience for exploring predictions.

    ### 🌠 Recommendation Model
    - Evaluates mining sites based on custom feature weights and a trained 
    ML model.
    - Normalizes input data and predicts suitability scores.
    - Adjusts recommendations according to user preferences.
    - Ranks sites to highlight the top recommendations based on input criteria.
    """,
)
