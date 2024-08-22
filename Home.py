import streamlit as st
from predict import show_predict_page
from recommend import show_recommend_page
from analyze import show_analysis_page
from visualize import show_visualize_page
from insights import show_insights_page
from about import show_about_page

st.set_page_config(page_title="Cosmic Mining Hub", page_icon="💫")

# Sidebar navigation
with st.sidebar:
    selection = st.radio("Go to", ["Home", "🚀 Predict", "✨ Recommend", "Analyze", "Visualize", "Insights", "About"])

# Page content
if selection == "Home":
    st.markdown("""
        <div align="center">
        ## 💫 Cosmic Mining Hub
        <samp>
        Uncover the Secrets of the Cosmos, One Site at a Time.
        </samp>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
        Cosmic Mining Hub is a cutting-edge web platform powered by 
        Machine Learning for exploring and analyzing data on potential 
        space mining sites. It provides recommendations based on user 
        input and advanced ML models.
        **👈 Select a model from the sidebar** to see it in action!
        ### 🚀 Prediction Model
        - Provides predictions on the potential of mining sites.
        - Uses historical data and trained ML models to estimate viability.
        - Offers insights into key features affecting site suitability.
        - Helps users make informed decisions about space mining.
        ### ✨ Recommendation Model
        - Evaluates mining sites based on user-defined feature weights.
        - Normalizes input data and predicts suitability scores.
        - Adjusts scores according to user preferences and ranks sites.
        - Recommends the best mining sites based on personalized criteria.
        """)
elif selection == "🚀 Predict":
    show_predict_page()
elif selection == "✨ Recommend":
    show_recommend_page()
elif selection == "Analyze":
    show_analysis_page()
elif selection == "Visualize":
    show_visualize_page()
elif selection == "Insights":
    show_insights_page()
elif selection == "About":
    show_about_page()
