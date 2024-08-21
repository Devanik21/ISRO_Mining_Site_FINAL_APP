import streamlit as st

st.set_page_config(
    page_title="👽 Cosmic Mining Hub",
    page_icon="🌌",
)

with st.sidebar:
    st.success("Explore models to begin your journey!")
    st.title("👽 Cosmic Mining Hub")
    st.subheader("Discovering the Treasures of the Universe.")
    st.markdown(
        """
        Welcome to the Cosmic Mining Hub, a cutting-edge 
        machine learning platform designed to analyze 
        extraterrestrial mining data and offer expert 
        recommendations.
        """
    )
    with st.expander("About This Project"):
        st.markdown(
            """
            ### ISRO Space Exploration Hackathon

            Cosmic Mining Hub is a submission for the 
            [ISRO Space Exploration Hackathon](https://dorahacks.io/hackathon/isro-space-exploration-hackathon/) 
            by Team PiklPython (Desh, Devanik, Shivam).
            """
        )

st.markdown(
    """
    <div style="text-align: center;">

    ## 🌌 Cosmic Mining Hub

    <em>
    Discovering the Treasures of the Universe.
    </em>

    </div>
    """,
    unsafe_allow_html=True
)
st.divider()
st.markdown(
    """
    The Cosmic Mining Hub leverages machine learning to explore, 
    analyze, and recommend prime mining sites across the cosmos.

    **🛩️ Select a model from the sidebar** to start exploring!

    ### 🛰️ Prediction Model
    - **Comprehensive Analysis:** Assesses vital factors such as distance from Earth, mineral richness, projected value (B USD), and sustainability indices to predict optimal mining sites.
    - **Advanced Algorithms:** Utilizes sophisticated models like Random Forest, XGBoost, and LightGBM for precise predictions.
    - **Interactive Input:** Adapts to user data, offering personalized mining site recommendations.
    - **Strategic Insights:** Provides actionable insights for cosmic resource exploration and extraction.

    ### 🌠 Recommendation Model
    - **Site Evaluation:** Analyzes and ranks mining sites based on user-defined criteria and trained ML models.
    - **Data Normalization:** Processes input data, predicting suitability scores aligned with user preferences.
    - **Custom Ranking:** Generates top site recommendations by prioritizing features according to user input.
    - **Dynamic Interaction:** Allows real-time adjustments to feature importance, enhancing recommendation accuracy.
    """
)
