import streamlit as st

# Configure the page with a modern theme and a custom icon
st.set_page_config(
    page_title="Galactic Mining Hub",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)
    # Adding selectboxes for user customization

# Sidebar content with advanced layout

with st.sidebar:
    mining_importance = st.selectbox(
        "🔧 **Select Mining Site Importance Level**",
        options=["Low", "Medium", "High", "Critical"]
    )
    distance_filter = st.selectbox(
        "🌐 **Filter Sites by Distance**",
        options=["< 100 light years", "100-500 light years", "500-1000 light years", "> 1000 light years"]
    )
    outlier_sensitivity = st.selectbox(
        "🔍 **Adjust Sensitivity for Outlier Detection**",
        options=["Low", "Medium", "High"]
    )
    st.title("🪐 **Galactic Mining Hub**")
    st.subheader("Deep dive into the infinite cosmic sea!")
    st.markdown(
        """
        **Galactic Mining Hub** is a cutting-edge platform that leverages advanced
        Machine Learning and Data Science techniques to revolutionize space mining 
        exploration. Dive deep into the cosmos to discover valuable mining sites
        across the galaxy.
        """
    )
    st.image("space_mining.png", use_column_width=True)

    with st.expander("🌟 **Project Overview**"):
        st.markdown(
            """
            ## 🚀 **Galactic Mining Hub**

            This initiative is at the forefront of space exploration, aiming to identify 
            and evaluate mining sites on distant celestial bodies using sophisticated 
            AI algorithms. Developed to push the boundaries of what's possible in 
            extraterrestrial resource extraction.

            **Developer:** [Devanik](https://www.linkedin.com/in/devanik/)
            """
        )
    st.markdown(
        """
        ## **Navigate the Hub**
        - **🚀 Prediction Model:** Classify mining sites based on their potential.
        - **✨ Recommendation Model:** Generate top mining site recommendations.
        - **📊 Analysis:** Perform in-depth data analysis and clustering.
        - **🔍 Insights:** Obtain actionable insights for decision-making.
        - **📈 Visualizations:** Explore data through advanced visual tools.
        """
    )



# Main content layout with a centered introduction and styled text
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px;">
        <h1>🛰️ <strong>Galactic Mining Hub</strong></h1>
        <h2><em>Explore, Analyze, and Discover Cosmic Mining Sites with Advanced AI</em></h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# Information and interactive section
st.markdown(
    f"""
    **Welcome to Galactic Mining Hub**, a premier platform that combines the power of 
    **Machine Learning** and **Data Science** to unlock the secrets of the universe. 
    Our hub provides a comprehensive toolkit for space mining analysis, from predictive 
    modeling to in-depth data insights, designed to support informed decision-making 
    in the field of space exploration.

    ### **🚀 Prediction Model**
    - **Predictive Analysis:** Identify potential mining sites based on critical features.
    - **Insightful Visualizations:** Gain insights into the suitability of sites for mining operations.
    - **Interactive Experience:** Engage with predictions to explore potential outcomes.

    ### **✨ Recommendation Model**
    - **Custom Recommendations:** Tailor mining site evaluations with custom feature weighting.
    - **Data-Driven Insights:** Utilize a trained ML model to score and rank sites.
    - **Optimized Selection:** Highlight top recommendations aligned with user-defined criteria.

    ### **📊 Advanced Analysis**
    - **Deep Data Exploration:** Use advanced clustering and outlier detection to understand data patterns.
    - **Multi-Dimensional Visualization:** Leverage techniques like PCA to uncover hidden trends.

    ### **🔍 Actionable Insights**
    - **Strategic Recommendations:** Get actionable advice based on comprehensive data analysis.
    - **Sustainability & Efficiency:** Focus on mining sites with optimal sustainability and efficiency indices.

    ### **📈 Cutting-Edge Visualizations**
    - **Interactive Charts:** Explore data through a wide range of visual tools, including heatmaps, scatter plots, and more.
    - **Dynamic Analysis:** Visualize correlations, distributions, and trends with real-time updates based on user input.

    ---
    **Ready to embark on your cosmic journey?** Use the sidebar to navigate through the hub’s capabilities and start your exploration!
    """
)
