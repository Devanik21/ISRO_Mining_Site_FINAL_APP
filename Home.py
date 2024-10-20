import streamlit as st
import pandas as pd
import time
import plotly.express as px

# Configure the page with a modern theme and a custom icon
st.set_page_config(
    page_title="Galactic Mining Hub",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Sidebar content with advanced layout

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0F0F0F;
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(45deg, #141E30, #243B55);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Sidebar content with user customization options
with st.sidebar:
    mining_importance = st.selectbox(
        "🔧 **Select Mining Site Importance Level**",
        options=["Low", "Medium", "High", "Critical"]
    )
    distance_filter = st.selectbox(
        "🌐 **Filter Sites by Distance**",
        options=["< 100 light years", "100-500 light years", "500-1000 light years", "> 1000 light years"]
    )
    outlier_sensitivity = st.slider(
        "🔍 **Adjust Sensitivity for Outlier Detection**",
        min_value=0, max_value=100, value=50
    )
    mining_site_types = st.multiselect(
        "🏔️ **Select Mining Site Types**",
        options=["Asteroid", "Moon", "Planet", "Comet"],
        default=["Asteroid", "Planet"]
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

tab1, tab2, tab3 = st.tabs(["Overview", "Prediction Model", "Visualizations"])

with tab1:
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
    st.markdown(
        """
        **Welcome to the Galactic Mining Hub**, where Machine Learning meets space exploration!
        """
    )

with tab2:
    st.header("🚀 Prediction Model")
    st.markdown("Engage with predictive analysis to identify the best mining sites.")
    if st.button("Run Prediction Model"):
        with st.spinner("Running the model, please wait..."):
            time.sleep(2)  # Simulate a long-running task
            st.success("Model prediction completed!")

    # Sample results for demonstration purposes
    results_df = pd.DataFrame({
        'Mining Site': ['Site A', 'Site B', 'Site C'],
        'Predicted Value': [0.85, 0.78, 0.92]
    })
    st.write(results_df)

    # Button to download the DataFrame as CSV
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name='mining_site_predictions.csv',
        mime='text/csv'
    )

with tab3:
    st.header("📈 Cutting-Edge Visualizations")
    st.markdown("Explore the data through a range of interactive visualizations.")

    # Sample data for visualization
    data = pd.DataFrame({
        'Distance (light years)': [50, 150, 300, 750, 1200],
        'Mining Importance': ['Low', 'Medium', 'High', 'Critical', 'Medium'],
        'Feasibility Score': [0.2, 0.6, 0.8, 0.95, 0.5]
    })

    # Interactive scatter plot using Plotly
    fig = px.scatter(
        data,
        x='Distance (light years)',
        y='Feasibility Score',
        color='Mining Importance',
        title="Mining Site Feasibility Analysis",
        labels={"Feasibility Score": "Feasibility", "Distance (light years)": "Distance"}
    )
    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("**Ready to embark on your cosmic journey?** Use the sidebar to navigate through the hub’s capabilities and start your exploration!")
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
