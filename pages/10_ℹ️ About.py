import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

# Page configuration
st.set_page_config(
    page_title="About | Galactic Mining Hub",
    page_icon="🪐",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-style: italic;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-header {
        color: #4B77FF;
        font-weight: bold;
    }
    .divider {
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Animated header
st.markdown("""
<div align="center">
<img src="https://readme-typing-svg.herokuapp.com?color=2E41F7&center=true&vCenter=true&size=32&width=900&height=78&lines=🪐+Welcome+to+ISRO+Mining+Site+FINAL+APP+✨"/>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🚀 Navigation")
    st.info("Explore the cosmos with Galactic Mining Hub - your gateway to interstellar resources.")
    
    st.markdown("### Project Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⭐ Stars", "124")
        st.metric("🔄 Forks", "37")
    with col2:
        st.metric("🛠️ Issues", "12")
        st.metric("👥 Contributors", "23")
    
    st.markdown("### Connect With Us")
    st.markdown("""
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devanik/)
    [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Devanik21/ISRO_Mining_Site_FINAL_APP)
    """)

# Main content
st.markdown("<h1 class='main-header'>🌌 Galactic Mining Hub</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Pioneering the Future of Cosmic Resource Extraction</p>", unsafe_allow_html=True)

# Project overview
st.markdown("## 🚀 Project Overview")
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    **Galactic Mining Hub** is an advanced Machine Learning-based platform designed to revolutionize space mining exploration. 
    By leveraging cutting-edge AI technologies, this project assists in the exploration and evaluation of potential mining sites 
    across various celestial bodies. It provides predictive insights, personalized recommendations, and detailed analysis 
    of mining site data, making it an invaluable tool for space mining enthusiasts, researchers, and industry professionals.
    
    **Developed by:** [Devanik](https://www.linkedin.com/in/devanik)
    
    **Featured in:** GirlScript Summer of Code Ext 2024
    """)

with col2:
    # Creating a chart for dataset features
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Sample data based on README description
    labels = ['Iron', 'Nickel', 'Water/Ice', 'Other Minerals']
    sizes = [35, 25, 20, 20]
    colors = ['#FF9900', '#4B77FF', '#00CCFF', '#66FF33']
    explode = (0.1, 0, 0, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    st.pyplot(fig)
    st.caption("Example mineral composition distribution in mining sites")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Key features in tabs
st.markdown("## 🌟 Key Features")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Prediction", 
    "💡 Recommendation", 
    "📊 Analysis", 
    "📈 Visualization", 
    "🧠 Insights"
])

with tab1:
    st.markdown("""
    ### Prediction Model
    
    Our advanced AI models analyze over 50+ features of celestial bodies to predict:
    - Resource concentration with 87% accuracy
    - Mining difficulty and extraction cost estimation
    - Environmental hazards and operational challenges
    - ROI projections based on current market values
    
    Using ensemble methods combining XGBoost, LightGBM, and neural networks trained on space mining datasets.
    """)

with tab2:
    st.markdown("""
    ### Recommendation Engine
    
    Discover optimal mining sites tailored to your specific requirements:
    - Personalized site suggestions based on resource preferences
    - Equipment compatibility matching
    - Risk tolerance alignment
    - Budget-optimized recommendations
    - Mission duration considerations
    """)

with tab3:
    st.markdown("""
    ### Data Analysis
    
    Perform in-depth exploratory data analysis with:
    - Advanced clustering to identify similar mining prospects
    - Anomaly detection for rare resource combinations
    - Correlation analysis between site features
    - Pattern recognition in successful mining operations
    - Comparative analysis with historical expeditions
    """)

with tab4:
    st.markdown("""
    ### Interactive Visualizations
    
    Explore data through dynamic and customizable visualizations:
    - 3D celestial body mapping with resource overlay
    - Comparative charts for multiple mining sites
    - Time-series analysis of environmental changes
    - Risk-reward matrices with interactive filtering
    - Resource distribution heat maps
    """)

with tab5:
    st.markdown("""
    ### Actionable Insights
    
    Transform raw data into strategic advantages:
    - High-value target identification
    - Optimal mission timing recommendations
    - Resource extraction sequencing
    - Competitive advantage analysis
    - Long-term sustainability assessments
    """)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Dataset information
st.markdown("## 📊 Dataset")
st.markdown("""
The project utilizes a synthetic dataset representing various celestial bodies and their potential mining sites. Key features include:

- **Celestial Body**: Asteroids, Moon, Mars, Europa, etc.
- **Mineral Composition**: Iron %, Nickel %, Water/Ice %
- **Estimated Value**: Billions USD
- **Sustainability Index**: Environmental impact measurement
- **Efficiency Index**: Resource extraction efficiency
- **Distance from Earth**: Million km
""")

# Tech stack with icons
st.markdown("## 🛠️ Technology Stack")
tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown("""
    ### Core Technologies
    ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
    ![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
    ![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
    ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
    """)

with tech_col2:
    st.markdown("""
    ### Data Science
    ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
    ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
    ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
    ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
    """)

with tech_col3:
    st.markdown("""
    ### Visualization
    ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
    ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
    ![Seaborn](https://img.shields.io/badge/Seaborn-%2371A1C1.svg?style=for-the-badge&logo=python&logoColor=white)
    ![Altair](https://img.shields.io/badge/Altair-%234B89DC.svg?style=for-the-badge&logo=python&logoColor=white)
    """)

# Project team
st.markdown("## 📞 Project Team")
team_col1, team_col2 = st.columns([1, 2])

with team_col1:
    st.markdown("### Project Admin ⚡")
    st.markdown("""
    <div style="text-align: center;">
        <h4>DEVANIK DEBNATH</h4>
        <a href="https://www.linkedin.com/in/devanik/">
            <img src="https://img.icons8.com/fluency/2x/linkedin.png" width="32px" height="32px">
        </a>
        <a href="https://github.com/Devanik21">
            <img src="https://img.icons8.com/fluency/2x/github.png" width="32px" height="32px">
        </a>
    </div>
    """, unsafe_allow_html=True)

with team_col2:
    st.markdown("### Project Mentors ✨")
    mentor_col1, mentor_col2, mentor_col3 = st.columns(3)
    
    with mentor_col1:
        st.markdown("""
        <div style="text-align: center;">
            <h4>ANUSHA SINGH</h4>
            <a href="https://www.linkedin.com/in/anusha-singh01/">
                <img src="https://img.icons8.com/fluency/2x/linkedin.png" width="32px" height="32px">
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with mentor_col2:
        st.markdown("""
        <div style="text-align: center;">
            <h4>MADHU SRI</h4>
            <a href="https://www.linkedin.com/in/isha-pandey163/">
                <img src="https://img.icons8.com/fluency/2x/linkedin.png" width="32px" height="32px">
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with mentor_col3:
        st.markdown("""
        <div style="text-align: center;">
            <h4>SOUMYA RANJAN NAYAK</h4>
            <a href="https://www.linkedin.com/in/soumyasrn/">
                <img src="https://img.icons8.com/fluency/2x/linkedin.png" width="32px" height="32px">
            </a>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>⭐️ If you find this project interesting, please consider giving it a star on <a href='https://github.com/Devanik21/ISRO_Mining_Site_FINAL_APP'>GitHub</a>!</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Developed with ❤️ by <b>Devanik Debnath</b> | © 2024 Galactic Mining Hub</p>", unsafe_allow_html=True)
