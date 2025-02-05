import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("RF_mining_model.pkl")

# Set the page configuration
st.set_page_config(
    page_title="Mining Site Prediction",
    page_icon="🛰️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #F5F5F5;
            color: #333;
        }
        .sidebar .sidebar-content {
            background-color: #EEEEEE;
            padding: 10px;
            border-radius: 10px;
        }
        .stSlider {
            color: #83e4f7;
        }
        .stButton>button {
            background-color: #042380;
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #000000;
        }
    </style>
""", unsafe_allow_html=True)

def show_decide_page():
    # Title and description
    st.title("Mining Site Prediction 🛰️")
    st.write("""
        **Discover the potential of your mining site!**  
        Enter the details in the sidebar to find out if your site is worth mining.
    """)
    
    st.sidebar.header("🔍 Input Features")

    # Sidebar sliders for user input
    distance_from_earth = st.sidebar.slider("Distance from Earth (M km)", 1.0, 1000.0, 100.0)
    iron = st.sidebar.slider("Iron (%)", 0.0, 100.0, 50.0)
    nickel = st.sidebar.slider("Nickel (%)", 0.0, 100.0, 50.0)
    water_ice = st.sidebar.slider("Water Ice (%)", 0.0, 100.0, 50.0)
    other_minerals = st.sidebar.slider("Other Minerals (%)", 0.0, 100.0, 50.0)
    estimated_value = st.sidebar.slider("Estimated Value (B USD)", 0.0, 500.0, 100.0)
    sustainability_index = st.sidebar.slider("Sustainability Index", 0.0, 100.0, 50.0)
    efficiency_index = st.sidebar.slider("Efficiency Index", 0.0, 100.0, 50.0)

    # Data preparation
    data = {
        'distance_from_earth': distance_from_earth,
        'iron': iron,
        'nickel': nickel,
        'water_ice': water_ice,
        'other_minerals': other_minerals,
        'Estimated Value (B USD)': estimated_value,
        'sustainability_index': sustainability_index,
        'efficiency_index': efficiency_index
    }

    # Prediction button
    if st.button("🔮 Predict"):
        features = pd.DataFrame(data, index=[0])
        st.subheader('🔍 User Input Features')
        st.table(features)
        
        prediction = model.predict(features)
        st.subheader('📊 Prediction Result')
        
        if prediction[0] == 1:
            st.success("✅ **This is a Potential Mining Site!**")
        else:
            st.error("❌ **This is Not a Potential Mining Site.**")

    # Additional info note
    st.markdown("""
        <div style="margin-top: 20px; padding: 15px; border: 2px solid #ccc; border-radius: 10px;">
            <strong>Note:</strong> The prediction is based on the model's analysis of key features such as distance from Earth, 
            mineral composition, estimated value, and sustainability indices. Use this as a guide, but further analysis may be needed.
        </div>
    """, unsafe_allow_html=True)

show_decide_page()
