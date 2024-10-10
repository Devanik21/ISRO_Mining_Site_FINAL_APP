import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model with error handling
try:
    model = joblib.load("FINAL_mining_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop() 
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop() 

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


def adjust_weights(data, base_weights):
    adjusted_weights = base_weights.copy()

    high_iron_threshold = 70
    low_iron_threshold = 10
    high_water_ice_threshold = 60
    high_nickel_threshold = 60

    # Winsorize iron content values
    iron_values = data['Iron (%)']
    iron_winsorized = np.clip(iron_values, np.percentile(iron_values, 5), np.percentile(iron_values, 95))

   
    if iron_winsorized > high_iron_threshold:
        adjusted_weights['Iron (%)'] += 0.3 
    elif iron_winsorized < low_iron_threshold:
        adjusted_weights['Iron (%)'] -= 0.1 
        
    if data['Water Ice (%)'] > high_water_ice_threshold:
        adjusted_weights['Water Ice (%)'] += 0.15

    if data['Nickel (%)'] > high_nickel_threshold:
        adjusted_weights['Nickel (%)'] += 0.1

    # Normalize the weights so the sum is still 1
    total_weight = sum(adjusted_weights.values())
    adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

    return adjusted_weights


def reset_inputs():
    st.session_state['Distance_from_Earth'] = 100.0
    st.session_state['Iron'] = 50.0
    st.session_state['Nickel'] = 50.0
    st.session_state['Water_Ice'] = 50.0
    st.session_state['Other_Minerals'] = 50.0
    st.session_state['Estimated_Value'] = 100.0
    st.session_state['Sustainability_Index'] = 50.0
    st.session_state['Efficiency_Index'] = 50.0


if 'Distance_from_Earth' not in st.session_state:
    reset_inputs()


def show_decide_page():
    # Title and description
    st.title("Mining Site Prediction 🛰️")
    st.write("""
        **Discover the potential of your mining site!**  
        Enter the details in the sidebar to find out if your site is worth mining.
    """)

    st.sidebar.header("🔍 Input Features")

    # Sidebar sliders for user input
    Distance_from_Earth = st.sidebar.slider("Distance from Earth (M km)", 1.0, 1000.0, st.session_state['Distance_from_Earth'], key='Distance_from_Earth')
    Iron = st.sidebar.slider("Iron (%)", 0.0, 100.0, st.session_state['Iron'], key='Iron')
    Nickel = st.sidebar.slider("Nickel (%)", 0.0, 100.0, st.session_state['Nickel'], key='Nickel')
    Water_Ice = st.sidebar.slider("Water Ice (%)", 0.0, 100.0, st.session_state['Water_Ice'], key='Water_Ice')
    Other_Minerals = st.sidebar.slider("Other Minerals (%)", 0.0, 100.0, st.session_state['Other_Minerals'], key='Other_Minerals')
    Estimated_Value = st.sidebar.slider("Estimated Value (B USD)", 0.0, 500.0, st.session_state['Estimated_Value'], key='Estimated_Value')
    Sustainability_Index = st.sidebar.slider("Sustainability Index", 0.0, 100.0, st.session_state['Sustainability_Index'], key='Sustainability_Index')
    Efficiency_Index = st.sidebar.slider("Efficiency Index", 0.0, 100.0, st.session_state['Efficiency_Index'], key='Efficiency_Index')

    # Reset button
    if st.sidebar.button("🔄 Reset All"):
        reset_inputs()
        st.experimental_rerun()

    # Data preparation
    data = {
        'Distance from Earth (M km)': Distance_from_Earth,
        'Iron (%)': Iron,
        'Nickel (%)': Nickel,
        'Water Ice (%)': Water_Ice,
        'Other Minerals (%)': Other_Minerals,
        'Estimated Value (B USD)': Estimated_Value,
        'Sustainability Index': Sustainability_Index,
        'Efficiency Index': Efficiency_Index
    }

    base_weights = {
        'Iron (%)': 0.2,
        'Nickel (%)': 0.2,
        'Water Ice (%)': 0.2,
        'Other Minerals (%)': 0.1,
        'Estimated Value (B USD)': 0.1,
        'Sustainability Index': 0.1,
        'Efficiency Index': 0.1
    }

    adjusted_weights = adjust_weights(data, base_weights)

    # Prediction button
    if st.button("🔮 Predict"):
        features = pd.DataFrame(data, index=[0])
        st.subheader('🔍 User Input Features')
        st.table(features)

        # Perform prediction with error handling
        try:
            with st.spinner("Predicting..."):
                prediction = model.predict(features)
            st.subheader('📊 Prediction Result')

            if prediction[0] == 1:
                st.success("✅ **This is a Potential Mining Site!**")
            else:
                st.error("❌ **This is Not a Potential Mining Site.**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # Additional info note
    st.markdown("""
        <div style="margin-top: 20px; padding: 15px; border: 2px solid #ccc; border-radius: 10px;">
            <strong>Note:</strong> The prediction is based on the model's analysis of key features such as distance from Earth, 
            mineral composition, estimated value, and sustainability indices. Use this as a guide, but further analysis may be needed.
        </div>
    """, unsafe_allow_html=True)


show_decide_page()
