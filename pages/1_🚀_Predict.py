import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("RF_mining_model.pkl")
st.set_page_config(page_title="Mining Site Prediction", page_icon="🚀")

def show_decide_page ():
    st.title("Mining Site Prediction")
    st.write("Check if your mining site is worth it or not by entering a few data in the sidebar")
    st.sidebar.header("Input Features")

    Distance_from_Earth = st.sidebar.slider("Distance from Earth (M km)", 1.0, 1000.0, 100.0)
    Iron = st.sidebar.slider("Iron (%)", 0.0, 100.0, 50.0)
    Nickel = st.sidebar.slider("Nickel (%)", 0.0, 100.0, 50.0)
    Water_Ice = st.sidebar.slider("Water Ice (%)", 0.0, 100.0, 50.0)
    Other_Minerals = st.sidebar.slider("Other Minerals (%)", 0.0, 100.0, 50.0)
    Estimated_Value = st.sidebar.slider("Estimated Value (B USD)", 0.0, 500.0, 100.0)
    Sustainability_Index = st.sidebar.slider("Sustainability Index", 0.0, 100.0, 0.5)
    Efficiency_Index = st.sidebar.slider("Efficiency Index", 0.0, 100.0, 0.5)

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

    ok = st.button("Predict")

    if ok:
        features = pd.DataFrame(data, index=[0])
        st.subheader('User Input Features')
        st.write(features)
        prediction = model.predict(features)
        st.subheader('Prediction Result')

        if prediction[0] == 1:
            st.success("✅ This is a **Potential Mining Site**.")
        else:
            st.error("❌ This is **Not a Potential Mining Site**.")

    st.markdown("""
<div style="margin-top: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
    <strong>Note:</strong> The prediction is based on the model's analysis of key features such as distance from Earth, mineral composition,estimated value(B USD) and sustainability indices.
</div>
""", unsafe_allow_html=True)

show_decide_page()
