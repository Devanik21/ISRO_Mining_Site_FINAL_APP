# STEPS TO `DEPLOY` A ML MODEL IN `Streamlit`

# 1. Create app.py or any name.py file in a folder, containing the following below code as an example w.r.t to a dataset :


Then `run` the file as `python` script

`CODE`
................................................................................................
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("C:\\imp\\1. ISRO\\RF_mining_model.pkl")

# Title of the web app
st.title("Mining Site Prediction")

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
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

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input Features')
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)


# Display the prediction result
st.subheader('Prediction Result')

# Customize the prediction message
if prediction[0] == 1:
    st.success("✅ This is a **Potential Mining Site**.")
else:
    st.error("❌ This is **Not a Potential Mining Site**.")

# Optionally, you can add more details or a description below the result
st.markdown("""
<div style="margin-top: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
    <strong>Note:</strong> The prediction is based on the model's analysis of key features such as distance from Earth, mineral composition,estimated value(B USD) and sustainability indices.
</div>
""", unsafe_allow_html=True)


................................................................................................................................

# 2. Now, open cmd and give the following command

# cd 'path/to the folder/ where ur app.py is present'

AND `Not` the path of ur app.py `file`

# Run the command `streamlit run app.py `

'CODE'

C:\Users\debna>`cd "C:\imp\1. ISRO"`

C:\imp\1. ISRO>`streamlit run app.py`

You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.118.96:8501


# Now, lets deploy it permanently as `app`



1. create a repository in github


2.Add requirements.txt in First Deployment folder containing all libraries required

3.Add all files of First Deployment folder and not the whole folder at once.


4.Easy, now create new app in Streamlit cloud and deploy it.