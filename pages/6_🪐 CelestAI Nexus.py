import streamlit as st
import google.generativeai as genai
import json
from streamlit_lottie import st_lottie
import requests

# Configure the page
st.set_page_config(
    page_title=" CelestAI Nexus",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a rare, futuristic, vibrant look
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
            color: #ffffff;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(to bottom, #ff512f, #dd2476);
            padding: 20px;
            border-radius: 15px;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #6a0572;
            color: white;
            border-radius: 10px;
            padding: 8px;
            font-weight: bold;
        }
        .stTextArea>div>textarea {
            background: #ff9966;
            color: black;
            border-radius: 15px;
            padding: 12px;
            font-weight: bold;
        }
        .stButton>button {
            background: linear-gradient(to right, #12c2e9, #c471ed, #f64f59);
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            box-shadow: 0px 5px 15px rgba(255, 255, 255, 0.3);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🚀 Welcome to  CelestAI Nexus")
    api_key = st.text_input("🔑 Enter Google Gemini API Key:", type="password")
    if api_key:
        genai.configure(api_key=api_key)
    
    st.markdown("#### 💡 Example Prompts:")
    example_prompts = [
        "What is space mining?",
        "How can AI help in space exploration?",
        "What are the key minerals found on asteroids?",
        "How does Galactic Mining Hub work?",
    ]
    for prompt in example_prompts:
        if st.button(prompt):
            st.session_state.user_input = prompt

    st.markdown("---")

# Function to load Lottie animation
def load_lottie_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)

# Load and display Lottie animation
animation_path = "src/AI.json"
try:
    animation = load_lottie_file(animation_path)
    st_lottie(animation, speed=1, loop=True, quality="high", height=250, key="animation")
except Exception as e:
    st.error(f"Error loading animation: {e}")

st.markdown(
    """
    <div style="text-align: center; margin-top: 50px;">
        <h1 style="color: #ffdd00; text-shadow: 2px 2px 8px rgba(255, 255, 255, 0.5);"><strong>🛰️ CelestAI Nexus 🌌</strong></h1>
        <p style="color: #ff9a8b; font-size: 20px; font-weight: bold;">Your AI companion for futuristic space exploration and asteroid mining insights! 🔥</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Chatbot Interaction
user_input = st.text_area("💬 Ask a question about space mining:")
if st.button("🚀 Send Query"):
    if not api_key:
        st.warning("⚠️ Please enter a valid Google Gemini API key in the sidebar.")
    elif not user_input.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(user_input)
            st.success("✨ AI Response:")
            st.markdown(f"<p style='background: linear-gradient(to right, #8e44ad, #c0392b); padding:12px; border-radius:12px; color:white; font-weight:bold;'>{response.text}</p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")
