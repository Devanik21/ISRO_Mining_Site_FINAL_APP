import streamlit as st
import json
import requests
from streamlit_lottie import st_lottie

# Configure the page
st.set_page_config(
    page_title=" CelestAI Nexus",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
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

# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🚀 Welcome to CelestAI Nexus")
    api_key = st.text_input("🔑 Enter Google Gemini API Key:", type="password")
    
    st.markdown("#### 💡 Example Prompts:")
    example_prompts = [
        "What is space mining?",
        "How can AI help in space exploration?",
        "What are the key minerals found on asteroids?",
        "How does Galactic Mining Hub work?",
    ]
    for i, prompt in enumerate(example_prompts):
        if st.button(prompt, key=f"example_{i}"):
            st.session_state.user_input = prompt
    st.markdown("---")

# Function to load Lottie animation from URL
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load animation from URL
lottie_url = "https://assets5.lottiefiles.com/packages/lf20_XZ3pkn.json"
animation = load_lottie_url(lottie_url)

# Display animation if available
if animation:
    st_lottie(animation, speed=1, loop=True, quality="high", height=250, key="animation")
else:
    try:
        # Fallback to local file
        with open("src/AI.json", "r", encoding="utf-8") as file:
            animation = json.load(file)
            st_lottie(animation, speed=1, loop=True, quality="high", height=250, key="animation")
    except:
        st.warning("Animation could not be loaded.")

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

# User input
user_input = st.text_area("💬 Ask a question about space mining:", value=st.session_state.user_input)

if st.button("🚀 Send Query"):
    if not api_key:
        st.warning("⚠️ Please enter a valid Google Gemini API Key in the sidebar.")
    elif not user_input.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        try:
            # Display loading spinner
            with st.spinner("🔄 Processing your query..."):
                # Import Gemini library
                import google.generativeai as genai
                
                # Configure the API
                genai.configure(api_key=api_key)
                
                # Use Gemini 2.0 Flash model
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(user_input)
                
                st.success("✨ AI Response:")
                st.markdown(
                    f"<p style='background: linear-gradient(to right, #8e44ad, #c0392b); padding:12px; border-radius:12px; color:white; font-weight:bold;'>{response.text}</p>", 
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("If you're having issues with the model, try these models: 'gemini-1.5-flash', 'gemini-1.5-pro', or check Google's documentation for the latest model names.")
