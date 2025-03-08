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

# Initialize session state for user input if not exists
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

# Alternative function to load Lottie from local file
def load_lottie_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading animation file: {e}")
        return None

# Load animation from URL (public Lottie animations)
lottie_url = "https://assets5.lottiefiles.com/packages/lf20_XZ3pkn.json"  # Space/AI related animation
animation = load_lottie_url(lottie_url)

# If URL loading fails, try local file or show error
if not animation:
    try:
        animation = load_lottie_file("src/AI.json")
    except:
        st.warning("Could not load animation. Continuing without it.")

# Display animation if available
if animation:
    st_lottie(animation, speed=1, loop=True, quality="high", height=250, key="animation")

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

# Use session state to maintain the user input
user_input = st.text_area("💬 Ask a question about space mining:", value=st.session_state.user_input)

if st.button("🚀 Send Query"):
    if not api_key:
        st.warning("⚠️ Please enter a valid Google Gemini API Key in the sidebar.")
    elif not user_input.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        try:
            # Display a loading message
            with st.spinner("🔄 Processing your query..."):
                # Import Gemini library only if API key is provided
                import google.generativeai as genai
                
                # Configure the API
                genai.configure(api_key=api_key)
                
                # Check available models and use the correct endpoint
                try:
                    model = genai.GenerativeModel("gemini-pro")
                    response = model.generate_content(user_input)
                    
                    st.success("✨ AI Response:")
                    st.markdown(
                        f"<p style='background: linear-gradient(to right, #8e44ad, #c0392b); padding:12px; border-radius:12px; color:white; font-weight:bold;'>{response.text}</p>", 
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error with Gemini API: {e}")
                    st.info("The model 'gemini-pro' might not be available. Please check your API key and the available models.")
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure you have installed the required packages: pip install google-generativeai streamlit-lottie")
