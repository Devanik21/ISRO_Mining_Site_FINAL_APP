import streamlit as st
import google.generativeai as genai
import json

from streamlit_lottie import st_lottie
import requests

# Configure the page
st.set_page_config(
    page_title="AstroMine AI",

    page_icon="🪐",

    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Configuration
with st.sidebar:
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")

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

    st.divider()

                            # Function to load Lottie animation
def load_lottie_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)

                            # Load and display Lottie animation
animation_path = "src/AI.json"
try:
    animation = load_lottie_file(animation_path)
    st_lottie(animation, speed=1, loop=True, quality="high", height=150, key="animation")
except Exception as e:
    st.error(f"Error loading animation: {e}")

st.markdown(

    """
    <div style="text-align: center; margin-top: 50px;">
        <h1><strong>AstroMine AI</strong></h1>

    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

                                        # Chatbot Interaction
user_input = st.text_area("Ask a question about space mining:")

if st.button("Send"):
    if not api_key:
        st.warning("⚠️ Please enter a valid Google Gemini API key in the sidebar.")
    elif not user_input.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(user_input)
            st.write("Response:")
            st.success(response.text)

        except Exception as e:
            
            st.error(f"Error: {e}")
