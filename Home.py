import streamlit as st

def main():
    st.set_page_config(page_title="👽 Cosmic Mining Hub", page_icon="🌌")

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Prediction", "Recommendation", "Analysis", "Visualization", "Insights", "About"])

    if selection == "Home":
        st.title("Welcome to Stellar Minesite")
        st.write("Use the navigation panel on the left to explore the different features of the app.")

    elif selection == "Prediction":
        from predict import show_decide_page
        show_decide_page()

    elif selection == "Recommendation":
        from recommend import show_recommend_page
        show_recommend_page()

    elif selection == "Analysis":
        from analyze import show_analyze_page
        show_analyze_page()

    elif selection == "Visualization":
        from visualize import show_visualize_page
        show_visualize_page()

    elif selection == "Insights":
        from insights import show_insights_page
        show_insights_page()

    elif selection == "About":
        from about import show_about_page
        show_about_page()

if __name__ == "__main__":
    main()
