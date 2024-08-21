import streamlit as st

st.set_page_config(
    page_title="Stellar Minesite",
    page_icon="💫",
)

with st.sidebar:
    st.success("Select a model above to try it!")
    st.title("💫 Stellar Minesite")
    st.subheader("Unlocking the Riches of the Cosmos, One Site at a Time.")
    st.markdown(
        """
        Stellar Minesite is an ML-driven web platform to 
        analyze data on mining sites, and generate 
        recommendations based on user input.
        """
    )
    with st.expander("Built for ISRO Space Exploration Hackathon"):
        st.markdown(
            """
            ## ISRO Space Exploration Hackathon

            This project is a submission for [ISRO Space Exploration Hackathon](https://dorahacks.io/hackathon/isro-space-exploration-hackathon/)
            and it was built by Team PiklPython (Desh, Devanik, Shivam).
            """
        )

st.markdown(
    """
    <div align="center">

    ## 💫 Stellar Minesite

    <samp>
    Unlocking the Riches of the Cosmos, One Site at a Time.
    </samp>

    </div>
    """,
    unsafe_allow_html=True
)
st.divider()
st.markdown("""
    Stellar Minesite is a Machine Learning driven web 
    platform for exploring analyzing data on mining sites,
    and generating recommendations based on user input. 
    Machine Learning and Data Science projects.

    **👈 Select a model from the sidebar** to see it in action!
    ### 🚀 The Prediction Model
    - brief description line
    - brief description line
    - brief description line
    - brief description line

    ### ✨ The Recommendation Model
    - The recommendation model evaluates space mining sites 
    based on user-defined feature weights and a trained 
    machine learning model.
    - It normalizes input data, predicts suitability scores, and adjusts for user preferences.
    - Finally, it ranks sites to provide the top recommendations.
    - Users interactively adjust feature weight/priority and the model recommends best minesites available.
    """, 
)
