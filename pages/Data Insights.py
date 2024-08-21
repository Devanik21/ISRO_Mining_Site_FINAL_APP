def show_insights_page():
    st.title("Data Insights")
    st.write("Discover key insights from the mining site dataset.")

    data = load_data()

    # Example insights
    st.subheader("Top 5 Mining Sites by Estimated Value")
    top_sites = data[['Celestial Body', 'Estimated Value (B USD)']].sort_values(by='Estimated Value (B USD)', ascending=False).head(5)
    st.table(top_sites)

    st.subheader("Average Mineral Composition")
    avg_composition = data[['Iron (%)', 'Nickel (%)', 'Water Ice (%)', 'Other Minerals (%)']].mean()
    st.write(avg_composition)

    st.subheader("Distribution of Sustainability and Efficiency Indices")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(data['Sustainability Index'], bins=20, kde=True, ax=ax[0])
    ax[0].set_title("Sustainability Index")
    sns.histplot(data['Efficiency Index'], bins=20, kde=True, ax=ax[1])
    ax[1].set_title("Efficiency Index")
    st.pyplot(fig)
