def show_future_trends_page():
    st.title("Future Trends")
    st.write("Explore predicted trends based on the current dataset.")

    # Example trends analysis
    data = load_data()

    st.subheader("Predicted Trends in Mineral Yields")
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x='Distance from Earth (M km)', y='Iron (%)', ax=ax, label='Iron')
    sns.lineplot(data=data, x='Distance from Earth (M km)', y='Nickel (%)', ax=ax, label='Nickel')
    sns.lineplot(data=data, x='Distance from Earth (M km)', y='Water Ice (%)', ax=ax, label='Water Ice')
    ax.set_title("Predicted Mineral Yields by Distance from Earth")
    st.pyplot(fig)
