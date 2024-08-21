def show_model_performance_page():
    st.title("Model Performance")
    st.write("Evaluate the performance of the trained models.")

    # Example performance metrics
    st.subheader("Random Forest Model Performance")
    model = joblib.load("RF_mining_model.pkl")
    data = load_data()

    X = data.drop(columns=['Potential Mining Site'])
    y = data['Potential Mining Site']

    # Predictions
    predictions = model.predict(X)

    # Calculate metrics
    accuracy = (predictions == y).mean()
    st.write(f"Accuracy: {accuracy:.2f}")

    # Add more metrics and visualizations as needed
