import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

def show_analyze_page():
    st.title("Mining Site Analysis")
    st.write("Analyze the characteristics of different mining sites.")
    
    # Load dataset
    df = pd.read_csv("space_mining_dataset.csv")
    
    # Display dataset summary
    st.subheader("Dataset Summary")
    st.write(df.describe())
    
    # Handle non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        st.subheader("Non-Numeric Columns Encoding")
        st.write("Encoding non-numeric columns for analysis:")
        for col in non_numeric_cols:
            st.write(f"- {col}")
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)
    
    # Clustering Analysis
    st.subheader("Clustering Analysis")
    st.write("Applying K-Means clustering to group similar mining sites.")
    
    # Select the number of clusters
    num_clusters = st.slider("Number of Clusters", 2, 10, 3)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df.select_dtypes(include=['number']))
    
    st.write("### Cluster Centers")
    st.write(pd.DataFrame(kmeans.cluster_centers_, columns=df.select_dtypes(include=['number']).columns))
    
    # Visualize clusters with PCA
    st.subheader("Cluster Visualization with PCA")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.select_dtypes(include=['number']).drop(columns=['Cluster']))
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='viridis', data=df)
    st.pyplot(plt)
    
    # Feature Importance Analysis
    st.subheader("Feature Importance")
    st.write("Analyzing feature importance using RandomForest.")
    
    from sklearn.ensemble import RandomForestClassifier
    
    features = df.drop(columns=['Cluster', 'PCA1', 'PCA2'])
    target = df['Cluster']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, target)
    
    importances = pd.Series(model.feature_importances_, index=features.columns)
    importances = importances.sort_values(ascending=False)
    
    st.write(importances)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=importances.index)
    plt.title("Feature Importance")
    st.pyplot(plt)
    
    # Display the DataFrame with clusters and PCA
    st.subheader("Analyzed Data")
    st.write(df)

