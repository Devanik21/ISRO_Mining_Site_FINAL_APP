import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def show_analyze_page():
    st.title("Mining Site Analysis")
    st.write("Analyze the characteristics of different mining sites with advanced data analytics.")

    # Load dataset
    df = pd.read_csv("space_mining_dataset.csv")
    
    # Display dataset summary
    st.write("### Dataset Summary")
    st.write(df.describe())
    
    # Display correlation matrix for numeric features
    st.write("### Correlation Matrix")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select numeric columns
    corr_matrix = numeric_df.corr()
    st.write(corr_matrix)

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)
    
    # Clustering analysis using KMeans
    st.write("### Clustering Analysis")
    st.write("Identifying clusters of similar mining sites using KMeans.")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_df)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    df['Cluster'] = clusters

    st.write("Cluster Centers:")
    st.write(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=numeric_df.columns))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='iron', y='nickel', hue='Cluster', data=df, palette='Set1')
    plt.title('KMeans Clustering of Mining Sites')
    st.pyplot(plt)

    # PCA for dimensionality reduction and visualization
    st.write("### Principal Component Analysis (PCA)")
    st.write("Reducing dimensionality for visualization.")

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    df['PCA1'] = pca_features[:, 0]
    df['PCA2'] = pca_features[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set1')
    plt.title('PCA of Mining Sites')
    st.pyplot(plt)

    # Outlier detection using IQR method
    st.write("### Outlier Detection")
    st.write("Detecting outliers using the IQR method.")

    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
    df['Outlier'] = outliers

    st.write(f"Number of outliers detected: {outliers.sum()}")
    st.write(df[outliers])

    # Correlation with non-numeric features using One-Hot Encoding
    st.write("### Correlation with Non-Numeric Features")
    st.write("Analyzing correlations with non-numeric features using One-Hot Encoding.")

    non_numeric_df = df.select_dtypes(exclude=['float64', 'int64'])
    encoded_df = pd.get_dummies(non_numeric_df)
    combined_df = pd.concat([numeric_df, encoded_df], axis=1)
    combined_corr_matrix = combined_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(combined_corr_matrix, annot=False, cmap='coolwarm')
    st.pyplot(plt)

    st.write("Analysis complete!")

show_analyze_page()
