import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def show_analyze_page():
    st.title("🔍 Mining Site Analysis")
    st.write("Analyze the characteristics of different mining sites with advanced data analytics.")

    # Load dataset
    df = pd.read_csv("space_mining_dataset.csv")
    
    # Display dataset summary
    st.write("## 📊 Dataset Summary")
    st.write(df.describe().style.background_gradient(cmap='coolwarm'))

    # User selects columns for correlation matrix and heatmap
    st.write("## 📈 Correlation Matrix & Heatmap")
    selected_columns = st.multiselect("Select columns for correlation matrix:", df.columns.tolist(), default=df.select_dtypes(include=['float64', 'int64']).columns.tolist())
    
    if selected_columns:
        numeric_df = df[selected_columns]
        corr_matrix = numeric_df.corr()
        st.write(corr_matrix.style.background_gradient(cmap='viridis', axis=None))

        # Correlation heatmap
        st.write("### 🔥 Correlation Heatmap")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        st.pyplot(plt)
    
    # Clustering analysis using KMeans
    st.write("## 🧩 Clustering Analysis")
    st.write("Identifying clusters of similar mining sites using KMeans.")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    df['Cluster'] = clusters

    st.write("### 🎯 Cluster Centers")
    st.write(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=df.select_dtypes(include=['float64', 'int64']).columns).style.background_gradient(cmap='coolwarm'))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='iron', y='nickel', hue='Cluster', data=df, palette='Set1')
    plt.title('KMeans Clustering of Mining Sites')
    st.pyplot(plt)

    # PCA for dimensionality reduction and visualization
    st.write("## 🔍 Principal Component Analysis (PCA)")
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
    st.write("## 🚨 Outlier Detection")
    st.write("Detecting outliers using the IQR method.")

    Q1 = df.select_dtypes(include=['float64', 'int64']).quantile(0.25)
    Q3 = df.select_dtypes(include=['float64', 'int64']).quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((df.select_dtypes(include=['float64', 'int64']) < (Q1 - 1.5 * IQR)) | (df.select_dtypes(include=['float64', 'int64']) > (Q3 + 1.5 * IQR))).any(axis=1)
    df['Outlier'] = outliers

    st.write(f"**Number of outliers detected:** `{outliers.sum()}`")
    st.write(df[outliers].style.background_gradient(cmap='Reds'))

    # Correlation with non-numeric features using One-Hot Encoding
    st.write("## 🔗 Correlation with Non-Numeric Features")
    st.write("Analyzing correlations with non-numeric features using One-Hot Encoding.")

    non_numeric_df = df.select_dtypes(exclude=['float64', 'int64'])
    encoded_df = pd.get_dummies(non_numeric_df)
    combined_df = pd.concat([df.select_dtypes(include=['float64', 'int64']), encoded_df], axis=1)
    combined_corr_matrix = combined_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(combined_corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    st.pyplot(plt)

    # Additional Visualizations
    st.write("## 📊 More Advanced Visualizations")

    

    # Bar Chart
    st.write("### 📊 Bar Chart")
    selected_bar_columns = st.selectbox("Select a column for bar chart:", df.columns.tolist())
    if selected_bar_columns:
        plt.figure(figsize=(10, 6))
        df[selected_bar_columns].value_counts().plot(kind='bar', color='skyblue')
        plt.title(f'Bar Chart of {selected_bar_columns}')
        st.pyplot(plt)

    st.write("### 📊 Histogram")
    selected_hist_columns = st.selectbox("Select a column for histogram:", df.columns.tolist(), index=0)
    if selected_hist_columns:
        plt.figure(figsize=(10, 6))
        df[selected_hist_columns].plot(kind='hist', bins=30, color='green')
        plt.title(f'Histogram of {selected_hist_columns}')
        st.pyplot(plt)

    # Pairplot
    st.write("### 🌐 Pairplot")
    selected_pair_columns = st.multiselect("Select columns for pairplot:", df.select_dtypes(include=['float64', 'int64']).columns.tolist(), default=df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:4])
    if selected_pair_columns:
        sns.pairplot(df[selected_pair_columns])
        st.pyplot(plt)

    # Boxplot
    st.write("### 📦 Boxplot")
    selected_box_columns = st.selectbox("Select a column for boxplot:", df.select_dtypes(include=['float64', 'int64']).columns.tolist())
    if selected_box_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[selected_box_columns], color='purple')
        plt.title(f'Boxplot of {selected_box_columns}')
        st.pyplot(plt)

    st.success("Analysis complete!")

show_analyze_page()
