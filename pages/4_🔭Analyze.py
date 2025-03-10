import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    selected_columns = st.multiselect("Select columns for correlation matrix:", df.columns.tolist(), default=df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:5], key="correlation_columns")
    
    if selected_columns:
        numeric_df = df[selected_columns]
        corr_matrix = numeric_df.corr()
        st.write(corr_matrix.style.background_gradient(cmap='viridis', axis=None))

        # Correlation heatmap
        st.write("### 🔥 Correlation Heatmap")
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)
    
    # Clustering analysis using KMeans
    st.write("## 🧩 Clustering Analysis")
    st.write("Identifying clusters of similar mining sites using KMeans.")

    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaled_features = scaler.fit_transform(df[numeric_cols])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    df['Cluster'] = clusters

    st.write("### 🎯 Cluster Centers")
    st.write(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=numeric_cols).style.background_gradient(cmap='coolwarm'))

    # Plotly scatter plot for clustering
    st.write("### 🔍 Interactive Cluster Visualization")
    fig = px.scatter(df, x='iron', y='nickel', color='Cluster', hover_data=df.columns)
    st.plotly_chart(fig)

    # PCA for dimensionality reduction and visualization
    st.write("## 🔍 Principal Component Analysis (PCA)")
    
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    df['PCA1'] = pca_features[:, 0]
    df['PCA2'] = pca_features[:, 1]

    # Explained variance
    st.write(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # PCA components visualization
    fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', hover_data=df.columns)
    st.plotly_chart(fig)

    # PCA Loading plot
    st.write("### 🧬 PCA Loading Plot")
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=numeric_cols)
    
    fig = px.scatter(loading_df, x='PC1', y='PC2', text=loading_df.index)
    fig.update_traces(textposition='top center')
    for i, feature in enumerate(loading_df.index):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loading_df.PC1[i],
            y1=loading_df.PC2[i]
        )
    st.plotly_chart(fig)

    # Outlier detection methods
    st.write("## 🚨 Outlier Detection")
    
    # IQR Method
    st.write("### 📏 IQR Method")
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    outliers_iqr = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                    (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    
    st.write(f"**Number of outliers detected (IQR):** `{outliers_iqr.sum()}`")
    
    # Z-Score Method
    st.write("### 📊 Z-Score Method")
    z_scores = stats.zscore(df[numeric_cols])
    abs_z_scores = np.abs(z_scores)
    outliers_z = (abs_z_scores > 3).any(axis=1)
    df['z_outlier'] = outliers_z
    
    st.write(f"**Number of outliers detected (Z-Score):** `{outliers_z.sum()}`")
    
    # Isolation Forest
    st.write("### 🌲 Isolation Forest")
    iso = IsolationForest(contamination=0.1, random_state=42)
    outliers_iso = iso.fit_predict(scaled_features) == -1
    df['iso_outlier'] = outliers_iso
    
    st.write(f"**Number of outliers detected (Isolation Forest):** `{sum(outliers_iso)}`")
    
    # Visualize outliers
    st.write("### 👁️ Outlier Visualization")
    selected_feature = st.selectbox("Select feature to visualize outliers:", numeric_cols)
    
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=("IQR Method", "Z-Score Method", "Isolation Forest"))
    
    fig.add_trace(go.Box(x=df[selected_feature], name="IQR", 
                         boxpoints='outliers', marker_color='blue'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df[selected_feature], y=np.zeros_like(df[selected_feature]),
                         mode='markers', marker=dict(color=['red' if x else 'blue' for x in outliers_z]),
                         name="Z-Score"), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df[selected_feature], y=np.zeros_like(df[selected_feature]),
                         mode='markers', marker=dict(color=['red' if x else 'blue' for x in outliers_iso]),
                         name="Isolation Forest"), row=3, col=1)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig)

    # DBSCAN Clustering
    st.write("## 🌐 DBSCAN Clustering")
    dbscan = DBSCAN(eps=1.0, min_samples=5)
    df['DBSCAN_cluster'] = dbscan.fit_predict(scaled_features)
    
    fig = px.scatter(df, x='PCA1', y='PCA2', color='DBSCAN_cluster', 
                     title="DBSCAN Clustering Results")
    st.plotly_chart(fig)

    # Feature distributions
    st.write("## 📊 Feature Distributions")
    
    # Histogram with KDE
    st.write("### 📈 Histogram with KDE")
    hist_feature = st.selectbox("Select feature for histogram:", numeric_cols, key="hist_kde")
    
    fig = px.histogram(df, x=hist_feature, marginal="box", color='Cluster',
                      title=f"Distribution of {hist_feature}")
    st.plotly_chart(fig)
    
    # Violin plots
    st.write("### 🎻 Violin Plots")
    violin_feature = st.selectbox("Select feature for violin plot:", numeric_cols, key="violin_feature")
    
    fig = px.violin(df, y=violin_feature, box=True, points="all", color='Cluster')
    st.plotly_chart(fig)
    
    # Parallel coordinates
    st.write("## 🌈 Parallel Coordinates")
    selected_features = st.multiselect("Select features for parallel coordinates:", 
                                      numeric_cols, default=numeric_cols[:5])
    
    if selected_features:
        fig = px.parallel_coordinates(df, dimensions=selected_features, color='Cluster')
        st.plotly_chart(fig)
    
    # Radar chart
    st.write("## 📡 Radar Chart")
    radar_features = st.multiselect("Select features for radar chart:", 
                                   numeric_cols, default=numeric_cols[:5])
    
    if radar_features:
        # Normalize the data for radar chart
        radar_data = df[radar_features].copy()
        min_max_scaler = MinMaxScaler()
        radar_data_scaled = min_max_scaler.fit_transform(radar_data)
        radar_df = pd.DataFrame(radar_data_scaled, columns=radar_features)
        
        # Calculate mean for each cluster
        cluster_means = {}
        for cluster in df['Cluster'].unique():
            cluster_means[f'Cluster {cluster}'] = radar_df.loc[df['Cluster'] == cluster].mean()
        
        # Create radar chart
        fig = go.Figure()
        
        for cluster_name, values in cluster_means.items():
            fig.add_trace(go.Scatterpolar(
                r=values.values,
                theta=values.index,
                fill='toself',
                name=cluster_name
            ))
            
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True
        )
        
        st.plotly_chart(fig)
    
    # Scatter matrix
    st.write("## 🔄 Scatter Matrix")
    scatter_features = st.multiselect("Select features for scatter matrix:", 
                                     numeric_cols, default=numeric_cols[:4])
    
    if scatter_features:
        fig = px.scatter_matrix(df, dimensions=scatter_features, color='Cluster')
        st.plotly_chart(fig)
    
    # 3D Scatter plot
    st.write("## 🌌 3D Scatter Plot")
    x_3d = st.selectbox("X axis:", numeric_cols, index=0, key="x_3d")
    y_3d = st.selectbox("Y axis:", numeric_cols, index=1, key="y_3d")
    z_3d = st.selectbox("Z axis:", numeric_cols, index=2, key="z_3d")
    
    fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color='Cluster')
    st.plotly_chart(fig)
    
    # Time series analysis (if there's any date column)
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if date_columns:
        st.write("## ⏱️ Time Series Analysis")
        date_col = st.selectbox("Select date column:", date_columns)
        value_col = st.selectbox("Select value to analyze:", numeric_cols)
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
        # Group by date and aggregate
        time_df = df.groupby(pd.Grouper(key=date_col, freq='M'))[value_col].mean().reset_index()
        
        fig = px.line(time_df, x=date_col, y=value_col, title=f"{value_col} over time")
        st.plotly_chart(fig)
    
    # Contour plot
    st.write("## 🗺️ Contour Plot")
    x_contour = st.selectbox("X axis:", numeric_cols, index=0, key="x_contour")
    y_contour = st.selectbox("Y axis:", numeric_cols, index=1, key="y_contour")
    z_contour = st.selectbox("Z axis (value):", numeric_cols, index=2, key="z_contour")
    
    fig = px.density_contour(df, x=x_contour, y=y_contour, z=z_contour, histfunc="avg")
    fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    st.plotly_chart(fig)
    
    # Sunburst chart for categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        st.write("## 🌞 Hierarchical Visualization")
        cat_cols = st.multiselect("Select categorical columns (in hierarchical order):", 
                                categorical_cols, default=categorical_cols[:min(2, len(categorical_cols))])
        
        if cat_cols and len(cat_cols) >= 1:
            value_col = st.selectbox("Value to aggregate:", numeric_cols, key="sunburst_value")
            
            fig = px.sunburst(df, path=cat_cols, values=value_col)
            st.plotly_chart(fig)
    
    # Treemap
    if categorical_cols:
        st.write("## 🌳 Treemap")
        treemap_cats = st.multiselect("Select categorical columns for treemap:", 
                                     categorical_cols, default=categorical_cols[:min(2, len(categorical_cols))],
                                     key="treemap_cats")
        
        if treemap_cats:
            treemap_value = st.selectbox("Value to size by:", numeric_cols, key="treemap_value")
            
            fig = px.treemap(df, path=treemap_cats, values=treemap_value, color=treemap_value)
            st.plotly_chart(fig)
    
    # Calendar heatmap
    if date_columns:
        st.write("## 📅 Calendar Heatmap")
        cal_date_col = st.selectbox("Select date column:", date_columns, key="cal_date")
        cal_value_col = st.selectbox("Select value for heatmap:", numeric_cols, key="cal_value")
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[cal_date_col]):
            df[cal_date_col] = pd.to_datetime(df[cal_date_col], errors='coerce')
            
        # Extract components
        df['year'] = df[cal_date_col].dt.year
        df['month'] = df[cal_date_col].dt.month
        df['day'] = df[cal_date_col].dt.day
        
        # Group by date
        cal_df = df.groupby(['year', 'month', 'day'])[cal_value_col].mean().reset_index()
        
        # Create heatmap
        fig = px.density_heatmap(cal_df, x='day', y='month', z=cal_value_col, 
                               color_continuous_scale='Viridis')
        st.plotly_chart(fig)
    
    # Ridgeline plot
    st.write("## 🏔️ Ridgeline Plot")
    ridge_feature = st.selectbox("Select feature for ridgeline plot:", numeric_cols, key="ridge_feature")
    
    # Creating ridgeline plot
    fig = go.Figure()
    clusters = sorted(df['Cluster'].unique())
    
    for i, cluster in enumerate(clusters):
        cluster_data = df[df['Cluster'] == cluster][ridge_feature]
        
        # Generate kernel density estimate
        kde_x = np.linspace(df[ridge_feature].min(), df[ridge_feature].max(), 100)
        kde_y = stats.gaussian_kde(cluster_data)(kde_x)
        
        # Scale
        kde_y = kde_y / kde_y.max() * 0.5
        
        # Add offset based on cluster
        kde_y = kde_y + i
        
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y,
            fill='tozeroy',
            name=f'Cluster {cluster}'
        ))
    
    fig.update_layout(title=f"Ridgeline Plot of {ridge_feature} by Cluster")
    st.plotly_chart(fig)
    
    # Bubble chart
    st.write("## 🔮 Bubble Chart")
    x_bubble = st.selectbox("X axis:", numeric_cols, index=0, key="x_bubble")
    y_bubble = st.selectbox("Y axis:", numeric_cols, index=1, key="y_bubble")
    size_bubble = st.selectbox("Size variable:", numeric_cols, index=2, key="size_bubble")
    
    fig = px.scatter(df, x=x_bubble, y=y_bubble, size=size_bubble, color='Cluster',
                   hover_name=df.index, size_max=50)
    st.plotly_chart(fig)
    
    # Hexbin plot
    st.write("## 🔷 Hexbin Plot")
    x_hex = st.selectbox("X axis:", numeric_cols, index=0, key="x_hex")
    y_hex = st.selectbox("Y axis:", numeric_cols, index=1, key="y_hex")
    
    fig = px.density_heatmap(df, x=x_hex, y=y_hex, marginal_x="histogram", marginal_y="histogram",
                           nbinsx=20, nbinsy=20)
    st.plotly_chart(fig)
    
    # Feature importance from a random forest (simplified)
    st.write("## 🌟 Feature Importance")
    from sklearn.ensemble import RandomForestRegressor
    
    target_col = st.selectbox("Select target variable:", numeric_cols, key="rf_target")
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance')
    st.plotly_chart(fig)
    
    # Partial dependence plot (simplified)
    st.write("## 🔍 Partial Dependence Plot")
    pdp_feature = st.selectbox("Select feature:", feature_cols, key="pdp_feature")
    
    from sklearn.inspection import partial_dependence
    
    pdp = partial_dependence(rf, X, features=[feature_cols.index(pdp_feature)])
    pdp_df = pd.DataFrame({
        pdp_feature: pdp["values"][0],
        'Partial Dependence': pdp["average"][0]
    })
    
    fig = px.line(pdp_df, x=pdp_feature, y='Partial Dependence')
    st.plotly_chart(fig)
    
    # Anomaly scores
    st.write("## 🚩 Anomaly Scores")
    # Use Isolation Forest scores
    df['anomaly_score'] = -1 * iso.score_samples(scaled_features)
    
    fig = px.histogram(df, x='anomaly_score', color='iso_outlier',
                     title='Distribution of Anomaly Scores')
    st.plotly_chart(fig)
    
    # Cumulative distribution
    st.write("## 📈 Cumulative Distribution")
    cdf_feature = st.selectbox("Select feature for CDF:", numeric_cols, key="cdf_feature")
    
    # Calculate ECDF
    x = np.sort(df[cdf_feature])
    y = np.arange(1, len(x)+1) / len(x)
    
    fig = px.line(x=x, y=y, title=f'ECDF of {cdf_feature}')
    fig.update_layout(xaxis_title=cdf_feature, yaxis_title='Cumulative Probability')
    st.plotly_chart(fig)
    
    # Scatter plot with regression line
    st.write("## 📏 Regression Analysis")
    x_reg = st.selectbox("X variable:", numeric_cols, index=0, key="x_reg")
    y_reg = st.selectbox("Y variable:", numeric_cols, index=1, key="y_reg")
    
    fig = px.scatter(df, x=x_reg, y=y_reg, trendline="ols", 
                   title=f'Regression: {y_reg} vs {x_reg}')
    st.plotly_chart(fig)
    
    # Residual plot
    from sklearn.linear_model import LinearRegression
    
    st.write("### 📊 Residual Plot")
    X_reg = df[[x_reg]]
    y_reg_data = df[y_reg]
    
    reg = LinearRegression().fit(X_reg, y_reg_data)
    df['predicted'] = reg.predict(X_reg)
    df['residuals'] = df[y_reg] - df['predicted']
    
    fig = px.scatter(df, x='predicted', y='residuals', color='Cluster',
                   title='Residual Plot')
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig)

    st.success("Analysis complete! 👏")

show_analyze_page()
