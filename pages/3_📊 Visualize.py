import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io
import base64
from datetime import datetime
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import folium
from streamlit_folium import st_folium
import networkx as nx
import umap
import hdbscan

# Set page configuration
st.set_page_config(page_title="Advanced Mining Site Analysis Suite", layout="wide")

# Custom CSS for enhanced aesthetics
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stSelectbox, .stSlider {font-size: 14px; background-color: #ffffff; border-radius: 5px;}
    .sidebar .sidebar-content {background-color: #e8f4f8;}
    </style>
""", unsafe_allow_html=True)

# Sidebar for global settings
st.sidebar.title("Global Settings")
theme = st.sidebar.selectbox("Select Theme", ["Darkgrid", "Whitegrid", "Dark", "White", "Ticks"], index=0)
sns.set_style(theme.lower())
palette = st.sidebar.selectbox("Color Palette", ["viridis", "coolwarm", "magma", "pastel", "cubehelix", "Set2"], index=0)
sns.set_palette(palette)
font_size = st.sidebar.slider("Font Size", 10, 20, 12)
plt.rcParams.update({'font.size': font_size})

# Load dataset with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("space_mining_dataset.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset 'space_mining_dataset.csv' not found. Please upload a CSV file.")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            return df
        return None

df = load_data()
if df is None:
    st.stop()

# Data preprocessing
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Main title and description
st.title("Advanced Mining Site Analysis Suite")
st.write("A comprehensive toolset for exploring, visualizing, and modeling mining site data with advanced analytics.")

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Basic Visualizations", "Advanced Visualizations", "3D & Animations", 
                                               "Statistical Insights", "Machine Learning", "Geospatial & Network Analysis"])

# --- Tab 1: Basic Visualizations ---
with tab1:
    st.header("Basic Visualizations")
    # Same as original with minor enhancements (e.g., log scale option)
    st.subheader("Interactive Scatter Plot")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis", numeric_columns, index=0, key="scatter_x")
        y_axis = st.selectbox("Y-axis", numeric_columns, index=1, key="scatter_y")
        hue = st.selectbox("Hue (optional)", [None] + categorical_columns, key="scatter_hue")
    with col2:
        size = st.selectbox("Size (optional)", [None] + numeric_columns, key="scatter_size")
        alpha = st.slider("Transparency", 0.1, 1.0, 0.8, key="scatter_alpha")
        marker = st.selectbox("Marker Style", ["o", "s", "^", "D", "x"], key="scatter_marker")
        log_scale = st.checkbox("Log Scale", key="scatter_log")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df[hue] if hue else None, 
                    size=df[size] if size else None, alpha=alpha, marker=marker, ax=ax)
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"{x_axis} vs {y_axis}")
    st.pyplot(fig)

    # Add more basic plots as in original...

# --- Tab 2: Advanced Visualizations ---
with tab2:
    st.header("Advanced Visualizations")
    # Enhanced with Pair Plot
    st.subheader("Pair Plot")
    pair_cols = st.multiselect("Select Columns", numeric_columns, default=numeric_columns[:4], key="pair_cols")
    if pair_cols:
        pair_fig = sns.pairplot(df[pair_cols], diag_kind="kde", palette=palette)
        st.pyplot(pair_fig.figure)

    # Add more advanced plots as in original...

# --- Tab 3: 3D & Animations ---
with tab3:
    st.header("3D Visualizations & Animations")
    # Enhanced with 3D Contour
    st.subheader("3D Contour Plot")
    cont_x = st.selectbox("X-axis", numeric_columns, key="cont_x")
    cont_y = st.selectbox("Y-axis", numeric_columns, key="cont_y")
    cont_z = st.selectbox("Z-axis", numeric_columns, key="cont_z")
    X, Y = np.meshgrid(df[cont_x], df[cont_y])
    Z = np.array(df[cont_z]).reshape(X.shape[0], -1)[:X.shape[0], :X.shape[1]]
    fig = go.Figure(data=[go.Contour(z=Z, x=df[cont_x], y=df[cont_y])])
    fig.update_layout(title=f"3D Contour: {cont_z}")
    st.plotly_chart(fig)

    # Add more 3D plots as in original...

# --- Tab 4: Statistical Insights ---
with tab4:
    st.header("Statistical Insights")
    # Enhanced with Distribution Fit
    st.subheader("Distribution Fitting")
    dist_col = st.selectbox("Column", numeric_columns, key="dist_col")
    dist_type = st.selectbox("Distribution", ["Normal", "Log-Normal", "Exponential"], key="dist_type")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[dist_col], kde=False, bins=30, stat="density", ax=ax)
    if dist_type == "Normal":
        mu, sigma = stats.norm.fit(df[dist_col].dropna())
        x = np.linspace(df[dist_col].min(), df[dist_col].max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
        st.write(f"Normal Fit: μ={mu:.2f}, σ={sigma:.2f}")
    # Add more distribution fits...
    st.pyplot(fig)

    # Add more stats as in original...

# --- Tab 5: Machine Learning ---
with tab5:
    st.header("Machine Learning Tools")

    # Clustering
    st.subheader("Clustering Analysis")
    cluster_cols = st.multiselect("Features for Clustering", numeric_columns, default=numeric_columns[:3], key="cluster_cols")
    cluster_method = st.selectbox("Method", ["KMeans", "DBSCAN", "HDBSCAN"], key="cluster_method")
    if cluster_cols and len(cluster_cols) >= 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[cluster_cols])
        if cluster_method == "KMeans":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="kmeans_n")
            model = KMeans(n_clusters=n_clusters)
        elif cluster_method == "DBSCAN":
            eps = st.slider("Epsilon", 0.1, 2.0, 0.5, key="dbscan_eps")
            model = DBSCAN(eps=eps)
        else:
            min_cluster_size = st.slider("Min Cluster Size", 5, 50, 15, key="hdbscan_size")
            model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        
        labels = model.fit_predict(scaled_data)
        df['Cluster'] = labels
        fig = px.scatter(df, x=cluster_cols[0], y=cluster_cols[1], color='Cluster', title=f"{cluster_method} Clustering")
        st.plotly_chart(fig)

    # Regression
    st.subheader("Random Forest Regression")
    target = st.selectbox("Target Variable", numeric_columns, key="reg_target")
    features = st.multiselect("Features", numeric_columns, default=[col for col in numeric_columns if col != target], key="reg_features")
    if features and target:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        feat_importance = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
        st.bar_chart(feat_importance.set_index('Feature'))

# --- Tab 6: Geospatial & Network Analysis ---
with tab6:
    st.header("Geospatial & Network Analysis")

    # Geospatial Visualization (assuming lat/lon columns exist)
    st.subheader("Geospatial Map")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5)
        for _, row in df.iterrows():
            folium.Marker([row['latitude'], row['longitude']]).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.write("No latitude/longitude columns detected.")

    # Network Analysis
    st.subheader("Network Visualization")
    edge_col1 = st.selectbox("Source Node", categorical_columns, key="net_source")
    edge_col2 = st.selectbox("Target Node", categorical_columns, key="net_target")
    if edge_col1 and edge_col2:
        G = nx.from_pandas_edgelist(df, edge_col1, edge_col2)
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10, ax=ax)
        st.pyplot(fig)

    # UMAP Dimensionality Reduction
    st.subheader("UMAP Visualization")
    umap_cols = st.multiselect("Features for UMAP", numeric_columns, default=numeric_columns[:5], key="umap_cols")
    if umap_cols and len(umap_cols) >= 2:
        reducer = umap.UMAP(n_components=2)
        embedding = reducer.fit_transform(df[umap_cols])
        fig = px.scatter(x=embedding[:, 0], y=embedding[:, 1], title="UMAP 2D Projection")
        st.plotly_chart(fig)

# Enhanced Export Functionality
st.sidebar.subheader("Export Options")
export_format = st.sidebar.selectbox("Export Format", ["PNG", "PDF", "CSV", "JSON"], key="export_format")
if st.sidebar.button("Export Current Visualization"):
    if export_format == "CSV":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="mining_data.csv">Download CSV</a>'
    elif export_format == "JSON":
        json_str = df.to_json()
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="mining_data.json">Download JSON</a>'
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format=export_format.lower(), dpi=300)
        b64 = base64.b64encode(buf.getvalue()).decode()
        href = f'<a href="data:image/{export_format.lower()};base64,{b64}" download="visualization.{export_format.lower()}">Download {export_format}</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown(f"<footer style='text-align: center; color: gray;'>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Grok 3 (xAI)</footer>", unsafe_allow_html=True)
