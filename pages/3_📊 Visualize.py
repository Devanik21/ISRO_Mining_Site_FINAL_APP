import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import io
import base64
from datetime import datetime
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # Corrected import
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import shap
import xgboost as xgb
import time

# Page configuration
st.set_page_config(page_title="Ultimate Mining Site Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #1f77b4; color: white; border-radius: 5px;}
    .stSelectbox, .stSlider {font-size: 14px; background-color: #ffffff; border-radius: 5px;}
    .sidebar .sidebar-content {background-color: #e6f1fa;}
    </style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("Control Panel")
theme = st.sidebar.selectbox("Theme", ["Darkgrid", "Whitegrid", "Dark", "White", "Ticks"], index=0)
sns.set_style(theme.lower())
palette = st.sidebar.selectbox("Color Palette", ["viridis", "coolwarm", "magma", "pastel", "cubehelix", "Set2", "Spectral"], index=0)
sns.set_palette(palette)
font_size = st.sidebar.slider("Font Size", 10, 24, 12)
plt.rcParams.update({'font.size': font_size})
update_freq = st.sidebar.slider("Real-Time Update Frequency (s)", 1, 10, 5)

# Load dataset with simulation option
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("space_mining_dataset.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Upload a CSV or use simulated data.")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            np.random.seed(42)
            df = pd.DataFrame({
                'Site_ID': [f'S{str(i).zfill(3)}' for i in range(100)],
                'Latitude': np.random.uniform(-90, 90, 100),
                'Longitude': np.random.uniform(-180, 180, 100),
                'Ore_Density': np.random.lognormal(2, 0.5, 100),
                'Depth': np.random.uniform(50, 500, 100),
                'Temperature': np.random.normal(25, 10, 100),
                'Radiation': np.random.exponential(1, 100),
                'Category': np.random.choice(['Iron', 'Copper', 'Gold', 'Rare Earth'], 100)
            })
    return df

df = load_data()
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Main title
st.title("Ultimate Mining Site Analytics Dashboard")
st.write("A comprehensive tool for visualizing, analyzing, and predicting mining site data.")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Core Visualizations", "Advanced Plots", "Geospatial & 3D", "ML & Predictive", "Real-Time Analytics"])

# Set black borders globally for Matplotlib
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5

# --- Tab 1: Core Visualizations ---
with tab1:
    st.header("Core Visualizations")
    
    # Enhanced Scatter with regression
    st.subheader("Enhanced Scatter Plot")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis", numeric_columns, key="scatter_x")
        y_axis = st.selectbox("Y-axis", numeric_columns, key="scatter_y")
        hue = st.selectbox("Hue", [None] + categorical_columns, key="scatter_hue")
    with col2:
        size = st.selectbox("Size", [None] + numeric_columns, key="scatter_size")
        reg_line = st.checkbox("Add Regression Line", key="scatter_reg")
        log_scale = st.checkbox("Log Scale", key="scatter_log")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df[hue] if hue else None, size=df[size] if size else None, ax=ax)
    if reg_line:
        sns.regplot(x=df[x_axis], y=df[y_axis], scatter=False, color='red', ax=ax)
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_title(f"{x_axis} vs {y_axis}")
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    st.pyplot(fig)

    # Dynamic Histogram
    st.subheader("Dynamic Histogram")
    hist_col = st.selectbox("Column", numeric_columns, key="hist")
    bins = st.slider("Bins", 10, 100, 30, key="hist_bins")
    stat = st.selectbox("Statistic", ["count", "density", "probability"], key="hist_stat")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(df[hist_col], bins=bins, kde=True, stat=stat, ax=ax)
    ax.set_title(f"Distribution of {hist_col}")
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    st.pyplot(fig)

    # Nested Pie Chart
    st.subheader("Nested Pie Chart")
    outer_col = st.selectbox("Outer Category", categorical_columns, key="pie_outer")
    inner_col = st.selectbox("Inner Category", categorical_columns, key="pie_inner")
    fig, ax = plt.subplots(figsize=(10, 10))
    outer_data = df[outer_col].value_counts()
    inner_data = df.groupby([outer_col, inner_col]).size().unstack().fillna(0)
    ax.pie(outer_data, labels=outer_data.index, radius=1.2, wedgeprops=dict(width=0.3))
    ax.pie(inner_data.values.flatten(), radius=0.9, wedgeprops=dict(width=0.3))
    ax.set_title(f"Nested Distribution: {outer_col} & {inner_col}")
    st.pyplot(fig)  # Pie charts don’t have spines, so no border adjustment needed

# --- Tab 2: Advanced Plots ---
with tab2:
    st.header("Advanced Visualizations")
    
    # Parallel Coordinates
    st.subheader("Parallel Coordinates")
    par_cols = st.multiselect("Features", numeric_columns, default=numeric_columns[:4], key="par_cols")
    if par_cols:
        fig = px.parallel_coordinates(df, dimensions=par_cols, color=par_cols[0] if par_cols else None)
        st.plotly_chart(fig)

    # Sankey Diagram
    st.subheader("Sankey Diagram")
    sankey_cols = st.multiselect("Categorical Columns", categorical_columns, default=categorical_columns[:2], key="sankey_cols")
    if len(sankey_cols) >= 2:
        sankey_data = df.groupby(sankey_cols).size().reset_index(name='value')
        nodes = pd.concat([sankey_data[sankey_cols[0]], sankey_data[sankey_cols[1]]]).unique()
        node_dict = {node: i for i, node in enumerate(nodes)}
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=list(node_dict.keys())),
            link=dict(source=sankey_data[sankey_cols[0]].map(node_dict),
                      target=sankey_data[sankey_cols[1]].map(node_dict),
                      value=sankey_data['value']))])
        st.plotly_chart(fig)

    # Ridge Plot
    st.subheader("Ridge Plot")
    ridge_cat = st.selectbox("Categorical", categorical_columns, key="ridge_cat")
    ridge_num = st.selectbox("Numeric", numeric_columns, key="ridge_num")
    g = sns.FacetGrid(df, row=ridge_cat, hue=ridge_cat, aspect=15, height=0.5)
    g.map(sns.kdeplot, ridge_num, fill=True)
    g.fig.suptitle(f"Ridge Plot: {ridge_num} by {ridge_cat}")
    for ax in g.axes.flat:
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
    st.pyplot(g.fig)

    # Violin + Swarm Combo
    st.subheader("Violin + Swarm Plot")
    combo_x = st.selectbox("Categorical", categorical_columns, key="combo_x")
    combo_y = st.selectbox("Numeric", numeric_columns, key="combo_y")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.violinplot(x=df[combo_x], y=df[combo_y], inner=None, ax=ax)
    sns.swarmplot(x=df[combo_x], y=df[combo_y], color="k", size=3, ax=ax)
    ax.set_title(f"{combo_x} vs {combo_y}")
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    st.pyplot(fig)

# --- Tab 3: Geospatial & 3D ---
with tab3:
    st.header("Geospatial & 3D Visualizations")
    
    # Geospatial Map
    st.subheader("Geospatial Map")
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        map_col = st.selectbox("Value Column", numeric_columns, key="map_col")
        m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=3)
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=row[map_col] / df[map_col].max() * 20,
                popup=f"{map_col}: {row[map_col]}",
                fill=True
            ).add_to(m)
        folium_static(m)
    else:
        st.warning("No Latitude/Longitude columns found for mapping.")

    # 3D Scatter with Clusters
    st.subheader("3D Scatter with Clustering")
    x_3d = st.selectbox("X-axis", numeric_columns, key="3d_x")
    y_3d = st.selectbox("Y-axis", numeric_columns, key="3d_y")
    z_3d = st.selectbox("Z-axis", numeric_columns, key="3d_z")
    k = st.slider("Number of Clusters", 2, 10, 3, key="3d_k")
    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(df[[x_3d, y_3d, z_3d]].dropna())
    fig_3d = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color=clusters, title=f"3D Scatter with {k} Clusters")
    st.plotly_chart(fig_3d)

    # 3D Contour Plot
    st.subheader("3D Contour Plot")
    cont_x = st.selectbox("X-axis", numeric_columns, key="cont_x")
    cont_y = st.selectbox("Y-axis", numeric_columns, key="cont_y")
    cont_z = st.selectbox("Z-axis", numeric_columns, key="cont_z")
    X, Y = np.meshgrid(df[cont_x], df[cont_y])
    Z = np.array(df[cont_z]).reshape(-1, X.shape[1])[:X.shape[0], :X.shape[1]]
    fig = go.Figure(data=[go.Contour(z=Z, x=df[cont_x], y=df[cont_y])])
    st.plotly_chart(fig)

# --- Tab 4: ML & Predictive ---
with tab4:
    st.header("Machine Learning & Predictive Analytics")
    
    # Feature Importance with XGBoost
    st.subheader("Feature Importance")
    target = st.selectbox("Target Variable", numeric_columns, key="ml_target")
    features = st.multiselect("Features", numeric_columns, default=numeric_columns[:5], key="ml_features")
    if features and target:
        X = df[features].dropna()
        y = df[target].loc[X.index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        fig, ax = plt.subplots(figsize=(12, 8))
        xgb.plot_importance(model, ax=ax)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        st.pyplot(fig)

    # SHAP Analysis
    st.subheader("SHAP Explainability")
    if features and target:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        st.pyplot(fig)

    # TSNE Visualization
    st.subheader("t-SNE Visualization")
    tsne_cols = st.multiselect("Features for t-SNE", numeric_columns, default=numeric_columns[:5], key="tsne_cols")
    if tsne_cols and len(tsne_cols) >= 2:
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(df[tsne_cols].dropna())
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(tsne_result[:, 0], tsne_result[:, 1])
        ax.set_title("t-SNE Projection")
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        st.pyplot(fig)

# --- Tab 5: Real-Time Analytics ---
with tab5:
    st.header("Real-Time Analytics")
    
    # Simulated Real-Time Data
    st.subheader("Real-Time Line Plot")
    rt_col = st.selectbox("Column", numeric_columns, key="rt_col")
    placeholder = st.empty()
    data_buffer = []
    for i in range(50):
        new_data = df[rt_col].iloc[np.random.randint(0, len(df))] + np.random.normal(0, 0.1)
        data_buffer.append(new_data)
        if len(data_buffer) > 20:
            data_buffer.pop(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data_buffer)
        ax.set_title(f"Real-Time {rt_col}")
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        placeholder.pyplot(fig)
        time.sleep(update_freq / 10)

    # Real-Time Heatmap
    st.subheader("Real-Time Correlation Heatmap")
    rt_cols = st.multiselect("Columns", numeric_columns, default=numeric_columns[:4], key="rt_cols")
    if rt_cols:
        placeholder_heatmap = st.empty()
        for _ in range(20):
            sample = df[rt_cols].sample(frac=0.1)
            corr = sample.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
            placeholder_heatmap.pyplot(fig)
            time.sleep(update_freq / 10)

# Export Options (Removed HTML)
st.sidebar.subheader("Export Options")
export_type = st.sidebar.selectbox("Export Type", ["PNG", "PDF", "CSV"], key="export_type")
if st.sidebar.button("Export"):
    if export_type == "CSV":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="mining_data.csv">Download CSV</a>'
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format=export_type.lower(), dpi=300)
        b64 = base64.b64encode(buf.getvalue()).decode()
        href = f'<a href="data:image/{export_type.lower()};base64,{b64}" download="visualization.{export_type.lower()}">Download {export_type}</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown(f"<footer style='text-align: center; color: gray;'>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Grok 3 (xAI)</footer>", unsafe_allow_html=True)
