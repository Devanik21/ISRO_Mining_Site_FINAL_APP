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
from sklearn.cluster import KMeans

# Set page configuration
st.set_page_config(page_title="Advanced Mining Site Visualization", layout="wide")

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stSelectbox {font-size: 14px;}
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
st.title("Advanced Mining Site Visualization Dashboard")
st.write("Explore and visualize mining site data with interactive, customizable plots and advanced analytics.")

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Basic Visualizations", "Advanced Visualizations", "3D & Animations", 
                                         "Statistical Insights", "Geospatial Analysis", "Machine Learning"])

# --- Tab 1: Basic Visualizations ---
with tab1:
    st.header("Basic Visualizations")

    # Scatter Plot with customization
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df[hue] if hue else None, 
                    size=df[size] if size else None, alpha=alpha, marker=marker, ax=ax)
    ax.set_title(f"{x_axis} vs {y_axis}")
    st.pyplot(fig)

    # Histogram with KDE and stats
    st.subheader("Histogram with KDE")
    col1, col2 = st.columns(2)
    with col1:
        hist_col = st.selectbox("Column", numeric_columns, key="hist")
        bins = st.slider("Number of Bins", 10, 100, 30, key="hist_bins")
    with col2:
        kde = st.checkbox("Show KDE", value=True, key="hist_kde")
        multiple = st.selectbox("Multiple", ["layer", "stack", "dodge"], key="hist_multiple")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[hist_col], kde=kde, bins=bins, multiple=multiple, ax=ax)
    st.pyplot(fig)
    
    # NEW: Density Plot
    st.subheader("Density Plot")
    density_cols = st.multiselect("Select Columns", numeric_columns, default=numeric_columns[:2], key="density_cols")
    if density_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in density_cols:
            sns.kdeplot(df[col], ax=ax, label=col, fill=True, alpha=0.3)
        plt.legend()
        st.pyplot(fig)

    # Pie Chart with explode option
    st.subheader("Pie Chart")
    pie_col = st.selectbox("Categorical Column", categorical_columns, key="pie")
    explode = st.checkbox("Explode Slices", key="pie_explode")
    pie_data = df[pie_col].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    explode_vals = [0.1 if explode else 0] * len(pie_data)
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, 
           explode=explode_vals, colors=sns.color_palette(palette))
    st.pyplot(fig)

    # NEW: Count Plot
    st.subheader("Count Plot")
    count_x = st.selectbox("Categorical Column", categorical_columns, key="count_x")
    hue_count = st.selectbox("Hue (optional)", [None] + categorical_columns, key="count_hue")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=df[count_x], hue=df[hue_count] if hue_count else None, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # NEW: Interactive Time Series
    if any("date" in col.lower() or "time" in col.lower() for col in df.columns):
        st.subheader("Time Series Plot")
        date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_cols:
            date_col = st.selectbox("Date/Time Column", date_cols, key="time_date")
            value_col = st.selectbox("Value Column", numeric_columns, key="time_value")
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                chart = alt.Chart(df).mark_line().encode(
                    x=date_col,
                    y=value_col,
                    tooltip=[date_col, value_col]
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            except:
                st.error("Could not convert to datetime format")

# --- Tab 2: Advanced Visualizations ---
with tab2:
    st.header("Advanced Visualizations")

    # Violin Plot with split option
    st.subheader("Violin Plot")
    violin_x = st.selectbox("Categorical Column", categorical_columns, key="violin_x")
    violin_y = st.selectbox("Numeric Column", numeric_columns, key="violin_y")
    split = st.checkbox("Split by Hue", key="violin_split")
    hue = st.selectbox("Hue (optional)", [None] + categorical_columns, key="violin_hue") if split else None
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x=df[violin_x], y=df[violin_y], hue=df[hue] if hue else None, split=split, ax=ax)
    st.pyplot(fig)

    # NEW: Radar Chart
    st.subheader("Radar Chart")
    radar_cols = st.multiselect("Select Numeric Columns (3-8 recommended)", numeric_columns, 
                                default=numeric_columns[:5], key="radar_cols")
    radar_cat = st.selectbox("Category Column", categorical_columns, key="radar_cat")
    if radar_cols and len(radar_cols) >= 3:
        categories = df[radar_cat].unique()[:5]  # Limit to first 5 categories
        fig = go.Figure()
        for cat in categories:
            subset = df[df[radar_cat] == cat]
            if not subset.empty:
                values = [subset[col].mean() for col in radar_cols]
                values.append(values[0])  # Close the loop
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=radar_cols + [radar_cols[0]],
                    fill='toself',
                    name=str(cat)
                ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig)

    '''# NEW: Parallel Coordinates
    st.subheader("Parallel Coordinates")
    parallel_cols = st.multiselect("Select Columns", numeric_columns, default=numeric_columns[:5], key="parallel_cols")
    parallel_color = st.selectbox("Color by", categorical_columns + numeric_columns, key="parallel_color")
    if parallel_cols:
        fig = px.parallel_coordinates(df, dimensions=parallel_cols, color=df[parallel_color], 
                                    color_continuous_scale=px.colors.diverging.Tealrose)
        st.plotly_chart(fig)'''

    # NEW: Interactive Heatmap
    st.subheader("Interactive Correlation Heatmap")
    corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"], key="corr_method")
    corr_matrix = df[numeric_columns].corr(method=corr_method)
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig)

    # NEW: Pairplot with custom settings
    st.subheader("Enhanced Pairplot")
    pair_cols = st.multiselect("Select Columns (2-5 recommended)", numeric_columns, 
                              default=numeric_columns[:3], key="pair_cols")
    pair_hue = st.selectbox("Hue", [None] + categorical_columns, key="pair_hue")
    if pair_cols and len(pair_cols) >= 2:
        with st.spinner("Generating pairplot..."):
            sns.set(style="ticks")
            pair_plot = sns.pairplot(df, vars=pair_cols, hue=pair_hue, corner=True)
            st.pyplot(pair_plot.fig)

    # NEW: Violin + Swarm Combined Plot
    st.subheader("Violin + Swarm Combined Plot")
    vs_x = st.selectbox("Categorical Column", categorical_columns, key="vs_x")
    vs_y = st.selectbox("Numeric Column", numeric_columns, key="vs_y")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x=df[vs_x], y=df[vs_y], inner=None, ax=ax, alpha=0.5)
    sns.swarmplot(x=df[vs_x], y=df[vs_y], color="white", edgecolor="black", ax=ax, alpha=0.7)
    st.pyplot(fig)

# --- Tab 3: 3D & Animations ---
with tab3:
    st.header("3D Visualizations & Animations")

    # 3D Scatter Plot with Plotly
    st.subheader("3D Scatter Plot")
    x_3d = st.selectbox("X-axis", numeric_columns, key="3d_x")
    y_3d = st.selectbox("Y-axis", numeric_columns, key="3d_y")
    z_3d = st.selectbox("Z-axis", numeric_columns, key="3d_z")
    color_3d = st.selectbox("Color (optional)", [None] + categorical_columns + numeric_columns, key="3d_color")
    fig_3d = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color=color_3d, 
                           title=f"3D Scatter: {x_3d} vs {y_3d} vs {z_3d}")
    st.plotly_chart(fig_3d)

    # NEW: 3D Mesh Plot
    st.subheader("3D Mesh Plot")
    mesh_x = st.selectbox("X-axis", numeric_columns, key="mesh_x")
    mesh_y = st.selectbox("Y-axis", numeric_columns, key="mesh_y")
    mesh_z = st.selectbox("Z-axis", numeric_columns, key="mesh_z")
    if len(df) > 10:
        # Create a mesh grid for 3D visualization
        fig = go.Figure(data=[go.Mesh3d(
            x=df[mesh_x],
            y=df[mesh_y],
            z=df[mesh_z],
            opacity=0.8,
            colorscale='Viridis'
        )])
        fig.update_layout(scene=dict(xaxis_title=mesh_x, yaxis_title=mesh_y, zaxis_title=mesh_z))
        st.plotly_chart(fig)

    # NEW: Bubble Chart Animation
    st.subheader("Bubble Chart Animation")
    bubble_x = st.selectbox("X-axis", numeric_columns, key="bubble_x")
    bubble_y = st.selectbox("Y-axis", numeric_columns, key="bubble_y")
    bubble_size = st.selectbox("Size", numeric_columns, key="bubble_size")
    bubble_color = st.selectbox("Color", categorical_columns + numeric_columns, key="bubble_color")
    bubble_frame = st.selectbox("Animation Frame", categorical_columns, key="bubble_frame")
    fig = px.scatter(df, x=bubble_x, y=bubble_y, size=bubble_size, color=bubble_color,
                    animation_frame=bubble_frame, size_max=55, range_x=[df[bubble_x].min(), df[bubble_x].max()],
                    range_y=[df[bubble_y].min(), df[bubble_y].max()])
    st.plotly_chart(fig)

    # NEW: 3D Cone Plot (Vector Field)
    st.subheader("3D Vector Field (Cone Plot)")
    if len(numeric_columns) >= 6:
        u_col = st.selectbox("U Component (X Direction)", numeric_columns, key="u_col")
        v_col = st.selectbox("V Component (Y Direction)", numeric_columns, key="v_col")
        w_col = st.selectbox("W Component (Z Direction)", numeric_columns, key="w_col")
        x_pos = st.selectbox("X Position", numeric_columns, key="x_pos")
        y_pos = st.selectbox("Y Position", numeric_columns, key="y_pos")
        z_pos = st.selectbox("Z Position", numeric_columns, key="z_pos")
        
        fig = go.Figure(data=go.Cone(
            x=df[x_pos],
            y=df[y_pos],
            z=df[z_pos],
            u=df[u_col],
            v=df[v_col],
            w=df[w_col],
            colorscale='Blues',
            sizemode="absolute"
        ))
        fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1)))
        st.plotly_chart(fig)

# --- Tab 4: Statistical Insights ---
with tab4:
    st.header("Statistical Insights")

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    stats_col = st.multiselect("Select Columns", numeric_columns, default=numeric_columns[:3], key="stats_col")
    if stats_col:
        st.write(df[stats_col].describe())

    # NEW: Distribution Comparison
    st.subheader("Distribution Comparison")
    dist_cols = st.multiselect("Select Columns to Compare", numeric_columns, default=numeric_columns[:2], key="dist_cols")
    if dist_cols and len(dist_cols) >= 2:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for i, col in enumerate(dist_cols[:2]):
            sns.histplot(df[col], kde=True, ax=axs[i])
            axs[i].set_title(f"Distribution of {col}")
        st.pyplot(fig)
        
        # QQ Plot for normality test
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for i, col in enumerate(dist_cols[:2]):
            stats.probplot(df[col].dropna(), plot=axs[i])
            axs[i].set_title(f"QQ Plot of {col}")
        st.pyplot(fig)

    # NEW: Multi-level Analysis
    st.subheader("Multi-level Statistical Analysis")
    if categorical_columns:
        cat_col = st.selectbox("Categorical Variable", categorical_columns, key="multi_cat")
        num_col = st.selectbox("Numeric Variable", numeric_columns, key="multi_num")
        
        # Group statistics
        grouped = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'min', 'max', 'count'])
        st.write(grouped)
        
        # ANOVA test
        categories = df[cat_col].unique()
        if len(categories) > 1:
            groups = [df[df[cat_col] == cat][num_col].dropna() for cat in categories]
            if all(len(g) > 0 for g in groups):
                f_stat, p_val = stats.f_oneway(*groups)
                st.write(f"ANOVA Test: F={f_stat:.4f}, p-value={p_val:.4f}")
                if p_val < 0.05:
                    st.write("Significant differences detected between groups.")
                else:
                    st.write("No significant differences detected between groups.")

    # NEW: Regression Analysis
    st.subheader("Simple Regression Analysis")
    reg_x = st.selectbox("Independent Variable", numeric_columns, key="reg_x")
    reg_y = st.selectbox("Dependent Variable", numeric_columns, key="reg_y")
    
    # Scatter with regression line
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=df[reg_x], y=df[reg_y], ax=ax)
    st.pyplot(fig)
    
    # Calculate regression results
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[reg_x], df[reg_y])
    st.write(f"Regression equation: y = {slope:.4f}x + {intercept:.4f}")
    st.write(f"R-squared: {r_value**2:.4f}, p-value: {p_value:.4f}")

# --- Tab 5: Geospatial Analysis (NEW) ---
with tab5:
    st.header("Geospatial Analysis")
    
    # Check if latitude and longitude columns exist
    lat_candidates = [col for col in df.columns if "lat" in col.lower()]
    lon_candidates = [col for col in df.columns if "lon" in col.lower() or "lng" in col.lower()]
    
    if lat_candidates and lon_candidates:
        lat_col = st.selectbox("Latitude Column", lat_candidates, key="lat_col")
        lon_col = st.selectbox("Longitude Column", lon_candidates, key="lon_col")
        color_col = st.selectbox("Color by", [None] + categorical_columns + numeric_columns, key="map_color")
        size_col = st.selectbox("Size by", [None] + numeric_columns, key="map_size")
        
        # Create map
        if color_col:
            fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col, color=df[color_col] if color_col else None,
                                  size=df[size_col] if size_col else None, zoom=1, height=600,
                                  mapbox_style="open-street-map")
        else:
            fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col, zoom=1, height=600,
                                  mapbox_style="open-street-map")
        st.plotly_chart(fig)
        
        # NEW: Density map
        st.subheader("Density Heatmap")
        fig = px.density_mapbox(df, lat=lat_col, lon=lon_col, z=size_col if size_col else None,
                              radius=10, zoom=1, height=600, mapbox_style="open-street-map")
        st.plotly_chart(fig)
    else:
        st.info("No latitude/longitude columns detected. Please upload data with geospatial information.")
        
        # NEW: Sample map for demonstration
        st.subheader("Sample Geospatial Visualization")
        sample_df = pd.DataFrame({
            'lat': np.random.uniform(20, 60, 100),
            'lon': np.random.uniform(-120, 20, 100),
            'value': np.random.randint(1, 100, 100)
        })
        fig = px.scatter_mapbox(sample_df, lat='lat', lon='lon', color='value',
                              zoom=1, height=600, mapbox_style="open-street-map")
        st.plotly_chart(fig)

# --- Tab 6: Machine Learning (NEW) ---
with tab6:
    st.header("Machine Learning Insights")
    
    # NEW: K-Means Clustering
    st.subheader("K-Means Clustering")
    cluster_cols = st.multiselect("Select Features for Clustering", numeric_columns, 
                                 default=numeric_columns[:3], key="cluster_cols")
    n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="n_clusters")
    
    if cluster_cols and len(cluster_cols) >= 2:
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[cluster_cols])
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to dataframe
        df_cluster = df.copy()
        df_cluster['Cluster'] = clusters
        
        # Visualize clusters in 2D
        if len(cluster_cols) >= 2:
            fig = px.scatter(df_cluster, x=cluster_cols[0], y=cluster_cols[1], 
                          color='Cluster', title="K-Means Clustering Result")
            st.plotly_chart(fig)
        
        # Visualize in 3D
        if len(cluster_cols) >= 3:
            fig = px.scatter_3d(df_cluster, x=cluster_cols[0], y=cluster_cols[1], z=cluster_cols[2],
                              color='Cluster', title="3D Cluster Visualization")
            st.plotly_chart(fig)
        
        # Cluster statistics
        st.write("Cluster Centers:")
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_cols)
        centers.index.name = 'Cluster'
        st.write(centers)
        
        # Cluster distribution
        st.write("Cluster Distribution:")
        fig, ax = plt.subplots(figsize=(8, 4))
        df_cluster['Cluster'].value_counts().sort_index().plot(kind='bar', ax=ax)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    # NEW: Anomaly Detection
    st.subheader("Anomaly Detection")
    anomaly_col = st.selectbox("Column for Anomaly Detection", numeric_columns, key="anomaly_col")
    threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, key="anomaly_threshold")
    
    if anomaly_col:
        z_scores = np.abs(stats.zscore(df[anomaly_col].dropna()))
        outliers = df[z_scores > threshold]
        
        st.write(f"Detected {len(outliers)} anomalies based on Z-score > {threshold}")
        
        # Visualize anomalies
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(range(len(df)), df[anomaly_col], c='blue', alpha=0.5, label='Normal')
        ax.scatter(outliers.index, outliers[anomaly_col], c='red', label='Anomaly')
        ax.set_xlabel('Index')
        ax.set_ylabel(anomaly_col)
        ax.legend()
        st.pyplot(fig)
        
        if len(outliers) > 0:
            st.write("Anomaly Data Points:")
            st.write(outliers)

# Export Functionality
st.sidebar.subheader("Export Options")
export_format = st.sidebar.selectbox("Export Format", ["PNG", "PDF", "CSV"], key="export_format")
if st.sidebar.button("Export Current Visualization"):
    if export_format == "CSV":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="mining_data.csv">Download CSV</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format=export_format.lower(), dpi=300)
        b64 = base64.b64encode(buf.getvalue()).decode()
        href = f'<a href="data:image/{export_format.lower()};base64,{b64}" download="visualization.{export_format.lower()}">Download {export_format}</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown(f"<footer style='text-align: center; color: gray;'>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</footer>", unsafe_allow_html=True)
