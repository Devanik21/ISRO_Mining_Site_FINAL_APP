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
tab1, tab2, tab3, tab4 = st.tabs(["Basic Visualizations", "Advanced Visualizations", "3D & Animations", "Statistical Insights"])

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
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"{x_axis} vs {y_axis}")
    st.pyplot(fig)

    # Histogram with KDE and stats
    st.subheader("Histogram with KDE")
    hist_col = st.selectbox("Column", numeric_columns, key="hist")
    bins = st.slider("Number of Bins", 10, 100, 30, key="hist_bins")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[hist_col], kde=True, bins=bins, color='purple', stat="density", ax=ax)
    ax.set_xlabel(hist_col)
    ax.set_title(f"Distribution of {hist_col}")
    st.pyplot(fig)
    st.write(f"Mean: {df[hist_col].mean():.2f}, Std: {df[hist_col].std():.2f}")

    # Pie Chart with explode option
    st.subheader("Pie Chart")
    pie_col = st.selectbox("Categorical Column", categorical_columns, key="pie")
    explode = st.checkbox("Explode Slices", key="pie_explode")
    pie_data = df[pie_col].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    explode_vals = [0.1 if explode else 0] * len(pie_data)
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, 
           explode=explode_vals, colors=sns.color_palette(palette))
    ax.set_title(f"Distribution of {pie_col}")
    st.pyplot(fig)

    # Boxplot with outlier detection
    st.subheader("Boxplot")
    box_x = st.selectbox("Categorical Column", categorical_columns, key="box_x")
    box_y = st.selectbox("Numeric Column", numeric_columns, key="box_y")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=df[box_x], y=df[box_y], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title(f"{box_x} vs {box_y}")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

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
    ax.set_title(f"{violin_x} vs {violin_y}")
    st.pyplot(fig)

    # Swarm Plot with dodge
    st.subheader("Swarm Plot")
    swarm_x = st.selectbox("Categorical Column", categorical_columns, key="swarm_x")
    swarm_y = st.selectbox("Numeric Column", numeric_columns, key="swarm_y")
    dodge = st.checkbox("Dodge by Hue", key="swarm_dodge")
    hue = st.selectbox("Hue (optional)", [None] + categorical_columns, key="swarm_hue") if dodge else None
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.swarmplot(x=df[swarm_x], y=df[swarm_y], hue=df[hue] if hue else None, dodge=dodge, ax=ax)
    ax.set_title(f"{swarm_x} vs {swarm_y}")
    st.pyplot(fig)

    # Joint Plot with regression
    st.subheader("Joint Plot")
    joint_x = st.selectbox("X-axis", numeric_columns, key="joint_x")
    joint_y = st.selectbox("Y-axis", numeric_columns, key="joint_y")
    kind = st.selectbox("Kind", ["scatter", "hex", "kde", "reg"], key="joint_kind")
    fig = sns.jointplot(x=df[joint_x], y=df[joint_y], kind=kind, color="purple")
    st.pyplot(fig.figure)

    # Clustered Heatmap
    st.subheader("Clustered Heatmap")
    fig = sns.clustermap(df[numeric_columns].corr(), cmap="coolwarm", annot=True, figsize=(10, 10))
    st.pyplot(fig.figure)

    # Facet Grid
    st.subheader("Facet Grid")
    facet_col = st.selectbox("Categorical Column", categorical_columns, key="facet_col")
    facet_num = st.selectbox("Numeric Column", numeric_columns, key="facet_num")
    g = sns.FacetGrid(df, col=facet_col, col_wrap=3, height=4)
    g.map(sns.histplot, facet_num, bins=20)
    st.pyplot(g.fig)

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

    # Animated Scatter Plot
    st.subheader("Animated Scatter Plot")
    anim_x = st.selectbox("X-axis", numeric_columns, key="anim_x")
    anim_y = st.selectbox("Y-axis", numeric_columns, key="anim_y")
    anim_frame = st.selectbox("Animation Frame", categorical_columns + numeric_columns, key="anim_frame")
    fig_anim = px.scatter(df, x=anim_x, y=anim_y, animation_frame=anim_frame, 
                          range_x=[df[anim_x].min(), df[anim_x].max()], 
                          range_y=[df[anim_y].min(), df[anim_y].max()])
    st.plotly_chart(fig_anim)

    # 3D Surface Plot
    st.subheader("3D Surface Plot")
    surf_x = st.selectbox("X-axis", numeric_columns, key="surf_x")
    surf_y = st.selectbox("Y-axis", numeric_columns, key="surf_y")
    surf_z = st.selectbox("Z-axis", numeric_columns, key="surf_z")
    X, Y = np.meshgrid(df[surf_x], df[surf_y])
    Z = np.array(df[surf_z]).reshape(X.shape[0], -1)[:X.shape[0], :X.shape[1]]
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title=f"3D Surface: {surf_z}", scene=dict(xaxis_title=surf_x, yaxis_title=surf_y, zaxis_title=surf_z))
    st.plotly_chart(fig)

# --- Tab 4: Statistical Insights ---
with tab4:
    st.header("Statistical Insights")

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    stats_col = st.multiselect("Select Columns", numeric_columns, default=numeric_columns[:3], key="stats_col")
    if stats_col:
        st.write(df[stats_col].describe())

    # PCA Analysis
    st.subheader("Principal Component Analysis (PCA)")
    pca_cols = st.multiselect("Select Features for PCA", numeric_columns, default=numeric_columns[:5], key="pca_cols")
    if pca_cols and len(pca_cols) >= 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[pca_cols])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(pca_result[:, 0], pca_result[:, 1])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA: 2D Projection")
        st.pyplot(fig)
        st.write(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

    # Statistical Tests
    st.subheader("Statistical Tests")
    test_x = st.selectbox("Variable 1", numeric_columns, key="test_x")
    test_y = st.selectbox("Variable 2", numeric_columns, key="test_y")
    test_type = st.selectbox("Test Type", ["T-Test", "Correlation", "ANOVA"], key="test_type")
    if test_type == "T-Test":
        t_stat, p_val = stats.ttest_ind(df[test_x].dropna(), df[test_y].dropna())
        st.write(f"T-Statistic: {t_stat:.4f}, P-Value: {p_val:.4f}")
    elif test_type == "Correlation":
        corr, p_val = stats.pearsonr(df[test_x].dropna(), df[test_y].dropna())
        st.write(f"Pearson Correlation: {corr:.4f}, P-Value: {p_val:.4f}")
    elif test_type == "ANOVA":
        if categorical_columns:
            anova_col = st.selectbox("Categorical Column", categorical_columns, key="anova_col")
            groups = [df[test_x][df[anova_col] == cat].dropna() for cat in df[anova_col].unique()]
            f_stat, p_val = stats.f_oneway(*groups)
            st.write(f"F-Statistic: {f_stat:.4f}, P-Value: {p_val:.4f}")

    # Outlier Detection
    st.subheader("Outlier Detection")
    outlier_col = st.selectbox("Column", numeric_columns, key="outlier_col")
    z_scores = np.abs(stats.zscore(df[outlier_col].dropna()))
    threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, key="outlier_threshold")
    outliers = df[outlier_col][z_scores > threshold]
    st.write(f"Outliers: {len(outliers)} found")
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
st.markdown(f"<footer style='text-align: center; color: gray;'>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Grok 3 (xAI)</footer>", unsafe_allow_html=True)
