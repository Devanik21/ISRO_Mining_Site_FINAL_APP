import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io
import base64
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set page configuration
st.set_page_config(page_title="Advanced Mining Visualization", layout="wide", initial_sidebar_state="expanded")

# Custom styling
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .reporting-selector {background-color: #262730; border-radius: 5px; padding: 10px;}
    h1, h2, h3 {color: #4CAF50;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title("🛠️ Visualization Controls")
    st.markdown("---")
    
    with st.expander("🎨 Theme Settings", expanded=True):
        theme = st.selectbox("Plot Theme", ["dark", "darkgrid", "whitegrid", "ticks"], index=0)
        color_palette = st.selectbox("Color Palette", 
                                   ["viridis", "plasma", "inferno", "magma", "cividis", "rocket", "mako", "flare", "crest"], 
                                   index=0)
        font_size = st.slider("Font Size", 8, 16, 12)
        
    with st.expander("💾 Export Options", expanded=True):
        export_format = st.selectbox("Export Format", ["PNG", "SVG", "PDF", "CSV"], index=0)
        if st.button("Export Current Visualization", key="export_btn"):
            if export_format == "CSV":
                if 'df' in locals():
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="mining_data.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)

# Apply theme settings
sns.set_theme(style=theme)
sns.set_palette(color_palette)
plt.rcParams.update({'font.size': font_size})

# Load dataset with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("space_mining_dataset.csv")
        return df
    except FileNotFoundError:
        st.info("Dataset not found. Please upload a CSV file.")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            return df
        # Generate sample data if no file is uploaded
        return generate_sample_data()

def generate_sample_data(n=1000):
    np.random.seed(42)
    sites = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']
    minerals = ['Iron', 'Copper', 'Gold', 'Silver', 'Platinum', 'Titanium']
    risk_levels = ['Low', 'Medium', 'High']
    
    data = {
        'site_id': np.random.choice(sites, n),
        'depth_m': np.random.normal(500, 150, n),
        'mineral_content_pct': np.random.normal(35, 15, n),
        'extraction_cost': np.random.normal(5000, 2000, n),
        'yield_tons': np.random.normal(200, 75, n),
        'purity_pct': np.random.normal(80, 10, n),
        'primary_mineral': np.random.choice(minerals, n),
        'risk_level': np.random.choice(risk_levels, n, p=[0.5, 0.3, 0.2]),
        'processing_time_hrs': np.random.normal(12, 4, n),
        'energy_consumption_kwh': np.random.normal(8000, 2500, n)
    }
    
    df = pd.DataFrame(data)
    df['revenue'] = df['yield_tons'] * df['purity_pct'] * np.random.normal(100, 20, n)
    df['profit'] = df['revenue'] - df['extraction_cost']
    df['efficiency_score'] = df['profit'] / df['energy_consumption_kwh']
    
    return df

df = load_data()
if df is None:
    st.stop()

# Data preprocessing
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Calculate key metrics for dashboard
if 'profit' in df.columns and 'yield_tons' in df.columns:
    total_profit = df['profit'].sum()
    total_yield = df['yield_tons'].sum()
    avg_purity = df['purity_pct'].mean() if 'purity_pct' in df.columns else 0
    roi = (df['profit'].sum() / df['extraction_cost'].sum() * 100) if 'extraction_cost' in df.columns else 0
else:
    total_profit = 0
    total_yield = 0
    avg_purity = 0
    roi = 0

# Dashboard header with key metrics
st.title("🚀 Advanced Mining Site Visualization")
st.markdown("Comprehensive data analysis platform for mining operations with interactive visualizations")

# Key metrics in columns
metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Total Profit", f"${total_profit:,.0f}")
with metric_cols[1]:
    st.metric("Total Yield", f"{total_yield:,.1f} tons")
with metric_cols[2]:
    st.metric("Avg. Purity", f"{avg_purity:.1f}%")
with metric_cols[3]:
    st.metric("ROI", f"{roi:.1f}%")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visual Insights",
    "📈 Detailed Analysis",
    "🔍 3D Exploration",
    "📉 Statistical Insights"
])

# Tab 1: Visual Insights
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Revenue vs Cost by Site")
        if all(col in df.columns for col in ['site_id', 'revenue', 'extraction_cost']):
            site_summary = df.groupby('site_id').agg({
                'revenue': 'sum',
                'extraction_cost': 'sum'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=site_summary['site_id'], y=site_summary['revenue'], 
                                name='Revenue', marker_color='#72B7B2'))
            fig.add_trace(go.Bar(x=site_summary['site_id'], y=site_summary['extraction_cost'], 
                                name='Cost', marker_color='#F2A5A5'))
            
            fig.update_layout(
                barmode='group',
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Mineral Distribution")
        if 'primary_mineral' in df.columns:
            mineral_counts = df['primary_mineral'].value_counts().reset_index()
            mineral_counts.columns = ['mineral', 'count']
            
            fig = px.pie(mineral_counts, values='count', names='mineral', 
                        color_discrete_sequence=px.colors.sequential.Viridis)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("Depth vs Yield Heatmap")
        if all(col in df.columns for col in ['depth_m', 'yield_tons']):
            fig, ax = plt.subplots(figsize=(10, 6))
            hb = ax.hexbin(df['depth_m'], df['yield_tons'], gridsize=20, 
                        cmap='viridis', mincnt=1)
            cb = plt.colorbar(hb, ax=ax)
            cb.set_label('Count')
            ax.set_xlabel('Depth (m)')
            ax.set_ylabel('Yield (tons)')
            ax.set_title('Depth vs Yield Distribution')
            st.pyplot(fig)
    
    with col4:
        st.subheader("Efficiency by Risk Level")
        if all(col in df.columns for col in ['risk_level', 'efficiency_score']):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='risk_level', y='efficiency_score', data=df, palette="viridis", ax=ax)
            ax.set_title('Efficiency Score by Risk Level')
            ax.set_xlabel('Risk Level')
            ax.set_ylabel('Efficiency Score')
            st.pyplot(fig)

# Tab 2: Detailed Analysis
with tab2:
    st.subheader("Custom Analysis")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        plot_type = st.selectbox("Plot Type", [
            "Scatter Plot", "Bar Chart", "Violin Plot", 
            "Correlation Heatmap", "Pairplot"
        ])
    
    with col2:
        x_var = st.selectbox("X Variable", numeric_columns + categorical_columns)
    
    with col3:
        y_var = st.selectbox("Y Variable", numeric_columns)
    
    if plot_type == "Scatter Plot":
        color_var = st.selectbox("Color By", [None] + categorical_columns)
        fig = px.scatter(df, x=x_var, y=y_var, color=color_var,
                        title=f"{x_var} vs {y_var}",
                        template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Bar Chart":
        if x_var in categorical_columns:
            agg_func = st.selectbox("Aggregation", ["mean", "sum", "count", "median"])
            df_grouped = df.groupby(x_var)[y_var].agg(agg_func).reset_index()
            fig = px.bar(df_grouped, x=x_var, y=y_var, 
                         title=f"{agg_func.capitalize()} of {y_var} by {x_var}",
                         template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("X variable should be categorical for bar charts")
    
    elif plot_type == "Violin Plot":
        split_var = st.selectbox("Split By", [None] + categorical_columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        if split_var:
            sns.violinplot(x=x_var, y=y_var, hue=split_var, data=df, split=True, ax=ax)
        else:
            sns.violinplot(x=x_var, y=y_var, data=df, ax=ax)
        ax.set_title(f"{x_var} vs {y_var}")
        st.pyplot(fig)
    
    elif plot_type == "Correlation Heatmap":
        selected_cols = st.multiselect("Select Columns", numeric_columns, default=numeric_columns[:6])
        if selected_cols:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[selected_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='viridis', 
                       fmt='.2f', linewidths=.5, ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
    
    elif plot_type == "Pairplot":
        selected_cols = st.multiselect("Select Columns (max 4)", numeric_columns, default=numeric_columns[:3])
        hue_var = st.selectbox("Hue Variable", [None] + categorical_columns)
        if len(selected_cols) > 0:
            fig = sns.pairplot(df[selected_cols + ([hue_var] if hue_var else [])], 
                              hue=hue_var, height=2.5, diag_kind="kde")
            st.pyplot(fig.fig)

# Tab 3: 3D Exploration
with tab3:
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("3D Visualization Parameters")
        x_3d = st.selectbox("X-axis", numeric_columns, key="3d_x")
        y_3d = st.selectbox("Y-axis", numeric_columns, key="3d_y", index=1 if len(numeric_columns) > 1 else 0)
        z_3d = st.selectbox("Z-axis", numeric_columns, key="3d_z", index=2 if len(numeric_columns) > 2 else 0)
        color_3d = st.selectbox("Color by", [None] + categorical_columns + numeric_columns, key="3d_color")
        
        if st.button("Apply K-Means Clustering"):
            if all(col in df.columns for col in [x_3d, y_3d, z_3d]):
                features = df[[x_3d, y_3d, z_3d]].dropna()
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                
                n_clusters = 3
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df['cluster'] = kmeans.fit_predict(scaled_features)
                color_3d = 'cluster'
                st.success(f"Applied K-means clustering with {n_clusters} clusters")
    
    with col1:
        st.subheader("3D Data Exploration")
        if all(col in df.columns for col in [x_3d, y_3d, z_3d]):
            fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color=color_3d,
                              title=f"3D Visualization: {x_3d} vs {y_3d} vs {z_3d}")
            
            fig.update_layout(
                scene=dict(
                    xaxis_title=x_3d,
                    yaxis_title=y_3d,
                    zaxis_title=z_3d,
                ),
                template='plotly_dark',
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 4: Statistical Insights
with tab4:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Descriptive Statistics")
        stats_col = st.multiselect("Select Columns", numeric_columns, default=numeric_columns[:4])
        if stats_col:
            st.dataframe(df[stats_col].describe().T.style.format("{:.2f}"))
    
    with col2:
        st.subheader("Distribution Analysis")
        dist_col = st.selectbox("Select Column", numeric_columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[dist_col], kde=True, bins=30, color='purple', ax=ax)
        ax.set_title(f"Distribution of {dist_col}")
        
        # Add descriptive stats to the plot
        stats_text = (f"Mean: {df[dist_col].mean():.2f}\n"
                    f"Median: {df[dist_col].median():.2f}\n"
                    f"Std Dev: {df[dist_col].std():.2f}\n"
                    f"Skew: {df[dist_col].skew():.2f}")
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)
        
        st.pyplot(fig)
    
    st.subheader("Principal Component Analysis")
    pca_cols = st.multiselect("Select Features for PCA", numeric_columns, default=numeric_columns[:5])
    
    if pca_cols and len(pca_cols) >= 2:
        n_components = min(3, len(pca_cols))
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[pca_cols])
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create column names for the transformed data
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
        
        if 'primary_mineral' in df.columns:
            pca_df['primary_mineral'] = df['primary_mineral'].values
        
        explained_var = pca.explained_variance_ratio_ * 100
        
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='primary_mineral' if 'primary_mineral' in pca_df else None,
                        title=f"PCA: {explained_var[0]:.1f}% & {explained_var[1]:.1f}% variance explained",
                        template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show feature importance
        loadings = pd.DataFrame(
            pca.components_.T, 
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=pca_cols
        )
        
        st.subheader("Feature Importance in Principal Components")
        st.dataframe(loadings.style.background_gradient(cmap='viridis').format("{:.3f}"))

# Footer
st.markdown(f"<footer style='text-align: center; color: gray; margin-top: 50px;'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Advanced Mining Dashboard v2.0</footer>", unsafe_allow_html=True)
