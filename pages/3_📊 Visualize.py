import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data class for configuration
@dataclass
class DashboardConfig:
    title: str = "🌌 Next-Gen Space Mining Analytics Platform 🌠"
    theme: str = "plotly_dark"
    layout: str = "wide"
    cache_ttl: int = 3600  # Cache time-to-live in seconds

# Utility Functions
def measure_time(func):
    """Decorator to measure execution time of functions."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} executed in {time.time() - start_time:.2f} seconds")
        return result
    return wrapper

@st.cache_data(ttl=DashboardConfig.cache_ttl)
@measure_time
def load_data() -> pd.DataFrame:
    """Load or generate space mining dataset with advanced error handling."""
    try:
        df = pd.read_csv("space_mining_dataset.csv")
        logger.info("Dataset loaded from file.")
    except FileNotFoundError:
        logger.warning("Dataset not found, generating synthetic data.")
        celestial_bodies = ['Mars', 'Moon', 'Asteroid Belt', 'Europa', 'Titan', 'Ceres', 'Ganymede']
        elements = ['Iron', 'Titanium', 'Platinum', 'Water', 'Helium-3', 'Rare Earth Elements', 'Silicates']
        np.random.seed(42)  # For reproducibility
        sample_data = {
            'Celestial Body': np.random.choice(celestial_bodies, 1000),
            'Element': np.random.choice(elements, 1000),
            'Element Concentration': np.random.lognormal(mean=2, sigma=1, size=1000),
            'Mining Difficulty': np.random.gamma(shape=2, scale=2, size=1000),
            'Extraction Cost': np.random.exponential(scale=5000, size=1000),
            'Resource Value': np.random.exponential(scale=10000, size=1000),
            'Distance (AU)': np.random.uniform(0.5, 50, 1000),
            'Environmental Risk': np.random.uniform(1, 5, 1000),
            'Site Age (M Years)': np.random.uniform(10, 5000, 1000),
            'Gravity (g)': np.random.uniform(0.1, 2.5, 1000)
        }
        df = pd.DataFrame(sample_data)
    return df

@st.cache_resource
@measure_time
def train_ml_model(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestRegressor, float]:
    """Train a RandomForest model and return it with its R2 score."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"RandomForest model trained with R2 score: {r2:.3f}")
    return model, r2

# Visualization Class
class SpaceMiningVisualizer:
    def __init__(self, df: pd.DataFrame, config: DashboardConfig):
        self.df = df
        self.config = config
        self.numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
    def setup_ui(self):
        """Configure Streamlit page and apply custom CSS."""
        #st.set_page_config(layout=self.config.layout, page_title=self.config.title, page_icon="🌠")
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            color: #00E5FF;
            text-align: center;
            margin-bottom: 1.5rem;
            text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
        }
        .sub-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #00B0FF;
            margin: 1.5rem 0 1rem 0;
            border-bottom: 2px solid #00B0FF;
            padding-bottom: 0.3rem;
        }
        .card {
            padding: 1.5rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #1A237E, #283593);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            margin-bottom: 1.5rem;
            color: #FFFFFF;
        }
        .metric-box {
            background: #3F51B5;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="main-header">{self.config.title}</div>', unsafe_allow_html=True)

    @measure_time
    def render_overview(self):
        """Render dataset overview with advanced statistics."""
        with st.expander("Dataset Intelligence Report", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="sub-header">Statistical Summary</div>', unsafe_allow_html=True)
                st.dataframe(self.df.describe().style.format("{:.2f}"))
            with col2:
                st.markdown('<div class="sub-header">Data Preview</div>', unsafe_allow_html=True)
                st.dataframe(self.df.head())
            with col3:
                st.markdown('<div class="sub-header">Data Health</div>', unsafe_allow_html=True)
                missing = self.df.isnull().sum()
                fig = px.bar(x=missing.index, y=missing.values, title="Missing Values", template=self.config.theme)
                st.plotly_chart(fig, use_container_width=True)

    def apply_filters(self) -> pd.DataFrame:
        """Apply interactive filters and return filtered DataFrame."""
        st.markdown('<div class="sub-header">Exploration Filters</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            bodies = st.multiselect("Celestial Bodies", sorted(self.df['Celestial Body'].unique()), default=sorted(self.df['Celestial Body'].unique()))
        with col2:
            metric = st.selectbox("Metric Filter", self.numeric_cols)
            min_val, max_val = float(self.df[metric].min()), float(self.df[metric].max())
            range_val = st.slider(f"{metric} Range", min_val, max_val, (min_val, max_val))
        with col3:
            sort_by = st.selectbox("Sort By", self.numeric_cols, index=self.numeric_cols.index('Resource Value'))
            sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

        filtered_df = self.df[self.df['Celestial Body'].isin(bodies)]
        filtered_df = filtered_df[(filtered_df[metric] >= range_val[0]) & (filtered_df[metric] <= range_val[1])]
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=(sort_order == "Ascending"))
        st.markdown(f"**Analyzing {len(filtered_df)} of {len(self.df)} sites**")
        return filtered_df

    @measure_time
    def render_key_visualizations(self, filtered_df: pd.DataFrame):
        """Render advanced interactive visualizations."""
        st.markdown('<div class="sub-header">Galactic Insights</div>', unsafe_allow_html=True)
        
        # 3D Visualization with Enhanced Features
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card">3D Cosmic Map</div>', unsafe_allow_html=True)
            x, y, z = [st.selectbox(f"{axis}-Axis", self.numeric_cols, index=i) for axis, i in zip(['X', 'Y', 'Z'], [0, 1, 2])]
            fig = px.scatter_3d(filtered_df, x=x, y=y, z=z, color='Celestial Body', size='Resource Value',
                                opacity=0.8, template=self.config.theme, hover_name='Celestial Body',
                                hover_data=self.numeric_cols)
            fig.update_layout(height=600, scene=dict(aspectmode="data"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="card">Profitability Nexus</div>', unsafe_allow_html=True)
            size_var = st.selectbox("Size Metric", self.numeric_cols, index=self.numeric_cols.index('Mining Difficulty'))
            fig = px.scatter(filtered_df, x='Extraction Cost', y='Resource Value', color='Celestial Body', size=size_var,
                             trendline="ols", template=self.config.theme, title="Economic Viability Matrix")
            max_val = max(filtered_df[['Extraction Cost', 'Resource Value']].max())
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(color='red', dash='dash'), name='Break-even'))
            st.plotly_chart(fig, use_container_width=True)

        # Advanced Distribution and Correlation
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="card">Distribution Cosmos</div>', unsafe_allow_html=True)
            dist_var = st.selectbox("Distribution Metric", self.numeric_cols)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Density Map', 'Statistical Spread'))
            for body in filtered_df['Celestial Body'].unique():
                data = filtered_df[filtered_df['Celestial Body'] == body][dist_var].dropna()
                if len(data) > 5:
                    fig.add_trace(go.Histogram(x=data, name=body, opacity=0.6, histnorm='probability density'), row=1, col=1)
                    kde = stats.gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), name=f"{body} KDE"), row=1, col=1)
            fig.add_trace(go.Box(x=filtered_df['Celestial Body'], y=filtered_df[dist_var], boxmean=True), row=2, col=1)
            fig.update_layout(height=600, template=self.config.theme)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown('<div class="card">Correlation Galaxy</div>', unsafe_allow_html=True)
            corr_vars = st.multiselect("Correlation Metrics", self.numeric_cols, default=self.numeric_cols[:4])
            if len(corr_vars) > 1:
                corr_matrix = filtered_df[corr_vars].corr()
                fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, template=self.config.theme)
                st.plotly_chart(fig, use_container_width=True)

    @measure_time
    def render_advanced_analysis(self, filtered_df: pd.DataFrame):
        """Render cutting-edge statistical analysis."""
        st.markdown('<div class="sub-header">Quantum Analysis Suite</div>', unsafe_allow_html=True)
        features = st.multiselect("Analysis Features", self.numeric_cols, default=self.numeric_cols[:3])
        
        if len(features) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card">PCA Space Reduction</div>', unsafe_allow_html=True)
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(filtered_df[features])
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(scaled_data)
                pca_df = pd.DataFrame({'PCA1': pca_result[:, 0], 'PCA2': pca_result[:, 1], 'Celestial Body': filtered_df['Celestial Body']})
                explained_var = pca.explained_variance_ratio_ * 100
                fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='Celestial Body', template=self.config.theme,
                                 title=f"PCA ({explained_var[0]:.1f}% + {explained_var[1]:.1f}% Variance)")
                for i, feature in enumerate(features):
                    fig.add_annotation(x=pca.components_[0, i] * 5, y=pca.components_[1, i] * 5, text=feature, showarrow=True)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown('<div class="card">K-Means Cluster Constellation</div>', unsafe_allow_html=True)
                n_clusters = st.slider("Cluster Count", 2, 10, 3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                filtered_df['Cluster'] = kmeans.fit_predict(scaled_data)
                centers = scaler.inverse_transform(kmeans.cluster_centers_)
                cluster_df = pd.DataFrame(centers, columns=features)
                fig = px.scatter_3d(filtered_df, x=features[0], y=features[1], z=features[2], color='Cluster', symbol='Celestial Body',
                                    template=self.config.theme, title=f"{n_clusters}-Cluster Analysis")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(cluster_df.style.format("{:.2f}"))

    @measure_time
    def render_predictive_insights(self, filtered_df: pd.DataFrame):
        """Render AI-driven predictive analytics."""
        st.markdown('<div class="sub-header">AI Prediction Core</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">Regression Hyperplane</div>', unsafe_allow_html=True)
            target = st.selectbox("Prediction Target", self.numeric_cols, index=self.numeric_cols.index('Resource Value'))
            predictors = st.multiselect("Predictors", [col for col in self.numeric_cols if col != target], default=self.numeric_cols[:3])
            if predictors:
                X = filtered_df[predictors]
                y = filtered_df[target]
                model, r2 = train_ml_model(X, y)
                filtered_df['Predicted'] = model.predict(X)
                fig = px.scatter(filtered_df, x='Predicted', y=target, color='Celestial Body', template=self.config.theme,
                                 title=f"Predicted vs Actual (R² = {r2:.3f})")
                min_val, max_val = min(filtered_df[[target, 'Predicted']].min()), max(filtered_df[[target, 'Predicted']].max())
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(color='red', dash='dash')))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="card">Real-Time Prediction Engine</div>', unsafe_allow_html=True)
            inputs = {var: st.slider(var, float(filtered_df[var].min()), float(filtered_df[var].max()), float(filtered_df[var].mean())) 
                      for var in predictors}
            if inputs:
                input_df = pd.DataFrame([inputs])
                prediction = model.predict(input_df)[0]
                st.markdown(f"<h3 style='text-align: center; color: #00E5FF;'>Predicted {target}: {prediction:.2f}</h3>", unsafe_allow_html=True)
                importances = pd.DataFrame({'Feature': predictors, 'Importance': model.feature_importances_ * 100})
                fig = px.bar(importances, x='Feature', y='Importance', template=self.config.theme, title="Feature Impact (%)")
                st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Execute the dashboard."""
        self.setup_ui()
        self.render_overview()
        filtered_df = self.apply_filters()
        tabs = st.tabs(["📊 Galactic Insights", "🔍 Quantum Analysis", "🤖 AI Predictions"])
        with tabs[0]:
            self.render_key_visualizations(filtered_df)
        with tabs[1]:
            self.render_advanced_analysis(filtered_df)
        with tabs[2]:
            self.render_predictive_insights(filtered_df)

# Main Execution
if __name__ == "__main__":
    config = DashboardConfig()
    df = load_data()
    visualizer = SpaceMiningVisualizer(df, config)
    visualizer.run()
