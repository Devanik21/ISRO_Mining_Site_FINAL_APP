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

def show_visualize_page():
    # Page Configuration
    st.set_page_config(layout="wide", page_title="Space Mining Analytics", page_icon="🌠")
    
    # Custom CSS to enhance the UI
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #6200EA;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #6200EA, #B388FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #304FFE;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Page Header
    st.markdown('<div class="main-header">🌌 Advanced Space Mining Analytics Dashboard 🌠</div>', unsafe_allow_html=True)
    
    # Data Loading with Error Handling
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv("space_mining_dataset.csv")
            return df
        except FileNotFoundError:
            # Generate sample data if file is not found
            celestial_bodies = ['Mars', 'Moon', 'Asteroid Belt', 'Europa', 'Titan', 'Ceres', 'Ganymede']
            elements = ['Iron', 'Titanium', 'Platinum', 'Water', 'Helium-3', 'Rare Earth Elements', 'Silicates']
            
            sample_data = {
                'Celestial Body': np.random.choice(celestial_bodies, 100),
                'Element Concentration': np.random.uniform(0.1, 15, 100),
                'Mining Difficulty': np.random.uniform(1, 10, 100),
                'Extraction Cost': np.random.uniform(500, 10000, 100),
                'Resource Value': np.random.uniform(1000, 50000, 100),
                'Distance': np.random.uniform(0.5, 50, 100),
                'Environmental Risk': np.random.uniform(1, 5, 100),
                'Site Age (Million Years)': np.random.uniform(10, 500, 100),
                'Gravity': np.random.uniform(0.1, 2.5, 100)
            }
            return pd.DataFrame(sample_data)
    
    with st.spinner('Loading dataset...'):
        df = load_data()
    
    # Display basic dataset information
    with st.expander("Dataset Overview", expanded=False):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="sub-header">Dataset Statistics</div>', unsafe_allow_html=True)
            st.dataframe(df.describe())
        with col2:
            st.markdown('<div class="sub-header">Data Sample</div>', unsafe_allow_html=True)
            st.dataframe(df.head())
            
        # Missing values analysis
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.markdown('<div class="sub-header">Missing Values</div>', unsafe_allow_html=True)
            st.bar_chart(missing_values[missing_values > 0])
    
    # Main Dashboard Layout
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Key Visualizations", "🔍 Advanced Analysis", "🤖 Predictive Insights"])
    
    with main_tab1:
        # Interactive Filters
        st.markdown('<div class="sub-header">Data Filters</div>', unsafe_allow_html=True)
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            selected_bodies = st.multiselect("Select Celestial Bodies", 
                                           options=sorted(df['Celestial Body'].unique()),
                                           default=sorted(df['Celestial Body'].unique()))
        with filter_col2:
            # Dynamic range selector based on numerical columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if numeric_cols:
                selected_metric = st.selectbox("Filter by Metric", numeric_cols)
                min_val, max_val = float(df[selected_metric].min()), float(df[selected_metric].max())
                metric_range = st.slider(f"{selected_metric} Range", min_val, max_val, (min_val, max_val))
        with filter_col3:
            sort_by = st.selectbox("Sort Results By", ['Resource Value', 'Extraction Cost', 'Mining Difficulty'])
            sort_order = st.radio("Sort Order", ["Descending", "Ascending"], horizontal=True)
        
        # Apply filters
        filtered_df = df[df['Celestial Body'].isin(selected_bodies)]
        if 'selected_metric' in locals():
            filtered_df = filtered_df[(filtered_df[selected_metric] >= metric_range[0]) & 
                                      (filtered_df[selected_metric] <= metric_range[1])]
        
        # Sort data
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=(sort_order=="Ascending"))
        
        # Display filtered data count
        st.markdown(f"**Showing {len(filtered_df)} sites out of {len(df)} total**")
        
        # Interactive Plotly Visualizations
        st.markdown('<div class="sub-header">Interactive Visualizations</div>', unsafe_allow_html=True)
        
        # Row 1: Key Metrics
        viz_row1_col1, viz_row1_col2 = st.columns(2)
        
        with viz_row1_col1:
            st.markdown('<div class="card">3D Scatter Plot</div>', unsafe_allow_html=True)
            x_axis = st.selectbox("X-axis", numeric_cols, index=0)
            y_axis = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            z_axis = st.selectbox("Z-axis", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)
            
            fig = px.scatter_3d(filtered_df, x=x_axis, y=y_axis, z=z_axis,
                              color='Celestial Body', size='Resource Value',
                              opacity=0.7, template='plotly_dark',
                              hover_name="Celestial Body",
                              hover_data=[col for col in filtered_df.columns if col != 'Celestial Body'])
            
            fig.update_layout(height=600, margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_row1_col2:
            st.markdown('<div class="card">Resource Value vs Extraction Cost</div>', unsafe_allow_html=True)
            bubble_size = st.selectbox("Bubble Size Represents", numeric_cols, 
                                     index=numeric_cols.index('Mining Difficulty') if 'Mining Difficulty' in numeric_cols else 0)
            
            fig = px.scatter(filtered_df, x='Extraction Cost', y='Resource Value', 
                           color='Celestial Body', size=bubble_size, 
                           hover_name="Celestial Body", template='plotly',
                           title="Mining Profitability Analysis",
                           labels={'Extraction Cost': 'Cost (Credits)', 'Resource Value': 'Value (Credits)'},
                           trendline="ols")
            
            # Add a reference line for profit threshold (Value = Cost)
            max_val = max(filtered_df['Extraction Cost'].max(), filtered_df['Resource Value'].max())
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', 
                                    line=dict(color='red', dash='dash'), 
                                    name='Break-even Line'))
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 2: More Advanced Visualizations
        viz_row2_col1, viz_row2_col2 = st.columns(2)
        
        with viz_row2_col1:
            st.markdown('<div class="card">Distribution Analysis</div>', unsafe_allow_html=True)
            dist_variable = st.selectbox("Select Variable for Distribution", numeric_cols)
            
            # Create subplot with histogram and boxplot
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.1, 
                              subplot_titles=('Histogram with KDE', 'Boxplot by Celestial Body'))
            
            # Histogram with KDE
            for body in selected_bodies:
                body_data = filtered_df[filtered_df['Celestial Body'] == body][dist_variable].dropna()
                if len(body_data) > 0:
                    # Histogram
                    fig.add_trace(go.Histogram(x=body_data, name=body, opacity=0.6, 
                                             histnorm='probability density'), row=1, col=1)
                    
                    # KDE (approximated with smoothed line)
                    if len(body_data) >= 5:  # Need sufficient data for KDE
                        kde = stats.gaussian_kde(body_data)
                        x_range = np.linspace(body_data.min(), body_data.max(), 1000)
                        fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines',
                                               name=f"{body} KDE", line=dict(width=2)), row=1, col=1)
            
            # Boxplot
            fig.add_trace(go.Box(x=filtered_df['Celestial Body'], y=filtered_df[dist_variable], 
                               name=dist_variable, boxmean=True), row=2, col=1)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_row2_col2:
            st.markdown('<div class="card">Correlation Matrix Heatmap</div>', unsafe_allow_html=True)
            
            # Multi-select for correlation columns
            corr_columns = st.multiselect("Select Variables for Correlation", 
                                         numeric_cols, 
                                         default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols)
            
            if len(corr_columns) > 1:
                # Calculate correlation matrix
                corr_matrix = filtered_df[corr_columns].corr()
                
                # Create heatmap
                fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                              zmin=-1, zmax=1, aspect="auto")
                fig.update_layout(height=600, title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least 2 variables to show correlations")
        
        # Row 3: Detailed Analysis
        st.markdown('<div class="sub-header">Detailed Resource Analysis</div>', unsafe_allow_html=True)
        
        if 'Celestial Body' in df.columns and 'Resource Value' in df.columns:
            # Grouped Bar Chart
            body_metrics = filtered_df.groupby('Celestial Body').agg({
                'Resource Value': 'mean',
                'Extraction Cost': 'mean',
                'Mining Difficulty': 'mean'
            }).reset_index()
            
            # Create a normalized version for comparison
            for col in ['Resource Value', 'Extraction Cost', 'Mining Difficulty']:
                if col in body_metrics.columns:
                    max_val = body_metrics[col].max()
                    if max_val > 0:  # Avoid division by zero
                        body_metrics[f'{col}_norm'] = body_metrics[col] / max_val
            
            fig = px.bar(body_metrics.melt(id_vars='Celestial Body', 
                                          value_vars=['Resource Value_norm', 'Extraction Cost_norm', 'Mining Difficulty_norm']
                                          if 'Resource Value_norm' in body_metrics.columns else
                                          ['Resource Value', 'Extraction Cost', 'Mining Difficulty']),
                        x='Celestial Body', y='value', color='variable', barmode='group',
                        title='Normalized Comparison Across Celestial Bodies',
                        labels={'value': 'Normalized Value (0-1)', 'variable': 'Metric'})
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Profitability Calculation
            if 'Resource Value' in filtered_df.columns and 'Extraction Cost' in filtered_df.columns:
                filtered_df['Profitability'] = filtered_df['Resource Value'] - filtered_df['Extraction Cost']
                filtered_df['ROI'] = (filtered_df['Profitability'] / filtered_df['Extraction Cost']).replace([np.inf, -np.inf], np.nan)
                
                profit_metrics = filtered_df.groupby('Celestial Body').agg({
                    'Profitability': ['mean', 'max', 'min', 'std'],
                    'ROI': ['mean', 'max']
                }).reset_index()
                
                st.markdown('<div class="card">Profitability Analysis by Celestial Body</div>', unsafe_allow_html=True)
                st.dataframe(profit_metrics)
    
    with main_tab2:
        st.markdown('<div class="sub-header">Advanced Statistical Analysis</div>', unsafe_allow_html=True)
        
        # Feature Selection for Advanced Analysis
        feature_cols = st.multiselect("Select Features for Analysis", 
                                    numeric_cols,
                                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)
        
        if len(feature_cols) >= 2:
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                # Dimensionality Reduction with PCA
                st.markdown('<div class="card">PCA Dimensionality Reduction</div>', unsafe_allow_html=True)
                
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(filtered_df[feature_cols])
                
                # Apply PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(scaled_data)
                
                # Create DataFrame for visualization
                pca_df = pd.DataFrame({
                    'PCA1': pca_result[:, 0],
                    'PCA2': pca_result[:, 1],
                    'Celestial Body': filtered_df['Celestial Body'].values
                })
                
                # Calculate explained variance
                explained_variance = pca.explained_variance_ratio_ * 100
                
                # Plot PCA
                fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='Celestial Body',
                               title=f"PCA Analysis (Explained Variance: {explained_variance[0]:.1f}% and {explained_variance[1]:.1f}%)",
                               template='plotly_white')
                
                # Add loadings (feature contributions)
                for i, feature in enumerate(feature_cols):
                    fig.add_annotation(
                        x=pca.components_[0, i] * 5,  # Scale for visibility
                        y=pca.components_[1, i] * 5,  # Scale for visibility
                        ax=0, ay=0,
                        xanchor="center", yanchor="center",
                        text=feature,
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explain PCA results
                st.markdown(f"""
                **PCA Analysis Insights:**
                - The two principal components explain {explained_variance.sum():.1f}% of total variance
                - PC1 ({explained_variance[0]:.1f}%) primarily represents: {feature_cols[np.argmax(abs(pca.components_[0]))]}
                - PC2 ({explained_variance[1]:.1f}%) primarily represents: {feature_cols[np.argmax(abs(pca.components_[1]))]}
                """)
            
            with adv_col2:
                # K-Means Clustering
                st.markdown('<div class="card">K-Means Clustering</div>', unsafe_allow_html=True)
                
                # Choose number of clusters
                n_clusters = st.slider("Number of Clusters", 2, 8, 3)
                
                # Apply K-Means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                filtered_df['Cluster'] = kmeans.fit_predict(scaled_data)
                
                # Cluster centers (transform back to original scale)
                centers = scaler.inverse_transform(kmeans.cluster_centers_)
                
                # Create a cluster profile table
                cluster_profiles = pd.DataFrame(centers, columns=feature_cols)
                cluster_profiles.index.name = 'Cluster'
                
                # Show 3D cluster plot
                if len(feature_cols) >= 3:
                    x_col, y_col, z_col = feature_cols[:3]
                    
                    fig = px.scatter_3d(filtered_df, x=x_col, y=y_col, z=z_col,
                                       color='Cluster', symbol='Celestial Body',
                                       title=f"K-Means Clustering ({n_clusters} clusters)",
                                       template='plotly_dark')
                    
                    # Add cluster centers
                    for i in range(n_clusters):
                        fig.add_trace(go.Scatter3d(
                            x=[centers[i, feature_cols.index(x_col)]],
                            y=[centers[i, feature_cols.index(y_col)]],
                            z=[centers[i, feature_cols.index(z_col)]],
                            mode='markers',
                            marker=dict(size=15, symbol='diamond', color=i, line=dict(width=2, color='black')),
                            name=f'Cluster {i} Center'
                        ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display cluster profiles
                st.markdown("**Cluster Profiles (Center Values)**")
                st.dataframe(cluster_profiles.round(2))
                
                # Cluster distribution by celestial body
                cluster_dist = pd.crosstab(filtered_df['Celestial Body'], filtered_df['Cluster'], 
                                          normalize='index') * 100
                
                st.markdown("**Cluster Distribution by Celestial Body (%)**")
                st.dataframe(cluster_dist.round(1))
    
    with main_tab3:
        st.markdown('<div class="sub-header">Predictive Analytics</div>', unsafe_allow_html=True)
        
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            # Regression Analysis
            st.markdown('<div class="card">Regression Model</div>', unsafe_allow_html=True)
            
            # Target and predictor selection
            target_var = st.selectbox("Target Variable (Y)", numeric_cols, 
                                     index=numeric_cols.index('Resource Value') if 'Resource Value' in numeric_cols else 0)
            
            predictor_vars = st.multiselect("Predictor Variables (X)", 
                                          [col for col in numeric_cols if col != target_var],
                                          default=[col for col in numeric_cols[:3] if col != target_var])
            
            if len(predictor_vars) > 0:
                # Prepare data
                X = filtered_df[predictor_vars]
                y = filtered_df[target_var]
                
                # Add constant for intercept
                X_with_const = sm.add_constant(X)
                
                # Fit model
                model = sm.OLS(y, X_with_const).fit()
                
                # Display model summary
                st.text(f"Model R-squared: {model.rsquared:.3f}")
                st.text(f"Model Adjusted R-squared: {model.rsquared_adj:.3f}")
                
                # Coefficient plot
                coef_data = pd.DataFrame({
                    'Variable': ['const'] + predictor_vars,
                    'Coefficient': model.params.values,
                    'StdError': model.bse.values
                })
                
                fig = px.bar(coef_data, x='Variable', y='Coefficient',
                           error_y='StdError', title=f"Regression Coefficients for {target_var}",
                           template='plotly_white')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Predictions vs. Actual
                filtered_df['Predicted'] = model.predict(X_with_const)
                
                fig = px.scatter(filtered_df, x='Predicted', y=target_var, 
                               color='Celestial Body',
                               labels={target_var: f'Actual {target_var}', 'Predicted': f'Predicted {target_var}'},
                               title=f"Actual vs. Predicted {target_var}")
                
                # Add a reference line (perfect predictions)
                min_val = min(filtered_df[target_var].min(), filtered_df['Predicted'].min())
                max_val = max(filtered_df[target_var].max(), filtered_df['Predicted'].max())
                
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                                       line=dict(color='black', dash='dash'), name='Perfect Prediction'))
                
                st.plotly_chart(fig, use_container_width=True)
        
        with pred_col2:
            # Interactive Prediction Tool
            st.markdown('<div class="card">Prediction Tool</div>', unsafe_allow_html=True)
            
            st.write("Adjust the parameters to predict resource value:")
            
            # Create sliders for each predictor
            input_values = {}
            for var in predictor_vars:
                min_val = float(filtered_df[var].min())
                max_val = float(filtered_df[var].max())
                mean_val = float(filtered_df[var].mean())
                
                input_values[var] = st.slider(
                    f"{var}", 
                    min_val, 
                    max_val, 
                    mean_val,
                    step=(max_val-min_val)/100
                )
            
            if len(input_values) > 0:
                # Create input array
                input_df = pd.DataFrame([input_values])
                input_with_const = sm.add_constant(input_df)
                
                # Make prediction
                prediction = model.predict(input_with_const)[0]
                
                # Display prediction
                st.markdown(f"""
                <div style="text-align: center; margin-top: 20px;">
                    <h3>Predicted {target_var}</h3>
                    <div style="font-size: 3rem; font-weight: bold; color: #304FFE;">
                        {prediction:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate feature importance
                importance = abs(np.array(list(input_values.values())) * model.params[1:].values)
                importance_df = pd.DataFrame({
                    'Feature': list(input_values.keys()),
                    'Importance': importance / importance.sum() * 100
                }).sort_values('Importance', ascending=False)
                
                # Plot importance
                fig = px.bar(importance_df, x='Feature', y='Importance',
                           title="Feature Contribution to Prediction (%)",
                           template='plotly_white')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Provide insights
                st.markdown("**Prediction Insights:**")
                top_feature = importance_df.iloc[0]['Feature']
                top_contribution = importance_df.iloc[0]['Importance']
                
                st.write(f"- {top_feature} is the most influential factor, contributing {top_contribution:.1f}% to the prediction")
                
                # Compare with similar sites
                st.markdown("**Similar Mining Sites:**")
                
                # Calculate Euclidean distance to all sites
                distances = []
                for i, row in filtered_df.iterrows():
                    dist = np.sqrt(sum((row[var] - input_values[var])**2 for var in predictor_vars))
                    distances.append((i, dist))
                
                # Get top 3 similar sites
                similar_indices = [idx for idx, _ in sorted(distances, key=lambda x: x[1])[:3]]
                similar_sites = filtered_df.iloc[similar_indices][['Celestial Body', target_var] + predictor_vars]
                
                st.dataframe(similar_sites)

# Call the function to show the visualization page
show_visualize_page()
