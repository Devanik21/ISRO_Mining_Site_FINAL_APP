import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def show_insights_page():
    # Set page configuration and title
    st.set_page_config(layout="wide", page_title="Space Mining Analytics", page_icon="🌌")
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4B0082;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #483D8B;
        margin-top: 1.5rem;
    }
    .metric-container {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .info-text {
        font-size: 1rem;
        color: #696969;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Dashboard Header
    st.markdown('<p class="main-header">🌌 Advanced Space Mining Analytics Dashboard</p>', unsafe_allow_html=True)
    
    # Load and cache data
    @st.cache_data
    def load_data():
        df = pd.read_csv("space_mining_dataset.csv")
       
    
    try:
        df = load_data()
        
        # Navigation tabs
        tabs = st.tabs(["📊 Overview", "🔍 Deep Analysis", "📈 Predictive Insights", "📋 Recommendations", "🔄 What-If Scenarios"])
        
        ############## OVERVIEW TAB ##############
        with tabs[0]:
            # Key Metrics Row
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(label="Total Mining Sites", value=f"{df.shape[0]:,}")
            with metric_cols[1]:
                st.metric(label="Unique Celestial Bodies", value=df['Celestial Body'].nunique())
            with metric_cols[2]:
                st.metric(label="Avg. Site Value (B USD)", value=f"${df['Estimated Value (B USD)'].mean():.2f}B")
            with metric_cols[3]:
                st.metric(label="Highest Value Site (B USD)", value=f"${df['Estimated Value (B USD)'].max():.2f}B")
            
            # Filters Section
            with st.expander("🔍 Advanced Filters", expanded=True):
                filter_cols = st.columns([1, 1, 1])
                
                with filter_cols[0]:
                    celestial_body_selected = st.multiselect(
                        "Celestial Bodies", 
                        options=df['Celestial Body'].unique(), 
                        default=df['Celestial Body'].unique()
                    )
                
                with filter_cols[1]:
                    min_value, max_value = st.slider(
                        "Estimated Value Range (B USD)", 
                        min_value=float(df['Estimated Value (B USD)'].min()), 
                        max_value=float(df['Estimated Value (B USD)'].max()), 
                        value=(float(df['Estimated Value (B USD)'].min()), float(df['Estimated Value (B USD)'].max()))
                    )
                
                with filter_cols[2]:
                    min_sustainability, max_sustainability = st.slider(
                        "Sustainability Index Range", 
                        min_value=float(df['sustainability_index'].min()), 
                        max_value=float(df['sustainability_index'].max()), 
                        value=(float(df['sustainability_index'].min()), float(df['sustainability_index'].max()))
                    )
            
            # Apply filters
            df_filtered = df[
                (df['Celestial Body'].isin(celestial_body_selected)) &
                (df['Estimated Value (B USD)'] >= min_value) & 
                (df['Estimated Value (B USD)'] <= max_value) &
                (df['sustainability_index'] >= min_sustainability) & 
                (df['sustainability_index'] <= max_sustainability)
            ]
            
            # Resource Distribution
            st.markdown('<p class="sub-header">Resource Distribution by Celestial Body</p>', unsafe_allow_html=True)
            chart_cols = st.columns([2, 1])
            
            with chart_cols[0]:
                resource_fig = go.Figure()
                
                for resource in ['iron', 'nickel', 'water_ice']:
                    resource_summary = df_filtered.groupby('Celestial Body')[resource].mean().reset_index()
                    resource_fig.add_trace(go.Bar(
                        x=resource_summary['Celestial Body'],
                        y=resource_summary[resource],
                        name=resource.capitalize()
                    ))
                
                resource_fig.update_layout(
                    barmode='group',
                    title='Average Resource Percentage by Celestial Body',
                    xaxis_title='Celestial Body',
                    yaxis_title='Average Percentage (%)',
                    legend_title='Resource Type',
                    height=400
                )
                st.plotly_chart(resource_fig, use_container_width=True)
            
            with chart_cols[1]:
                value_summary = df_filtered.groupby('Celestial Body')['Estimated Value (B USD)'].sum().reset_index()
                value_fig = px.pie(
                    value_summary, 
                    values='Estimated Value (B USD)', 
                    names='Celestial Body',
                    title='Total Estimated Value by Celestial Body',
                    hole=0.4
                )
                value_fig.update_traces(textposition='inside', textinfo='percent+label')
                value_fig.update_layout(height=400)
                st.plotly_chart(value_fig, use_container_width=True)
            
            # Site Characteristics Table
            st.markdown('<p class="sub-header">Site Characteristics Summary</p>', unsafe_allow_html=True)
            
            summary_cols = st.columns([2, 1])
            with summary_cols[0]:
                celestial_body_summary = df_filtered.groupby('Celestial Body').agg({
                    'iron': ['mean', 'std'],
                    'nickel': ['mean', 'std'],
                    'water_ice': ['mean', 'std'],
                    'Estimated Value (B USD)': ['mean', 'max'],
                    'sustainability_index': ['mean'],
                    'efficiency_index': ['mean'],
                    'distance_from_earth': ['mean']
                }).style.background_gradient(cmap='Blues',axis=None)
                
                st.dataframe(celestial_body_summary, height=300)
            
            with summary_cols[1]:
                site_count = df_filtered['Celestial Body'].value_counts().reset_index()
                site_count.columns = ['Celestial Body', 'Number of Sites']
                
                count_fig = px.bar(
                    site_count,
                    x='Celestial Body',
                    y='Number of Sites',
                    color='Number of Sites',
                    title='Number of Mining Sites per Celestial Body'
                )
                count_fig.update_layout(height=300)
                st.plotly_chart(count_fig, use_container_width=True)
        
        ############## DEEP ANALYSIS TAB ##############
        with tabs[1]:
            st.markdown('<p class="sub-header">Resource Correlation Analysis</p>', unsafe_allow_html=True)
            
            # Correlation Analysis
            analysis_cols = st.columns([1, 1])
            
            with analysis_cols[0]:
                # Correlation Matrix
                corr_features = ['iron', 'nickel', 'water_ice', 'Estimated Value (B USD)', 
                               'sustainability_index', 'efficiency_index', 'distance_from_earth']
                corr_matrix = df_filtered[corr_features].corr()
                
                corr_fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix of Key Metrics',
                    aspect="auto"
                )
                corr_fig.update_layout(height=400)
                st.plotly_chart(corr_fig, use_container_width=True)
            
            with analysis_cols[1]:
                # Scatter plot with custom parameters
                x_variable = st.selectbox("X-Axis Variable", corr_features)
                y_variable = st.selectbox("Y-Axis Variable", corr_features, index=3)
                
                scatter_fig = px.scatter(
                    df_filtered,
                    x=x_variable,
                    y=y_variable,
                    color='Celestial Body',
                    size='Estimated Value (B USD)',
                    hover_data=['sustainability_index', 'efficiency_index'],
                    title=f'{y_variable} vs {x_variable} by Celestial Body'
                )
                scatter_fig.update_layout(height=400)
                st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Resource Distribution by Parameters
            st.markdown('<p class="sub-header">Resource Distribution Analysis</p>', unsafe_allow_html=True)
            
            # 3D Scatter Plot
            plot_cols = st.columns([2, 1])
            with plot_cols[0]:
                fig3d = px.scatter_3d(
                    df_filtered, 
                    x='iron', 
                    y='nickel', 
                    z='water_ice',
                    color='Celestial Body', 
                    size='Estimated Value (B USD)',
                    opacity=0.7,
                    title='3D Resource Distribution'
                )
                fig3d.update_layout(height=500)
                st.plotly_chart(fig3d, use_container_width=True)
            
            with plot_cols[1]:
                # Sustainability vs Efficiency
                sustain_fig = px.scatter(
                    df_filtered,
                    x='sustainability_index',
                    y='efficiency_index',
                    color='Celestial Body',
                    size='Estimated Value (B USD)',
                    title='Sustainability vs Efficiency Index'
                )
                sustain_fig.update_layout(height=500)
                st.plotly_chart(sustain_fig, use_container_width=True)
        
        ############## PREDICTIVE INSIGHTS TAB ##############
        with tabs[2]:
            st.markdown('<p class="sub-header">Site Clustering Analysis</p>', unsafe_allow_html=True)
            
            # Prepare data for clustering
            cluster_cols = ['iron', 'nickel', 'water_ice', 'sustainability_index', 
                           'efficiency_index', 'distance_from_earth']
            
            X = df_filtered[cluster_cols].copy()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Number of clusters slider
            n_clusters = st.slider("Number of Site Clusters", min_value=2, max_value=6, value=3)
            
            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_filtered['cluster'] = kmeans.fit_predict(X_scaled)
            
            # PCA for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            df_filtered['pca_x'] = pca_result[:, 0]
            df_filtered['pca_y'] = pca_result[:, 1]
            
            cluster_cols = st.columns([2, 1])
            
            with cluster_cols[0]:
                # PCA Cluster Plot
                cluster_fig = px.scatter(
                    df_filtered,
                    x='pca_x',
                    y='pca_y',
                    color='cluster',
                    symbol='Celestial Body',
                    size='Estimated Value (B USD)',
                    hover_data=cluster_cols + ['Estimated Value (B USD)'],
                    title='Site Clusters (PCA Visualization)'
                )
                cluster_fig.update_layout(height=500)
                st.plotly_chart(cluster_fig, use_container_width=True)
            
            with cluster_cols[1]:
                # Cluster Characteristics
                cluster_summary = df_filtered.groupby('cluster').agg({
                    'iron': 'mean',
                    'nickel': 'mean',
                    'water_ice': 'mean',
                    'Estimated Value (B USD)': ['mean', 'count'],
                    'sustainability_index': 'mean',
                    'efficiency_index': 'mean',
                    'distance_from_earth': 'mean'
                })
                
                cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
                cluster_summary = cluster_summary.rename(columns={
                    'Estimated Value (B USD)_count': 'Number of Sites'
                })
                
                st.dataframe(cluster_summary.style.background_gradient(cmap='viridis', axis=None), height=500)
            
            # Optimal Sites Identification
            st.markdown('<p class="sub-header">Optimal Site Identification</p>', unsafe_allow_html=True)
            
            # Weighted scoring system
            st.markdown("### Customize Site Scoring Weights")
            weight_cols = st.columns(5)
            
            with weight_cols[0]:
                value_weight = st.slider("Value Importance", 0.0, 1.0, 0.4, 0.1)
            with weight_cols[1]:
                resource_weight = st.slider("Resource Importance", 0.0, 1.0, 0.3, 0.1)
            with weight_cols[2]:
                sustainability_weight = st.slider("Sustainability Importance", 0.0, 1.0, 0.1, 0.1)
            with weight_cols[3]:
                efficiency_weight = st.slider("Efficiency Importance", 0.0, 1.0, 0.1, 0.1)
            with weight_cols[4]:
                distance_weight = st.slider("Distance Importance", 0.0, 1.0, 0.1, 0.1)
            
            # Normalize features for scoring
            df_scoring = df_filtered.copy()
            
            for col in ['iron', 'nickel', 'water_ice', 'Estimated Value (B USD)', 'sustainability_index', 'efficiency_index']:
                df_scoring[f'{col}_norm'] = (df_scoring[col] - df_scoring[col].min()) / (df_scoring[col].max() - df_scoring[col].min())
            
            # Inverse distance (closer is better)
            df_scoring['distance_norm'] = 1 - (df_scoring['distance_from_earth'] - df_scoring['distance_from_earth'].min()) / (df_scoring['distance_from_earth'].max() - df_scoring['distance_from_earth'].min())
            
            # Calculate composite score
            df_scoring['resource_score'] = (df_scoring['iron_norm'] + df_scoring['nickel_norm'] + df_scoring['water_ice_norm']) / 3
            df_scoring['total_score'] = (
                value_weight * df_scoring['Estimated Value (B USD)_norm'] + 
                resource_weight * df_scoring['resource_score'] + 
                sustainability_weight * df_scoring['sustainability_index_norm'] + 
                efficiency_weight * df_scoring['efficiency_index_norm'] + 
                distance_weight * df_scoring['distance_norm']
            )
            
            # Display top sites
            top_n = st.slider("Show Top N Sites", 5, 20, 10)
            top_sites = df_scoring.sort_values('total_score', ascending=False).head(top_n)
            
            st.dataframe(
                top_sites[['Celestial Body', 'iron', 'nickel', 'water_ice', 
                          'Estimated Value (B USD)', 'sustainability_index', 
                          'efficiency_index', 'distance_from_earth', 'total_score']].reset_index(drop=True),
                height=400
            )
        
        ############## RECOMMENDATIONS TAB ##############
        with tabs[3]:
            st.markdown('<p class="sub-header">Strategic Recommendations</p>', unsafe_allow_html=True)
            
            # Generate insights based on data
            high_value_sites = df_filtered[df_filtered['Estimated Value (B USD)'] > df_filtered['Estimated Value (B USD)'].median()]
            
            # Feature importance for value
            value_corr = df_filtered.corr()['Estimated Value (B USD)'].sort_values(ascending=False)
            
            # Key insights
            insight_cols = st.columns(2)
            
            with insight_cols[0]:
                st.markdown("### Value Drivers")
                st.markdown(f"**Top Value-Correlated Features:**")
                
                value_drivers = value_corr.drop('Estimated Value (B USD)').head(3)
                for feature, corr in value_drivers.items():
                    st.markdown(f"- **{feature.capitalize()}**: {corr:.2f} correlation")
                
                st.markdown("### Resource Strategy")
                
                # Identify primary resource for each celestial body
                resource_cols = ['iron', 'nickel', 'water_ice']
                celestial_resources = {}
                
                for body in df_filtered['Celestial Body'].unique():
                    body_df = df_filtered[df_filtered['Celestial Body'] == body]
                    avg_resources = body_df[resource_cols].mean()
                    primary_resource = avg_resources.idxmax()
                    celestial_resources[body] = {
                        'primary': primary_resource,
                        'value': avg_resources[primary_resource]
                    }
                
                for body, data in celestial_resources.items():
                    st.markdown(f"- **{body}**: Focus on {data['primary']} ({data['value']:.1f}%)")
            
            with insight_cols[1]:
                st.markdown("### Optimization Opportunities")
                
                # Identify sites with high value but low sustainability
                optimization_sites = df_filtered[
                    (df_filtered['Estimated Value (B USD)'] > df_filtered['Estimated Value (B USD)'].quantile(0.75)) &
                    (df_filtered['sustainability_index'] < df_filtered['sustainability_index'].median())
                ]
                
                st.markdown(f"**{len(optimization_sites)} high-value sites** need sustainability improvements")
                
                # Distance vs Value Analysis
                st.markdown("### Accessibility Analysis")
                
                close_value_sites = df_filtered[
                    (df_filtered['distance_from_earth'] < df_filtered['distance_from_earth'].median()) &
                    (df_filtered['Estimated Value (B USD)'] > df_filtered['Estimated Value (B USD)'].median())
                ]
                
                st.markdown(f"**{len(close_value_sites)} sites** combine above-average value with closer proximity")
            
            # Investment Recommendations
            st.markdown("### Investment Priority Matrix")
            
            # Create priority score
            df_priority = df_filtered.copy()
            df_priority['value_rank'] = df_priority['Estimated Value (B USD)'].rank(ascending=False)
            df_priority['accessibility_rank'] = df_priority['distance_from_earth'].rank()
            df_priority['sustainability_rank'] = df_priority['sustainability_index'].rank(ascending=False)
            
            df_priority['priority_score'] = (
                0.5 * df_priority['value_rank'] + 
                0.3 * df_priority['accessibility_rank'] + 
                0.2 * df_priority['sustainability_rank']
            )
            
            # Top 5 investment priorities
            top_priorities = df_priority.sort_values('priority_score').head(5)
            
            priority_fig = go.Figure()
            
            priority_fig.add_trace(go.Scatter(
                x=top_priorities['Estimated Value (B USD)'],
                y=top_priorities['sustainability_index'],
                mode='markers',
                marker=dict(
                    size=top_priorities['distance_from_earth'],
                    sizemode='area',
                    sizeref=2.*max(top_priorities['distance_from_earth'])/(40.**2),
                    sizemin=4,
                    color=top_priorities['priority_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Priority Score')
                ),
                text=range(1, len(top_priorities) + 1),
                hovertemplate='Priority #%{text}<br>Value: %{x:.2f}B USD<br>Sustainability: %{y:.2f}<extra></extra>'
            ))
            
            priority_fig.update_layout(
                title='Top 5 Investment Priorities',
                xaxis_title='Estimated Value (B USD)',
                yaxis_title='Sustainability Index',
                height=400
            )
            
            st.plotly_chart(priority_fig, use_container_width=True)
            
            # Detailed Recommendations
            st.markdown("### Detailed Recommendations")
            
            recommendation_tabs = st.tabs(["Value Maximization", "Sustainability", "Operational Efficiency"])
            
            with recommendation_tabs[0]:
                st.markdown("""
                - **Focus Resource Extraction**: Prioritize mining sites with iron content >50% for maximum ROI
                - **Celestial Body Selection**: Moon and Mars sites show best value-to-distance ratio
                - **Portfolio Balancing**: Maintain 70% high-value / 30% strategic resource distribution
                """)
            
            with recommendation_tabs[1]:
                st.markdown("""
                - **Technology Investment**: Increase efficiency on lower sustainability sites
                - **Site Rotation Strategy**: Implement 3-phase mining rotation for sustainable resource extraction
                - **Environmental Monitoring**: Deploy advanced sensor arrays on high-resource sites
                """)
            
            with recommendation_tabs[2]:
                st.markdown("""
                - **Transportation Optimization**: Establish relay points for distant high-value sites
                - **Resource Processing**: Build in-situ refineries at sites with >40% iron content
                - **Automation Deployment**: Prioritize automated systems for hazardous environments
                """)
        
        ############## WHAT-IF SCENARIOS TAB ##############
        with tabs[4]:
            st.markdown('<p class="sub-header">What-If Scenario Analysis</p>', unsafe_allow_html=True)
            
            scenario_cols = st.columns([1, 1])
            
            with scenario_cols[0]:
                st.markdown("### Resource Price Sensitivity")
                
                # Price adjustment sliders
                iron_price_change = st.slider("Iron Price Change (%)", -50, 100, 0)
                nickel_price_change = st.slider("Nickel Price Change (%)", -50, 100, 0)
                water_ice_price_change = st.slider("Water Ice Price Change (%)", -50, 100, 0)
                
                # Simplified model for demonstration
                df_price_sim = df_filtered.copy()
                
                # Assume value is proportionally affected by resource prices
                # This is a simplified model for demonstration
                base_iron_contribution = 0.4
                base_nickel_contribution = 0.3
                base_water_contribution = 0.3
                
                df_price_sim['adjusted_value'] = df_price_sim['Estimated Value (B USD)'] * (
                    base_iron_contribution * (1 + iron_price_change/100) * df_price_sim['iron'] +
                    base_nickel_contribution * (1 + nickel_price_change/100) * df_price_sim['nickel'] +
                    base_water_contribution * (1 + water_ice_price_change/100) * df_price_sim['water_ice']
                ) / (
                    base_iron_contribution * df_price_sim['iron'] +
                    base_nickel_contribution * df_price_sim['nickel'] +
                    base_water_contribution * df_price_sim['water_ice']
                )
                
                # Display results
                price_impact_fig = px.bar(
                    df_price_sim.groupby('Celestial Body').agg({
                        'Estimated Value (B USD)': 'sum',
                        'adjusted_value': 'sum'
                    }).reset_index(),
                    x='Celestial Body',
                    y=['Estimated Value (B USD)', 'adjusted_value'],
                    barmode='group',
                    title='Impact of Price Changes on Total Value',
                    labels={'value': 'Total Value (B USD)'},
                    color_discrete_sequence=['#636EFA', '#EF553B']
                )
                
                price_impact_fig.update_layout(height=400, legend_title="Scenario")
                price_impact_fig.data[0].name = 'Current Value'
                price_impact_fig.data[1].name = 'Adjusted Value'
                
                st.plotly_chart(price_impact_fig, use_container_width=True)
            
            with scenario_cols[1]:
                st.markdown("### Technology Improvement Simulation")
                
                # Technology improvement factors
                extraction_efficiency = st.slider("Extraction Efficiency Improvement (%)", 0, 50, 20)
                sustainability_improvement = st.slider("Sustainability Technology Improvement (%)", 0, 50, 15)
                
                # Apply improvements to efficiency and sustainability
                df_tech_sim = df_filtered.copy()
                
                df_tech_sim['improved_efficiency'] = df_tech_sim['efficiency_index'] * (1 + extraction_efficiency/100)
                df_tech_sim['improved_efficiency'] = df_tech_sim['improved_efficiency'].clip(upper=10)  # Cap at 10
                
                df_tech_sim['improved_sustainability'] = df_tech_sim['sustainability_index'] * (1 + sustainability_improvement/100)
                df_tech_sim['improved_sustainability'] = df_tech_sim['improved_sustainability'].clip(upper=10)  # Cap at 10
                
                # Calculate estimated value increase (simplified model)
                efficiency_value_factor = 0.15  # 15% value boost from max efficiency improvement
                sustainability_value_factor = 0.08  # 8% value boost from max sustainability improvement
                
                df_tech_sim['tech_adjusted_value'] = df_tech_sim['Estimated Value (B USD)'] * (
                    1 + efficiency_value_factor * (df_tech_sim['improved_efficiency'] - df_tech_sim['efficiency_index']) / df_tech_sim['efficiency_index'] +
                    sustainability_value_factor * (df_tech_sim['improved_sustainability'] - df_tech_sim['sustainability_index']) / df_tech_sim['sustainability_index']
                )
                
                # Plot tech improvement impact
                tech_impact_fig = go.Figure()
                
                # Current values for comparison
                tech_impact_fig.add_trace(go.Scatter(
                    x=df_tech_sim['efficiency_index'],
                    y=df_tech_sim['sustainability_index'],
                    mode='markers',
                    marker=dict(
                        size=df_tech_sim['Estimated Value (B USD)'] * 3,
                        color='blue',
                        opacity=0.5
                    ),
                    name='Current Technology'
                ))
                
                # Improved values
                tech_impact_fig.add_trace(go.Scatter(
                    x=df_tech_sim['improved_efficiency'],
                    y=df_tech_sim['improved_sustainability'],
                    mode='markers',
                    marker=dict(
                        size=df_tech_sim['tech_adjusted_value'] * 3,
                        color='red',
                        opacity=0.5
                    ),
                    name='Improved Technology'
                ))
                
                tech_impact_fig.update_layout(
                    title='Technology Improvement Impact',
                    xaxis_title='Efficiency Index',
                    yaxis_title='Sustainability Index',
                    height=400
                )
                
                st.plotly_chart(tech_impact_fig, use_container_width=True)
            
            # ROI Calculator for Technology Investment
            st.markdown("### ROI Calculator for Technology Investment")
            
            roi_cols = st.columns([1, 1, 1])
            
            with roi_cols[0]:
                tech_investment = st.number_input("Technology Investment (Million USD)", 50, 1000, 200, 50)
                tech_lifespan = st.number_input("Technology Lifespan (Years)", 1, 20, 5, 1)
            
            with roi_cols[1]:
                # Calculate ROI (simplified)
                total_value_increase = df_tech_sim['tech_adjusted_value'].sum() - df_tech_sim['Estimated Value (B USD)'].sum()
                annual_return = total_value_increase / tech_lifespan
                roi_percent = (annual_return * 1000 / tech_investment) * 100  # Convert B to M
                
                st.metric(
                    label="Annual Value Increase (B USD)",
                    value=f"${annual_return:.2f}B"
                )
                
                st.metric(
                    label="ROI per Year",
                    value=f"{roi_percent:.1f}%"
                )
                
                payback_period = tech_investment / (annual_return * 1000)
                st.metric(
                    label="Payback Period (Years)",
                    value=f"{payback_period:.1f}"
                )
            
            with roi_cols[2]:
                # Simple NPV calculation
                discount_rate = st.slider("Discount Rate (%)", 5, 20, 10)
                
                # Calculate NPV
                npv = -tech_investment
                for year in range(1, tech_lifespan + 1):
                    npv += (annual_return * 1000) / ((1 + discount_rate/100) ** year)
                
                st.metric(
                    label="Net Present Value (Million USD)",
                    value=f"${npv:.1f}M"
                )
                
                # Investment recommendation
                if npv > 0:
                    st.success("✅ Investment Recommended")
                else:
                    st.error("❌ Investment Not Recommended")
    
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.info("Please ensure 'space_mining_dataset.csv' is available with the expected columns.")

if __name__ == "__main__":
    show_insights_page()
