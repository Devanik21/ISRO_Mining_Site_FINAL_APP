import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

def show_insights_page():
    # Set page configuration for a wide layout and custom title
    st.set_page_config(page_title="🌌 Mining Site Insights", layout="wide", initial_sidebar_state="expanded")
    st.title("🌌 Mining Site Insights")
    st.markdown("Gain **actionable insights** based on the characteristics of mining sites across celestial bodies.")

    # Load dataset
    df = pd.read_csv("space_mining_dataset.csv")
    
    # -------------------------------
    # Sidebar: Advanced Filtering & Ranking Parameters
    # -------------------------------
    st.sidebar.header("Filter & Ranking Options")
    
    # Celestial Body Filter
    celestial_options = df['Celestial Body'].unique()
    celestial_body_selected = st.sidebar.multiselect(
        "Select Celestial Body",
        options=celestial_options,
        default=celestial_options
    )
    df_filtered = df[df['Celestial Body'].isin(celestial_body_selected)]
    
    # Estimated Value Filter
    min_val = float(df['Estimated Value (B USD)'].min())
    max_val = float(df['Estimated Value (B USD)'].max())
    value_range = st.sidebar.slider(
        "Estimated Value Range (B USD)",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val)
    )
    df_filtered = df_filtered[
        (df_filtered['Estimated Value (B USD)'] >= value_range[0]) &
        (df_filtered['Estimated Value (B USD)'] <= value_range[1])
    ]
    
    # Ranking Parameters: Weights for sustainability and efficiency
    st.sidebar.markdown("### Ranking Parameters")
    sustainability_weight = st.sidebar.slider("Sustainability Weight", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    efficiency_weight = st.sidebar.slider("Efficiency Weight", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    
    # -------------------------------
    # Main Dashboard: Key Metrics & General Insights
    # -------------------------------
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Mining Sites", df.shape[0])
    col2.metric("Filtered Mining Sites", df_filtered.shape[0])
    col3.metric("Unique Celestial Bodies", df['Celestial Body'].nunique())
    
    st.markdown("---")
    
    st.subheader("General Insights")
    st.write(f"🔢 **Total Mining Sites:** {df.shape[0]}")
    st.write(f"🌕 **Unique Celestial Bodies:** {df['Celestial Body'].nunique()}")
    
    st.subheader("Insights by Celestial Body")
    celestial_body_summary = df_filtered.groupby('Celestial Body').agg({
        'iron': ['mean', 'std'],
        'nickel': ['mean', 'std'],
        'water_ice': ['mean', 'std'],
        'Estimated Value (B USD)': ['mean', 'std'],
        'sustainability_index': ['mean', 'std'],
        'efficiency_index': ['mean', 'std'],
        'distance_from_earth': ['mean', 'std']
    }).round(2)
    st.dataframe(celestial_body_summary.style.background_gradient(cmap='Blues'))
    
    st.markdown("---")
    
    st.subheader("Estimated Value Analysis")
    median_value = df_filtered['Estimated Value (B USD)'].median()
    high_value_sites = df_filtered[df_filtered['Estimated Value (B USD)'] > median_value]
    low_value_sites = df_filtered[df_filtered['Estimated Value (B USD)'] <= median_value]
    
    col4, col5 = st.columns(2)
    col4.metric("High-Value Sites", high_value_sites.shape[0])
    col5.metric("Low-Value Sites", low_value_sites.shape[0])
    
    st.markdown("#### High-Value Sites Overview")
    st.dataframe(high_value_sites[['Celestial Body', 'iron', 'nickel', 'water_ice', 'Estimated Value (B USD)']]
                 .describe().T.style.background_gradient(cmap='Greens'))
    
    st.markdown("#### Low-Value Sites Overview")
    st.dataframe(low_value_sites[['Celestial Body', 'iron', 'nickel', 'water_ice', 'Estimated Value (B USD)']]
                 .describe().T.style.background_gradient(cmap='Oranges'))
    
    st.markdown("---")
    
    # -------------------------------
    # New Tool: Site Ranking Calculator
    # -------------------------------
    st.subheader("🔝 Top Ranked Mining Sites")
    df_filtered = df_filtered.copy()  # Avoid SettingWithCopyWarning
    df_filtered['ranking_score'] = (
        df_filtered['Estimated Value (B USD)'] * 
        (sustainability_weight * df_filtered['sustainability_index'] + efficiency_weight * df_filtered['efficiency_index'])
    ) / (df_filtered['distance_from_earth'] + 1)
    
    top_sites = df_filtered.sort_values(by='ranking_score', ascending=False).head(5)
    st.write("Based on your selected parameters, the top 5 mining sites are:")
    st.dataframe(top_sites[['Celestial Body', 'Estimated Value (B USD)', 'sustainability_index', 'efficiency_index', 'distance_from_earth', 'ranking_score']])
    
    st.markdown("---")
    
    # -------------------------------
    # New Tool: Data Export Utility
    # -------------------------------
    st.subheader("Export Data")
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_space_mining_data.csv',
        mime='text/csv'
    )
    
    st.markdown("---")
    
    # -------------------------------
    # Additional Advanced Tools
    # -------------------------------
    st.header("Advanced Tools")
    
    # 1. Site Comparison Tool
    with st.expander("🔍 Site Comparison Tool", expanded=False):
        st.markdown("Select up to 3 mining sites to compare their key metrics side-by-side.")
        df_filtered = df_filtered.copy()
        # Create a unique identifier for each site based on the index
        df_filtered['Site ID'] = df_filtered.index.astype(str)
        site_options = df_filtered['Site ID'].tolist()
        selected_sites = st.multiselect("Select Site IDs", options=site_options, default=site_options[:2])
        if selected_sites:
            comparison_df = df_filtered[df_filtered['Site ID'].isin(selected_sites)]
            st.write("### Comparison of Selected Mining Sites")
            st.dataframe(comparison_df[['Celestial Body', 'Estimated Value (B USD)', 'iron', 'nickel', 'water_ice',
                                        'sustainability_index', 'efficiency_index', 'distance_from_earth']])
    
    # 2. Risk Assessment Tool
    with st.expander("⚠️ Risk Assessment Tool", expanded=False):
        st.markdown("Adjust parameters to compute a risk score for each site. Higher scores indicate higher operational risk.")
        risk_distance_weight = st.slider("Distance Risk Weight", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        risk_sustainability_weight = st.slider("Sustainability Mitigation Weight", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        risk_efficiency_weight = st.slider("Efficiency Mitigation Weight", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        df_risk = df_filtered.copy()
        # Compute Risk Score: Higher distance increases risk while higher sustainability and efficiency reduce it.
        df_risk['risk_score'] = (df_risk['distance_from_earth'] * risk_distance_weight) - (df_risk['sustainability_index'] * risk_sustainability_weight) - (df_risk['efficiency_index'] * risk_efficiency_weight)
        st.write("### Top 5 High-Risk Mining Sites")
        high_risk_sites = df_risk.sort_values(by='risk_score', ascending=False).head(5)
        st.dataframe(high_risk_sites[['Celestial Body', 'distance_from_earth', 'sustainability_index', 'efficiency_index', 'risk_score']])
        st.write("### Top 5 Low-Risk Mining Sites")
        low_risk_sites = df_risk.sort_values(by='risk_score', ascending=True).head(5)
        st.dataframe(low_risk_sites[['Celestial Body', 'distance_from_earth', 'sustainability_index', 'efficiency_index', 'risk_score']])
    
    # 3. Custom Query Tool
    with st.expander("🔎 Custom Query Tool", expanded=False):
        st.markdown("Apply custom filters to further refine the mining sites dataset.")
        min_iron = st.number_input("Minimum Iron (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        max_nickel = st.number_input("Maximum Nickel (%)", min_value=0.0, max_value=100.0, value=100.0, step=0.1)
        min_water_ice = st.number_input("Minimum Water Ice (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        min_sustainability = st.number_input("Minimum Sustainability Index", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        min_efficiency = st.number_input("Minimum Efficiency Index", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        
        df_custom = df_filtered[
            (df_filtered['iron'] >= min_iron) &
            (df_filtered['nickel'] <= max_nickel) &
            (df_filtered['water_ice'] >= min_water_ice) &
            (df_filtered['sustainability_index'] >= min_sustainability) &
            (df_filtered['efficiency_index'] >= min_efficiency)
        ]
        st.write("### Custom Query Results")
        st.dataframe(df_custom[['Celestial Body', 'Estimated Value (B USD)', 'iron', 'nickel', 'water_ice',
                                'sustainability_index', 'efficiency_index', 'distance_from_earth']])
    
    # 4. Clustering Analysis Tool
    with st.expander("🧩 Clustering Analysis Tool", expanded=False):
        st.markdown("Perform clustering analysis on mining sites based on key metrics.")
        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3, step=1)
        features = ['iron', 'nickel', 'water_ice', 'Estimated Value (B USD)', 'sustainability_index', 'efficiency_index', 'distance_from_earth']
        df_cluster = df_filtered[features].copy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        # Assign clusters to the filtered dataset
        df_filtered['cluster'] = kmeans.fit_predict(df_cluster)
        st.write("### Cluster Assignment for Each Mining Site")
        st.dataframe(df_filtered[['Celestial Body', 'Estimated Value (B USD)', 'sustainability_index', 'efficiency_index', 'distance_from_earth', 'cluster']])
        st.write("### Cluster Centers")
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
        st.dataframe(cluster_centers)
    
    st.markdown("---")
    
    # Recommendations & Conclusion
    st.subheader("Recommendations")
    st.markdown("""
    - **High-Value Sites:** Investigate sites with high estimated values along with strong sustainability and efficiency metrics.
    - **Optimized Operations:** Prioritize sites that are closer to Earth to reduce logistical challenges.
    - **Balanced Strategy:** Use the ranking and risk assessment tools to find the optimal balance between economic potential and operational feasibility.
    - **Custom Analysis:** Leverage the custom query and clustering tools to uncover hidden patterns and site-specific insights.
    """)
    
    st.success("Insights analysis complete! Adjust the sidebar filters and explore the advanced tools to dive even deeper into the data.")

if __name__ == "__main__":
    show_insights_page()
