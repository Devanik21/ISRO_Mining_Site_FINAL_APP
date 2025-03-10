import streamlit as st
import pandas as pd

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
    # Compute ranking score based on provided parameters
    # Formula: (Estimated Value * (sustainability_weight * sustainability_index + efficiency_weight * efficiency_index)) / (distance_from_earth + 1)
    df_filtered = df_filtered.copy()  # avoid SettingWithCopyWarning
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
    
    # -------------------------------
    # Recommendations & Conclusion
    # -------------------------------
    st.subheader("Recommendations")
    st.markdown("""
    - **High-Value Sites:** Investigate sites with high estimated values along with strong sustainability and efficiency metrics.
    - **Optimized Operations:** Prioritize sites that are closer to Earth to reduce logistical challenges.
    - **Balanced Strategy:** Use the ranking tool to find the optimal balance between economic potential and operational feasibility.
    """)
    
    st.success("Insights analysis complete! Adjust the sidebar filters and ranking parameters to explore top-ranked mining sites.")

if __name__ == "__main__":
    show_insights_page()
