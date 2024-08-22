import streamlit as st
import pandas as pd

def show_insights_page():
    # Set page title with an icon
    st.title("🌌 Mining Site Insights")
    st.write("Gain **actionable insights** based on the characteristics of mining sites.")
    
    # Load dataset
    df = pd.read_csv("space_mining_dataset.csv")
    
    # General Insights
    st.write("## 🌍 General Insights")
    st.write(f"**🔢 Total Number of Mining Sites:** `{df.shape[0]}`")
    st.write(f"**🌕 Number of Unique Celestial Bodies:** `{df['Celestial Body'].nunique()}`")

    # User selection for Celestial Body
    celestial_body_selected = st.multiselect(
        "Select Celestial Body to Filter Insights", 
        options=df['Celestial Body'].unique(), 
        default=df['Celestial Body'].unique()
    )
    
    df_filtered = df[df['Celestial Body'].isin(celestial_body_selected)]
    
    # Insights by Celestial Body
    st.write("## 🌟 Insights by Celestial Body")
    st.write("Here’s an overview of key metrics per celestial body.")
    celestial_body_summary = df_filtered.groupby('Celestial Body').agg({
        'iron': ['mean', 'std'],
        'nickel': ['mean', 'std'],
        'water_ice': ['mean', 'std'],
        'Estimated Value (B USD)': ['mean', 'std'],
        'sustainability_index': ['mean', 'std'],
        'efficiency_index': ['mean', 'std'],
        'distance_from_earth': ['mean', 'std']
    }).style.background_gradient(cmap='Blues')
    st.dataframe(celestial_body_summary)

    # User selection for Estimated Value range
    min_value, max_value = st.slider(
        "Select Estimated Value Range (B USD)", 
        min_value=float(df['Estimated Value (B USD)'].min()), 
        max_value=float(df['Estimated Value (B USD)'].max()), 
        value=(float(df['Estimated Value (B USD)'].min()), float(df['Estimated Value (B USD)'].max()))
    )
    
    df_value_filtered = df_filtered[(df_filtered['Estimated Value (B USD)'] >= min_value) & (df_filtered['Estimated Value (B USD)'] <= max_value)]
    
    # Insights by Estimated Value
    st.write("## 💰 Insights by Estimated Value")
    high_value_sites = df_value_filtered[df_value_filtered['Estimated Value (B USD)'] > df_value_filtered['Estimated Value (B USD)'].median()]
    low_value_sites = df_value_filtered[df_value_filtered['Estimated Value (B USD)'] <= df_value_filtered['Estimated Value (B USD)'].median()]
    
    st.write(f"**💎 Number of High-Value Sites:** `{high_value_sites.shape[0]}`")
    st.write(f"**📉 Number of Low-Value Sites:** `{low_value_sites.shape[0]}`")
    
    st.write("### 🔝 High-Value Sites Overview")
    st.dataframe(high_value_sites[['Celestial Body', 'iron', 'nickel', 'water_ice', 'Estimated Value (B USD)']].describe().style.background_gradient(cmap='Greens'))
    
    st.write("### 📉 Low-Value Sites Overview")
    st.dataframe(low_value_sites[['Celestial Body', 'iron', 'nickel', 'water_ice', 'Estimated Value (B USD)']].describe().style.background_gradient(cmap='Oranges'))
    
    # Recommendations
    st.write("## 🔍 Recommendations")
    st.markdown("**For Higher Value Sites:** Focus on sites with **higher iron** and **nickel percentages**.")
    st.markdown("**For Sustainable Mining:** Prioritize sites with **higher sustainability** and **efficiency indices**.")
    st.markdown("**For Proximity:** Consider sites that are **closer to Earth** for easier access.")
    
    st.success("Insights complete!")

show_insights_page()
