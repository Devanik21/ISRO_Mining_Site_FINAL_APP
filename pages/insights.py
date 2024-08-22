import streamlit as st
import pandas as pd

def show_insights_page():
    st.title("Mining Site Insights")
    st.write("Get actionable insights based on the characteristics of mining sites.")

    # Load dataset
    df = pd.read_csv("space_mining_dataset.csv")

    # General Insights
    st.write("### General Insights")
    st.write(f"**Total Number of Mining Sites:** {df.shape[0]}")
    st.write(f"**Number of Unique Celestial Bodies:** {df['Celestial Body'].nunique()}")

    # Insights by Celestial Body
    st.write("### Insights by Celestial Body")
    celestial_body_summary = df.groupby('Celestial Body').agg({
        'iron': ['mean', 'std'],
        'nickel': ['mean', 'std'],
        'water_ice': ['mean', 'std'],
        'Estimated Value (B USD)': ['mean', 'std'],
        'sustainability_index': ['mean', 'std'],
        'efficiency_index': ['mean', 'std'],
        'distance_from_earth': ['mean', 'std']
    })
    st.write(celestial_body_summary)

    # Insights by Estimated Value
    st.write("### Insights by Estimated Value")
    high_value_sites = df[df['Estimated Value (B USD)'] > df['Estimated Value (B USD)'].median()]
    low_value_sites = df[df['Estimated Value (B USD)'] <= df['Estimated Value (B USD)'].median()]

    st.write(f"**Number of High-Value Sites:** {high_value_sites.shape[0]}")
    st.write(f"**Number of Low-Value Sites:** {low_value_sites.shape[0]}")

    st.write("#### High-Value Sites Overview")
    st.write(high_value_sites[['Celestial Body', 'iron', 'nickel', 'water_ice', 'Estimated Value (B USD)']].describe())

    st.write("#### Low-Value Sites Overview")
    st.write(low_value_sites[['Celestial Body', 'iron', 'nickel', 'water_ice', 'Estimated Value (B USD)']].describe())

    # Recommendations
    st.write("### Recommendations")
    st.write("**For Higher Value Sites:** Focus on sites with higher iron and nickel percentages.")
    st.write("**For Sustainable Mining:** Prioritize sites with higher sustainability and efficiency indices.")
    st.write("**For Proximity:** Consider sites that are closer to Earth for easier access.")

    st.write("Insights complete!")

show_insights_page()
