import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_visualize_page():
    st.title("📊 Mining Site Visualization")
    st.write("Visualize mining site data to gain insights.")

    # Load dataset
    df = pd.read_csv("space_mining_dataset.csv")

    # Iron vs. Nickel Scatter Plot
    st.write("### 🧲 Iron vs. Nickel Composition")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='iron', y='nickel', data=df, hue='Celestial Body', palette='Set1', s=100, edgecolor='black')
    plt.title('Iron vs. Nickel Composition (%)')
    plt.xlabel('Iron (%)')
    plt.ylabel('Nickel (%)')
    plt.grid(True)
    st.pyplot(plt)

    # Histogram of Iron Composition
    st.write("### 🏭 Distribution of Iron Composition")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['iron'], kde=True, color='red', bins=20, edgecolor='black')
    plt.title('Distribution of Iron Composition (%)')
    plt.xlabel('Iron (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    st.pyplot(plt)

    # Histogram of Water/Ice Composition
    st.write("### 💧 Distribution of Water/Ice Composition")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['water_ice'], kde=True, color='blue', bins=20, edgecolor='black')
    plt.title('Distribution of Water/Ice Composition (%)')
    plt.xlabel('Water/Ice (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    st.pyplot(plt)

    # Pie Chart of Celestial Bodies
    st.write("### 🌌 Celestial Body Distribution")
    body_counts = df['Celestial Body'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(body_counts, labels=body_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"), 
            wedgeprops={'edgecolor': 'black'})
    plt.title('Celestial Body Distribution')
    st.pyplot(plt)

    # Boxplot of Estimated Value by Celestial Body
    st.write("### 💵 Estimated Value by Celestial Body")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Celestial Body', y='Estimated Value (B USD)', data=df, palette='Set3')
    plt.xticks(rotation=45)
    plt.title('Estimated Value (B USD) by Celestial Body')
    plt.grid(True)
    st.pyplot(plt)

    # Correlation Heatmap (only for numeric columns)
    st.write("### 🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.5})
    plt.title('Correlation Heatmap of Numeric Features')
    st.pyplot(plt)

    # Pairplot of Selected Features
    st.write("### 🔗 Pairplot of Selected Features")
    selected_features = ['iron', 'nickel', 'water_ice', 'Estimated Value (B USD)', 'sustainability_index', 'efficiency_index']
    sns.pairplot(df[selected_features], diag_kind='kde', palette='coolwarm', plot_kws={'edgecolor': 'black'})
    plt.suptitle('Pairplot of Selected Features', y=1.02)
    st.pyplot(plt)

show_visualize_page()
