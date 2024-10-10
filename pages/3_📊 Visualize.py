import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_visualize_page():
    st.title("📊 Mining Site Visualization")
    st.write("Visualize mining site data to gain insights.")

    # Load dataset
    df = pd.read_csv("space_mining_dataset.csv")
    
    # Set seaborn style and palette
    sns.set_style("whitegrid")
    sns.set_palette("coolwarm")

    # Select columns for visualizations
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Select Columns to Visualize", columns, default=columns[:3])

    if not selected_columns:
        st.warning("Please select at least one column.")
        return

    # Iron vs. Nickel Scatter Plot (or any two selected columns)
    if len(selected_columns) >= 2:
        st.write(f"### 🧲 {selected_columns[0]} vs. {selected_columns[1]} Composition")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=selected_columns[0], y=selected_columns[1], data=df, hue='Celestial Body', 
                        palette='Spectral', s=100, edgecolor='black')
        plt.title(f'{selected_columns[0]} vs. {selected_columns[1]} Composition', fontsize=16, fontweight='bold')
        plt.xlabel(f'{selected_columns[0]}', fontsize=14)
        plt.ylabel(f'{selected_columns[1]}', fontsize=14)
        plt.grid(True)
        st.pyplot(plt)

    # Histogram of selected column
    if len(selected_columns) >= 1:
        st.write(f"### 🏭 Distribution of {selected_columns[0]}")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[selected_columns[0]], kde=True, color='crimson', bins=20, edgecolor='black')
        plt.title(f'Distribution of {selected_columns[0]}', fontsize=16, fontweight='bold')
        plt.xlabel(f'{selected_columns[0]}', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True)
        st.pyplot(plt)

    # Violin Plot: Distribution by Celestial Body
    if len(selected_columns) >= 1:
        st.write(f"### 🎻 {selected_columns[0]} Distribution by Celestial Body (Violin Plot)")
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='Celestial Body', y=selected_columns[0], data=df, palette='muted')
        plt.title(f'{selected_columns[0]} Distribution by Celestial Body (Violin Plot)', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, fontsize=12)
        plt.grid(True)
        st.pyplot(plt)

    # FacetGrid to Compare Distributions Across Celestial Bodies
    if len(selected_columns) >= 1:
        st.write(f"### 🪐 {selected_columns[0]} Distribution by Celestial Body (FacetGrid)")
        g = sns.FacetGrid(df, col='Celestial Body', height=4, aspect=1.2)
        g.map(sns.histplot, selected_columns[0], kde=True, bins=15, color='orange')
        g.set_axis_labels(selected_columns[0], 'Frequency')
        g.fig.suptitle(f'{selected_columns[0]} Distribution by Celestial Body (FacetGrid)', fontsize=16, fontweight='bold')
        st.pyplot(g.fig)

    # Pie Chart of Celestial Bodies
    st.write("### 🌌 Celestial Body Distribution")
    body_counts = df['Celestial Body'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(body_counts, labels=body_counts.index, autopct='%1.1f%%', startangle=140, 
            colors=sns.color_palette("Set2"), wedgeprops={'edgecolor': 'black'})
    plt.title('Celestial Body Distribution', fontsize=16, fontweight='bold')
    st.pyplot(plt)

    # Boxplot of Selected Columns by Celestial Body
    if len(selected_columns) >= 1:
        st.write(f"### 💵 {selected_columns[0]} by Celestial Body (Boxplot)")
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Celestial Body', y=selected_columns[0], data=df, palette='rocket')
        plt.xticks(rotation=45, fontsize=12)
        plt.title(f'{selected_columns[0]} by Celestial Body', fontsize=16, fontweight='bold')
        plt.grid(True)
        st.pyplot(plt)

    # Correlation Heatmap (only for selected numeric columns)
    st.write("### 🔥 Correlation Heatmap")
    numeric_df = df[selected_columns].select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        corr_matrix = numeric_df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, 
                    cbar_kws={'shrink': 0.5}, mask=mask)
        plt.title('Correlation Heatmap of Selected Features', fontsize=16, fontweight='bold')
        st.pyplot(plt)
    else:
        st.warning("No numeric columns selected for correlation heatmap.")

    # Pairplot of Selected Features
    if len(selected_columns) > 1:
        st.write("### 🔗 Pairplot of Selected Features")
        sns.pairplot(df[selected_columns], diag_kind='kde', palette='coolwarm', plot_kws={'edgecolor': 'black'})
        plt.suptitle('Pairplot of Selected Features', y=1.02, fontsize=16, fontweight='bold')
        st.pyplot(plt)

    # Regression Plot: Iron vs Nickel with a trendline
    if 'iron' in df.columns and 'nickel' in df.columns:
        st.write("### 📈 Regression Plot: Iron vs Nickel")
        plt.figure(figsize=(10, 6))
        sns.regplot(x='iron', y='nickel', data=df, scatter_kws={'s': 50, 'color': 'green'}, line_kws={'color': 'red'})
        plt.title('Regression Plot: Iron vs Nickel', fontsize=16, fontweight='bold')
        plt.xlabel('Iron Content', fontsize=14)
        plt.ylabel('Nickel Content', fontsize=14)
        plt.grid(True)
        st.pyplot(plt)

show_visualize_page()
