import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_visualize_page():
    st.title("🌌 Mining Site Visualization 🌟")
    st.write("Explore and visualize mining site data interactively!")

    # Load dataset
    df = pd.read_csv("space_mining_dataset.csv")
    
    # Set seaborn style and palette
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Plot 1: Scatter Plot with column selection
    st.write("### 🧲 Scatter Plot")
    scatter_x = st.selectbox("Choose X-axis for Scatter Plot", df.columns, index=0)
    scatter_y = st.selectbox("Choose Y-axis for Scatter Plot", df.columns, index=1)
    if scatter_x and scatter_y:
        st.write(f"### {scatter_x} vs. {scatter_y} Composition 🌀")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=scatter_x, y=scatter_y, data=df, hue='Celestial Body', 
                        palette='Spectral', s=100, edgecolor='black')
        plt.title(f'{scatter_x} vs. {scatter_y} Composition', fontsize=16, fontweight='bold')
        plt.xlabel(f'{scatter_x}', fontsize=14)
        plt.ylabel(f'{scatter_y}', fontsize=14)
        plt.grid(True)
        st.pyplot(plt)

    # Plot 2: Histogram with column selection
    st.write("### 📊 Histogram")
    hist_col = st.selectbox("Choose a Column for Histogram", df.columns, index=0)
    if hist_col:
        st.write(f"### Distribution of {hist_col} 📉")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[hist_col], kde=True, color='crimson', bins=20, edgecolor='black')
        plt.title(f'Distribution of {hist_col}', fontsize=16, fontweight='bold')
        plt.xlabel(f'{hist_col}', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True)
        st.pyplot(plt)

    # Plot 3: Violin Plot with column selection
    st.write("### 🎻 Violin Plot")
    violin_col = st.selectbox("Choose a Column for Violin Plot", df.columns, index=0)
    if violin_col:
        st.write(f"### {violin_col} Distribution by Celestial Body 🎶")
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='Celestial Body', y=violin_col, data=df, palette='muted')
        plt.title(f'{violin_col} Distribution by Celestial Body (Violin Plot)', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, fontsize=12)
        plt.grid(True)
        st.pyplot(plt)

    # Plot 4: Pie Chart for Celestial Body Distribution
    st.write("### 🌌 Celestial Body Distribution (Pie Chart) 🍰")
    body_counts = df['Celestial Body'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(body_counts, labels=body_counts.index, autopct='%1.1f%%', startangle=140, 
            colors=sns.color_palette("Set2"), wedgeprops={'edgecolor': 'black'})
    plt.title('Celestial Body Distribution', fontsize=16, fontweight='bold')
    st.pyplot(plt)

    # Plot 5: Correlation Heatmap with selected numeric columns
    st.write("### 🔥 Correlation Heatmap")
    selected_columns = st.multiselect("Choose Columns for Correlation Heatmap", df.columns)
    if selected_columns:
        numeric_df = df[selected_columns].select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            plt.figure(figsize=(10, 8))
            corr_matrix = numeric_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, 
                        cbar_kws={'shrink': 0.5})
            plt.title('Correlation Heatmap of Selected Features', fontsize=16, fontweight='bold')
            st.pyplot(plt)
        else:
            st.warning("Please select numeric columns for the heatmap.")

    # Plot 6: Pairplot with selected columns
    st.write("### 🔗 Pairplot")
    pairplot_columns = st.multiselect("Choose Columns for Pairplot", df.columns, default=df.columns[:3])
    if pairplot_columns:
        sns.pairplot(df[pairplot_columns], diag_kind='kde', palette='coolwarm', plot_kws={'edgecolor': 'black'})
        plt.suptitle('Pairplot of Selected Features', y=1.02, fontsize=16, fontweight='bold')
        st.pyplot(plt)

    # Plot 7: Boxplot with column selection
    st.write("### 📦 Boxplot")
    box_col = st.selectbox("Choose a Column for Boxplot", df.columns, index=0)
    if box_col:
        st.write(f"### {box_col} by Celestial Body 📦")
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Celestial Body', y=box_col, data=df, palette='rocket')
        plt.xticks(rotation=45, fontsize=12)
        plt.title(f'{box_col} by Celestial Body', fontsize=16, fontweight='bold')
        plt.grid(True)
        st.pyplot(plt)

    # Plot 8: Hexbin Plot with column selection
    st.write("### 🧮 Hexbin Plot")
    hexbin_x = st.selectbox("Choose X-axis for Hexbin Plot", df.columns, index=0)
    hexbin_y = st.selectbox("Choose Y-axis for Hexbin Plot", df.columns, index=1)
    if hexbin_x and hexbin_y:
        st.write(f"### {hexbin_x} vs. {hexbin_y} (Hexbin Plot) 🎲")
        plt.figure(figsize=(10, 6))
        plt.hexbin(df[hexbin_x], df[hexbin_y], gridsize=30, cmap='Purples', edgecolors='black')
        plt.colorbar(label='Count')
        plt.title(f'Hexbin Plot of {hexbin_x} vs {hexbin_y}', fontsize=16, fontweight='bold')
        plt.grid(True)
        st.pyplot(plt)

show_visualize_page()
