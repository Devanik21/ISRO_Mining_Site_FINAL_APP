import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_visualize_page():
    st.title("📊 Mining Site Visualization")
    st.write("Visualize mining site data to gain insights.")

    # Load dataset
    df = pd.read_csv("space_mining_dataset.csv")
    
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
        sns.scatterplot(x=selected_columns[0], y=selected_columns[1], data=df, hue='Celestial Body', palette='Set1', s=100, edgecolor='black')
        plt.title(f'{selected_columns[0]} vs. {selected_columns[1]} Composition')
        plt.xlabel(f'{selected_columns[0]}')
        plt.ylabel(f'{selected_columns[1]}')
        plt.grid(True)
        st.pyplot(plt)

    # Histogram of selected column
    if len(selected_columns) >= 1:
        st.write(f"### 🏭 Distribution of {selected_columns[0]}")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[selected_columns[0]], kde=True, color='red', bins=20, edgecolor='black')
        plt.title(f'Distribution of {selected_columns[0]}')
        plt.xlabel(f'{selected_columns[0]}')
        plt.ylabel('Frequency')
        plt.grid(True)
        st.pyplot(plt)

    # Histogram of another selected column
    if len(selected_columns) >= 2:
        st.write(f"### 💧 Distribution of {selected_columns[1]}")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[selected_columns[1]], kde=True, color='blue', bins=20, edgecolor='black')
        plt.title(f'Distribution of {selected_columns[1]}')
        plt.xlabel(f'{selected_columns[1]}')
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

    # Boxplot of Selected Columns by Celestial Body
    if len(selected_columns) >= 1:
        st.write(f"### 💵 {selected_columns[0]} by Celestial Body")
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Celestial Body', y=selected_columns[0], data=df, palette='Set3')
        plt.xticks(rotation=45)
        plt.title(f'{selected_columns[0]} by Celestial Body')
        plt.grid(True)
        st.pyplot(plt)

    # Correlation Heatmap (only for selected numeric columns)
    st.write("### 🔥 Correlation Heatmap")
    numeric_df = df[selected_columns].select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.5})
        plt.title('Correlation Heatmap of Selected Features')
        st.pyplot(plt)
    else:
        st.warning("No numeric columns selected for correlation heatmap.")

    # Pairplot of Selected Features
    if len(selected_columns) > 1:
        st.write("### 🔗 Pairplot of Selected Features")
        sns.pairplot(df[selected_columns], diag_kind='kde', palette='coolwarm', plot_kws={'edgecolor': 'black'})
        plt.suptitle('Pairplot of Selected Features', y=1.02)
        st.pyplot(plt)

show_visualize_page()
