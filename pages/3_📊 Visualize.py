import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_visualize_page():
    st.title("Mining Site Visualization")
    st.write("Visualize mining site data to gain insights.")
    
    # Load dataset
    df = pd.read_csv("space_mining_dataset.csv")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Seaborn Style
    sns.set_style("darkgrid")
    sns.set_palette("viridis")
    
    # Scatter Plot
    st.write("### Scatter Plot")
    x_axis = st.selectbox("Select X-axis column", numeric_columns, index=0, key="scatter_x")
    y_axis = st.selectbox("Select Y-axis column", numeric_columns, index=1, key="scatter_y")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df[categorical_columns[0]] if categorical_columns else None, palette="coolwarm")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'Scatter Plot: {x_axis} vs {y_axis}')
    st.pyplot(plt)
    
    # Histogram
    st.write("### Histogram")
    hist_col = st.selectbox("Select column for histogram", numeric_columns, index=0, key="hist")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[hist_col], kde=True, bins=30, color='purple')
    plt.xlabel(hist_col)
    plt.ylabel("Frequency")
    plt.title(f'Distribution of {hist_col}')
    st.pyplot(plt)
    
    # Pie Chart
    st.write("### Pie Chart")
    pie_col = st.selectbox("Select categorical column for Pie Chart", categorical_columns, index=0, key="pie")
    pie_data = df[pie_col].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title(f'Distribution of {pie_col}')
    st.pyplot(plt)
    
    # Boxplot
    st.write("### Boxplot")
    box_x = st.selectbox("Select categorical column for Boxplot", categorical_columns, index=0, key="box_x")
    box_y = st.selectbox("Select numeric column for Boxplot", numeric_columns, index=0, key="box_y")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df[box_x], y=df[box_y], palette="coolwarm")
    plt.xticks(rotation=45)
    plt.title(f'Boxplot: {box_x} vs {box_y}')
    st.pyplot(plt)
    
    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='black')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)
    
    # Pairplot
    st.write("### Pairplot")
    selected_features = st.multiselect("Select features for Pairplot", numeric_columns, default=numeric_columns[:3], key="pairplot")
    if selected_features:
        sns.pairplot(df[selected_features], diag_kind='kde', palette='magma')
        st.pyplot(plt)
    
    # Additional 10 Visualizations
    visualizations = {
        "Violin Plot": lambda: sns.violinplot(x=df[st.selectbox("Select categorical column for Violin Plot", categorical_columns, key="violin_x")],
                                               y=df[st.selectbox("Select numeric column for Violin Plot", numeric_columns, key="violin_y")], palette="cubehelix"),
        
        "Swarm Plot": lambda: sns.swarmplot(x=df[st.selectbox("Select categorical column for Swarm Plot", categorical_columns, key="swarm_x")],
                                             y=df[st.selectbox("Select numeric column for Swarm Plot", numeric_columns, key="swarm_y")], palette="coolwarm"),
        
        "Reg Plot": lambda: sns.regplot(x=df[st.selectbox("Select X for Reg Plot", numeric_columns, key="reg_x")],
                                         y=df[st.selectbox("Select Y for Reg Plot", numeric_columns, key="reg_y")], scatter_kws={'alpha':0.5}, line_kws={'color':'red'}),
        
        "Bar Plot": lambda: sns.barplot(x=df[st.selectbox("Select categorical column for Bar Plot", categorical_columns, key="bar_x")],
                                         y=df[st.selectbox("Select numeric column for Bar Plot", numeric_columns, key="bar_y")], palette="Set2"),
        
        "Count Plot": lambda: sns.countplot(x=df[st.selectbox("Select categorical column for Count Plot", categorical_columns, key="count_x")], palette="viridis"),
        
        "Heatmap (Clustered)": lambda: sns.clustermap(df[numeric_columns].corr(), cmap="coolwarm", annot=True),
        
        "KDE Plot": lambda: sns.kdeplot(x=df[st.selectbox("Select X for KDE Plot", numeric_columns, key="kde_x")],
                                         y=df[st.selectbox("Select Y for KDE Plot", numeric_columns, key="kde_y")], cmap="mako", fill=True),
        
        "Hexbin Plot": lambda: plt.hexbin(df[st.selectbox("Select X for Hexbin Plot", numeric_columns, key="hexbin_x")],
                                           df[st.selectbox("Select Y for Hexbin Plot", numeric_columns, key="hexbin_y")], gridsize=30, cmap="coolwarm"),
        
        "Facet Grid": lambda: sns.FacetGrid(df, col=st.selectbox("Select categorical column for Facet Grid", categorical_columns, key="facet"))
                              .map(sns.histplot, st.selectbox("Select numeric column for Facet Grid", numeric_columns, key="facet_num")),
        
        "Joint Plot": lambda: sns.jointplot(x=df[st.selectbox("Select X for Joint Plot", numeric_columns, key="joint_x")],
                                             y=df[st.selectbox("Select Y for Joint Plot", numeric_columns, key="joint_y")], kind="hex", cmap="coolwarm")
    }
    
    selected_viz = st.selectbox("Select Additional Visualization", list(visualizations.keys()))
    plt.figure(figsize=(10, 6))
    visualizations[selected_viz]()
    st.pyplot(plt)
    
show_visualize_page()
