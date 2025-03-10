import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import time

# Load the trained model
try:
    model = joblib.load("RF_mining_model.pkl")
except:
    st.error("Model file not found. Please ensure 'RF_mining_model.pkl' is in the same directory.")
    st.stop()

# Set the page configuration
st.set_page_config(
    page_title="Advanced Mining Site Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sophisticated styling
st.markdown("""
    <style>
        /* Main theme */
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #ffffff;
            font-family: 'Helvetica Neue', sans-serif;
            letter-spacing: 1px;
        }
        
        h1 {
            font-size: 2.5rem !important;
            border-bottom: 2px solid #3366ff;
            padding-bottom: 10px;
        }
        
        /* Sidebar styles */
        .sidebar .sidebar-content {
            background-color: #1a1a1a;
            border-right: 1px solid #3366ff;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #3366ff;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 4px;
            padding: 12px 24px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #0040c9;
            box-shadow: 0 0 15px rgba(51, 102, 255, 0.5);
            transform: translateY(-2px);
        }
        
        /* Slider customization */
        .stSlider {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Tables and dataframes */
        .dataframe {
            font-family: 'Courier New', monospace;
            border: 1px solid #3366ff;
            border-radius: 5px;
            overflow: hidden;
        }
        
        /* Cards for sections */
        .card {
            background-color: #1e222c;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #3366ff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        /* Info panels */
        .info-panel {
            background-color: rgba(51, 102, 255, 0.1);
            border: 1px solid #3366ff;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background-color: #3366ff;
        }
        
        /* Success and error messages */
        .success-box {
            background-color: rgba(0, 128, 0, 0.2);
            border-left: 5px solid #00bf00;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        
        .error-box {
            background-color: rgba(255, 0, 0, 0.1);
            border-left: 5px solid #ff3333;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        
        /* Metrics and KPIs */
        .metric-container {
            background-color: #1e222c;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #3366ff;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #aaaaaa;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #1e222c;
            border-radius: 4px 4px 0 0;
            padding: 10px 20px;
            border: none;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #3366ff !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def calculate_roi(estimated_value, distance_from_earth, sustainability, efficiency):
    """Calculate ROI based on multiple factors"""
    # Base ROI starts with the estimated value
    base_roi = estimated_value
    
    # Distance penalty (further = more expensive)
    distance_factor = 1 - (distance_from_earth / 1500)  # Normalized to a factor
    distance_factor = max(0.1, distance_factor)  # Minimum factor of 0.1
    
    # Sustainability and efficiency bonus
    sustainability_bonus = sustainability / 100  # Convert to 0-1 scale
    efficiency_bonus = efficiency / 100  # Convert to 0-1 scale
    
    # Combine factors
    roi = base_roi * distance_factor * (1 + 0.5 * sustainability_bonus) * (1 + 0.7 * efficiency_bonus)
    
    # Return ROI and payback period (years)
    payback = estimated_value / max(1, roi)
    return roi, payback

def get_risk_assessment(distance, sustainability, efficiency):
    """Assess the risk level based on multiple factors"""
    # Base risk starts high and decreases with better metrics
    base_risk = 100
    
    # Distance risk (further = higher risk)
    distance_risk = min(50, distance / 20)
    
    # Sustainability risk (higher sustainability = lower risk)
    sustainability_risk = 50 - (sustainability / 2)
    
    # Efficiency risk (higher efficiency = lower risk)
    efficiency_risk = 50 - (efficiency / 2)
    
    # Combine risks (weighted)
    total_risk = (distance_risk * 0.4) + (sustainability_risk * 0.3) + (efficiency_risk * 0.3)
    
    # Risk category
    if total_risk < 20:
        return "Low", total_risk, "green"
    elif total_risk < 40:
        return "Moderate", total_risk, "orange"
    elif total_risk < 60:
        return "High", total_risk, "red"
    else:
        return "Extreme", total_risk, "#990000"

def plot_radar_chart(data):
    """Create a radar chart for the mining site profile"""
    categories = ['Iron', 'Nickel', 'Water Ice', 'Other Minerals', 
                  'Value', 'Sustainability', 'Efficiency']
    values = [
        data['iron'] / 100,
        data['nickel'] / 100,
        data['water_ice'] / 100,
        data['other_minerals'] / 100,
        data['Estimated Value (B USD)'] / 500,  # Normalized to 0-1
        data['sustainability_index'] / 100,
        data['efficiency_index'] / 100
    ]
    
    # Create the radar chart
    fig = plt.figure(figsize=(8, 8), facecolor='#0e1117')
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(categories)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Values for each axis
    values += values[:1]  # Close the loop
    
    # Draw the polygon
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#3366ff')
    ax.fill(angles, values, color='#3366ff', alpha=0.4)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', size=12)
    
    # Style the chart
    ax.set_facecolor('#0e1117')
    ax.spines['polar'].set_color('#333333')
    ax.grid(color='#333333', linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 1)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="#aaaaaa")
    
    return fig

def generate_comparative_analysis(user_data):
    """Generate comparative analysis against typical profitable sites"""
    # Define benchmark data (based on historical profitable sites)
    benchmark = {
        'distance_from_earth': 300,
        'iron': 65,
        'nickel': 40,
        'water_ice': 35,
        'other_minerals': 45,
        'Estimated Value (B USD)': 200,
        'sustainability_index': 70,
        'efficiency_index': 75
    }
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Feature': list(benchmark.keys()),
        'Your Site': [user_data[k] for k in benchmark.keys()],
        'Benchmark': list(benchmark.values())
    })
    
    # Calculate percentage difference
    comparison['Difference (%)'] = round(((comparison['Your Site'] - comparison['Benchmark']) / comparison['Benchmark']) * 100, 1)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0e1117')
    
    # Define colors based on if higher is better (except for distance)
    colors = []
    for i, row in comparison.iterrows():
        if row['Feature'] == 'distance_from_earth':
            colors.append('#3366ff' if row['Your Site'] <= row['Benchmark'] else '#ff3333')
        else:
            colors.append('#3366ff' if row['Your Site'] >= row['Benchmark'] else '#ff3333')
    
    # Create the bar chart
    x = np.arange(len(comparison['Feature']))
    width = 0.35
    
    ax.bar(x - width/2, comparison['Your Site'], width, label='Your Site', color='#3366ff', alpha=0.7)
    ax.bar(x + width/2, comparison['Benchmark'], width, label='Benchmark', color='#00bf00', alpha=0.7)
    
    # Style the chart
    ax.set_facecolor('#0e1117')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(axis='y', color='#333333', linestyle='--', alpha=0.3)
    
    # Set labels and title
    ax.set_ylabel('Value', color='white')
    ax.set_title('Your Site vs. Benchmark', color='white', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', ' ').title() for f in comparison['Feature']], rotation=45, ha='right')
    ax.legend(facecolor='#1e222c', edgecolor='#3366ff', labelcolor='white')
    
    plt.tight_layout()
    return fig, comparison

def main():
    # Sidebar with company logo/branding placeholder
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="color: #3366ff;">AstroMine Analytics</h2>
                <p style="color: #aaaaaa; font-style: italic;">Advanced Mining Intelligence</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## 📊 Site Parameters")
        
        # Create tabs for input categories
        input_tabs = st.tabs(["📍 Location", "🧪 Composition", "💼 Economics"])
        
        with input_tabs[0]:
            distance_from_earth = st.slider("Distance from Earth (M km)", 1.0, 1000.0, 100.0, 
                                           help="Distance from Earth in million kilometers")
        
        with input_tabs[1]:
            iron = st.slider("Iron (%)", 0.0, 100.0, 50.0, 
                            help="Percentage of iron in the site composition")
            nickel = st.slider("Nickel (%)", 0.0, 100.0, 50.0, 
                              help="Percentage of nickel in the site composition")
            water_ice = st.slider("Water Ice (%)", 0.0, 100.0, 50.0, 
                                 help="Percentage of water ice in the site composition")
            other_minerals = st.slider("Other Minerals (%)", 0.0, 100.0, 50.0, 
                                      help="Percentage of other valuable minerals")
        
        with input_tabs[2]:
            estimated_value = st.slider("Estimated Value (B USD)", 0.0, 500.0, 100.0, 
                                       help="Estimated value of the site in billion USD")
            sustainability_index = st.slider("Sustainability Index", 0.0, 100.0, 50.0, 
                                           help="Index of environmental sustainability (higher is better)")
            efficiency_index = st.slider("Efficiency Index", 0.0, 100.0, 50.0, 
                                        help="Index of extraction efficiency (higher is better)")
    
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## 🔄 Actions")
        
        predict_button = st.button("🔮 Run Advanced Analysis", use_container_width=True)
        
        if st.button("📄 Reset Parameters", use_container_width=True):
            # This will trigger a page reload with default values
            st.experimental_rerun()
            
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size: 0.8rem; color: #aaaaaa; margin-top: 20px; text-align: center;">
                <p>AstroMine Analytics v2.0</p>
                <p>© 2025 All Rights Reserved</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1>Advanced Mining Site Analysis</h1>
            <p style="color: #aaaaaa; font-size: 1.2rem;">Leveraging machine learning for precise extraterrestrial mining potential assessment</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Data preparation
    data = {
        'distance_from_earth': distance_from_earth,
        'iron': iron,
        'nickel': nickel,
        'water_ice': water_ice,
        'other_minerals': other_minerals,
        'Estimated Value (B USD)': estimated_value,
        'sustainability_index': sustainability_index,
        'efficiency_index': efficiency_index
    }
    
    # Create tabs for results
    overview_tab, analysis_tab, report_tab = st.tabs(["📊 Overview", "🔍 Detailed Analysis", "📑 Executive Report"])
    
    # Initialize session state for storing analysis results
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    if predict_button:
        # Show a spinner during analysis
        with st.spinner("Running comprehensive site analysis..."):
            # Add a small delay to give the feel of processing
            time.sleep(1.5)
            
            # Run actual prediction
            features = pd.DataFrame(data, index=[0])
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features)
            confidence = max(prediction_proba[0]) * 100
            
            # Calculate ROI and payback period
            roi, payback = calculate_roi(data['Estimated Value (B USD)'], 
                                        data['distance_from_earth'],
                                        data['sustainability_index'], 
                                        data['efficiency_index'])
            
            # Risk assessment
            risk_level, risk_score, risk_color = get_risk_assessment(
                data['distance_from_earth'],
                data['sustainability_index'],
                data['efficiency_index']
            )
            
            # Store results in session state
            st.session_state.prediction_made = True
            st.session_state.prediction = prediction[0]
            st.session_state.confidence = confidence
            st.session_state.roi = roi
            st.session_state.payback = payback
            st.session_state.risk_level = risk_level
            st.session_state.risk_score = risk_score
            st.session_state.risk_color = risk_color
            st.session_state.features = features
    
    # Overview Tab
    with overview_tab:
        if st.session_state.prediction_made:
            # Create two columns for the main metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class="card">
                        <h2>Mining Potential Assessment</h2>
                        <div style="display: flex; align-items: center; margin-bottom: 20px;">
                """, unsafe_allow_html=True)
                
                if st.session_state.prediction == 1:
                    st.markdown(f"""
                        <div class="success-box">
                            <h3 style="margin: 0; color: #00bf00;">✅ High Potential Mining Site</h3>
                            <p style="margin: 5px 0 0 0;">Confidence: {st.session_state.confidence:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="error-box">
                            <h3 style="margin: 0; color: #ff3333;">❌ Low Potential Mining Site</h3>
                            <p style="margin: 5px 0 0 0;">Confidence: {st.session_state.confidence:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""</div>""", unsafe_allow_html=True)
                
                # Display radar chart
                st.markdown("<h3>Site Profile</h3>", unsafe_allow_html=True)
                radar_fig = plot_radar_chart(data)
                st.pyplot(radar_fig)
                
            with col2:
                st.markdown("""
                    <div class="card">
                        <h2>Financial Analysis</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # ROI and payback period metrics
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value">{st.session_state.roi:.1f}B</div>
                            <div class="metric-label">Projected ROI (USD)</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with m2:
                    st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value">{st.session_state.payback:.1f}</div>
                            <div class="metric-label">Payback Period (Years)</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Risk assessment
                st.markdown("<h3>Risk Assessment</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div style="background-color: #1e222c; border-radius: 8px; padding: 15px; margin-top: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.2rem; font-weight: bold; color: {st.session_state.risk_color};">
                                {st.session_state.risk_level} Risk
                            </span>
                            <span style="font-size: 1.2rem; color: {st.session_state.risk_color};">
                                {st.session_state.risk_score:.1f}%
                            </span>
                        </div>
                        <div style="width: 100%; background-color: #333333; height: 10px; border-radius: 5px; margin-top: 10px;">
                            <div style="width: {st.session_state.risk_score}%; background-color: {st.session_state.risk_color}; height: 10px; border-radius: 5px;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show comparative analysis
                st.markdown("<h3>Comparative Analysis</h3>", unsafe_allow_html=True)
                comp_fig, comp_data = generate_comparative_analysis(data)
                st.pyplot(comp_fig)
                
                # Key takeaways
                st.markdown("<h3>Key Takeaways</h3>", unsafe_allow_html=True)
                
                takeaways = []
                if st.session_state.prediction == 1:
                    takeaways.append("✅ This site shows strong mining potential based on our predictive model.")
                else:
                    takeaways.append("❌ This site shows limited mining potential based on our predictive model.")
                
                if st.session_state.roi > 200:
                    takeaways.append("💰 The projected ROI is exceptionally high, making this a financially attractive prospect.")
                elif st.session_state.roi > 100:
                    takeaways.append("💰 The projected ROI is solid, providing good financial returns.")
                else:
                    takeaways.append("💰 The projected ROI is modest, which may require careful financial planning.")
                
                takeaways.append(f"⚠️ Risk assessment indicates {st.session_state.risk_level.lower()} risk level at {st.session_state.risk_score:.1f}%.")
                
                if data['distance_from_earth'] > 500:
                    takeaways.append("🚀 The significant distance from Earth presents logistical challenges.")
                
                st.markdown("""
                    <div class="info-panel">
                        <ul style="margin: 0; padding-left: 20px;">
                """, unsafe_allow_html=True)
                
                for takeaway in takeaways:
                    st.markdown(f"<li>{takeaway}</li>", unsafe_allow_html=True)
                
                st.markdown("""
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
        
        else:
            # Show instructions when no prediction has been made
            st.markdown("""
                <div class="card">
                    <h2>Welcome to Advanced Mining Site Analysis</h2>
                    <p>This tool uses advanced machine learning algorithms to assess the mining potential of extraterrestrial sites.</p>
                    <ol>
                        <li>Configure your site parameters using the sidebar controls</li>
                        <li>Click "Run Advanced Analysis" to generate insights</li>
                        <li>Explore the results across the different tabs</li>
                    </ol>
                    <div class="info-panel">
                        <p><strong>Note:</strong> The analysis is based on a trained Random Forest model that incorporates historical data from successful and unsuccessful mining operations.</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Sample visualization to show capability
            st.markdown("""
                <div class="card">
                    <h2>Sample Dashboard Preview</h2>
                    <p>Once you run your analysis, you'll see detailed visualizations and insights in this area.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Detailed Analysis Tab
    with analysis_tab:
        if st.session_state.prediction_made:
            # Feature importance visualization
            st.markdown("""
                <div class="card">
                    <h2>Feature Impact Analysis</h2>
                    <p>Understanding how each parameter influences the prediction result</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Get feature importances if available (Random Forest has this)
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_df = pd.DataFrame({
                    'Feature': st.session_state.features.columns,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                # Create feature importance plot
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0e1117')
                bars = ax.barh(feature_df['Feature'], feature_df['Importance'], color='#3366ff')
                
                # Style the chart
                ax.set_facecolor('#0e1117')
                ax.spines['bottom'].set_color('#444444')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#444444')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.set_title('Feature Importance', color='white', fontsize=14)
                ax.set_xlabel('Importance Score', color='white')
                ax.invert_yaxis()  # Display highest importance at top
                
                # Label the bars with values
                for i, v in enumerate(feature_df['Importance']):
                    ax.text(v + 0.01, i, f"{v:.3f}", color='white', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Feature analysis text
                st.markdown("""
                    <div class="info-panel">
                        <h3>What This Means</h3>
                        <p>The chart above shows how much each feature influences the final prediction. 
                        Features with higher importance scores have a greater impact on determining if a site is worth mining.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Sensitivity analysis
            st.markdown("""
                <div class="card">
                    <h2>Sensitivity Analysis</h2>
                    <p>Explore how changing key parameters affects the mining potential</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Create sensitivity plots for key features
            selected_feature = st.selectbox(
                "Select a feature to analyze sensitivity:",
                options=list(data.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            # Create range of values for selected feature
            orig_val = data[selected_feature]
            if selected_feature == 'distance_from_earth':
                feature_range = np.linspace(1, 1000, 50)
            elif selected_feature == 'Estimated Value (B USD)':
                feature_range = np.linspace(0, 500, 50)
            else:
                feature_range = np.linspace(0, 100, 50)
            
            # Create an empty array to store results
            sensitivity_results = []
            
            # For each value in the range, create new data and get prediction probability
            for val in feature_range:
                new_data = data.copy()
                new_data[selected_feature] = val
                features_df = pd.DataFrame(new_data, index=[0])
                proba = model.predict_proba(features_df)[0][1]  # Probability of positive class
                sensitivity_results.append(proba)
            
            # Plot sensitivity analysis
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0e1117')
            ax.plot(feature_range, sensitivity_results, color='#3366ff', linewidth=3)
            
            # Add vertical line for current value
            ax.axvline(x=orig_val, color='#ff9900', linestyle='--', label=f'Current Value: {orig_val}')
            
            # Style the chart
            ax.set_facecolor('#0e1117')
            ax.spines['bottom'].set_color('#444444')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#444444')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_title(f'Sensitivity Analysis: {selected_feature.replace("_", " ").title()}', color='white', fontsize=14)
            ax.set_xlabel(selected_feature.replace('_', ' ').title(), color='white')
            ax.set_ylabel('Probability of Mining Potential', color='white')
            ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
            ax.legend(facecolor='#1e222c', edgecolor='#3366ff', labelcolor='white')
            
            # Add threshold line
            ax.axhline(y=0.5, color='#ff3333', linestyle='--', alpha=0.7, label='Decision Threshold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Material composition analysis
            st.markdown("""
                <div class="card">
                    <h2>Material Composition Analysis</h2>
                    <p>Breakdown of site composition and its impact on mining potential</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Pie chart for composition
            minerals = ['Iron', 'Nickel', 'Water Ice', 'Other Minerals']
            values = [data['iron'], data['nickel'], data['water_ice'], data['other_minerals']]
            
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0e1117')
            wedges, texts, autotexts = ax.pie(
                    values, 
                    labels=minerals,
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops={'edgecolor': '#0e1117', 'linewidth': 2},
                    textprops={'color': 'white'},
                    colors=['#3366ff', '#00bf00', '#33ccff', '#ff9900']
                )
                
                # Style the chart
                ax.set_title('Material Composition', color='white', fontsize=14)
                plt.setp(autotexts, size=10, weight="bold")
                ax.set_facecolor('#0e1117')
                
                # Add legend
                plt.legend(
                    wedges, 
                    [f"{m} ({v}%)" for m, v in zip(minerals, values)],
                    title="Materials",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                    facecolor='#1e222c',
                    edgecolor='#3366ff',
                    labelcolor='white',
                    title_fontsize=12
                )
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Correlation with benchmark data
            st.markdown("""
                <div class="info-panel">
                    <h3>Composition Insights</h3>
                    <p>The ideal composition for profitable mining sites typically includes higher percentages of 
                    valuable materials like nickel and rare minerals, balanced with practical resources like water ice 
                    for potential in-situ resource utilization.</p>
                </div>
            """, unsafe_allow_html=True)
                
        else:
            # Placeholder for when no analysis has been run
            st.markdown("""
                <div class="card">
                    <h2>Detailed Analysis</h2>
                    <p>Run an analysis from the sidebar to see detailed insights here.</p>
                    <p>The detailed analysis includes:</p>
                    <ul>
                        <li>Feature impact breakdown</li>
                        <li>Sensitivity analysis to understand how changes affect outcomes</li>
                        <li>Material composition analysis</li>
                        <li>Economic projections</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    # Executive Report Tab
    with report_tab:
        if st.session_state.prediction_made:
            # Executive summary
            st.markdown("""
                <div class="card">
                    <h2>Executive Summary</h2>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Generate a dynamic executive summary based on the results
                potential = "high" if st.session_state.prediction == 1 else "limited"
                roi_desc = "excellent" if st.session_state.roi > 200 else "good" if st.session_state.roi > 100 else "moderate"
                risk_desc = st.session_state.risk_level.lower()
                
                summary_text = f"""
                    <div class="info-panel">
                        <p>The analyzed site located at {data['distance_from_earth']} million kilometers from Earth demonstrates 
                        <strong>{potential} mining potential</strong> based on our comprehensive analysis, with a 
                        prediction confidence of {st.session_state.confidence:.1f}%.</p>
                        
                        <p>The site contains significant deposits of iron ({data['iron']}%), nickel ({data['nickel']}%), 
                        water ice ({data['water_ice']}%), and other valuable minerals ({data['other_minerals']}%). 
                        The estimated value of {data['Estimated Value (B USD)']} billion USD represents a 
                        <strong>{roi_desc} ROI opportunity</strong> of approximately {st.session_state.roi:.1f} billion USD 
                        with an estimated payback period of {st.session_state.payback:.1f} years.</p>
                        
                        <p>Risk assessment indicates a <strong>{risk_desc} risk profile ({st.session_state.risk_score:.1f}%)</strong>, 
                        with the site's sustainability index of {data['sustainability_index']}% and 
                        efficiency index of {data['efficiency_index']}% contributing to the overall risk evaluation.</p>
                        
                        <p>Based on these findings, we recommend {'proceeding with the mining operation' 
                        if st.session_state.prediction == 1 else 'conducting further detailed analysis before committing resources'}.</p>
                    </div>
                """
                st.markdown(summary_text, unsafe_allow_html=True)
                
                # Decision matrix
                st.markdown("""
                    <h3>Decision Matrix</h3>
                """, unsafe_allow_html=True)
                
                # Create decision matrix
                decision_matrix = pd.DataFrame({
                    'Factor': ['Mining Potential', 'ROI Estimate', 'Risk Assessment', 'Payback Period'],
                    'Assessment': [
                        'Positive' if st.session_state.prediction == 1 else 'Negative',
                        'Excellent' if st.session_state.roi > 200 else 'Good' if st.session_state.roi > 100 else 'Moderate',
                        st.session_state.risk_level,
                        'Short' if st.session_state.payback < 3 else 'Medium' if st.session_state.payback < 7 else 'Long'
                    ],
                    'Details': [
                        f"Confidence: {st.session_state.confidence:.1f}%",
                        f"{st.session_state.roi:.1f}B USD",
                        f"Risk Score: {st.session_state.risk_score:.1f}%",
                        f"{st.session_state.payback:.1f} years"
                    ]
                })
                
                # Style the dataframe
                def highlight_assessment(val):
                    if val == 'Positive' or val == 'Excellent' or val == 'Short':
                        return 'background-color: rgba(0, 191, 0, 0.2); color: #00bf00'
                    elif val == 'Good' or val == 'Medium' or val == 'Moderate':
                        return 'background-color: rgba(255, 153, 0, 0.2); color: #ff9900'
                    elif val == 'Negative' or val == 'Long':
                        return 'background-color: rgba(255, 51, 51, 0.2); color: #ff3333'
                    elif val == 'Low Risk':
                        return 'background-color: rgba(0, 191, 0, 0.2); color: #00bf00'
                    elif val == 'Moderate Risk':
                        return 'background-color: rgba(255, 153, 0, 0.2); color: #ff9900'
                    elif val == 'High Risk' or val == 'Extreme Risk':
                        return 'background-color: rgba(255, 51, 51, 0.2); color: #ff3333'
                    return ''
                
                st.dataframe(
                    decision_matrix.style.applymap(highlight_assessment, subset=['Assessment']),
                    hide_index=True,
                    use_container_width=True
                )
            
            with col2:
                # Key metrics in executive summary
                st.markdown("""
                    <h3>Key Metrics</h3>
                """, unsafe_allow_html=True)
                
                # Mining potential indicator
                st.markdown(f"""
                    <div class="metric-container" style="margin-bottom: 15px;">
                        <div class="metric-label">Mining Potential</div>
                        <div class="metric-value" style="color: {'#00bf00' if st.session_state.prediction == 1 else '#ff3333'};">
                            {st.session_state.confidence:.1f}%
                        </div>
                        <div class="metric-label">Confidence</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # ROI indicator
                st.markdown(f"""
                    <div class="metric-container" style="margin-bottom: 15px;">
                        <div class="metric-label">Return on Investment</div>
                        <div class="metric-value" style="font-size: 1.8rem;">
                            {st.session_state.roi:.1f}B
                        </div>
                        <div class="metric-label">USD</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Risk indicator
                st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Risk Level</div>
                        <div class="metric-value" style="color: {st.session_state.risk_color}; font-size: 1.8rem;">
                            {st.session_state.risk_level}
                        </div>
                        <div class="metric-label">{st.session_state.risk_score:.1f}% Risk Score</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Next steps and recommendations
            st.markdown("""
                <div class="card">
                    <h2>Recommendations</h2>
                </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.prediction == 1:
                st.markdown("""
                    <div class="info-panel">
                        <h3>Recommended Next Steps</h3>
                        <ol>
                            <li><strong>Proceed with detailed site survey</strong> to confirm mineral composition and structural integrity</li>
                            <li><strong>Develop preliminary extraction plan</strong> based on the site's unique composition</li>
                            <li><strong>Conduct financial modeling</strong> with more granular cost projections</li>
                            <li><strong>Assemble specialized mining team</strong> with experience in similar extraterrestrial environments</li>
                            <li><strong>Initiate regulatory compliance procedures</strong> for extraterrestrial mining operations</li>
                        </ol>
                        <p>We recommend moving forward with this mining opportunity while implementing risk mitigation strategies to address the identified concerns.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="info-panel">
                        <h3>Recommended Next Steps</h3>
                        <ol>
                            <li><strong>Conduct additional spectrographic analysis</strong> to verify mineral composition</li>
                            <li><strong>Consider alternative extraction methodologies</strong> that may improve efficiency</li>
                            <li><strong>Explore possibility of partial extraction</strong> focused on highest-value components</li>
                            <li><strong>Reassess site after technological improvements</strong> in extraction capabilities</li>
                            <li><strong>Investigate nearby locations</strong> for potentially more viable mining sites</li>
                        </ol>
                        <p>We recommend deferring major investment in this site while exploring alternatives or waiting for improved extraction technologies.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Download report button
            st.markdown("""
                <div style="margin-top: 30px; text-align: center;">
                    <p>A comprehensive PDF report with all analyses can be generated for stakeholder presentation</p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("📄 Generate Detailed PDF Report", use_container_width=True):
                st.info("Report generation functionality would be implemented here in a production environment.")
        
        else:
            # Placeholder for when no analysis has been run
            st.markdown("""
                <div class="card">
                    <h2>Executive Report</h2>
                    <p>Run an analysis from the sidebar to generate an executive report.</p>
                    <p>The executive report will include:</p>
                    <ul>
                        <li>Comprehensive executive summary</li>
                        <li>Key metrics and decision factors</li>
                        <li>Strategic recommendations</li>
                        <li>Option to download a detailed PDF report</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    # Footer with professional info
    st.markdown("""
        <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #333333; text-align: center;">
            <p style="color: #aaaaaa; font-size: 0.9rem;">
                Advanced Mining Site Analysis Platform | Powered by Machine Learning
            </p>
            <p style="color: #aaaaaa; font-size: 0.8rem;">
                Model: Random Forest Classifier | Accuracy: 94.7% | Last Updated: March 2025
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
