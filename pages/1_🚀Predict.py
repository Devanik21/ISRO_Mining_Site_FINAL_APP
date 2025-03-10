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

# Page configuration
st.set_page_config(
    page_title="Advanced Mining Site Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: #ffffff; }
        h1, h2, h3 { color: #ffffff; font-family: 'Helvetica Neue', sans-serif; }
        h1 { font-size: 2.5rem !important; border-bottom: 2px solid #3366ff; padding-bottom: 10px; }
        
        .stButton>button { background-color: #3366ff; color: white; font-weight: bold; }
        
        .card { background-color: #1e222c; border-radius: 10px; padding: 20px; 
                margin-bottom: 20px; border-left: 5px solid #3366ff; }
        
        .info-panel { background-color: rgba(51, 102, 255, 0.1); border: 1px solid #3366ff; 
                     border-radius: 5px; padding: 15px; margin: 15px 0; }
        
        .success-box { background-color: rgba(0, 128, 0, 0.2); border-left: 5px solid #00bf00; 
                      padding: 20px; border-radius: 5px; margin: 20px 0; }
        
        .error-box { background-color: rgba(255, 0, 0, 0.1); border-left: 5px solid #ff3333; 
                    padding: 20px; border-radius: 5px; margin: 20px 0; }
        
        .metric-container { background-color: #1e222c; border-radius: 8px; padding: 15px; 
                           text-align: center; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2); }
        
        .metric-value { font-size: 2rem; font-weight: bold; color: #3366ff; }
        .metric-label { font-size: 0.9rem; color: #aaaaaa; }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def calculate_roi(estimated_value, distance_from_earth, sustainability, efficiency):
    """Calculate ROI based on multiple factors"""
    distance_factor = max(0.1, 1 - (distance_from_earth / 1500))
    sustainability_bonus = sustainability / 100
    efficiency_bonus = efficiency / 100
    
    roi = estimated_value * distance_factor * (1 + 0.5 * sustainability_bonus) * (1 + 0.7 * efficiency_bonus)
    payback = estimated_value / max(1, roi)
    return roi, payback

def get_risk_assessment(distance, sustainability, efficiency):
    """Assess risk level based on multiple factors"""
    distance_risk = min(50, distance / 20)
    sustainability_risk = 50 - (sustainability / 2)
    efficiency_risk = 50 - (efficiency / 2)
    
    total_risk = (distance_risk * 0.4) + (sustainability_risk * 0.3) + (efficiency_risk * 0.3)
    
    if total_risk < 20:
        return "Low", total_risk, "green"
    elif total_risk < 40:
        return "Moderate", total_risk, "orange"
    elif total_risk < 60:
        return "High", total_risk, "red"
    else:
        return "Extreme", total_risk, "#990000"

def plot_radar_chart(data):
    """Create radar chart for site profile"""
    categories = ['Iron', 'Nickel', 'Water Ice', 'Other Minerals', 
                  'Value', 'Sustainability', 'Efficiency']
    values = [
        data['iron'] / 100,
        data['nickel'] / 100,
        data['water_ice'] / 100,
        data['other_minerals'] / 100,
        data['Estimated Value (B USD)'] / 500,
        data['sustainability_index'] / 100,
        data['efficiency_index'] / 100
    ]
    
    fig = plt.figure(figsize=(8, 8), facecolor='#0e1117')
    ax = fig.add_subplot(111, polar=True)
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    values += values[:1]  # Close the loop
    
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#3366ff')
    ax.fill(angles, values, color='#3366ff', alpha=0.4)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', size=12)
    
    ax.set_facecolor('#0e1117')
    ax.spines['polar'].set_color('#333333')
    ax.grid(color='#333333', linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 1)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="#aaaaaa")
    
    return fig

def generate_comparative_analysis(user_data):
    """Generate comparative analysis against typical profitable sites"""
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
    
    comparison = pd.DataFrame({
        'Feature': list(benchmark.keys()),
        'Your Site': [user_data[k] for k in benchmark.keys()],
        'Benchmark': list(benchmark.values())
    })
    
    comparison['Difference (%)'] = round(((comparison['Your Site'] - comparison['Benchmark']) / comparison['Benchmark']) * 100, 1)
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0e1117')
    
    x = np.arange(len(comparison['Feature']))
    width = 0.35
    
    ax.bar(x - width/2, comparison['Your Site'], width, label='Your Site', color='#3366ff', alpha=0.7)
    ax.bar(x + width/2, comparison['Benchmark'], width, label='Benchmark', color='#00bf00', alpha=0.7)
    
    ax.set_facecolor('#0e1117')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(axis='y', color='#333333', linestyle='--', alpha=0.3)
    
    ax.set_ylabel('Value', color='white')
    ax.set_title('Your Site vs. Benchmark', color='white', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', ' ').title() for f in comparison['Feature']], rotation=45, ha='right')
    ax.legend(facecolor='#1e222c', edgecolor='#3366ff', labelcolor='white')
    
    plt.tight_layout()
    return fig, comparison

def main():
    # Sidebar with branding and inputs
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="color: #3366ff;">AstroMine Analytics</h2>
                <p style="color: #aaaaaa; font-style: italic;">Advanced Mining Intelligence</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## 📊 Site Parameters")
        
        # Input tabs
        input_tabs = st.tabs(["📍 Location", "🧪 Composition", "💼 Economics"])
        
        with input_tabs[0]:
            distance_from_earth = st.slider("Distance from Earth (M km)", 1.0, 1000.0, 100.0)
        
        with input_tabs[1]:
            iron = st.slider("Iron (%)", 0.0, 100.0, 50.0)
            nickel = st.slider("Nickel (%)", 0.0, 100.0, 50.0)
            water_ice = st.slider("Water Ice (%)", 0.0, 100.0, 50.0)
            other_minerals = st.slider("Other Minerals (%)", 0.0, 100.0, 50.0)
        
        with input_tabs[2]:
            estimated_value = st.slider("Estimated Value (B USD)", 0.0, 500.0, 100.0)
            sustainability_index = st.slider("Sustainability Index", 0.0, 100.0, 50.0)
            efficiency_index = st.slider("Efficiency Index", 0.0, 100.0, 50.0)
    
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## 🔄 Actions")
        
        predict_button = st.button("🔮 Run Advanced Analysis", use_container_width=True)
        
        if st.button("📄 Reset Parameters", use_container_width=True):
            st.rerun()
s

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size: 0.8rem; color: #aaaaaa; margin-top: 20px; text-align: center;">
                <p>AstroMine Analytics v2.0</p>
                <p>© 2025 All Rights Reserved</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Header
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
    
    # Initialize session state for results
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    # Run analysis when button clicked
    if predict_button:
        with st.spinner("Running comprehensive site analysis..."):
            time.sleep(1.5)
            
            features = pd.DataFrame(data, index=[0])
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features)
            confidence = max(prediction_proba[0]) * 100
            
            roi, payback = calculate_roi(data['Estimated Value (B USD)'], 
                                        data['distance_from_earth'],
                                        data['sustainability_index'], 
                                        data['efficiency_index'])
            
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
    
    # Overview Tab Content
    with overview_tab:
        if st.session_state.prediction_made:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="card"><h2>Mining Potential Assessment</h2></div>', 
                           unsafe_allow_html=True)
                
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
                
                st.markdown("<h3>Site Profile</h3>", unsafe_allow_html=True)
                radar_fig = plot_radar_chart(data)
                st.pyplot(radar_fig)
                
            with col2:
                st.markdown('<div class="card"><h2>Financial Analysis</h2></div>', 
                           unsafe_allow_html=True)
                
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
                
                # Comparative analysis
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
                    takeaways.append("💰 The projected ROI is exceptionally high.")
                elif st.session_state.roi > 100:
                    takeaways.append("💰 The projected ROI is solid, providing good financial returns.")
                else:
                    takeaways.append("💰 The projected ROI is modest, which may require careful financial planning.")
                
                takeaways.append(f"⚠️ Risk assessment indicates {st.session_state.risk_level.lower()} risk level at {st.session_state.risk_score:.1f}%.")
                
                st.markdown("""<div class="info-panel"><ul style="margin: 0; padding-left: 20px;">""", 
                           unsafe_allow_html=True)
                
                for takeaway in takeaways:
                    st.markdown(f"<li>{takeaway}</li>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
        
        else:
            # Instructions when no prediction has been made
            st.markdown("""
                <div class="card">
                    <h2>Welcome to Advanced Mining Site Analysis</h2>
                    <p>This tool uses advanced machine learning algorithms to assess the mining potential of extraterrestrial sites.</p>
                    <ol>
                        <li>Configure your site parameters using the sidebar controls</li>
                        <li>Click "Run Advanced Analysis" to generate insights</li>
                        <li>Explore the results across the different tabs</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
    
    # Analysis Tab Content
    with analysis_tab:
        if st.session_state.prediction_made:
            st.markdown("""
                <div class="card">
                    <h2>Feature Impact Analysis</h2>
                    <p>Understanding how each parameter influences the prediction result</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_df = pd.DataFrame({
                    'Feature': st.session_state.features.columns,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0e1117')
                bars = ax.barh(feature_df['Feature'], feature_df['Importance'], color='#3366ff')
                
                ax.set_facecolor('#0e1117')
                ax.spines['bottom'].set_color('#444444')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#444444')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.set_title('Feature Importance', color='white', fontsize=14)
                ax.invert_yaxis()  # Display highest importance at top
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Sensitivity analysis
            st.markdown("""
                <div class="card">
                    <h2>Sensitivity Analysis</h2>
                    <p>Explore how changing key parameters affects the mining potential</p>
                </div>
            """, unsafe_allow_html=True)
            
            selected_feature = st.selectbox(
                "Select a feature to analyze sensitivity:",
                options=list(data.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            # Create material composition chart if time permits
            st.markdown("""
                <div class="card">
                    <h2>Material Composition Analysis</h2>
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
            
            ax.set_title('Material Composition', color='white', fontsize=14)
            plt.setp(autotexts, size=10, weight="bold")
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            # Placeholder for when no analysis has been run
            st.markdown("""
                <div class="card">
                    <h2>Detailed Analysis</h2>
                    <p>Run an analysis from the sidebar to see detailed insights here.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Report Tab Content
    with report_tab:
        if st.session_state.prediction_made:
            st.markdown("""
                <div class="card">
                    <h2>Executive Summary</h2>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Generate executive summary
                potential = "high" if st.session_state.prediction == 1 else "limited"
                roi_desc = "excellent" if st.session_state.roi > 200 else "good" if st.session_state.roi > 100 else "moderate"
                risk_desc = st.session_state.risk_level.lower()
                
                summary_text = f"""
                    <div class="info-panel">
                        <p>The analyzed site located at {data['distance_from_earth']} million kilometers from Earth demonstrates 
                        <strong>{potential} mining potential</strong> with {st.session_state.confidence:.1f}% confidence.</p>
                        
                        <p>The site contains iron ({data['iron']}%), nickel ({data['nickel']}%), 
                        water ice ({data['water_ice']}%), and other minerals ({data['other_minerals']}%). 
                        Estimated value: {data['Estimated Value (B USD)']} billion USD with
                        <strong>{roi_desc} ROI</strong> of {st.session_state.roi:.1f} billion USD 
                        and payback period of {st.session_state.payback:.1f} years.</p>
                        
                        <p>Risk assessment: <strong>{risk_desc} risk ({st.session_state.risk_score:.1f}%)</strong></p>
                        
                        <p>Recommendation: {'Proceed with mining operation' 
                        if st.session_state.prediction == 1 else 'Conduct further analysis before committing resources'}.</p>
                    </div>
                """
                st.markdown(summary_text, unsafe_allow_html=True)
            
            with col2:
                # Key metrics
                st.markdown("<h3>Key Metrics</h3>", unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="metric-container" style="margin-bottom: 15px;">
                        <div class="metric-label">Mining Potential</div>
                        <div class="metric-value" style="color: {'#00bf00' if st.session_state.prediction == 1 else '#ff3333'};">
                            {st.session_state.confidence:.1f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="metric-container" style="margin-bottom: 15px;">
                        <div class="metric-label">Return on Investment</div>
                        <div class="metric-value">{st.session_state.roi:.1f}B USD</div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Risk Level</div>
                        <div class="metric-value" style="color: {st.session_state.risk_color};">
                            {st.session_state.risk_level}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Recommendations section
            st.markdown('<div class="card"><h2>Recommendations</h2></div>', unsafe_allow_html=True)
            
            if st.session_state.prediction == 1:
                st.markdown("""
                    <div class="info-panel">
                        <h3>Recommended Next Steps</h3>
                        <ol>
                            <li><strong>Proceed with detailed site survey</strong></li>
                            <li><strong>Develop preliminary extraction plan</strong></li>
                            <li><strong>Conduct financial modeling</strong></li>
                            <li><strong>Assemble specialized mining team</strong></li>
                            <li><strong>Initiate regulatory compliance procedures</strong></li>
                        </ol>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="info-panel">
                        <h3>Recommended Next Steps</h3>
                        <ol>
                            <li><strong>Conduct additional analysis</strong></li>
                            <li><strong>Consider alternative extraction methodologies</strong></li>
                            <li><strong>Explore partial extraction options</strong></li>
                            <li><strong>Investigate nearby locations</strong></li>
                        </ol>
                    </div>
                """, unsafe_allow_html=True)
            
            if st.button("📄 Generate Detailed PDF Report", use_container_width=True):
                st.info("Report generation functionality would be implemented here in production.")
        
        else:
            # Placeholder when no analysis has been run
            st.markdown("""
                <div class="card">
                    <h2>Executive Report</h2>
                    <p>Run an analysis from the sidebar to generate an executive report.</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
