import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.db_setup import get_engine, load_data
from src.eda import null_report, cohort_sizes, overall_funnel_rates, plot_class_imbalance
from src.q1_cohort_analysis import compute_cohort_metrics, rank_cohorts, top_3_cohorts, METRIC_JUSTIFICATION
from src.q2_sql_query import get_q2_result, run_query, COHORT_DATE_AGGREGATION_QUERY, COHORT_ROLLUP_QUERY
from src.feature_engineering import build_features, get_train_test_split
from src.models import train_all_models, load_models
from src.evaluation import summary_table, plot_all_confusion_matrices, plot_feature_importances, evaluate_model

# Streamlit Page Config
st.set_page_config(
    page_title="Vahan.ai — Product Analytics Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Visual Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E293B;
        background: linear-gradient(135deg, #0F172A 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .justification-box {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_app_data():
    engine = get_engine()
    df = load_data(engine)
    return df

try:
    df = load_app_data()
except Exception as e:
    st.error(f"Error loading database: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.image("https://vahan.ai/wp-content/uploads/2021/04/vahan-logo.png", width=160)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "📊 Executive Overview",
        "🏆 Q1: Cohort Performance Ranking",
        "⚡ Q2: SQL Query Engine (PostgreSQL)",
        "🤖 Q3: ML Benchmarks & Fine-Tuning",
        "🎯 Q3: Key Drivers & Features",
        "🔮 Real-Time FT Conversion Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Vahan.ai Product Analytics Case Study")
st.sidebar.caption("Built with Streamlit & PostgreSQL")

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if page == "📊 Executive Overview":
    st.markdown('<div class="main-header">Vahan.ai — Product Analytics Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Lead-to-First-Trip (FT) Conversion Funnel Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    total_leads = len(df)
    ft_count = df['ft_after_upload'].sum()
    ft_rate = (ft_count / total_leads) * 100
    connect_rate = (df['connected'].sum() / df['attempted'].sum()) * 100
    
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_leads:,}</div><div class="metric-label">Total Uploaded Leads</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{ft_count:,}</div><div class="metric-label">First Trip Conversions</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{ft_rate:.2f}%</div><div class="metric-label">FT Conversion Rate</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{connect_rate:.1f}%</div><div class="metric-label">Call Connect Rate</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Conversion Funnel Stage Drop-off")
        funnel_df = overall_funnel_rates(df)
        st.dataframe(funnel_df, use_container_width=True)
        
    with col_right:
        st.subheader("Target Class Imbalance (FT=0 vs FT=1)")
        fig_imb = plot_class_imbalance(df)
        st.pyplot(fig_imb)

# ==========================================
# PAGE 2: Q1 COHORT PERFORMANCE RANKING
# ==========================================
elif page == "🏆 Q1: Cohort Performance Ranking":
    st.markdown('<div class="main-header">Q1 — Top Lead-Source Cohorts</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating cohort performance on bottom-line business metrics</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="justification-box"><b>Metric Justification & Strategy:</b><br>{METRIC_JUSTIFICATION}</div>', unsafe_allow_html=True)
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        min_leads_threshold = st.slider("Minimum Lead Volume Filter (Exclude noise)", min_value=1, max_value=200, value=50)
    with col_filter2:
        sort_metric = st.selectbox("Primary Ranking Metric", ["FT Conversion Rate (%)", "Connect Rate (%)", "Interest Rate (%)", "Total Leads"])
        
    metrics_df = compute_cohort_metrics(df)
    ranked_df = rank_cohorts(metrics_df, min_leads=min_leads_threshold)
    
    st.subheader("Top 3 Cohorts")
    top3 = top_3_cohorts(ranked_df)
    st.dataframe(top3[['rank', 'lead_source', 'total_leads', 'ft_conversions', 'ft_conversion_rate_pct', 'connect_rate_pct', 'interest_rate_pct']], use_container_width=True)
    
    st.subheader("Full Cohort Performance Table")
    st.dataframe(ranked_df, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=ranked_df, y='lead_source', x='ft_conversion_rate_pct', palette='Blues_r', ax=ax)
    ax.set_title('Cohort FT Conversion Rate (%)', fontweight='bold')
    st.pyplot(fig)

# ==========================================
# PAGE 3: Q2 SQL QUERY ENGINE
# ==========================================
elif page == "⚡ Q2: SQL Query Engine (PostgreSQL)":
    st.markdown('<div class="main-header">Q2 — PostgreSQL SQL Aggregation Query</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executing native SQL queries against local PostgreSQL database <code>vahan_case_study</code></div>', unsafe_allow_html=True)
    
    engine = get_engine()
    
    query_type = st.radio("Select SQL Query View", ["Cohort x Upload Date Grain (Q2 Standard)", "Cohort Rollup Summary"])
    
    if query_type == "Cohort x Upload Date Grain (Q2 Standard)":
        query_text = st.text_area("PostgreSQL SQL Query", COHORT_DATE_AGGREGATION_QUERY, height=260)
    else:
        query_text = st.text_area("PostgreSQL SQL Query", COHORT_ROLLUP_QUERY, height=260)
        
    if st.button("Execute SQL Query"):
        with st.spinner("Executing query against PostgreSQL..."):
            res_df = run_query(engine, query_text)
            st.success(f"Query returned {len(res_df)} rows.")
            st.dataframe(res_df, use_container_width=True)
            
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Query Result CSV", data=csv, file_name="vahan_sql_aggregation.csv", mime="text/csv")

# ==========================================
# PAGE 4: Q3 ML BENCHMARKS & FINE-TUNING
# ==========================================
elif page == "🤖 Q3: ML Benchmarks & Fine-Tuning":
    st.markdown('<div class="main-header">Q3 — Predictive Machine Learning Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparing Logistic Regression, Random Forest, XGBoost, and Fine-Tuned XGBoost</div>', unsafe_allow_html=True)
    
    X, y, encoders = build_features(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = load_models()
    if not models or len(models) < 3:
        with st.spinner("Training ML models..."):
            models = train_all_models(X_train, y_train)
            
    st.subheader("Model Performance Benchmark")
    sum_df = summary_table(models, X_test, y_test)
    st.dataframe(sum_df, use_container_width=True)
    
    st.subheader("Confusion Matrices (Class Imbalance Evaluation)")
    fig_cm = plot_all_confusion_matrices(models, X_test, y_test)
    st.pyplot(fig_cm)

# ==========================================
# PAGE 5: Q3 KEY DRIVERS & FEATURES
# ==========================================
elif page == "🎯 Q3: Key Drivers & Features":
    st.markdown('<div class="main-header">Q3 — Operational Drivers of FT Conversion</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Feature importances across trained models</div>', unsafe_allow_html=True)
    
    X, y, _ = build_features(df)
    models = load_models()
    
    fig_imp = plot_feature_importances(models, list(X.columns))
    st.pyplot(fig_imp)
    
    st.markdown("""
    ### **Business Insights & Key Takeaways:**
    1. **Connection & Intent dominate:** `connected` and `interested` are consistently ranked as the #1 and #2 predictors across Random Forest, XGBoost, and Logistic Regression.
    2. **Channel Sourcing vs Call Responsiveness:** Lead source cohort identity has secondary predictive weight compared to whether a candidate actually picks up and expresses interest.
    3. **Actionable Operational Strategy:** Focus tele-calling speed on high-intent candidates (`interested = 1`) to drive them to onboarding (`OB_after_upload = 1`) and complete their First Trip (FT).
    """)

# ==========================================
# PAGE 6: REAL-TIME FT CONVERSION PREDICTOR
# ==========================================
elif page == "🔮 Real-Time FT Conversion Predictor":
    st.markdown('<div class="main-header">Real-Time Candidate FT Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive tool to estimate First Trip conversion probability for candidate leads</div>', unsafe_allow_html=True)
    
    X, y, encoders = build_features(df)
    models = load_models()
    model = models.get('Fine-Tuned XGBoost', models.get('XGBoost', list(models.values())[0]))
    
    st.sidebar.subheader("Model Selected for Prediction")
    st.sidebar.info(f"Using **{type(model).__name__}**")
    
    col_a, col_b = st.columns(2)
    with col_a:
        lead_source_input = st.selectbox("Lead Source Cohort", encoders['lead_source_encoder'].classes_)
        attempted_input = st.selectbox("Was Candidate Attempted/Called?", [1, 0])
        connected_input = st.selectbox("Did the Call Connect?", [1, 0])
        tag_filled_input = st.selectbox("Is Candidate Tag Form Filled?", [1, 0])
        
    with col_b:
        interested_input = st.selectbox("Expressed Interest?", [1, 0])
        ob_upload_input = st.selectbox("Onboarding After Upload Initiated?", [1, 0])
        ob_first_attempt_input = st.selectbox("Onboarding After First Attempt?", [1, 0])
        attempt_per_lead_input = st.number_input("Call Attempts per Lead", min_value=0.0, max_value=10.0, value=1.0)
        p50_hrs_input = st.number_input("Speed to First Call (P50 Hours)", min_value=0.0, max_value=500.0, value=float(encoders['p50_median']))
        
    if st.button("Predict FT Conversion Probability"):
        input_data = pd.DataFrame([{
            'attempted': attempted_input,
            'connected': connected_input,
            'attempt_per_lead': attempt_per_lead_input,
            'tag_filled': tag_filled_input,
            'interested': interested_input,
            'ob_after_upload': ob_upload_input,
            'ob_after_first_attempt': ob_first_attempt_input,
            'upload_to_first_attempt_p50_hrs': p50_hrs_input,
            'day_of_week': 3,
            'day_of_month': 18,
            'lead_source_encoded': encoders['lead_source_encoder'].transform([lead_source_input])[0]
        }])
        
        # Ensure column order matches X
        input_data = input_data[X.columns]
        
        prob = model.predict_proba(input_data)[0, 1]
        pred_class = int(prob >= 0.5)
        
        st.markdown("---")
        st.subheader("Prediction Result")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Predicted FT Probability", f"{prob * 100:.2f}%")
        with res_col2:
            if prob >= 0.5:
                st.success("High Conversion Potential (Likely to Complete First Trip)")
            else:
                st.warning("Low Conversion Probability (Follow-up Recommended)")
