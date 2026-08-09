import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import importlib
import src.models
import src.evaluation
import src.feature_engineering
importlib.reload(src.models)
importlib.reload(src.evaluation)
importlib.reload(src.feature_engineering)

from src.db_setup import get_engine, load_data
from src.eda import null_report, cohort_sizes, overall_funnel_rates, plot_class_imbalance
from src.q1_cohort_analysis import compute_cohort_metrics, rank_cohorts, top_3_cohorts, METRIC_JUSTIFICATION
from src.q2_sql_query import get_q2_result, run_query, COHORT_DATE_AGGREGATION_QUERY, COHORT_ROLLUP_QUERY
from src.feature_engineering import build_features, get_train_test_split
from src.models import train_all_models, load_models, tune_best_model, save_models
from src.evaluation import summary_table, plot_all_confusion_matrices, plot_feature_importances, evaluate_model
import plotly.graph_objects as go

def plot_plotly_funnel(df: pd.DataFrame):
    """Generate a clean Plotly horizontal bar chart for conversion funnel stage drop-off."""
    funnel_df = overall_funnel_rates(df)
    fig = go.Figure(go.Bar(
        x=funnel_df['pct_of_total'],
        y=funnel_df['stage'],
        orientation='h',
        text=[f"<b>{count:,}</b> ({pct:.2f}%)" for count, pct in zip(funnel_df['count'], funnel_df['pct_of_total'])],
        textposition='outside',
        marker=dict(
            color=['#F59E0B', '#EAB308', '#D97706', '#B45309', '#78350F', '#451A03'],
            line=dict(color='#FFFFFF', width=1)
        ),
        cliponaxis=False
    ))
    fig.update_layout(
        margin=dict(l=10, r=90, t=10, b=30),
        height=330,
        yaxis=dict(autorange="reversed", title="", tickfont=dict(size=12, color="#0F172A", weight="bold")),
        xaxis=dict(title="% of Total Uploaded Leads", range=[0, 115], gridcolor="#E2E8F0"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155")
    )
    return fig

def plot_plotly_class_imbalance(df: pd.DataFrame):
    """Generate a clean Plotly bar chart for target class imbalance."""
    counts = df['ft_after_upload'].value_counts()
    no_ft = int(counts.get(0, 0))
    ft = int(counts.get(1, 0))
    total = no_ft + ft
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['No FT (0)', 'FT Conversion (1)'],
        y=[no_ft, ft],
        text=[f"<b>{no_ft:,}</b><br>({(no_ft/total)*100:.2f}%)", f"<b>{ft:,}</b><br>({(ft/total)*100:.2f}%)"],
        textposition='outside',
        marker=dict(
            color=['#94A3B8', '#F59E0B'],
            line=dict(color='#FFFFFF', width=1.5)
        ),
        width=0.45,
        cliponaxis=False
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=30),
        height=330,
        yaxis=dict(title="Candidate Count", range=[0, total * 1.18], gridcolor="#E2E8F0"),
        xaxis=dict(title="Target Class (First Trip Status)", tickfont=dict(size=12, color="#0F172A", weight="bold")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155")
    )
    return fig

# Streamlit Page Config
st.set_page_config(
    page_title="Vahan.ai — Product Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Banana Yellow Visual Styling & Top Bar Navigation
st.markdown("""
<style>
    /* Global Clean Font & Background */
    .stApp {
        background-color: #FAFAF9;
    }
    
    /* Constrain Main Container Width for Comfortable Layout */
    .main .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 1450px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        margin: 0 auto !important;
    }

    /* Hide Default Sidebar for Clean Top Nav Layout */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Main Brand Header */
    .brand-banner {
        background: linear-gradient(135deg, #FEF08A 0%, #FACC15 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        border: 1px solid #EAB308;
        box-shadow: 0 4px 12px rgba(250, 204, 21, 0.15);
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 900;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .brand-subtitle {
        font-size: 1.05rem;
        font-weight: 500;
        color: #713F12;
        margin-top: 0.25rem;
    }

    /* Top Radio Button Navigation Styling (Banana Yellow Pills) */
    div[data-testid="stRadio"] {
        margin-bottom: 1.5rem;
    }
    div[data-testid="stRadio"] > div {
        flex-direction: row !important;
        justify-content: flex-start !important;
        background-color: #FEFCE8;
        padding: 8px 12px;
        border-radius: 12px;
        border: 1px solid #FDE047;
        gap: 8px;
        flex-wrap: wrap;
    }
    div[data-testid="stRadio"] label {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #334155;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: #FACC15;
        background-color: #FEF9C3;
        color: #0F172A;
    }
    div[data-testid="stRadio"] label[aria-checked="true"] {
        background-color: #FACC15 !important;
        color: #0F172A !important;
        border-color: #CA8A04 !important;
        font-weight: 800 !important;
        box-shadow: 0 2px 6px rgba(234, 179, 8, 0.3);
    }

    /* Metric Card Styling */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3px solid #F59E0B;
        border-radius: 12px;
        padding: 1.1rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.12);
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.76rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.35rem;
    }

    /* Justification & Alert Box */
    .justification-box {
        background-color: #FEF9C3;
        border-left: 5px solid #FACC15;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        color: #1E293B;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }

    /* Quick Access Module Cards */
    .module-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3px solid #FACC15;
        border-radius: 10px;
        padding: 1.25rem;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    .module-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.4rem;
    }
    .module-desc {
        font-size: 0.88rem;
        color: #64748B;
        margin-bottom: 1rem;
        line-height: 1.4;
    }

    /* Styled Buttons (Banana Yellow Primary) */
    .stButton > button {
        background-color: #FACC15;
        color: #0F172A;
        font-weight: 700;
        border: 1px solid #CA8A04;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #EAB308;
        border-color: #854D0E;
        color: #0F172A;
        box-shadow: 0 4px 8px rgba(234, 179, 8, 0.25);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_app_data():
    engine = get_engine()
    df = load_data(engine)
    return df

@st.cache_resource(show_spinner=False)
def load_app_models(num_features: int):
    models = load_models(expected_features=num_features)
    if not models or len(models) < 3:
        engine = get_engine()
        raw_df = load_data(engine)
        X_feat, y_feat, _ = build_features(raw_df)
        X_tr, X_te, y_tr, y_te = get_train_test_split(X_feat, y_feat, test_size=0.2, random_state=42)
        models = train_all_models(X_tr, y_tr)
        best_rf, _ = tune_best_model(X_tr, y_tr, model_name='Random Forest')
        best_xgb, _ = tune_best_model(X_tr, y_tr, model_name='XGBoost')
        models['Fine-Tuned Random Forest'] = best_rf
        models['Fine-Tuned XGBoost'] = best_xgb
        save_models(models)
    return models

try:
    df = load_app_data()
except Exception as e:
    st.error(f"Error loading database: {e}")
    st.stop()

# Define Navigation Options
PAGES = [
    "Overview",
    "Q1: Cohort Ranking",
    "Q2: SQL Engine",
    "Q3: ML Benchmarks",
    "Q3: Feature Drivers",
    "Lead Predictor"
]

# Initialize Session State for Active Page
if "active_page" not in st.session_state:
    st.session_state.active_page = PAGES[0]

# Top Brand Header Banner
st.markdown("""
<div class="brand-banner">
    <div class="brand-title">Vahan.ai — Product Analytics Platform</div>
    <div class="brand-subtitle">Lead-to-First-Trip (FT) Funnel Performance & Predictive Lead Scoring Engine</div>
</div>
""", unsafe_allow_html=True)

# Top Bar Horizontal Switchable Navigation
def on_page_change():
    st.session_state.active_page = st.session_state.top_nav_selection

selected_page = st.radio(
    "Select View",
    PAGES,
    index=PAGES.index(st.session_state.active_page) if st.session_state.active_page in PAGES else 0,
    horizontal=True,
    key="top_nav_selection",
    on_change=on_page_change,
    label_visibility="collapsed"
)

# Active page sync
page = st.session_state.active_page

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW (HOME LAYOUT)
# ==========================================
if page == "Overview":
    st.subheader("Executive Overview & Funnel Metrics")
    st.caption("High-level conversion benchmarks and core operational KPIs across all uploaded candidate leads.")
    
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

    # Conversion Funnel & Class Imbalance Overview
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Conversion Funnel Stage Drop-off")
        tab_chart, tab_table = st.tabs(["📊 Funnel Chart View", "📋 Funnel Data Table"])
        with tab_chart:
            fig_funnel = plot_plotly_funnel(df)
            st.plotly_chart(fig_funnel, use_container_width=True, config={'displayModeBar': False})
        with tab_table:
            funnel_df = overall_funnel_rates(df)
            st.dataframe(
                funnel_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "stage": st.column_config.TextColumn("Stage Name"),
                    "count": st.column_config.NumberColumn("Candidate Volume", format="%d"),
                    "pct_of_total": st.column_config.ProgressColumn(
                        "% of Total Leads",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100
                    )
                }
            )
        
    with col_right:
        st.subheader("Target Class Imbalance (FT=0 vs FT=1)")
        fig_imb = plot_plotly_class_imbalance(df)
        st.plotly_chart(fig_imb, use_container_width=True, config={'displayModeBar': False})
        
        st.info("**Class Imbalance Insight:** Severe positive class imbalance (**0.30% baseline rate**, 54 / 18,198 leads). ML evaluation requires Precision-Recall / PR-AUC focus over standard accuracy.")

# ==========================================
# PAGE 2: Q1 COHORT PERFORMANCE RANKING
# ==========================================
elif page == "Q1: Cohort Ranking":
    st.subheader("Q1 — Top Lead-Source Cohorts Evaluation")
    st.caption("Evaluating cohort performance on bottom-line business metrics and lead quality.")
    
    st.markdown(f'<div class="justification-box"><b>Metric Justification & Strategy:</b><br>{METRIC_JUSTIFICATION}</div>', unsafe_allow_html=True)
    
    metrics_df = compute_cohort_metrics(df)
    # Filter noise cohorts (min_leads=50 default threshold) and rank cohorts
    ranked_df = rank_cohorts(metrics_df, min_leads=50)
    
    st.subheader("Top 3 Cohorts")
    top3 = top_3_cohorts(ranked_df)
    st.dataframe(top3[['rank', 'lead_source', 'total_leads', 'ft_conversions', 'ft_conversion_rate_pct', 'connect_rate_pct', 'interest_rate_pct']], use_container_width=True)
    
    st.subheader("Full Cohort Performance Table")
    st.dataframe(ranked_df, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=ranked_df, y='lead_source', x='ft_conversion_rate_pct', hue='lead_source', palette='YlOrBr_r', legend=False, ax=ax)
    ax.set_title('Cohort FT Conversion Rate (%)', fontweight='bold', color='#0F172A')
    st.pyplot(fig)

# ==========================================
# PAGE 3: Q2 SQL QUERY ENGINE
# ==========================================
elif page == "Q2: SQL Engine":
    st.subheader("Q2 — PostgreSQL SQL Aggregation Query Engine")
    st.caption("Executing native SQL queries directly against local PostgreSQL database vahan_case_study.")
    
    engine = get_engine()
    
    query_type = st.radio("Select Query Grain View", ["Cohort x Upload Date Grain (Q2 Standard)", "Cohort Rollup Summary"], horizontal=True)
    
    if query_type == "Cohort x Upload Date Grain (Q2 Standard)":
        query_text = st.text_area("PostgreSQL SQL Query", COHORT_DATE_AGGREGATION_QUERY, height=260)
    else:
        query_text = st.text_area("PostgreSQL SQL Query", COHORT_ROLLUP_QUERY, height=260)
        
    if st.button("Execute SQL Query"):
        with st.spinner("Executing query against PostgreSQL..."):
            res_df = run_query(engine, query_text)
            st.success(f"Query executed successfully — returned {len(res_df)} rows.")
            st.dataframe(res_df, use_container_width=True)
            
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Result CSV", data=csv, file_name="vahan_sql_aggregation.csv", mime="text/csv")

# ==========================================
# PAGE 4: Q3 ML BENCHMARKS & SELECTION
# ==========================================
elif page == "Q3: ML Benchmarks":
    st.subheader("Q3 — Candidate Model Benchmarking (3 Baseline Models)")
    st.caption("Step 1: Benchmark candidate algorithms (Logistic Regression, Random Forest, XGBoost) on statistical metrics and feature usability.")
    
    X, y, encoders = build_features(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = load_app_models(X.shape[1])
    baseline_models = {
        'Logistic Regression': models['Logistic Regression'],
        'Random Forest': models['Random Forest'],
        'XGBoost': models['XGBoost']
    }
            
    st.subheader("Candidate Model Benchmark Table")
    sum_df = summary_table(baseline_models, X_test, y_test)
    st.dataframe(sum_df, use_container_width=True)
    
    st.subheader("Confusion Matrices (Class Imbalance Evaluation)")
    fig_cm = plot_all_confusion_matrices(baseline_models, X_test, y_test)
    st.pyplot(fig_cm)

    st.subheader("Feature Driver Comparison (Candidate Usability Assessment)")
    tree_baselines = {'Random Forest': baseline_models['Random Forest'], 'XGBoost': baseline_models['XGBoost']}
    fig_imp = plot_feature_importances(tree_baselines, list(X.columns))
    st.pyplot(fig_imp)

# ==========================================
# PAGE 5: Q3 PRODUCTION MODEL SELECTION & FINE-TUNING
# ==========================================
elif page == "Q3: Feature Drivers":
    st.subheader("Q3 — Production Model Selection & Hyperparameter Fine-Tuning")
    st.caption("Step 2: Select top production candidate (Random Forest), fine-tune hyperparameters via GridSearchCV, and evaluate operational drivers.")
    
    X, y, _ = build_features(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2, random_state=42)
    models = load_app_models(X.shape[1])
    
    rf_baseline = models['Random Forest']
    rf_tuned = models.get('Fine-Tuned Random Forest', rf_baseline)
    
    st.subheader("Baseline vs. Fine-Tuned Random Forest Comparison")
    rf_comp_models = {
        'Baseline Random Forest': rf_baseline,
        'Fine-Tuned Random Forest': rf_tuned
    }
    comp_df = summary_table(rf_comp_models, X_test, y_test)
    st.dataframe(comp_df, use_container_width=True)
    
    st.subheader("Production Feature Drivers (Fine-Tuned Random Forest)")
    fig_imp = plot_feature_importances({'Fine-Tuned Random Forest': rf_tuned}, list(X.columns))
    st.pyplot(fig_imp)
    
    st.markdown("""
    <div class="justification-box">
    <h3><b>Why Random Forest was Selected & Fine-Tuned for Production Deployment:</b></h3>
    <ol>
        <li><b>Selection Rationale:</b> Random Forest distributes importance across early-stage, operationally controllable levers (speed to call <code>upload_to_first_attempt_p50_hrs</code>, call attempt frequency <code>attempt_per_lead</code>, and channel <code>lead_source_encoded</code>).</li>
        <li><b>Hyperparameter Tuning:</b> Fine-tuning via <code>GridSearchCV</code> optimizes class weighting, tree depth, and minimum split sizes to improve ROC-AUC and recall on the minority FT conversion class.</li>
        <li><b>Operational Recommendation:</b> Tele-calling teams should optimize speed-to-first-call (<code>p50_hrs</code>) and call attempt frequency for top-performing lead channels to maximize First Trip completions.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 6: REAL-TIME FT CONVERSION PREDICTOR
# ==========================================
elif page == "Lead Predictor":
    st.subheader("Real-Time Candidate FT Predictor")
    st.caption("Interactive lead scoring tool using trained production Random Forest model to estimate First Trip conversion probability.")
    
    X, y, encoders = build_features(df)
    models = load_app_models(X.shape[1])
    model = models.get('Fine-Tuned Random Forest', models.get('Random Forest', list(models.values())[0]))
    
    st.info("**Production Model Active:** Fine-Tuned Random Forest | Features scored at upload time — onboarding status intentionally excluded (post-event signal not available for new leads).")
    
    with st.container():
        st.markdown("<div class='justification-box'><b>Candidate Features Input Panel</b><br><small>Score a lead using only information available at the time of upload and initial call attempts.</small></div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            lead_source_input = st.selectbox("Lead Source Cohort", encoders['lead_source_encoder'].classes_)
            attempted_input = st.selectbox("Was Candidate Attempted/Called?", [1, 0])
            connected_input = st.selectbox("Did the Call Connect?", [1, 0])
            tag_filled_input = st.selectbox("Is Candidate Tag Form Filled?", [1, 0])
            
        with col_b:
            interested_input = st.selectbox("Expressed Interest?", [1, 0])
            attempt_per_lead_input = st.number_input("Call Attempts per Lead", min_value=0.0, max_value=10.0, value=1.0)
            p50_hrs_input = st.number_input("Speed to First Call (P50 Hours)", min_value=0.0, max_value=500.0, value=float(encoders['p50_median']))
            upload_date_input = st.date_input("Lead Upload Date", value=datetime.today())
            
    if st.button("Predict FT Conversion Probability"):
        upload_dt = pd.to_datetime(upload_date_input)
        input_data = pd.DataFrame([{
            'attempted': attempted_input,
            'connected': connected_input,
            'attempt_per_lead': attempt_per_lead_input,
            'tag_filled': tag_filled_input,
            'interested': interested_input,
            'upload_to_first_attempt_p50_hrs': p50_hrs_input,
            'day_of_week': upload_dt.dayofweek,
            'day_of_month': upload_dt.day,
            'lead_source_encoded': encoders['lead_source_encoder'].transform([lead_source_input])[0]
        }])
        
        # Ensure column order matches X
        input_data = input_data[X.columns]
        
        prob = model.predict_proba(input_data)[0, 1]
        
        st.markdown("---")
        st.subheader("Prediction Result & Lead Score")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Predicted FT Conversion Probability", f"{prob * 100:.2f}%")
        with res_col2:
            if prob >= 0.5:
                st.success("High Conversion Potential (Likely to Complete First Trip)")
            else:
                st.warning("Low Conversion Probability (Priority Follow-up Recommended)")

