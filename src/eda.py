from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def null_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary of missing values per column."""
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df)) * 100
    report = pd.DataFrame({
        'column': df.columns,
        'null_count': null_counts.values,
        'null_pct': null_pcts.values.round(2)
    })
    return report.sort_values(by='null_count', ascending=False).reset_index(drop=True)

def cohort_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Return lead count per lead_source and flag small cohorts (< 50 leads)."""
    counts = df['lead_source'].value_counts().reset_index()
    counts.columns = ['lead_source', 'total_leads']
    counts['is_small_cohort'] = counts['total_leads'] < 50
    return counts

def overall_funnel_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall conversion funnel numbers across the entire dataset."""
    total_leads = len(df)
    attempted = int(df['attempted'].sum())
    connected = int(df['connected'].sum())
    interested = int(df['interested'].sum())
    onboarded = int(df['ob_after_upload'].sum())
    ft_conversions = int(df['ft_after_upload'].sum())
    
    funnel = pd.DataFrame([
        {'stage': '1. Uploaded Leads', 'count': total_leads, 'pct_of_total': 100.0},
        {'stage': '2. Attempted', 'count': attempted, 'pct_of_total': round(attempted / total_leads * 100, 2)},
        {'stage': '3. Connected', 'count': connected, 'pct_of_total': round(connected / total_leads * 100, 2)},
        {'stage': '4. Interested', 'count': interested, 'pct_of_total': round(interested / total_leads * 100, 2)},
        {'stage': '5. Onboarded (OB)', 'count': onboarded, 'pct_of_total': round(onboarded / total_leads * 100, 2)},
        {'stage': '6. First Trip (FT)', 'count': ft_conversions, 'pct_of_total': round(ft_conversions / total_leads * 100, 2)},
    ])
    return funnel

def plot_class_imbalance(df: pd.DataFrame, output_path: str = None) -> plt.Figure:
    """Plot target variable class imbalance (FT=0 vs FT=1)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['ft_after_upload'].value_counts()
    
    sns.barplot(x=['No FT (0)', 'FT (1)'], y=counts.values, palette=['#e74c3c', '#2ecc71'], ax=ax)
    ax.set_title('Class Imbalance: First Trip (FT) Conversions', fontsize=12, fontweight='bold')
    ax.set_ylabel('Candidate Count')
    
    for i, v in enumerate(counts.values):
        pct = (v / len(df)) * 100
        ax.text(i, v + 200, f"{v:,}\n({pct:.2f}%)", ha='center', va='bottom', fontweight='bold')
        
    ax.set_ylim(0, len(df) * 1.15)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig

def plot_funnel_by_cohort(df: pd.DataFrame, output_path: str = None) -> plt.Figure:
    """Plot total leads per cohort with funnel breakdown."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cohort_summary = df.groupby('lead_source').agg(
        total=('uploaded_leads', 'sum'),
        attempted=('attempted', 'sum'),
        connected=('connected', 'sum'),
        interested=('interested', 'sum'),
        ft=('ft_after_upload', 'sum')
    ).sort_values(by='total', ascending=True)
    
    cohort_summary[['total', 'attempted', 'connected', 'interested']].plot(
        kind='barh', ax=ax, width=0.8, colormap='viridis'
    )
    
    ax.set_title('Lead Volume & Funnel Drop-off by Lead Source Cohort', fontsize=12, fontweight='bold')
    ax.set_xlabel('Count')
    ax.set_ylabel('Lead Source')
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig
