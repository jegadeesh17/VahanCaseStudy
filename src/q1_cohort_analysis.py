from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

METRIC_JUSTIFICATION = (
    "FT Conversion Rate (ft_after_upload / total_leads) is selected as the primary metric "
    "for ranking lead-source cohorts. As a recruitment platform for blue-collar drivers/riders, "
    "Vahan's core value proposition and monetization rely on candidates completing their first trip (FT). "
    "While upstream funnel metrics (Attempted %, Connect %, Interested %) reflect lead reachability and quality, "
    "they are intermediate steps. FT Conversion Rate directly measures bottom-line business impact. "
    "Cohorts with fewer than 50 leads are excluded from ranking to ensure statistical reliability, "
    "and Connect/Interested rates serve as secondary tiebreakers."
)

def compute_cohort_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw candidate data by lead_source cohort and compute key funnel metrics."""
    aggregated = df.groupby('lead_source').agg(
        total_leads=('uploaded_leads', 'sum'),
        attempted=('attempted', 'sum'),
        connected=('connected', 'sum'),
        interested=('interested', 'sum'),
        onboarded=('ob_after_upload', 'sum'),
        ft_conversions=('ft_after_upload', 'sum')
    ).reset_index()

    aggregated['attempt_rate_pct'] = (aggregated['attempted'] / aggregated['total_leads'] * 100).round(2)
    aggregated['connect_rate_pct'] = (aggregated['connected'] / aggregated['attempted'].replace(0, pd.NA)).astype(float).fillna(0.0).mul(100).round(2)
    aggregated['interest_rate_pct'] = (aggregated['interested'] / aggregated['connected'].replace(0, pd.NA)).astype(float).fillna(0.0).mul(100).round(2)
    aggregated['ob_rate_pct'] = (aggregated['onboarded'] / aggregated['interested'].replace(0, pd.NA)).astype(float).fillna(0.0).mul(100).round(2)
    aggregated['ft_conversion_rate_pct'] = (aggregated['ft_conversions'] / aggregated['total_leads'] * 100).round(2)

    return aggregated

def rank_cohorts(metrics_df: pd.DataFrame, min_leads: int = 50) -> pd.DataFrame:
    """Filter small cohorts (< min_leads) and rank remaining cohorts by FT Conversion Rate."""
    filtered = metrics_df[metrics_df['total_leads'] >= min_leads].copy()
    
    # Sort by ft_conversion_rate_pct DESC, then interest_rate_pct DESC, then connect_rate_pct DESC
    sorted_df = filtered.sort_values(
        by=['ft_conversion_rate_pct', 'interest_rate_pct', 'connect_rate_pct', 'total_leads'],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)
    
    sorted_df['rank'] = range(1, len(sorted_df) + 1)
    return sorted_df

def top_3_cohorts(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """Return top 3 cohorts with justification commentary."""
    top3 = ranked_df.head(3).copy()
    top3['justification'] = METRIC_JUSTIFICATION
    return top3

def plot_cohort_ranking(ranked_df: pd.DataFrame, output_path: str = None) -> plt.Figure:
    """Plot horizontal bar chart of top cohorts by FT Conversion Rate."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        data=ranked_df,
        y='lead_source',
        x='ft_conversion_rate_pct',
        palette='Blues_r',
        ax=ax
    )
    
    ax.set_title('Lead-Source Cohorts Ranked by FT Conversion Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('FT Conversion Rate (%)')
    ax.set_ylabel('Lead Source Cohort')
    
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(f'{width:.2f}%',
                        (width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center',
                        xytext=(5, 0), textcoords='offset points', fontweight='bold')

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig
