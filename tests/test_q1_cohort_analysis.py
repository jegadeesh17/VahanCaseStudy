import pytest
import pandas as pd
from src.db_setup import get_engine, load_data
from src.q1_cohort_analysis import compute_cohort_metrics, rank_cohorts, top_3_cohorts

@pytest.fixture(scope="module")
def sample_df():
    engine = get_engine()
    return load_data(engine)

def test_compute_cohort_metrics(sample_df):
    metrics = compute_cohort_metrics(sample_df)
    assert isinstance(metrics, pd.DataFrame)
    assert 'ft_conversion_rate_pct' in metrics.columns
    assert len(metrics) == sample_df['lead_source'].nunique()
    assert metrics['total_leads'].sum() == len(sample_df)

def test_rank_cohorts(sample_df):
    metrics = compute_cohort_metrics(sample_df)
    ranked = rank_cohorts(metrics, min_leads=50)
    assert isinstance(ranked, pd.DataFrame)
    assert 'rank' in ranked.columns
    assert ranked['rank'].iloc[0] == 1
    # Check that small cohorts were filtered out
    assert (ranked['total_leads'] >= 50).all()

def test_top_3_cohorts(sample_df):
    metrics = compute_cohort_metrics(sample_df)
    ranked = rank_cohorts(metrics, min_leads=50)
    top3 = top_3_cohorts(ranked)
    assert len(top3) == 3
    assert 'justification' in top3.columns
