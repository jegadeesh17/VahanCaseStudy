import pytest
import pandas as pd
from src.db_setup import get_engine, load_data
from src.eda import null_report, cohort_sizes, overall_funnel_rates

@pytest.fixture(scope="module")
def sample_df():
    engine = get_engine()
    return load_data(engine)

def test_null_report(sample_df):
    report = null_report(sample_df)
    assert isinstance(report, pd.DataFrame)
    assert 'column' in report.columns
    assert 'null_count' in report.columns
    assert len(report) == len(sample_df.columns)

def test_cohort_sizes(sample_df):
    sizes = cohort_sizes(sample_df)
    assert isinstance(sizes, pd.DataFrame)
    assert 'lead_source' in sizes.columns
    assert 'is_small_cohort' in sizes.columns
    assert len(sizes) == sample_df['lead_source'].nunique()

def test_overall_funnel_rates(sample_df):
    funnel = overall_funnel_rates(sample_df)
    assert isinstance(funnel, pd.DataFrame)
    assert len(funnel) == 6
    assert funnel.iloc[0]['count'] == 18198
    assert funnel.iloc[-1]['pct_of_total'] < 1.0  # FT is ~0.3%
