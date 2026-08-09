import pytest
import pandas as pd
from src.db_setup import get_engine, load_data
from src.q2_sql_query import get_q2_result, run_query, COHORT_ROLLUP_QUERY

@pytest.fixture(scope="module")
def setup_db():
    engine = get_engine()
    load_data(engine)
    return engine

def test_q2_result_execution(setup_db):
    df_q2 = get_q2_result(setup_db)
    assert isinstance(df_q2, pd.DataFrame)
    assert 'upload_date' in df_q2.columns
    assert 'lead_source' in df_q2.columns
    assert 'total_leads' in df_q2.columns
    assert 'ft_conversion_rate_pct' in df_q2.columns
    assert len(df_q2) > 0
    assert df_q2['total_leads'].sum() == 18198

def test_cohort_rollup_query(setup_db):
    df_rollup = run_query(setup_db, COHORT_ROLLUP_QUERY)
    assert isinstance(df_rollup, pd.DataFrame)
    assert len(df_rollup) == 16  # 16 unique lead sources
    assert df_rollup['total_leads'].sum() == 18198
