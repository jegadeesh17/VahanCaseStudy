import pytest
import pandas as pd
from sqlalchemy import text
from src.db_setup import get_engine, load_data, sanitize_column_name

def test_sanitize_column_name():
    assert sanitize_column_name('upload_date') == 'upload_date'
    assert sanitize_column_name('Uploaded Leads') == 'uploaded_leads'
    assert sanitize_column_name('upload_to_first_attempt_P50 (hrs)') == 'upload_to_first_attempt_p50_hrs'
    assert sanitize_column_name('Attempt → Connected %') == 'attempt_to_connected_pct'

def test_db_setup_and_load():
    engine = get_engine()
    df = load_data(engine)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 18198
    assert 'ft_after_upload' in df.columns
    assert 'lead_source' in df.columns
    
    # Query PostgreSQL directly to verify persistence
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM leads")).scalar()
        assert result == 18198
