import pandas as pd
from sqlalchemy import text
from src.db_setup import get_engine

COHORT_DATE_AGGREGATION_QUERY = """
SELECT
    upload_date,
    lead_source,
    COUNT(*)                                                                            AS total_leads,
    SUM(attempted)                                                                      AS total_attempted,
    SUM(connected)                                                                      AS total_connected,
    SUM(interested)                                                                     AS total_interested,
    SUM(ob_after_upload)                                                                AS total_onboarded,
    SUM(ft_after_upload)                                                                AS total_ft,

    ROUND((100.0 * SUM(attempted) / COUNT(*))::numeric, 2)                              AS attempt_rate_pct,
    ROUND((100.0 * SUM(connected) / NULLIF(SUM(attempted), 0))::numeric, 2)            AS connect_rate_pct,
    ROUND((100.0 * SUM(interested) / NULLIF(SUM(connected), 0))::numeric, 2)           AS interest_rate_pct,
    ROUND((100.0 * SUM(ob_after_upload) / NULLIF(SUM(interested), 0))::numeric, 2)     AS ob_rate_pct,
    ROUND((100.0 * SUM(ft_after_upload) / COUNT(*))::numeric, 2)                        AS ft_conversion_rate_pct,
    ROUND(AVG(upload_to_first_attempt_p50_hrs)::numeric, 1)                            AS avg_p50_hrs_to_first_call

FROM leads
GROUP BY upload_date, lead_source
ORDER BY upload_date DESC, ft_conversion_rate_pct DESC;
"""

COHORT_ROLLUP_QUERY = """
SELECT
    lead_source,
    COUNT(*)                                                                            AS total_leads,
    SUM(attempted)                                                                      AS total_attempted,
    SUM(connected)                                                                      AS total_connected,
    SUM(interested)                                                                     AS total_interested,
    SUM(ob_after_upload)                                                                AS total_onboarded,
    SUM(ft_after_upload)                                                                AS total_ft,

    ROUND((100.0 * SUM(attempted) / COUNT(*))::numeric, 2)                              AS attempt_rate_pct,
    ROUND((100.0 * SUM(connected) / NULLIF(SUM(attempted), 0))::numeric, 2)            AS connect_rate_pct,
    ROUND((100.0 * SUM(interested) / NULLIF(SUM(connected), 0))::numeric, 2)           AS interest_rate_pct,
    ROUND((100.0 * SUM(ob_after_upload) / NULLIF(SUM(interested), 0))::numeric, 2)     AS ob_rate_pct,
    ROUND((100.0 * SUM(ft_after_upload) / COUNT(*))::numeric, 2)                        AS ft_conversion_rate_pct,
    ROUND(AVG(upload_to_first_attempt_p50_hrs)::numeric, 1)                            AS avg_p50_hrs_to_first_call

FROM leads
GROUP BY lead_source
ORDER BY ft_conversion_rate_pct DESC, total_leads DESC;
"""

def run_query(engine, query: str) -> pd.DataFrame:
    """Execute SQL query against PostgreSQL engine and return DataFrame."""
    if engine is None:
        engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)
    return df

def get_q2_result(engine=None) -> pd.DataFrame:
    """Return Q2 aggregated dataset at upload_date x lead_source grain from PostgreSQL."""
    return run_query(engine, COHORT_DATE_AGGREGATION_QUERY)
