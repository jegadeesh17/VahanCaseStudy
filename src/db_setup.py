import os
import re
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def sanitize_column_name(col_name: str) -> str:
    """Sanitize Excel column names for PostgreSQL compatibility."""
    s = col_name.strip()
    s = s.replace('(hrs)', 'hrs')
    s = s.replace('%', 'pct')
    s = s.replace('→', 'to')
    s = re.sub(r'[\s\-\/\(\)]+', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_').lower()

def get_connection_str() -> str:
    """Construct PostgreSQL SQLAlchemy connection string from environment variables."""
    host = os.getenv('DB_HOST', '127.0.0.1')
    port = os.getenv('DB_PORT', '5432')
    db = os.getenv('DB_NAME', 'vahan_case_study')
    user = os.getenv('DB_USER', 'postgres')
    password = os.getenv('DB_PASSWORD', '')
    
    if password:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return f"postgresql+psycopg2://{user}@{host}:{port}/{db}"

def get_engine():
    """Create and return SQLAlchemy engine."""
    return create_engine(get_connection_str())

def load_data(engine=None, xlsx_path: str = None) -> pd.DataFrame:
    """Read raw Excel sheet, clean column names, load into PostgreSQL table 'leads', and return DataFrame."""
    if engine is None:
        engine = get_engine()
        
    if xlsx_path is None:
        project_root = Path(__file__).resolve().parent.parent
        xlsx_path = project_root / "data" / "Vahan_Case_Study.xlsx"
        
    try:
        df = pd.read_excel(xlsx_path, sheet_name="Raw Data", engine="openpyxl")
    except Exception:
        try:
            with engine.connect() as conn:
                df = pd.read_sql("SELECT * FROM leads", conn)
                if 'upload_date' in df.columns:
                    df['upload_date'] = pd.to_datetime(df['upload_date']).dt.date
                return df
        except Exception as db_err:
            raise RuntimeError(f"Could not load data from Excel ({xlsx_path}) or PostgreSQL 'leads' table.") from db_err

    # Sanitize column names
    col_mapping = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=col_mapping)
    
    # Standardize upload_date to date object / string
    df['upload_date'] = pd.to_datetime(df['upload_date']).dt.date
    
    # Load into PostgreSQL
    with engine.begin() as conn:
        df.to_sql('leads', conn, if_exists='replace', index=False)
        
    return df

if __name__ == "__main__":
    engine = get_engine()
    df = load_data(engine)
    print(f"Data successfully loaded to PostgreSQL table 'leads'. Total rows: {len(df)}")
