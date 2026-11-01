# Vahan.ai: Product Analytics & ML Case Study

An end-to-end product analytics, database engineering and machine learning pipeline for Vahan.ai, a recruitment platform for blue-collar drivers and delivery riders. It ranks lead sources by first-trip conversion, aggregates the funnel in PostgreSQL, and scores leads with a model that only uses information available right after upload.

Vahan sources candidate lead numbers and tele-calls them to drive conversion to First Trip (FT), the point where a driver or rider completes their first delivery or ride.

- **Total leads:** 18,198 candidate rows across 16 lead sources (cohorts) and 11 upload dates.
- **Overall FT conversions:** 54 candidates (about 0.30%).
- **Main challenge:** extreme class imbalance (0.3% positive class), which needs specialized sampling or weighting and early-funnel feature selection.

## Features

- **Cohort performance ranking (Q1):** the primary metric is the FT conversion rate, `ft_after_upload / total_leads * 100`. Monetization depends on candidates completing a first trip, so attempt, connect and interest rates are treated as intermediate steps. Cohorts with fewer than 50 leads are excluded from rankings to remove statistical noise.
- **PostgreSQL aggregation (Q2):** aggregates at the `upload_date × lead_source` grain against the `leads` table in the `vahan_case_study` database. Division by zero is handled with `NULLIF()`, and float averages are rounded with an explicit `::numeric` cast.
- **Predictive model (Q3):** evaluates Logistic Regression, Random Forest and XGBoost, then tunes with 3-fold cross-validated `GridSearchCV`. `tune_best_model()` supports any of the three, and fine-tuned weights are saved for Random Forest and XGBoost.
- **Leakage control:** `ob_after_upload` and `ob_after_first_attempt` are excluded from the features. They only become observable after a lead has connected and expressed interest, which is after the point where scoring is useful.
- **Production model choice:** Random Forest over XGBoost for early-funnel usability. With `ob_after_upload` excluded, XGBoost's advantage shrinks, and Random Forest spreads importance across early operational levers that tele-calling teams can act on: `upload_to_first_attempt_p50_hrs`, `lead_source_encoded` and `attempt_per_lead`.
- **Streamlit dashboard:** an interactive app over the same data.

## Quick start

Requires Python 3.11 or later, and PostgreSQL 12 or later running locally on port 5432.

```bash
git clone https://github.com/jegadeesh17/VahanCaseStudy.git
cd VahanCaseStudy
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; use .venv/bin/activate elsewhere
pip install -r requirements.txt
cp .env.example .env               # then fill in your PostgreSQL password
```

## Usage

Load the raw Excel data into the PostgreSQL `leads` table, then train and save the models:

```bash
python -m src.db_setup
python -c "from src.db_setup import get_engine, load_data; from src.feature_engineering import build_features, get_train_test_split; from src.models import train_all_models, save_models; df = load_data(get_engine()); X, y, _ = build_features(df); X_train, X_test, y_train, y_test = get_train_test_split(X, y); save_models(train_all_models(X_train, y_train))"
```

Launch the dashboard:

```bash
streamlit run app/app.py
```

The notebook `notebooks/VahanCaseStudy.ipynb` walks through the same steps and calls the modules in `src/`.

## Running tests

```bash
python -m pytest tests/                    # all 17 unit tests
python -m pytest tests/test_models.py      # one file
```

## Configuration

| Variable | Purpose |
|---|---|
| `DB_HOST` | PostgreSQL host (for example `127.0.0.1`) |
| `DB_PORT` | PostgreSQL port (5432) |
| `DB_NAME` | Database name (`vahan_case_study`) |
| `DB_USER` | PostgreSQL user |
| `DB_PASSWORD` | PostgreSQL password |

Connection strings are built from these variables with `os.getenv()`. Raw candidate phone numbers are stored as SHA-256 hashes (`candidate_phone`), not as plain numbers.

## Project structure

```
app/         Streamlit dashboard
data/        raw dataset (Vahan_Case_Study.xlsx)
docs/        the case-study write-up (PDF)
models/      serialized model weights (.pkl)
notebooks/   walkthrough notebook
src/         database loader, EDA, cohort analysis, SQL query, features, models, evaluation
tests/       unit tests, one file per module
```

## Documentation

- [docs/Vahan.ai Case Study_ Candidate Conversion Optimization _Jegadeesh D.pdf](<docs/Vahan.ai Case Study_ Candidate Conversion Optimization _Jegadeesh D.pdf>): the case-study write-up

## License

No license granted; shared for review.
