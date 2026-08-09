# Vahan.ai — Product Analytics & ML Case Study

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/database-PostgreSQL-336791.svg)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/tests-17%2F17%20passed-success.svg)](https://docs.pytest.org/)

An end-to-end product analytics, database engineering, and machine learning pipeline for **Vahan.ai** — a recruitment platform for blue-collar drivers and delivery riders.

---

## 📌 Executive Overview

Vahan sources candidate lead numbers and tele-calls them to drive conversion to **First Trip (FT)** — the point where a driver/rider completes their first delivery/ride.

### **Key Metrics & Data Snapshot:**
- **Total Leads:** 18,198 candidate rows across 16 lead sources (cohorts) and 11 upload dates.
- **Overall FT Conversions:** 54 candidates (~0.30% positive conversion rate).
- **Primary Challenge:** Extreme class imbalance (0.3% positive class) requiring specialized sampling/weighting and early-funnel feature selection.

---

## 🚀 Key Deliverables

### **1. Q1 — Cohort Performance Ranking**
* **Primary Metric:** `FT Conversion Rate = (ft_after_upload / total_leads) * 100`
* **Product Rationale:** Monetization and core business value rely on candidates completing their first trip. Upstream metrics (Attempt %, Connect %, Interest %) are intermediate steps.
* **Filtering Rule:** Cohorts with fewer than 50 leads are excluded from rankings to eliminate statistical noise.

### **2. Q2 — PostgreSQL Aggregation Query**
* **Execution Grain:** `upload_date × lead_source` (Cohort-Date level).
* **Database Integration:** Executes natively against local PostgreSQL database `vahan_case_study` table `leads`. Handles division-by-zero using `NULLIF()` and rounds float averages via explicit `::numeric` type casting.

### **3. Q3 — Predictive ML & Production Usability**
* **Models Evaluated:** Logistic Regression, Random Forest, XGBoost.
* **Leakage Exclusions:** `ob_after_upload` and `ob_after_first_attempt` are excluded from the feature matrix. Both columns are mid-to-late funnel events — they only become observable after a lead has already connected and expressed interest, which is after the point where scoring is useful. Including them would give the model information that doesn't exist for freshly uploaded leads.
* **Production Model Selection:** **Random Forest** chosen over XGBoost for **early-funnel usability**. When `ob_after_upload` is properly excluded, XGBoost's advantage shrinks — and Random Forest distributes importance across early operational levers: `upload_to_first_attempt_p50_hrs`, `lead_source_encoded`, and `attempt_per_lead`. These features are available immediately after lead upload and are directly actionable by tele-calling teams.
* **Fine-Tuning:** 3-fold cross-validated `GridSearchCV` hyperparameter tuning. The `tune_best_model()` function supports tuning any of the three models. Serialized fine-tuned weights are saved for both Random Forest and XGBoost.

---

## 📁 Repository Architecture

```
VahanCaseStudy/
│
├── app/
│   ├── app.py                        # Streamlit interactive web dashboard
│   ├── assets/                       # Static assets (placeholder)
│   ├── components/                   # Reusable UI components (placeholder)
│   └── pages/                        # Multi-page Streamlit pages (placeholder)
│
├── notebooks/
│   └── VahanCaseStudy.ipynb          # Thin orchestrator notebook (Step 1 to Step 11)
│
├── src/                              # Standalone Python modules (Source of truth)
│   ├── __init__.py                   # Package initializer
│   ├── db_setup.py                   # PostgreSQL connection & data loader
│   ├── eda.py                        # Exploratory analysis & funnel plots
│   ├── q1_cohort_analysis.py         # Cohort metrics computation & ranking
│   ├── q2_sql_query.py               # Native PostgreSQL aggregation queries
│   ├── feature_engineering.py        # Feature matrix, leakage exclusion & split
│   ├── models.py                     # ML model training & GridSearchCV fine-tuning
│   └── evaluation.py                 # Confusion matrices, ROC-AUC & feature plots
│
├── tests/                            # 17 automated unit tests
│   ├── test_db_setup.py
│   ├── test_eda.py
│   ├── test_q1_cohort_analysis.py
│   ├── test_q2_sql_query.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── data/
│   └── Vahan_Case_Study.xlsx         # Raw dataset
│
├── docs/                             # Documentation (reserved)
│
├── models/                           # Serialized model weights (.pkl)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── fine-tuned_random_forest.pkl
│   └── fine-tuned_xgboost.pkl
│
├── .env.example                      # Environment variables template
├── .gitignore                        # Git exclusion rules
└── requirements.txt                  # Python dependencies (core subset)
```

> **Note:** `requirements.txt` includes all core production dependencies. Run `pip install -r requirements.txt` to install everything in one step.

---

## 🛠️ Getting Started

### **1. Prerequisites**
- Python 3.11+
- PostgreSQL 12+ running locally on port 5432

### **2. Clone & Install Dependencies**
```powershell
git clone https://github.com/jegadeesh17/VahanCaseStudy.git
cd VahanCaseStudy

pip install -r requirements.txt
```

### **3. Environment Setup**
Copy `.env.example` to `.env` and fill in your PostgreSQL password:
```env
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=vahan_case_study
DB_USER=postgres
DB_PASSWORD=your_postgres_password
```

### **4. Run Automated Test Suite**
Verify that all 17 unit tests pass cleanly:
```powershell
python -m pytest tests/
```

### **5. Execute Full Pipeline via CLI**
```powershell
# Load raw Excel data into PostgreSQL table 'leads'
python -m src.db_setup

# Run model training and fine-tuning
python -c "from src.db_setup import get_engine, load_data; from src.feature_engineering import build_features, get_train_test_split; from src.models import train_all_models, save_models; df = load_data(get_engine()); X, y, _ = build_features(df); X_train, X_test, y_train, y_test = get_train_test_split(X, y); save_models(train_all_models(X_train, y_train))"
```

### **6. Launch Interactive Streamlit App**
```powershell
streamlit run app/app.py
```

---

## 🔒 Security Audit & Compliance

- **No Hardcoded Credentials:** Database connection strings are dynamically constructed via `os.getenv()` reading from `.env`.
- **Git Ignore Safeguards:** `.env`, `.pytest_cache/`, `venv/`, `__pycache__/`, `.ipynb_checkpoints/`, and `*.pyc` files are excluded from version control via `.gitignore`.
- **Data Privacy:** Raw candidate phone numbers are stored as anonymized SHA-256 hashes (`candidate_phone`), preserving PII compliance.

---

## ✒️ Author
**Jegadeesh** — Product Analytics & Engineering  
[GitHub Repository](https://github.com/jegadeesh17/VahanCaseStudy.git)
