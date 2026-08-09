from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def train_logistic_regression(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """Train Logistic Regression classifier with balanced class weights."""
    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf

def train_random_forest(X_train, y_train, n_estimators: int = 200, random_state: int = 42) -> RandomForestClassifier:
    """Train Random Forest classifier with balanced class weights."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

def train_xgboost(X_train, y_train, n_estimators: int = 200, random_state: int = 42) -> XGBClassifier:
    """Train XGBoost classifier with scale_pos_weight to handle class imbalance."""
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    clf = XGBClassifier(
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

def train_all_models(X_train, y_train, random_state: int = 42) -> dict:
    """Train Logistic Regression, Random Forest, and XGBoost models on training set."""
    models = {
        'Logistic Regression': train_logistic_regression(X_train, y_train, random_state=random_state),
        'Random Forest': train_random_forest(X_train, y_train, random_state=random_state),
        'XGBoost': train_xgboost(X_train, y_train, random_state=random_state)
    }
    return models

def tune_best_model(X_train, y_train, model_name: str = 'XGBoost', random_state: int = 42):
    """
    Perform hyperparameter fine-tuning using GridSearchCV on the selected best model.
    Returns the fine-tuned model and best parameters dictionary.
    """
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    if model_name == 'XGBoost':
        base_clf = XGBClassifier(random_state=random_state, eval_metric='logloss', n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 0.8]
        }
    elif model_name == 'Random Forest':
        base_clf = RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    else:
        base_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state)
        param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}

    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params

def save_models(models_dict: dict, output_dir: str = None) -> None:
    """Persist trained models to disk using joblib."""
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "models"
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models_dict.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, output_dir / filename)

def load_models(input_dir: str = None, expected_features: int = None) -> dict:
    """Load persisted models from disk."""
    if input_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        input_dir = project_root / "models"
    else:
        input_dir = Path(input_dir)
        
    models = {}
    mapping = {
        'logistic_regression.pkl': 'Logistic Regression',
        'random_forest.pkl': 'Random Forest',
        'xgboost.pkl': 'XGBoost',
        'fine-tuned_random_forest.pkl': 'Fine-Tuned Random Forest',
        'fine-tuned_xgboost.pkl': 'Fine-Tuned XGBoost',
    }
    for file, name in mapping.items():
        filepath = input_dir / file
        if filepath.exists():
            try:
                m = joblib.load(filepath)
                if expected_features is not None:
                    n_feat = getattr(m, 'n_features_in_', getattr(m, 'feature_importances_', None))
                    if n_feat is not None:
                        count = n_feat if isinstance(n_feat, (int, np.integer)) else len(n_feat)
                        if count != expected_features:
                            continue
                models[name] = m
            except Exception:
                continue
    return models


