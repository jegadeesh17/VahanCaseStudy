from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """Evaluate fitted model on test set and return dict of key metrics and confusion matrix."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    f1_pos = f1_score(y_test, y_pred, pos_label=1)
    prec_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        'model_name': model_name,
        'confusion_matrix': cm,
        'roc_auc': round(roc_auc, 4),
        'f1_weighted': round(f1_w, 4),
        'f1_ft_class': round(f1_pos, 4),
        'precision_ft_class': round(prec_pos, 4),
        'recall_ft_class': round(rec_pos, 4),
        'classification_report': report
    }

def summary_table(models_dict: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Generate model comparison table across key evaluation metrics."""
    records = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        records.append({
            'Model': name,
            'ROC-AUC': metrics['roc_auc'],
            'F1 (Weighted)': metrics['f1_weighted'],
            'F1 (FT Class)': metrics['f1_ft_class'],
            'Precision (FT Class)': metrics['precision_ft_class'],
            'Recall (FT Class)': metrics['recall_ft_class']
        })
    return pd.DataFrame(records)

def plot_all_confusion_matrices(models_dict: dict, X_test: pd.DataFrame, y_test: pd.Series, output_path: str = None) -> plt.Figure:
    """Plot side-by-side confusion matrix heatmaps for all 3 models."""
    fig, axes = plt.subplots(1, len(models_dict), figsize=(15, 4))
    
    for ax, (name, model) in zip(axes, models_dict.items()):
        metrics = evaluate_model(model, X_test, y_test, name)
        cm = metrics['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['No FT (0)', 'FT (1)'],
                    yticklabels=['No FT (0)', 'FT (1)'])
        ax.set_title(f"{name}\nROC-AUC: {metrics['roc_auc']}", fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig

def get_feature_importance_df(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importance or coefficients from estimator."""
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_[0])
    else:
        scores = np.zeros(len(feature_names))

    if len(scores) != len(feature_names):
        if hasattr(model, "feature_names_in_") and len(model.feature_names_in_) == len(scores):
            feature_names = list(model.feature_names_in_)
        elif len(scores) < len(feature_names):
            feature_names = feature_names[:len(scores)]
        else:
            feature_names = list(feature_names) + [f"feature_{i}" for i in range(len(feature_names), len(scores))]
            
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': scores
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    return df_imp

def plot_feature_importances(models_dict: dict, feature_names: list, output_path: str = None) -> plt.Figure:
    """Plot feature importance bar charts side-by-side for Random Forest and XGBoost."""
    tree_models = {k: v for k, v in models_dict.items() if hasattr(v, "feature_importances_")}
    
    fig, axes = plt.subplots(1, len(tree_models), figsize=(14, 5))
    if len(tree_models) == 1:
        axes = [axes]
        
    for ax, (name, model) in zip(axes, tree_models.items()):
        df_imp = get_feature_importance_df(model, feature_names)
        sns.barplot(data=df_imp, y='feature', x='importance', hue='feature', palette='viridis', legend=False, ax=ax)
        ax.set_title(f"{name} — Top Drivers of FT Conversion", fontsize=11, fontweight='bold')
        ax.set_xlabel('Relative Importance')
        ax.set_ylabel('Feature')

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig
