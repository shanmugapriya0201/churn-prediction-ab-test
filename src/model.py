
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_models():
    
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"]
            }
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt", "log2"]
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0]
            }
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = (
            XGBClassifier(
                random_state=42,
                eval_metric="logloss"
            ),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        )
    else:
        print("[Model] XGBoost not installed — using GradientBoosting (equivalent concept)")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Nested Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def nested_cross_validation(X: np.ndarray,
                             y: np.ndarray,
                             model_name: str,
                             estimator,
                             param_grid: dict,
                             outer_folds: int = 5,
                             inner_folds: int = 3,
                             smote_fn=None) -> dict:
    
    print(f"\n[NestedCV] Running: {model_name}")
    print("-" * 50)

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)

    outer_aucs, outer_f1s, outer_precs, outer_recs = [], [], [], []

    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        # Scale features — fit on train fold only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        # Apply SMOTE to training fold only (never to test fold)
        if smote_fn is not None:
            X_train_scaled, y_train_fold = smote_fn.fit_resample(
                X_train_scaled, y_train_fold
            )

        # Inner CV: hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True
        )
        grid_search.fit(X_train_scaled, y_train_fold)
        best_model = grid_search.best_estimator_

        # Evaluate on outer test fold (completely unseen data)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = best_model.predict(X_test_scaled)

        auc = roc_auc_score(y_test_fold, y_pred_proba)
        f1 = f1_score(y_test_fold, y_pred, zero_division=0)
        prec = precision_score(y_test_fold, y_pred, zero_division=0)
        rec = recall_score(y_test_fold, y_pred, zero_division=0)

        outer_aucs.append(auc)
        outer_f1s.append(f1)
        outer_precs.append(prec)
        outer_recs.append(rec)

        print(f"  Fold {fold_num}: AUC={auc:.4f} | F1={f1:.4f} | "
              f"Best params: {grid_search.best_params_}")

    results = {
        "Model": model_name,
        "AUC_mean": np.mean(outer_aucs),
        "AUC_std": np.std(outer_aucs),
        "F1_mean": np.mean(outer_f1s),
        "F1_std": np.std(outer_f1s),
        "Precision_mean": np.mean(outer_precs),
        "Recall_mean": np.mean(outer_recs),
    }

    print(f"\n  {model_name} — AUC: {results['AUC_mean']:.4f} ± {results['AUC_std']:.4f} | "
          f"F1: {results['F1_mean']:.4f} ± {results['F1_std']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Final Model Training on Full Train Set
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      estimator,
                      param_grid: dict,
                      model_name: str,
                      output_dir: str = "outputs"):
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[FinalModel] Training {model_name} on full training set...")

    # Tune with 5-fold CV on the full training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"[FinalModel] Best hyperparameters: {grid_search.best_params_}")

    # Evaluate on held-out test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "AUC-ROC": roc_auc_score(y_test, y_pred_proba),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_test, y_pred),
    }

    print(f"\n[FinalModel] Test Set Results — {model_name}")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    print(f"\n[FinalModel] Classification Report:\n")
    print(classification_report(y_test, y_pred,
                                target_names=["No Churn", "Churn"],
                                zero_division=0))

    # Plot confusion matrix
    _plot_confusion_matrix(y_test, y_pred, model_name, output_dir)

    # Plot ROC curve
    _plot_roc_curve(y_test, y_pred_proba, metrics["AUC-ROC"], model_name, output_dir)

    # Feature importances (if available)
    _plot_feature_importance(best_model, model_name, output_dir)

    return best_model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_test, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {fname}")


def _plot_roc_curve(y_test, y_proba, auc, model_name, output_dir):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2E75B6", lw=2,
            label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Random Classifier (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"roc_curve_{model_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {fname}")


def _plot_feature_importance(model, model_name, output_dir):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(indices)), importances[indices],
           color="#2E75B6", alpha=0.85)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([f"F{i}" for i in indices], rotation=45, ha="right")
    ax.set_xlabel("Feature Index", fontsize=11)
    ax.set_ylabel("Importance Score", fontsize=11)
    ax.set_title(f"Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fname = os.path.join(output_dir, f"feature_importance_{model_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {fname}")


def plot_model_comparison(results: list, output_dir: str = "outputs"):

    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results).sort_values("AUC_mean", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2E75B6" if i == 0 else "#9DC3E6" for i in range(len(df))]
    bars = ax.bar(df["Model"], df["AUC_mean"], color=colors,
                  yerr=df["AUC_std"], capsize=5, alpha=0.9)

    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("AUC-ROC (Nested CV)", fontsize=12)
    ax.set_title("Model Comparison — AUC-ROC with Nested Cross-Validation",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + row["AUC_std"] + 0.005,
                f"{row['AUC_mean']:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fname = os.path.join(output_dir, "model_comparison_auc.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] Model comparison saved: {fname}")
