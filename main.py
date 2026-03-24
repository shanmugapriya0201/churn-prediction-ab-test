"""
main.py
-------
Full end-to-end pipeline for Churn Prediction + A/B Test Simulation.

Run this file to execute the complete project:
  python main.py

Steps:
  1.  Load and clean data
  2.  Exploratory Data Analysis (EDA)
  3.  Statistical feature selection (T-test + Chi-square)
  4.  Encode categorical features
  5.  Train/test split (stratified, 80/20)
  6.  Apply SMOTE to training set only
  7.  Benchmark 3 models with nested cross-validation
  8.  Select best model
  9.  Train final model on full training set
  10. Evaluate on held-out test set
  11. Run A/B test simulation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, get_feature_types
from src.feature_engineering import (
    ttest_feature_selection,
    chisquare_feature_selection,
    encode_categoricals,
    build_feature_matrix
)
from src.smote import SMOTE
from src.model import (
    get_models,
    nested_cross_validation,
    train_final_model,
    plot_model_comparison
)
from src.ab_test import (
    required_sample_size,
    simulate_ab_test,
    plot_ab_results,
    plot_churn_distribution
)

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2
ALPHA = 0.05  

os.makedirs(OUTPUT_DIR, exist_ok=True)


# STEP 1 & 2: Load Data + EDA

def run_eda(df: pd.DataFrame, cat_cols: list, cont_cols: list):
    """
    Quick EDA — save key plots to outputs/.
    Full EDA is in notebooks/01_EDA.ipynb
    """
    print("\n" + "=" * 60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\nDataset shape:  {df.shape}")
    print(f"Churn rate:     {df['Churn'].mean():.2%}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nBasic stats:\n{df[cont_cols].describe().round(2)}")

    # Plot 1: Churn distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df["Churn"].value_counts().rename({0: "No Churn", 1: "Churn"}).plot(
        kind="bar", ax=axes[0], color=["#2E75B6", "#D85A30"], alpha=0.85,
        edgecolor="white"
    )
    axes[0].set_title("Class Distribution (Churn vs No Churn)",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(["No Churn", "Churn"], rotation=0)
    axes[0].grid(True, alpha=0.3, axis="y")

    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height()):,}",
                         (p.get_x() + p.get_width() / 2, p.get_height() + 20),
                         ha="center", fontsize=11, fontweight="bold")

    # Plot 2: Continuous feature distributions
    df[cont_cols].hist(bins=30, ax=axes[1], color="#2E75B6", alpha=0.7,
                       edgecolor="white", figsize=(12, 4))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_overview.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Churn rate by contract type
    if "Contract" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        contract_churn = df.groupby("Contract")["Churn"].mean().sort_values()
        contract_churn.plot(kind="bar", ax=ax, color="#2E75B6", alpha=0.85,
                            edgecolor="white")
        ax.set_title("Churn Rate by Contract Type",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Churn Rate")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, alpha=0.3, axis="y")
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2%}",
                        (p.get_x() + p.get_width() / 2, p.get_height() + 0.005),
                        ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "churn_by_contract.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # Plot 4: Correlation heatmap (continuous features)
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[cont_cols + ["Churn"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, mask=mask, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix (Continuous Features + Churn)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[EDA] Plots saved to {OUTPUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("CUSTOMER CHURN PREDICTION + A/B TEST SIMULATION")
    print("Author: Shanmugapriya A")
    print("=" * 60)

    # ── STEP 1: Load Data ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: LOADING AND CLEANING DATA")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Dataset not found at: {DATA_PATH}")
        print("Please download the Telco Customer Churn dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        sys.exit(1)

    df = load_data(DATA_PATH)
    cat_cols, cont_cols = get_feature_types(df)

    # ── STEP 2: EDA ──────────────────────────────────────────────────────────
    run_eda(df, cat_cols, cont_cols)

    # ── STEP 3: Statistical Feature Selection ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: STATISTICAL FEATURE SELECTION")
    print("=" * 60)
    print("\nRationale: Apply T-test and Chi-square BEFORE modelling")
    print("to identify statistically significant features (p < 0.05).")
    print("This is more principled than correlation-based selection.")

    sig_continuous = ttest_feature_selection(df, cont_cols, alpha=ALPHA)
    sig_categorical = chisquare_feature_selection(df, cat_cols, alpha=ALPHA)

    print(f"\n[FeatureSelection] Summary:")
    print(f"  Continuous features kept: {len(sig_continuous)} / {len(cont_cols)}")
    print(f"  Categorical features kept: {len(sig_categorical)} / {len(cat_cols)}")

    # ── STEP 4: Encoding ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: ENCODING CATEGORICAL FEATURES")
    print("=" * 60)

    df_encoded = encode_categoricals(df.copy(), cat_cols)
    X, y, feature_names = build_feature_matrix(
        df_encoded, sig_continuous, sig_categorical
    )

    X = X.values
    y = y.values

    # ── STEP 5: Train/Test Split ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: TRAIN / TEST SPLIT (Stratified 80/20)")
    print("=" * 60)
    print("\nStratified split preserves the churn ratio in both sets.")
    print("The test set is HELD OUT and never used for training or tuning.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y           # Preserves churn ratio in both sets
    )

    print(f"\nTrain size: {len(X_train):,} | Churn rate: {y_train.mean():.2%}")
    print(f"Test size:  {len(X_test):,}  | Churn rate: {y_test.mean():.2%}")

    # ── STEP 6: Scale + SMOTE (Training Set Only) ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: FEATURE SCALING + SMOTE (Training Set Only)")
    print("=" * 60)
    print("\nCRITICAL: SMOTE is applied ONLY to training data.")
    print("The test set remains untouched to prevent data leakage.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
    X_test_scaled = scaler.transform(X_test)         # Apply to test

    smote = SMOTE(k_neighbours=5, random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled, y_train
    )

    # ── STEP 7: Nested Cross-Validation on Training Set ──────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: NESTED CROSS-VALIDATION (Model Benchmarking)")
    print("=" * 60)
    print("\nOuter loop (5-fold): evaluates generalisation performance")
    print("Inner loop (3-fold): tunes hyperparameters via GridSearchCV")
    print("This prevents optimistic bias from tuning on the same data")
    print("that performance is measured on.\n")

    models = get_models()
    all_results = []

    # Create a SMOTE instance for use inside nested CV
    smote_for_cv = SMOTE(k_neighbours=5, random_state=RANDOM_STATE)

    for name, (estimator, param_grid) in models.items():
        result = nested_cross_validation(
            X=X_train_scaled,      # Use unSMOTEd data — SMOTE applied inside each fold
            y=y_train,
            model_name=name,
            estimator=estimator,
            param_grid=param_grid,
            outer_folds=5,
            inner_folds=3,
            smote_fn=smote_for_cv
        )
        all_results.append(result)

    # ── Model Comparison ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL COMPARISON — Nested CV Results")
    print("=" * 60)

    results_df = pd.DataFrame(all_results).sort_values("AUC_mean", ascending=False)
    print(results_df[["Model", "AUC_mean", "AUC_std",
                       "F1_mean", "Precision_mean", "Recall_mean"]].to_string(index=False))

    plot_model_comparison(all_results, OUTPUT_DIR)

    # ── STEP 8: Select Best Model ─────────────────────────────────────────────
    best_model_name = results_df.iloc[0]["Model"]
    baseline_auc = results_df[results_df["Model"] == "Logistic Regression"]["AUC_mean"].values[0]
    best_auc = results_df.iloc[0]["AUC_mean"]
    improvement = (best_auc - baseline_auc) / baseline_auc * 100

    print(f"\n[Selection] Best model: {best_model_name}")
    print(f"[Selection] Best AUC:   {best_auc:.4f}")
    print(f"[Selection] Baseline (LR) AUC: {baseline_auc:.4f}")
    print(f"[Selection] Improvement over baseline: {improvement:.1f}%")

    # ── STEP 9: Train Final Model ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 9: TRAINING FINAL MODEL ({best_model_name})")
    print("=" * 60)

    best_estimator, best_param_grid = models[best_model_name]

    final_model, test_metrics = train_final_model(
        X_train=X_train_resampled,
        y_train=y_train_resampled,
        X_test=X_test_scaled,
        y_test=y_test,
        estimator=best_estimator,
        param_grid=best_param_grid,
        model_name=best_model_name,
        output_dir=OUTPUT_DIR
    )

    # ── STEP 10: A/B Test Simulation ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 10: A/B TEST SIMULATION")
    print("=" * 60)

    # Get predictions on the full dataset for A/B simulation
    X_all_scaled = scaler.transform(X)
    all_predictions = final_model.predict_proba(X_all_scaled)[:, 1]

    # Calculate required sample size first
    baseline_churn_rate = y.mean()
    required_sample_size(
        baseline_rate=baseline_churn_rate,
        expected_lift=0.05,
        alpha=ALPHA,
        power=0.80
    )

    # Run the simulation
    ab_results = simulate_ab_test(
        df=df,
        model_predictions=all_predictions,
        churn_threshold=0.5,
        discount_effect=0.08,
        sample_fraction=0.5,
        alpha=ALPHA,
        random_state=RANDOM_STATE
    )

    plot_ab_results(ab_results, OUTPUT_DIR)
    plot_churn_distribution(all_predictions, OUTPUT_DIR)

    # ── STEP 11: Final Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"""
Model Performance:
  Best Model:        {best_model_name}
  AUC-ROC (test):    {test_metrics['AUC-ROC']:.4f}
  F1 Score (test):   {test_metrics['F1 Score']:.4f}
  Precision (test):  {test_metrics['Precision']:.4f}
  Recall (test):     {test_metrics['Recall']:.4f}
  Accuracy (test):   {test_metrics['Accuracy']:.4f}
  Improvement over LR baseline: ~{improvement:.0f}%

A/B Test Results:
  Control churn rate:    {ab_results['control_churn_rate']:.2%}
  Treatment churn rate:  {ab_results['treatment_churn_rate']:.2%}
  Absolute reduction:    {ab_results['absolute_reduction']:.2%}
  Relative reduction:    {ab_results['relative_reduction']:.2%}
  Z-statistic:           {ab_results['z_statistic']:.4f}
  P-value:               {ab_results['p_value']:.6f}
  Conclusion:            {ab_results['conclusion']}

Outputs saved to: {OUTPUT_DIR}/
""")

    print("complete.")


if __name__ == "__main__":
    main()
