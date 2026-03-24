import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1. Statistical Feature Selection
# ─────────────────────────────────────────────────────────────────────────────

def ttest_feature_selection(df: pd.DataFrame,
                            continuous_cols: list,
                            target: str = "Churn",
                            alpha: float = 0.05) -> list:
    
    print("\n[FeatureSelection] T-Test Results (Continuous Features)")
    print("=" * 60)

    significant = []
    results = []

    churned = df[df[target] == 1]
    retained = df[df[target] == 0]

    for col in continuous_cols:
        t_stat, p_val = stats.ttest_ind(
            churned[col].dropna(),
            retained[col].dropna(),
            equal_var=False  # Welch's t-test — does not assume equal variance
        )
        significant_flag = p_val < alpha
        if significant_flag:
            significant.append(col)
        results.append({
            "Feature": col,
            "T-statistic": round(t_stat, 4),
            "P-value": round(p_val, 6),
            "Significant": "YES" if significant_flag else "NO"
        })

    results_df = pd.DataFrame(results).sort_values("P-value")
    print(results_df.to_string(index=False))
    print(f"\n[FeatureSelection] Significant continuous features: {significant}")

    return significant


def chisquare_feature_selection(df: pd.DataFrame,
                                categorical_cols: list,
                                target: str = "Churn",
                                alpha: float = 0.05) -> list:
    
    print("\n[FeatureSelection] Chi-Square Test Results (Categorical Features)")
    print("=" * 60)

    significant = []
    results = []

    for col in categorical_cols:
        contingency_table = pd.crosstab(df[col], df[target])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)
        significant_flag = p_val < alpha
        if significant_flag:
            significant.append(col)
        results.append({
            "Feature": col,
            "Chi2": round(chi2, 4),
            "P-value": round(p_val, 6),
            "DoF": dof,
            "Significant": "YES" if significant_flag else "NO"
        })

    results_df = pd.DataFrame(results).sort_values("P-value")
    print(results_df.to_string(index=False))
    print(f"\n[FeatureSelection] Significant categorical features: {significant}")

    return significant


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame,
                        categorical_cols: list) -> pd.DataFrame:

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"[Preprocessing] Encoded {len(categorical_cols)} categorical columns")
    return df


def build_feature_matrix(df: pd.DataFrame,
                         selected_continuous: list,
                         selected_categorical: list,
                         target: str = "Churn"):

    selected_features = selected_continuous + selected_categorical
    X = df[selected_features].copy()
    y = df[target].copy()

    print(f"[Preprocessing] Feature matrix shape: {X.shape}")
    print(f"[Preprocessing] Selected features: {selected_features}")

    return X, y, selected_features


def scale_features(X_train: np.ndarray,
                   X_test: np.ndarray):
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # Fit on train only
    X_test_scaled = scaler.transform(X_test)          # Apply to test
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from src.data_loader import load_data, get_feature_types

    df = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    cat_cols, cont_cols = get_feature_types(df)

    sig_cont = ttest_feature_selection(df, cont_cols)
    sig_cat = chisquare_feature_selection(df, cat_cols)

    df = encode_categoricals(df, cat_cols)
    X, y, features = build_feature_matrix(df, sig_cont, sig_cat)
    print(X.head())
