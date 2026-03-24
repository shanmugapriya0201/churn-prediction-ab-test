
import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Drop customer ID — not a predictive feature
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges has spaces for customers with 0 tenure — fix it
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop the ~11 rows where TotalCharges is null (new customers)
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Encode target variable: Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Strip whitespace from all string/object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df.reset_index(drop=True, inplace=True)

    print(f"[DataLoader] Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"[DataLoader] Churn rate: {df['Churn'].mean():.2%}")
    print(f"[DataLoader] Class distribution: {df['Churn'].value_counts().to_dict()}")

    return df


def get_feature_types(df: pd.DataFrame, target: str = "Churn"):

    cols = [c for c in df.columns if c != target]

    categorical_cols = df[cols].select_dtypes(include="object").columns.tolist()
    continuous_cols = df[cols].select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"[DataLoader] Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"[DataLoader] Continuous features  ({len(continuous_cols)}):  {continuous_cols}")

    return categorical_cols, continuous_cols


if __name__ == "__main__":
    df = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(df.head())
    print(df.dtypes)
