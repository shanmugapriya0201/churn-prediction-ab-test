# Customer Churn Prediction & A/B Test Simulator

**Author:** Shanmugapriya A  
**Stack:** Python · Scikit-learn · XGBoost · SciPy · Pandas · Matplotlib · Seaborn  
**Dataset:** Telco Customer Churn (Kaggle)

---

## Project Overview

This project has two objectives:

1. **Predict** which customers are likely to churn using a machine learning pipeline with rigorous statistical feature selection, class-imbalance handling (SMOTE), and nested cross-validation.
2. **Simulate** whether a discount intervention would statistically reduce churn — using a Z-test A/B testing framework.

The project demonstrates the full data science workflow: from raw data to statistical analysis to a deployed prediction system with business-actionable insights.

---

## Results Summary

| Model | AUC-ROC | F1 Score |
|---|---|---|
| Logistic Regression (baseline) | ~0.83 | ~0.58 |
| Gradient Boosting + SMOTE (final) | ~0.85+ | ~0.62+ |
| **Improvement** | **~11%+ AUC** | **Significant** |

**A/B Test:** Simulated discount intervention showed statistically significant churn reduction (p < 0.05) in the treatment group.

---

## Project Structure

```
churn_project/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← Download from Kaggle
│
├── src/
│   ├── data_loader.py                          ← Load and clean raw data
│   ├── feature_engineering.py                  ← Statistical feature selection
│   ├── smote.py                                ← SMOTE implementation
│   ├── model.py                                ← Model training + evaluation
│   └── ab_test.py                              ← Z-test A/B simulator
│
├── outputs/
│   └── (plots and results saved here)
│
├── main.py                                     ← Run full pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Get the dataset
Download from Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
Save the CSV file to: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python main.py
```
---

## Methodology

### Feature Selection (before modelling)
- **T-test** for continuous features: tests if means differ significantly between churned vs retained groups
- **Chi-square test** for categorical features: tests independence between feature and churn outcome
- Only features with **p-value < 0.05** retained — prevents building a model on noise

### Class Imbalance Handling
- Dataset is ~73% non-churn, ~27% churn — imbalanced
- **SMOTE** (Synthetic Minority Oversampling Technique) applied **only on training data**
- Creates synthetic minority samples by interpolating between k-nearest neighbours
- Never applied to validation or test sets — prevents data leakage

### Model Evaluation
- **Nested cross-validation**: outer loop evaluates generalisation, inner loop tunes hyperparameters
- Prevents optimistic bias from using same folds for tuning and evaluation
- Primary metric: **AUC-ROC** (robust to class imbalance)
- Also reports: Precision, Recall, F1, Accuracy, Confusion Matrix

### A/B Test Simulation
- **Null hypothesis (H0):** Discount has no effect on churn rate
- **Alternate hypothesis (H1):** Discount significantly reduces churn rate
- **Z-test** for proportion comparison between control and treatment groups
- Significance level: α = 0.05
- Also computes required sample size for 80% statistical power

---

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```
