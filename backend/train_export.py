"""
Training script for Loan Default Prediction System.
Trains a single model: Gradient Boosting Classifier.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "Loan_default.csv")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    'age', 'income', 'loanamount', 'creditscore', 'monthsemployed',
    'numcreditlines', 'interestrate', 'loanterm', 'dtiratio', 'hasmortgage',
    'hasdependents', 'hascosigner', "education_bachelor's",
    'education_high_school', "education_master's", 'education_phd',
    'employmenttype_full-time', 'employmenttype_part-time',
    'employmenttype_self-employed', 'employmenttype_unemployed',
    'loanpurpose_auto', 'loanpurpose_business', 'loanpurpose_education',
    'loanpurpose_home', 'loanpurpose_other', 'maritalstatus_divorced',
    'maritalstatus_married', 'maritalstatus_single'
]

NUMERIC_COLS = [
    'age', 'income', 'loanamount', 'creditscore', 'monthsemployed',
    'numcreditlines', 'interestrate', 'loanterm', 'dtiratio'
]


def load_and_preprocess():
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw records: {len(df)}")

    df.drop_duplicates(inplace=True)
    print(f"  After dedup:  {len(df)}")

    df_cleaned = df.copy()

    # One-hot encode categoricals
    df_cleaned = pd.get_dummies(df_cleaned, columns=['Education'], drop_first=False)
    df_cleaned['HasMortgage'] = df_cleaned['HasMortgage'].map({"Yes": 1, "No": 0})
    df_cleaned['HasDependents'] = df_cleaned['HasDependents'].map({"Yes": 1, "No": 0})
    df_cleaned['HasCoSigner'] = df_cleaned['HasCoSigner'].map({"Yes": 1, "No": 0})
    df_cleaned = pd.get_dummies(df_cleaned, columns=['EmploymentType'], drop_first=False)
    df_cleaned = pd.get_dummies(df_cleaned, columns=['LoanPurpose'], drop_first=False)
    df_cleaned = pd.get_dummies(df_cleaned, columns=['MaritalStatus'], drop_first=False)

    # Normalize column names
    df_cleaned.columns = df_cleaned.columns.str.strip().str.lower().str.replace(' ', '_')
    if 'loanid' in df_cleaned.columns:
        df_cleaned.drop(columns=['loanid'], inplace=True)

    # Convert bool columns to int
    bool_cols = df_cleaned.select_dtypes(include='bool').columns
    df_cleaned[bool_cols] = df_cleaned[bool_cols].astype(int)

    X = df_cleaned.drop('default', axis=1)
    y = df_cleaned['default']

    print(f"  Features: {X.shape[1]}")
    print(f"  Target distribution: 0={int((y==0).sum())}, 1={int((y==1).sum())} "
          f"({(y==1).mean()*100:.1f}% default)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    return X_train, X_test, y_train, y_test, scaler


def train_and_export():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    # Save scaler and feature columns
    print("\nSaving scaler and feature columns...")
    joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(SAVED_MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    # Train Gradient Boosting Classifier
    print("\nTraining Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=42
    )
    gb.fit(X_train, y_train)
    print("  Training complete.")

    # Save model
    model_path = os.path.join(SAVED_MODELS_DIR, "gradient_boosting.joblib")
    joblib.dump(gb, model_path)
    print(f"  Model saved: {model_path}")

    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = gb.predict(X_test)
    y_prob = gb.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Accuracy:  {acc:.6f}  ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.6f}  ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.6f}  ({rec*100:.2f}%)")
    print(f"  F1-Score:  {f1:.6f}  ({f1*100:.2f}%)")
    print(f"  ROC-AUC:   {roc:.6f}  ({roc*100:.2f}%)")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default (0)', 'Default (1)']))

    # Cross-validation (5-fold)
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(gb, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"  CV Mean Accuracy: {cv_mean:.6f} (+/- {cv_std:.6f})")

    # Save metrics
    metrics_data = {
        "model": "Gradient Boosting Classifier",
        "model_id": "gradient_boosting",
        "algorithm": f"GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)",
        "description": "Gradient Boosting is an ensemble classification algorithm that builds a sequence of decision trees, where each new tree attempts to correct the errors made by previous trees.",
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "roc_auc": round(roc, 6),
        "cv_mean_accuracy": round(cv_mean, 6),
        "cv_std_dev": round(cv_std, 6),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "dataset": {
            "total_records": 255347,
            "non_default": int((pd.concat([y_test, y_train]) == 0).sum()),
            "default": int((pd.concat([y_test, y_train]) == 1).sum()),
            "imbalance_note": "Dataset is imbalanced (~88.4% non-default, ~11.6% default). High accuracy alone does not indicate good default detection. Evaluate Recall and F1-Score."
        }
    }

    metrics_file = os.path.join(SAVED_MODELS_DIR, "models_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")

    # Clean up old model files
    old_files = [
        "logistic_regression.joblib",
        "naive_bayes.joblib",
        "decision_tree.joblib",
        "random_forest.joblib",
        "knn.joblib"
    ]
    for old in old_files:
        old_path = os.path.join(SAVED_MODELS_DIR, old)
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f"  Removed old model: {old}")

    print("\nTraining and export completed successfully!")
    print(f"Final model: Gradient Boosting Classifier")
    print(f"Saved at: {model_path}")


if __name__ == "__main__":
    train_and_export()
