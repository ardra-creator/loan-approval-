"""
Training and Evaluation Script for Loan Approval Prediction
Trains 4 machine learning models using a scikit-learn Pipeline and ColumnTransformer,
evaluates them on the test set, selects the best model based on F1-score,
and saves the final pipeline as models/loan_model.pkl.
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def train_and_evaluate(csv_path="loan_approval_dataset.csv",
                       models_dir="models",
                       plots_dir="static/plots"):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading and Preprocessing Dataset")
    print("=" * 60)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    initial_shape = df.shape
    print(f"Loaded dataset: {initial_shape[0]} rows, {initial_shape[1]} columns")

    # Remove duplicate rows
    df = df.drop_duplicates()
    duplicates_removed = initial_shape[0] - df.shape[0]
    print(f"Removed duplicate rows: {duplicates_removed} (Current rows: {len(df)})")

    # Drop Customer_ID as it is an arbitrary identifier
    if 'Customer_ID' in df.columns:
        df = df.drop(columns=['Customer_ID'])
        print("Dropped 'Customer_ID' column (non-predictive identifier).")

    # Identify features X and target y
    target_col = 'Loan_Status'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Target distribution:\n{y.value_counts()}")

    # Define column groups
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    num_cols = ['Dependents', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Credit_History']

    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    # Build preprocessing pipeline
    # Numerical: median imputation + standard scaling
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: most frequent imputation + one-hot encoding
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # 80/20 train-test split with fixed random_state and stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")

    print("\n" + "=" * 60)
    print("STEP 2: Training & Evaluating Candidate Models")
    print("=" * 60)

    candidate_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42, probability=True)
    }

    results = {}
    trained_pipelines = {}

    for name, classifier in candidate_models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, pos_label='Y'))
        rec = float(recall_score(y_test, y_pred, pos_label='Y'))
        f1 = float(f1_score(y_test, y_pred, pos_label='Y'))
        cm = confusion_matrix(y_test, y_pred, labels=['N', 'Y']).tolist()

        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'confusion_matrix': cm
        }
        trained_pipelines[name] = pipeline

        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall:    {rec * 100:.2f}%")
        print(f"  F1 Score:  {f1 * 100:.2f}%")

    # Select final model based on F1-score, tie-breaking on accuracy
    best_model_name = max(
        results.keys(),
        key=lambda m: (results[m]['f1'], results[m]['accuracy'])
    )

    best_pipeline = trained_pipelines[best_model_name]
    best_metrics = results[best_model_name]

    print("\n" + "=" * 60)
    print("STEP 3: Model Comparison & Selection")
    print("=" * 60)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 75)
    for model_name, m in results.items():
        print(f"{model_name:<25} | {m['accuracy']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f}")

    print(f"\n--> Selected Best Model: {best_model_name}")
    print(f"    Selected based on F1 Score: {best_metrics['f1']:.4f} (Accuracy: {best_metrics['accuracy']:.4f})")

    # Generate Confusion Matrix Plot for Selected Best Model
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm_array = np.array(best_metrics['confusion_matrix'])
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Approved (N)', 'Approved (Y)'],
                yticklabels=['Not Approved (N)', 'Approved (Y)'],
                cbar=False, annot_kws={"size": 14, "fontweight": "bold"})
    ax.set_title(f"Confusion Matrix: {best_model_name}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    cm_plot_path = os.path.join(plots_dir, "confusion_matrix.png")
    fig.savefig(cm_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    # Save final pipeline
    model_save_path = os.path.join(models_dir, "loan_model.pkl")
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved complete pipeline to: {model_save_path}")

    # Save metrics metadata for Flask app
    metrics_data = {
        'selected_model': best_model_name,
        'models_comparison': results,
        'feature_names': {
            'numerical': num_cols,
            'categorical': cat_cols
        },
        'test_set_size': len(X_test),
        'train_set_size': len(X_train)
    }
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved model metrics and metadata to: {metrics_path}")

    print("\nTraining and evaluation finished successfully!")
    return best_model_name, results

if __name__ == "__main__":
    train_and_evaluate()
