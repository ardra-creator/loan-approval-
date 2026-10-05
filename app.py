import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

# Load model and metrics on startup (no retraining)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train_model.py first.")

model = joblib.load(MODEL_PATH)

metrics_data = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)

@app.route("/")
def index():
    """Home page with project overview, dataset stats, model comparison table, and EDA plots."""
    plots = [
        {"filename": "loan_approval_distribution.png", "title": "Loan Approval Distribution (Target)"},
        {"filename": "numerical_feature_distributions.png", "title": "Numerical Feature Distributions"},
        {"filename": "correlation_heatmap.png", "title": "Correlation Heatmap"},
        {"filename": "credit_history_vs_loan_approval.png", "title": "Credit History vs Loan Status"},
        {"filename": "income_vs_loan_approval.png", "title": "Applicant Income vs Loan Status"}
    ]
    return render_template("index.html", metrics=metrics_data, plots=plots)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction form and submission handler."""
    if request.method == "GET":
        return render_template("predict.html")

    try:
        # Extract inputs matching the actual dataset columns
        data = {
            "Gender": request.form.get("Gender", "").strip(),
            "Married": request.form.get("Married", "").strip(),
            "Dependents": int(request.form.get("Dependents", 0)),
            "Education": request.form.get("Education", "").strip(),
            "Self_Employed": request.form.get("Self_Employed", "").strip(),
            "Applicant_Income": float(request.form.get("Applicant_Income", 0)),
            "Coapplicant_Income": float(request.form.get("Coapplicant_Income", 0)),
            "Loan_Amount": float(request.form.get("Loan_Amount", 0)),
            "Loan_Amount_Term": float(request.form.get("Loan_Amount_Term", 360)),
            "Credit_History": int(request.form.get("Credit_History", 1)),
            "Property_Area": request.form.get("Property_Area", "").strip()
        }

        # Convert input into a pandas DataFrame matching trained schema
        input_df = pd.DataFrame([data])

        # Generate prediction using the loaded pipeline
        raw_pred = model.predict(input_df)[0]
        is_approved = (raw_pred == "Y" or raw_pred == 1)

        # Calculate prediction probability if supported
        probability = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            if "Y" in classes:
                y_idx = classes.index("Y")
                approval_prob = probs[y_idx]
            elif 1 in classes:
                y_idx = classes.index(1)
                approval_prob = probs[y_idx]
            else:
                approval_prob = probs[1]
            probability = round(float(approval_prob) * 100, 2)

        return render_template(
            "result.html",
            is_approved=is_approved,
            probability=probability,
            applicant_data=data,
            selected_model=metrics_data.get("selected_model", "Machine Learning Model")
        )

    except Exception as e:
        return render_template(
            "result.html",
            error=str(e)
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
