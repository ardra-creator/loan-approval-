# Loan Approval Prediction System

A Machine Learning college Data Science mini-project to predict loan eligibility based on applicant financial, demographic, and credit history variables.

---

## 1. Problem Statement

Manual loan approval workflows in retail banking and financial institutions are slow, labor-intensive, and prone to human inconsistency. Evaluating credit risk effectively requires analyzing multiple applicant attributes—including credit history, applicant income, co-applicant income, loan amount, education, and dependents—to minimize default risk (False Positives) while maintaining business revenue by approving creditworthy applicants (minimizing False Negatives).

---

## 2. Objective

To develop, evaluate, and deploy an end-to-end Machine Learning classification pipeline that:
1. Accurately predicts loan approval status (`APPROVED` vs `NOT APPROVED`).
2. Evaluates multiple classical classification algorithms using actual dataset statistics.
3. Automatically selects the optimal model using **F1-score** (balanced harmonic mean of precision and recall) alongside accuracy.
4. Provides an interactive web application (Flask) where users input applicant profiles and receive real-time predictions and approval probabilities.

---

## 3. Dataset Information

The project utilizes the actual dataset provided in `loan_approval_dataset.csv`.

* **Total Records:** 614 rows
* **Total Columns:** 13 (1 identifier, 11 features, 1 target)
* **Target Variable:** `Loan_Status`
  * `Y` (Approved): 422 records (~68.7%)
  * `N` (Not Approved): 192 records (~31.3%)
* **Missing Values:** Handled automatically in pipeline via median (numerical) and mode (categorical).
* **Duplicates:** Handled automatically (`df.drop_duplicates()`).
* **Non-predictive Identifier:** `Customer_ID` is excluded from modeling.

---

## 4. Features

| Feature Column | Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| `Gender` | Categorical | Applicant's biological sex | `Male`, `Female` |
| `Married` | Categorical | Marital status | `Yes`, `No` |
| `Dependents` | Numerical | Number of financial dependents | `0`, `1`, `2`, `4` (represents 3+) |
| `Education` | Categorical | Educational qualification | `Graduate`, `Not Graduate` |
| `Self_Employed` | Categorical | Self-employment status | `No`, `Yes` |
| `Applicant_Income` | Numerical | Primary applicant monthly income | $150 to $81,000 |
| `Coapplicant_Income` | Numerical | Co-applicant monthly income | $0 to $41,667 |
| `Loan_Amount` | Numerical | Loan amount applied for (thousands) | $9k to $700k |
| `Loan_Amount_Term` | Numerical | Term of loan in months | 12 to 480 months |
| `Credit_History` | Numerical/Binary | Credit history meets criteria | `1` (Good), `0` (Poor) |
| `Property_Area` | Categorical | Property location area | `Urban`, `Semiurban`, `Rural` |

---

## 5. Data Preprocessing & Pipeline

Data preprocessing is encapsulated in a Scikit-Learn `ColumnTransformer` and `Pipeline` to prevent data leakage and ensure identical transformations during training and web inference:

* **Numerical Pipeline (`num_pipeline`):**
  * `SimpleImputer(strategy='median')`: Imputes missing continuous values using feature medians.
  * `StandardScaler()`: Normalizes numerical values to zero mean and unit variance.
* **Categorical Pipeline (`cat_pipeline`):**
  * `SimpleImputer(strategy='most_frequent')`: Imputes missing categorical values using the mode.
  * `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: Encodes categorical features into binary indicator columns.
* **Train / Test Split:**
  * 80% Training set (491 samples)
  * 20% Testing set (123 samples)
  * Stratification enabled (`stratify=y`) to maintain target class distribution across splits.
  * Random seed fixed (`random_state=42`) for reproducibility.

---

## 6. Exploratory Data Analysis (EDA)

The `eda.py` script generates only the required informative visual analysis plots, saved to `static/plots/`:

1. `loan_approval_distribution.png`: Target variable distribution (`Y` vs `N`).
2. `numerical_feature_distributions.png`: Histograms with KDE for all continuous features.
3. `correlation_heatmap.png`: Pearson correlation heatmap among numerical features.
4. `credit_history_vs_loan_approval.png`: Loan approval frequency conditioned on credit history.
5. `income_vs_loan_approval.png`: Box plots comparing applicant income by approval status.

---

## 7. Algorithms Trained & Evaluation Results

Four classification algorithms were trained on the identical preprocessing pipeline and evaluated on the 123 held-out test samples. All metrics below are actual calculated values:

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **80.49%** | **80.20%** | **95.29%** | **87.10%** |
| Decision Tree | 75.61% | 83.13% | 81.18% | 82.14% |
| Random Forest | 78.05% | 80.21% | 90.59% | 85.08% |
| Support Vector Machine (SVM) | 80.49% | 80.20% | 95.29% | 87.10% |

### Confusion Matrix for Selected Best Model (`Logistic Regression`):

```text
                  Predicted Negative (N)   Predicted Positive (Y)
True Negative (N)         18                       20
True Positive (Y)          4                       81
```

* **True Positives (TP):** 81
* **True Negatives (TN):** 18
* **False Positives (FP):** 20
* **False Negatives (FN):** 4

---

## 8. Final Model Selection

* **Selected Model:** **Logistic Regression**
* **Selection Criteria:** Highest **F1-Score (0.8710)** and **Accuracy (80.49%)**, combined with superior calibration for probability estimation via `predict_proba()`.
* **Saved Artifact:** Complete pipeline saved at `models/loan_model.pkl` (includes `ColumnTransformer` + model).

---

## 9. Project Directory Structure

```text
loan-approval-prediction/
│
├── app.py                      # Flask web server and routing
├── train_model.py              # Data preprocessing, training, and evaluation
├── eda.py                      # Exploratory Data Analysis plotting script
├── requirements.txt            # Minimal required Python dependencies
├── README.md                   # Project documentation and instructions
├── loan_approval_dataset.csv   # Actual dataset
│
├── models/
│   ├── loan_model.pkl          # Saved end-to-end trained pipeline
│   └── metrics.json            # Actual calculated metrics and metadata
│
├── static/
│   ├── css/
│   │   └── style.css           # Clean, responsive CSS styling
│   └── plots/
│       ├── confusion_matrix.png
│       ├── correlation_heatmap.png
│       ├── credit_history_vs_loan_approval.png
│       ├── income_vs_loan_approval.png
│       ├── loan_approval_distribution.png
│       └── numerical_feature_distributions.png
│
└── templates/
    ├── index.html              # Overview, performance table, EDA gallery
    ├── predict.html            # Input form using dataset feature columns
    └── result.html             # Prediction result, probability & disclaimer
```

---

## 10. How to Run the Project

### Step 1: Install Dependencies
Open a terminal in the project directory and install the requirements:
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Re-train and Evaluate Model
Run the model training script to train the 4 models and save the pipeline:
```bash
python train_model.py
```

### Step 3: Run the Flask Web Application
Start the local web server:
```bash
python app.py
```

Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 11. Testing the Workflow

1. Navigate to `http://127.0.0.1:5000` to review the dataset overview, model evaluation comparison table, confusion matrix, and EDA charts.
2. Click **"Check Loan Eligibility"** or navigate to `http://127.0.0.1:5000/predict`.
3. Enter applicant details:
   * **Test Case 1 (Approval Expected):** Graduate, Married, Income: $5000, Coapplicant: $2000, Loan Amount: $120k, Term: 360, Credit History: Good (1), Property Area: Semiurban.
     * Expected Output: `Loan Status: APPROVED` with estimated approval probability ~80%+.
   * **Test Case 2 (Rejection Expected):** Income: $1500, Coapplicant: $0, Loan Amount: $250k, Term: 360, Credit History: Poor (0), Property Area: Rural.
     * Expected Output: `Loan Status: NOT APPROVED` with estimated approval probability <15%.
