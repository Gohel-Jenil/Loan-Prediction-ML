# Loan Default Prediction System

A machine learning web application that predicts the likelihood of loan default using a **Gradient Boosting Classifier**.

## Machine Learning Model

**Gradient Boosting Classifier** — An ensemble classification algorithm that builds a sequence of decision trees, where each new tree corrects the errors of the previous one.

## Project Architecture

```
Loan_default.csv (Dataset)
        |
  Data Cleaning & Preprocessing
        |
  Train/Test Split (80/20)
        |
  StandardScaler (numeric features)
        |
  Gradient Boosting Classifier
        |
  Model Evaluation & Export
        |
  Flask Backend (app.py)
        |
  HTML/CSS/JavaScript Frontend
```

## Project Structure

```
Loan_project_Frontend/
├── backend/
│   ├── app.py                    # Flask backend
│   ├── train_export.py           # Model training script
│   └── saved_models/
│       ├── gradient_boosting.joblib  # Trained model
│       ├── scaler.joblib             # StandardScaler
│       ├── feature_columns.json      # Feature column order (28 columns)
│       └── models_metrics.json       # Evaluation metrics
├── templates/
│   ├── index.html                # Home page
│   ├── predict.html              # Prediction form
│   ├── models.html               # Model metrics
│   └── about.html                # Project information
├── static/
│   ├── css/style.css
│   └── js/app.js
├── Loan_default.csv              # Dataset (255,347 records)
├── requirements.txt
└── README.md
```

## How to Install

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python backend/app.py
```

Open browser: **http://localhost:5000**

## Pages

| Route | Page |
|-------|------|
| `/` | Home |
| `/predict` | Prediction Form |
| `/models` | Model Metrics |
| `/about` | Project Information |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server health check |
| GET | `/api/models` | Model metrics |
| GET | `/api/default-applicant` | Sample applicant presets |
| POST | `/api/predict` | Generate prediction |

## Dataset

- **Records:** 255,347 loan applications
- **Target:** Default (0 = No Default, 1 = Default)
- **Class Distribution:** ~88.4% non-default, ~11.6% default (imbalanced)

## Features

| Feature | Type | Range |
|---------|------|-------|
| Age | Numeric | 18–69 |
| Income | Numeric | $15,000–$149,999 |
| LoanAmount | Numeric | $5,000–$249,999 |
| CreditScore | Numeric | 300–849 |
| MonthsEmployed | Numeric | 0–119 |
| NumCreditLines | Numeric | 1–4 |
| InterestRate | Numeric | 2%–25% |
| LoanTerm | Numeric | 12, 24, 36, 48, 60 months |
| DTIRatio | Numeric | 0.1–0.9 |
| Education | Categorical | Bachelor's, Master's, High School, PhD |
| EmploymentType | Categorical | Full-time, Part-time, Self-employed, Unemployed |
| MaritalStatus | Categorical | Married, Divorced, Single |
| LoanPurpose | Categorical | Auto, Business, Education, Home, Other |
| HasMortgage | Binary | Yes/No |
| HasDependents | Binary | Yes/No |
| HasCoSigner | Binary | Yes/No |

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| ML Model | scikit-learn (GradientBoostingClassifier) |
| Data | Pandas, NumPy |
| Serialization | Joblib |
