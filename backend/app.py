"""
Flask backend for Loan Default Prediction System.
Uses a single model: Gradient Boosting Classifier.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates"),
    static_folder=os.path.join(ROOT_DIR, "static")
)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

# Load preprocessing artifacts
scaler = joblib.load(os.path.join(SAVED_MODELS_DIR, "scaler.joblib"))
with open(os.path.join(SAVED_MODELS_DIR, "feature_columns.json"), "r") as f:
    FEATURE_COLUMNS = json.load(f)
with open(os.path.join(SAVED_MODELS_DIR, "models_metrics.json"), "r") as f:
    MODEL_METRICS = json.load(f)

# Load single model: Gradient Boosting
model = joblib.load(os.path.join(SAVED_MODELS_DIR, "gradient_boosting.joblib"))

NUMERIC_COLS = [
    'age', 'income', 'loanamount', 'creditscore', 'monthsemployed',
    'numcreditlines', 'interestrate', 'loanterm', 'dtiratio'
]


def preprocess_applicant(data):
    """Transform raw applicant dict into a scaled DataFrame matching FEATURE_COLUMNS."""
    row = {col: 0 for col in FEATURE_COLUMNS}

    row['age'] = float(data.get('age', 38))
    row['income'] = float(data.get('income', 82000))
    row['loanamount'] = float(data.get('loanAmount', 28000))
    row['creditscore'] = float(data.get('creditScore', 740))
    row['monthsemployed'] = float(data.get('monthsEmployed', 60))
    row['numcreditlines'] = float(data.get('numCreditLines', 4))
    row['interestrate'] = float(data.get('interestRate', 9.5))
    row['loanterm'] = float(data.get('loanTerm', 36))
    row['dtiratio'] = float(data.get('dtiRatio', 0.28))

    def parse_bool(val):
        if isinstance(val, bool): return int(val)
        return 1 if str(val).lower() in ['true', '1', 'yes'] else 0

    row['hasmortgage'] = parse_bool(data.get('hasMortgage', True))
    row['hasdependents'] = parse_bool(data.get('hasDependents', False))
    row['hascosigner'] = parse_bool(data.get('hasCoSigner', True))

    edu = str(data.get('education', "Bachelor's")).strip().lower().replace("'", "")
    if "bachelor" in edu:
        row["education_bachelor's"] = 1
    elif "high" in edu:
        row["education_high_school"] = 1
    elif "master" in edu:
        row["education_master's"] = 1
    elif "phd" in edu or "doctor" in edu:
        row["education_phd"] = 1

    emp = str(data.get('employmentType', "Full-time")).strip().lower()
    if "full" in emp:
        row["employmenttype_full-time"] = 1
    elif "part" in emp:
        row["employmenttype_part-time"] = 1
    elif "self" in emp:
        row["employmenttype_self-employed"] = 1
    elif "unemploy" in emp:
        row["employmenttype_unemployed"] = 1

    purpose = str(data.get('loanPurpose', "Business")).strip().lower()
    if "auto" in purpose:
        row["loanpurpose_auto"] = 1
    elif "business" in purpose:
        row["loanpurpose_business"] = 1
    elif "education" in purpose:
        row["loanpurpose_education"] = 1
    elif "home" in purpose:
        row["loanpurpose_home"] = 1
    else:
        row["loanpurpose_other"] = 1

    marital = str(data.get('maritalStatus', "Married")).strip().lower()
    if "divorced" in marital:
        row["maritalstatus_divorced"] = 1
    elif "single" in marital:
        row["maritalstatus_single"] = 1
    else:
        row["maritalstatus_married"] = 1

    df_single = pd.DataFrame([row])[FEATURE_COLUMNS]
    df_single[NUMERIC_COLS] = scaler.transform(df_single[NUMERIC_COLS])
    return df_single


# ---- HTML Page Routes ----

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/models')
def models_page():
    return render_template('models.html')

@app.route('/about')
def about():
    return render_template('about.html')


# ---- API Routes ----

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": "Gradient Boosting Classifier",
        "model_loaded": model is not None,
        "accuracy": MODEL_METRICS.get("accuracy")
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(MODEL_METRICS)

@app.route('/api/default-applicant', methods=['GET'])
def get_default_applicant():
    presets = {
        "default": {
            "name": "Standard Applicant",
            "age": 38, "education": "Bachelor's", "maritalStatus": "Married",
            "income": 82000, "employmentType": "Full-time", "monthsEmployed": 60,
            "creditScore": 740, "dtiRatio": 0.28, "numCreditLines": 4,
            "loanAmount": 28000, "interestRate": 9.5, "loanTerm": 36,
            "loanPurpose": "Business",
            "hasMortgage": True, "hasDependents": False, "hasCoSigner": True
        },
        "high_risk": {
            "name": "High Risk Applicant",
            "age": 22, "education": "High School", "maritalStatus": "Single",
            "income": 22000, "employmentType": "Unemployed", "monthsEmployed": 6,
            "creditScore": 510, "dtiRatio": 0.65, "numCreditLines": 4,
            "loanAmount": 48000, "interestRate": 22.5, "loanTerm": 60,
            "loanPurpose": "Other",
            "hasMortgage": False, "hasDependents": True, "hasCoSigner": False
        },
        "borderline": {
            "name": "Borderline Applicant",
            "age": 31, "education": "Bachelor's", "maritalStatus": "Divorced",
            "income": 48000, "employmentType": "Part-time", "monthsEmployed": 24,
            "creditScore": 635, "dtiRatio": 0.44, "numCreditLines": 3,
            "loanAmount": 26000, "interestRate": 15.0, "loanTerm": 48,
            "loanPurpose": "Auto",
            "hasMortgage": False, "hasDependents": False, "hasCoSigner": False
        }
    }
    return jsonify(presets)

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"})

    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON in request body"}), 400

    if not data:
        return jsonify({"status": "error", "message": "No input data provided"}), 400

    try:
        X_processed = preprocess_applicant(data)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Preprocessing error: {str(e)}"}), 400

    try:
        pred_class = int(model.predict(X_processed)[0])
        prob_default = float(model.predict_proba(X_processed)[0][1]) * 100.0
    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction error: {str(e)}"}), 500

    is_no_default = pred_class == 0

    # Key factors
    positive_factors = []
    risk_factors = []

    credit = float(data.get('creditScore', 740))
    dti = float(data.get('dtiRatio', 0.28))
    income = float(data.get('income', 82000))
    loan_amt = float(data.get('loanAmount', 28000))
    cosigner = str(data.get('hasCoSigner', 'true')).lower() in ['true', '1', 'yes']

    if credit >= 700:
        positive_factors.append(f"Credit Score: {int(credit)}/850")
    elif credit < 620:
        risk_factors.append(f"Low Credit Score: {int(credit)}/850")

    if dti <= 0.35:
        positive_factors.append(f"DTI Ratio: {round(dti*100, 1)}%")
    else:
        risk_factors.append(f"High DTI Ratio: {round(dti*100, 1)}%")

    if cosigner:
        positive_factors.append("Co-Signer present")
    else:
        risk_factors.append("No Co-Signer")

    if income > 0 and (loan_amt / income) <= 0.4:
        positive_factors.append(f"Loan-to-Income Ratio: {round((loan_amt/income)*100, 1)}%")
    elif income > 0 and (loan_amt / income) > 0.7:
        risk_factors.append(f"High Loan-to-Income Ratio: {round((loan_amt/income)*100, 1)}%")

    # Risk tier
    if prob_default < 20:
        risk_tier = "Low Risk"
    elif prob_default < 50:
        risk_tier = "Moderate Risk"
    else:
        risk_tier = "High Risk"

    return jsonify({
        "status": "success",
        "model_used": "Gradient Boosting Classifier",
        "prediction": pred_class,
        "prediction_text": "Yes" if pred_class == 1 else "No",
        "default_probability": round(prob_default, 2),
        "risk_tier": risk_tier,
        "positive_factors": positive_factors,
        "risk_factors": risk_factors
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 50)
    print("  Loan Default Prediction System")
    print(f"  http://localhost:{port}")
    print("  Model: Gradient Boosting Classifier")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port, debug=False)
