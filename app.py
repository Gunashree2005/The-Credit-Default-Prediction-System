from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import pickle
import os
from datetime import timedelta

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'dF83nLwq!uZkP@r7^b$X2sH9eM1#yVcQ'
app.permanent_session_lifetime = timedelta(minutes=30)

# Load model, scaler, and encoders
model = pickle.load(open("credit_score_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

REQUIRED_COLS = ['AGE', 'INCOME', 'EMPLOYMENT_STATUS', 'LOAN_AMOUNT', 'LOAN_TERM',
                 'INTEREST_RATE', 'NUM_OF_DEPENDENTS', 'MARITAL_STATUS', 'EDUCATION_LEVEL']

@app.route('/')
def home():
    return render_template("homepage.html")

@app.route('/login', methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    if username == "admin" and password == "123":
        return redirect(url_for('creditscore'))
    else:
        return "Invalid credentials. Please try again.", 401

@app.route('/signup', methods=["POST"])
def signup():
    username = request.form.get("username")
    password = request.form.get("password")
    print(f"📝 New signup -> Username: {username}, Password: {password}")
    return redirect(url_for('creditscore'))

@app.route('/creditscore')
def creditscore():
    return render_template("creditscore.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        filename = file.filename

        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        if not set(REQUIRED_COLS).issubset(df.columns):
            missing = list(set(REQUIRED_COLS) - set(df.columns))
            return f"Missing columns: {missing}", 400

        # Encode categorical columns
        for col in df.select_dtypes(include='object').columns:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            else:
                df[col] = df[col].astype("category").cat.codes

        scaled = scaler.transform(df[REQUIRED_COLS])
        preds = model.predict(scaled)

        output = []
        for i, pred in enumerate(preds):
            score = round(pred, 2)
            if score >= 750:
                risk = "Low Risk"
                decision = "Likely Loan Approval"
            elif score >= 650:
                risk = "Medium Risk"
                decision = "Possible Approval with Caution"
            else:
                risk = "High Risk"
                decision = "Loan Likely Rejected. Try Budget Tracking."

            output.append({
                "score": score,
                "risk": risk,
                "decision": decision
            })

        # Store result in session
        session['prediction_result'] = output

        # Redirect to result page
        return redirect(url_for('analysis'))

    except Exception as e:
        print("🔥 SERVER ERROR:", str(e))
        return f"Server error: {str(e)}", 500

@app.route('/analysis')
def analysis():
    results = session.get('prediction_result')
    return render_template("analysis.html", results=results)

@app.route('/signup.html')
def budget_page():
    return render_template("budgettracking.html")

@app.route('/budget')
def budget():
    return render_template("budget.html")

@app.route('/clear_session')
def clear_session():
    session.pop('prediction_result', None)
    return redirect(url_for('creditscore'))

if __name__ == '__main__':
    app.run(debug=True)
