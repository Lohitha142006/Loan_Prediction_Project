from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model/loan_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get form values
        gender = float(request.form["gender"])
        married = float(request.form["married"])
        dependents = float(request.form["dependents"])
        education = float(request.form["education"])
        self_employed = float(request.form["self_employed"])

        applicant_income = float(request.form["applicant_income"])
        coapplicant_income = float(request.form["coapplicant_income"])
        loan_amount = float(request.form["loan_amount"])
        loan_amount_term = float(request.form["loan_amount_term"])

        credit_history = float(request.form["credit_history"])
        property_area = float(request.form["property_area"])

        # Arrange features in same order used during training
        features = np.array([[
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            property_area
        ]])

        # Prediction
        prediction = model.predict(features)[0]

        # Prediction probability
        probability = model.predict_proba(features)[0]

        confidence = round(max(probability) * 100, 2)

        # Probability of approval
        if len(probability) > 1:
            approval_probability = round(probability[1] * 100, 2)
        else:
            approval_probability = confidence

        # AI Score
        if confidence >= 90:
            ai_score = "EXCELLENT"
        elif confidence >= 80:
            ai_score = "VERY GOOD"
        elif confidence >= 70:
            ai_score = "GOOD"
        elif confidence >= 60:
            ai_score = "MODERATE"
        else:
            ai_score = "WEAK"

        # Result
        if prediction == 1:

            result = "🎉 LOAN APPROVED"

            risk = "🟢 LOW RISK"

            message = """
            Applicant profile meets lending requirements.
            Credit history and financial indicators suggest
            a strong repayment capability.
            """

        else:

            result = "❌ LOAN REJECTED"

            risk = "🔴 HIGH RISK"

            message = """
            Applicant profile does not meet the current
            lending requirements. Financial indicators
            suggest a higher repayment risk.
            """

        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence,
            approval_probability=approval_probability,
            ai_score=ai_score,
            risk=risk,
            message=message
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction="Error",
            confidence=0,
            approval_probability=0,
            ai_score="N/A",
            risk="N/A",
            message=str(e)
        )


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
