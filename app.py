import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# from sklearn.preprocessing import LabelEncoder

# Load models
logreg = joblib.load("logistic_regression_model.joblib")
dt = joblib.load("decision_tree_model.joblib")
rf = joblib.load("random_forest_model.joblib")
lgbm = joblib.load("lightgbm_model.joblib")
xgb = joblib.load("xgboost_model.joblib")


st.set_page_config(page_title="Loan Default Prediction", layout="wide")

st.title("📊 Loan Default Risk Prediction App")

st.markdown("""
This application uses 5 machine learning models to predict whether a loan will be fully paid or defaulted based on user input.
""")

with st.form("loan_form"):
    st.subheader("Enter Loan Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_policy = st.selectbox("Credit Policy (1: Meets policy)", [0, 1])
        purpose = st.selectbox("Loan Purpose", ['debt_consolidation', 'credit_card', 'all_other', 'home_improvement', 'small_business', 'major_purchase'])
        int_rate = st.number_input("Interest Rate", value=0.12)
        installment = st.number_input("Installment", value=300.0)

    with col2:
        log_annual_inc = st.number_input("Log Annual Income", value=10.5)
        dti = st.number_input("Debt-to-Income Ratio (DTI)", value=15.0)
        fico = st.number_input("FICO Score", min_value=300, max_value=850, value=720)
        days_with_cr_line = st.number_input("Days With Credit Line", value=2000)

    with col3:
        revol_bal = st.number_input("Revolving Balance", value=15000)
        revol_util = st.number_input("Revolving Utilization (%)", value=50.0)
        inq_last_6mths = st.number_input("Inquiries Last 6 Months", value=1)
        delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", value=0)
        pub_rec = st.number_input("Public Records", value=0)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Encode categorical feature
    purpose_encoded = purpose_encoder.transform([purpose])[0]

    # Create input DataFrame
    input_data = pd.DataFrame([[
        credit_policy, purpose_encoded, int_rate, installment,
        log_annual_inc, dti, fico, days_with_cr_line,
        revol_bal, revol_util, inq_last_6mths, delinq_2yrs, pub_rec
    ]], columns=[
        'credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc',
        'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
        'inq.last.6mths', 'delinq.2yrs', 'pub.rec'
    ])

    # Prediction from models
    st.subheader("📈 Predictions")
    models = {
        "Logistic Regression": logreg,
        "Decision Tree": dt,
        "Random Forest": rf,
        "LightGBM": lgbm,
        "XGBoost": xgb
    }

    results = {}
    for name, model in models.items():
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]  # Probability of default
        results[name] = {"Prediction": "Default" if prediction else "Fully Paid", "Probability": prob}

    results_df = pd.DataFrame(results).T
    st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)

    # SHAP explanation (only one model for simplicity)
    st.subheader("🔍 Model Explanation (SHAP)")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(input_data)

    st.markdown("**SHAP Feature Impact for Random Forest:**")
    fig, ax = plt.subplots()
    shap.bar_plot(shap_values[1], feature_names=input_data.columns)
    st.pyplot(fig)

    # Insights Section
    st.subheader("📌 Insights")
    high_prob_model = max(results.items(), key=lambda x: x[1]['Probability'])
    if high_prob_model[1]['Prediction'] == "Default":
        st.warning(f"⚠️ Model **{high_prob_model[0]}** predicts a high risk of default with **{high_prob_model[1]['Probability']:.2f}** probability.")
    else:
        st.success(f"✅ Model **{high_prob_model[0]}** predicts the loan will likely be **fully paid** with **{1 - high_prob_model[1]['Probability']:.2f}** probability.")
