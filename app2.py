import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

model_files = {
    "Logistic Regression": "logistic_regression_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "LightGBM": "lightgbm_model.joblib",
    "XGBoost": "xgboost_model.joblib"
}
models = {name: joblib.load(path) for name, path in model_files.items()}

# Purpose encoding
purpose_mapping = {
    'Debt Consolidation': 0,
    'Credit Card': 1,
    'Home Improvement': 2,
    'Major Purchase': 3,
    'Small Business': 4,
    'Educational': 5,
    'All Other': 6
}

# Feature descriptions
def display_feature_info():
    st.markdown("### 🔍 Feature Descriptions")
    st.markdown("""
    - **credit.policy**: 1 if the customer meets the credit underwriting criteria.
    - **purpose**: Reason for the loan (e.g. Debt Consolidation, Major Purchase).
    - **int.rate**: Interest rate on the loan.
    - **installment**: Monthly installment paid by the borrower.
    - **log.annual.inc**: Natural log of the self-reported annual income.
    - **dti**: Debt-to-income ratio.
    - **fico**: FICO credit score.
    - **days.with.cr.line**: Number of days the borrower has had a credit line.
    - **revol.bal**: Revolving balance on credit card (amount unpaid at end of month).
    - **revol.util**: Revolving line utilization rate (credit used vs. total available).
    - **inq.last.6mths**: Number of inquiries in the last 6 months.
    - **delinq.2yrs**: Number of times borrower was 30+ days past due in the last 2 years.
    - **pub.rec**: Number of derogatory public records.
    """)

# Sidebar
with st.sidebar:
    st.title("🔧 Input Features")
    credit_policy = st.selectbox("Credit Policy", [0, 1])
    purpose = st.selectbox("Purpose", list(purpose_mapping.keys()))
    int_rate = st.number_input("Interest Rate", min_value=0.0, value=0.12)
    installment = st.number_input("Installment", min_value=0.0, value=250.0)
    log_annual_inc = st.number_input("Log Annual Income", min_value=0.0, value=10.5)
    dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=15.0)
    fico = st.slider("FICO Score", min_value=300, max_value=850, value=720)
    days_with_cr_line = st.number_input("Days with Credit Line", min_value=0.0, value=2000.0)
    revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=15000.0)
    revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, value=45.0)
    inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0, value=1)
    delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", min_value=0, value=0)
    pub_rec = st.number_input("Public Records", min_value=0, value=0)

# Encode purpose
purpose_encoded = purpose_mapping.get(purpose, 6)

# Prepare input
input_data = pd.DataFrame([{
    "credit.policy": credit_policy,
    "purpose": purpose_encoded,
    "int.rate": int_rate,
    "installment": installment,
    "log.annual.inc": log_annual_inc,
    "dti": dti,
    "fico": fico,
    "days.with.cr.line": days_with_cr_line,
    "revol.bal": revol_bal,
    "revol.util": revol_util,
    "inq.last.6mths": inq_last_6mths,
    "delinq.2yrs": delinq_2yrs,
    "pub.rec": pub_rec
}])

# Main UI
st.title("🏦 Loan Approval Prediction App")
st.markdown("#### Made with ❤️ by Team Numerixa")

# Subsections
with st.expander("📊 Prediction Results"):
    if st.button("Predict with All Models"):
        results = {}
        for name, model in models.items():
            prediction = model.predict(input_data)[0]
            label = "Approved ✅" if prediction == 1 else "Not Approved ❌"
            results[name] = label
        st.write("### 🔮 Predictions:")
        for model, label in results.items():
            st.success(f"{model.upper()}: {label}")

with st.expander("📈 Insights"):
    st.markdown("""
    - High FICO score usually indicates good creditworthiness.
    - A lower DTI ratio increases chances of loan approval.
    - Purpose of loan plays a key role in underwriting.
    - Fewer recent credit inquiries are seen positively by lenders.
    """)

with st.expander("💡 Feature Guide"):
    display_feature_info()

# Optional chatbot UI
with st.expander("💬 Ask the Bot (Beta)"):
    st.markdown("This is a placeholder for an assistant to answer your loan-related questions.")
    user_question = st.text_input("Ask a question about loan eligibility:")
    if user_question:
        st.info("🤖: I am still learning. In future updates, I will give you a smart answer!")
