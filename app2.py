import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
from huggingface_hub import InferenceClient

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

# Background and styling
st.markdown("""
    <style>
    body { background-color: #f5f7fa; color: #333; }
    .main { background-color: #ffffff; border-radius: 10px; padding: 2rem;
            box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    .stApp { padding: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ----------------- TITLE -----------------
st.title("💰 Loan Approval Prediction System")
st.markdown("An intelligent system to predict loan approval using multiple ML models.")

# ----------------- FEATURE DESCRIPTIONS -----------------
with st.expander("🧠 Feature Descriptions"):
    st.markdown("""
    - **credit.policy**: 1 if the customer meets the credit underwriting criteria.
    - **purpose**: Purpose of the loan (debt consolidation, educational, etc.).
    - **int.rate**: Interest rate of the loan.
    - **installment**: Monthly payment for the loan.
    - **log.annual.inc**: Log of annual income.
    - **dti**: Debt-to-income ratio.
    - **fico**: FICO credit score.
    - **days.with.cr.line**: Days with credit line open.
    - **revol.bal**: Revolving balance.
    - **revol.util**: Revolving utilization rate.
    - **inq.last.6mths**: Inquiries in last 6 months.
    - **delinq.2yrs**: Delinquencies in last 2 years.
    - **pub.rec**: Public records.
    """)

# ----------------- LOAD MODELS -----------------
model_files = {
    "Logistic Regression": "logistic_regression_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "LightGBM": "lightgbm_model.joblib",
    "XGBoost": "xgboost_model.joblib"
}

models = {}
for name, path in model_files.items():
    try:
        models[name] = joblib.load(path)
    except:
        st.warning(f"⚠️ Could not load model: {name}")

# Purpose mapping
purpose_mapping = {
    'credit_card': 0, 'debt_consolidation': 1, 'educational': 2,
    'home_improvement': 3, 'major_purchase': 4, 'small_business': 5,
    'all_other': 6
}

# ----------------- SIDEBAR INPUT -----------------
st.sidebar.header("📋 Input Borrower Information")
user_input = {
    "credit.policy": st.sidebar.selectbox("Credit Policy", [0, 1]),
    "purpose": st.sidebar.selectbox("Purpose", list(purpose_mapping.keys())),
    "int.rate": st.sidebar.slider("Interest Rate", 0.05, 0.30, 0.12),
    "installment": st.sidebar.slider("Installment", 50.0, 1000.0, 250.0),
    "log.annual.inc": st.sidebar.slider("Log Annual Income", 8.0, 12.0, 10.0),
    "dti": st.sidebar.slider("Debt-to-Income Ratio", 0.0, 40.0, 18.0),
    "fico": st.sidebar.slider("FICO Score", 300, 850, 700),
    "days.with.cr.line": st.sidebar.slider("Days with Credit Line", 1000, 8000, 4000),
    "revol.bal": st.sidebar.slider("Revolving Balance", 0, 100000, 15000),
    "revol.util": st.sidebar.slider("Revolving Utilization (%)", 0.0, 100.0, 45.0),
    "inq.last.6mths": st.sidebar.slider("Inquiries Last 6 Months", 0, 10, 1),
    "delinq.2yrs": st.sidebar.slider("Delinquencies Last 2 Years", 0, 5, 0),
    "pub.rec": st.sidebar.slider("Public Records", 0, 5, 0)
}

input_df = pd.DataFrame([user_input])
input_df['purpose'] = input_df['purpose'].map(purpose_mapping)

# ----------------- PREDICTION -----------------
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

if st.sidebar.button("Predict"):
    prediction = selected_model.predict(input_df)[0]
    proba = selected_model.predict_proba(input_df)[0][1]

    st.subheader("🎯 Prediction Result")
    st.markdown(f"**Prediction:** {'Approved ✅' if prediction == 1 else 'Rejected ❌'}")
    st.markdown(f"**Probability of Approval:** {round(proba * 100, 2)}%")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 2),
        title={'text': "Approval Probability"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "green" if proba >= 0.5 else "red"},
               'steps': [
                   {'range': [0, 50], 'color': '#ffdddd'},
                   {'range': [50, 100], 'color': '#ddffdd'}]}))
    st.plotly_chart(fig, use_container_width=True)

    # SHAP Explanation
    try:
        explainer = shap.Explainer(selected_model)
        shap_values = explainer(input_df)

        st.markdown("**Top Features Impacting the Decision:**")
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(bbox_inches="tight")

    except:
        st.warning("SHAP explanation not available for this model.")

# ----------------- CHATBOT -----------------
st.markdown("---")
st.subheader("💬 Loan Advisor Chatbot")

HF_TOKEN = st.secrets["HF_TOKEN"]  # Add this in .streamlit/secrets.toml
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token=HF_TOKEN)

SYSTEM_PROMPT = """
You are a friendly and knowledgeable loan advisor. 
Explain loan terms, eligibility, interest rates, and repayment strategies clearly.
"""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_msg = st.text_input("You:", key="chat_input")

if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

    with st.spinner("Thinking..."):
        response_text = ""
        for message in client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *st.session_state.chat_history
            ],
            max_tokens=300,
            stream=True
        ):
            if message.choices[0].delta.content:
                response_text += message.choices[0].delta.content

    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# Display chat
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ by Team Numerixa</center>", unsafe_allow_html=True)
