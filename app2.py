import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline

# Set Streamlit page config
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

# Set background color and header
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
        color: #333;
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .stApp {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("💰 Loan Approval Prediction System")
st.markdown("An intelligent system to predict loan approval using multiple ML models.")

# Feature meanings
with st.expander("🧠 Feature Descriptions"):
    st.markdown("""
    - **credit.policy**: 1 if the customer meets the credit underwriting criteria.
    - **purpose**: The purpose of the loan (debt consolidation, educational, etc.).
    - **int.rate**: Interest rate of the loan.
    - **installment**: Monthly payment for the loan.
    - **log.annual.inc**: Natural log of the annual income.
    - **dti**: Debt-to-income ratio.
    - **fico**: FICO credit score.
    - **days.with.cr.line**: Number of days with credit line open.
    - **revol.bal**: Revolving balance (amount unpaid on credit card).
    - **revol.util**: Revolving line utilization rate.
    - **inq.last.6mths**: Number of inquiries in the last 6 months.
    - **delinq.2yrs**: Number of times delinquent in the past 2 years.
    - **pub.rec**: Number of derogatory public records.
    """)

# Load models
model_files = {
    "Logistic Regression": "logistic_regression_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "LightGBM": "lightgbm_model.joblib",
    "XGBoost": "xgboost_model.joblib"
}
models = {name: joblib.load(path) for name, path in model_files.items()}

# Purpose mapping
purpose_mapping = {
    'credit_card': 0, 'debt_consolidation': 1, 'educational': 2,
    'home_improvement': 3, 'major_purchase': 4, 'small_business': 5,
    'all_other': 6
}

# Sidebar input form
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

# Prepare input DataFrame
input_df = pd.DataFrame([user_input])
input_df['purpose'] = input_df['purpose'].map(purpose_mapping)

# Model selection
st.sidebar.markdown("---")
selected_model_name = st.sidebar.selectbox("Select Model", list(model_files.keys()))
selected_model = models[selected_model_name]

# Predict
if st.sidebar.button("Predict"):
    prediction = selected_model.predict(input_df)[0]
    proba = selected_model.predict_proba(input_df)[0][1]

    # Show prediction
    st.subheader("🎯 Prediction Result")
    st.markdown(f"**Prediction:** {'Approved ✅' if prediction == 1 else 'Rejected ❌'}")
    st.markdown(f"**Probability of Approval:** {round(proba * 100, 2)}%")

    # Gauge meter using plotly
    import plotly.graph_objects as go
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

    # SHAP explanation
    st.subheader("📊 SHAP Explanation")
    try:
        explainer = shap.Explainer(selected_model)
        shap_values = explainer(input_df)

        st.markdown("**Top Features Impacting the Decision:**")
        fig_shap_bar = shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig_shap_bar.figure)

        with st.expander("See SHAP Waterfall Explanation"):
            fig_waterfall = shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig_waterfall.figure)
    except Exception as e:
        st.warning("SHAP explanation not available for this model.")

# Insights section
st.subheader("📈 Insights")
st.markdown("""
- Interest rate and FICO score are strong indicators of loan approval.
- High revolving balance or utilization may negatively affect approval.
- Fewer recent inquiries and delinquencies improve approval chances.
""")

# Chatbot section
st.subheader("💬 Ask Our Chatbot")
@st.cache_resource
def get_chatbot():
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", 
                    token=st.secrets["HF_TOKEN"])

try:
    chatbot = get_chatbot()
    chat_input = st.text_input("Ask something about loan prediction or finance:")
    if chat_input:
        result = chatbot(chat_input, max_new_tokens=100)[0]["generated_text"]
        st.markdown(f"**Chatbot:** {result}")
except Exception as e:
    st.warning("Chatbot is currently unavailable. Please check your HuggingFace token or internet connection.")

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ by Team Numerixa</center>", unsafe_allow_html=True)
