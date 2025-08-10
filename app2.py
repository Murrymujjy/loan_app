import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
from huggingface_hub import InferenceClient
import traceback
import matplotlib.pyplot as plt
import numpy as np

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
    st.markdown(f"**Loan Decision:** {'Approved ✅' if prediction == 1 else 'Rejected ❌'}")
    st.markdown(f"**Probability of Loan Approval:** {round(proba * 100, 2)}%")

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

            # ----------------- SHAP Explanation -----------------
    try:
        model_type = type(selected_model).__name__.lower()

        if "lightgbm" in model_type or "lgbm" in model_type or "xgb" in model_type:
            explainer = shap.TreeExplainer(selected_model)
            shap_values = explainer.shap_values(input_df)

            # If binary classification, select class 1
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Calculate mean absolute SHAP values for each feature
            feature_importance = np.abs(shap_values).mean(axis=0)
            feature_names = input_df.columns

            # Sort features by importance
            sorted_idx = np.argsort(feature_importance)[::-1]
            feature_names = feature_names[sorted_idx]
            feature_importance = feature_importance[sorted_idx]

            # Plot bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(feature_names, feature_importance, color="skyblue")
            ax.set_xlabel("Mean Absolute SHAP Value")
            ax.set_ylabel("Feature")
            ax.set_title("Feature Importance (SHAP)")
            ax.invert_yaxis()
            st.pyplot(fig)

        else:
            st.info("SHAP explanation available only for LightGBM and XGBoost in this app.")

    except Exception as e:
        st.warning(f"SHAP explanation not available. Error: {e}")

# ----------------- CHATBOT -----------------
import streamlit as st
from huggingface_hub import InferenceClient

# Set your HF API key in Streamlit secrets or env variable
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
client = InferenceClient(token=HF_API_TOKEN)

# Preferred models list (will try in this order)
PREFERRED_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-2-13b-chat-hf",
    "tiiuae/falcon-7b-instruct"
]

def get_available_model():
    """Check which preferred model works for chat completion."""
    for model in PREFERRED_MODELS:
        try:
            client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5
            )
            return model
        except:
            continue
    return None

# Cache working model
if "selected_model" not in st.session_state:
    st.session_state.selected_model = get_available_model()

if not st.session_state.selected_model:
    st.error("❌ No available chat models found from the preferred list.")
else:
    model_name = st.session_state.selected_model
    st.info(f"✅ Using model: **{model_name}**")

    # Tabs for single-turn and multi-turn
    tab1, tab2 = st.tabs(["💬 Single-Turn Chat", "🗨️ Multi-Turn Chat"])

    with tab1:
        st.markdown("### Single-Turn Loan Advisor")
        user_input = st.text_input("Ask me anything about loans, eligibility, or financial planning:")

        if user_input:
            try:
                completion = client.chat_completion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful financial loan advisor. Give detailed, practical advice with clear steps."},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                bot_reply = completion.choices[0].message["content"]
                st.markdown(f"**Bot:** {bot_reply}")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

    with tab2:
        st.markdown("### Multi-Turn Loan Advisor")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "system", "content": "You are a helpful financial loan advisor. Give detailed, practical advice with clear steps."}
            ]

        user_message = st.text_input("You:", key="multi_input")

        if st.button("Send", key="send_multi"):
            if user_message.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_message})
                try:
                    completion = client.chat_completion(
                        model=model_name,
                        messages=st.session_state.chat_history,
                        max_tokens=300,
                        temperature=0.7
                    )
                    bot_reply = completion.choices[0].message["content"]
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**Bot:** {msg['content']}")


# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ by Team Numerixa</center>", unsafe_allow_html=True)
