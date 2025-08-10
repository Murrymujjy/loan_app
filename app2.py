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

# Styling
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
    except Exception:
        st.warning(f"⚠️ Could not load model file for: {name} (expected: {path})")

if not models:
    st.error("No models loaded. Put your .joblib files in the app folder and restart.")
    st.stop()

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
    "int.rate": st.sidebar.slider("Interest Rate", 0.0, 0.5, 0.12),
    "installment": st.sidebar.slider("Installment", 0.0, 5000.0, 250.0),
    "log.annual.inc": st.sidebar.slider("Log Annual Income", 0.0, 15.0, 10.0),
    "dti": st.sidebar.slider("Debt-to-Income Ratio", 0.0, 100.0, 18.0),
    "fico": st.sidebar.slider("FICO Score", 300, 850, 700),
    "days.with.cr.line": st.sidebar.slider("Days with Credit Line", 0, 20000, 4000),
    "revol.bal": st.sidebar.slider("Revolving Balance", 0, 1000000, 15000),
    "revol.util": st.sidebar.slider("Revolving Utilization (%)", 0.0, 100.0, 45.0),
    "inq.last.6mths": st.sidebar.slider("Inquiries Last 6 Months", 0, 50, 1),
    "delinq.2yrs": st.sidebar.slider("Delinquencies Last 2 Years", 0, 20, 0),
    "pub.rec": st.sidebar.slider("Public Records", 0, 20, 0)
}

input_df = pd.DataFrame([user_input])
input_df['purpose'] = input_df['purpose'].map(purpose_mapping)

# ----------------- PREDICTION -----------------
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

if st.sidebar.button("Predict"):
    try:
        prediction = selected_model.predict(input_df)[0]
        proba = selected_model.predict_proba(input_df)[0][1]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

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

    # ----------------- SHAP Explanation (LightGBM & XGBoost only) -----------------
    try:
        model_type = type(selected_model).__name__.lower()

        if ("lightgbm" in model_type) or ("lgbm" in model_type) or ("xgb" in model_type) or ("xgboost" in model_type):
            # Use TreeExplainer for tree models
            explainer = shap.TreeExplainer(selected_model)
            # depending on model, shap_values may be list (binary)
            shap_values = explainer.shap_values(input_df if ("xgb" in model_type or "xgboost" in model_type) else input_df.to_numpy())

            if isinstance(shap_values, list) and len(shap_values) >= 2:
                shap_vals_pos = shap_values[1]
            else:
                shap_vals_pos = shap_values

            # compute mean absolute shap for each feature
            feature_importance = np.abs(shap_vals_pos).mean(axis=0)
            feature_names = np.array(input_df.columns)

            # sort desc
            order = np.argsort(feature_importance)[::-1]
            feature_importance = feature_importance[order]
            feature_names = feature_names[order]

            # plot horizontal bar chart
            fig, ax = plt.subplots(figsize=(8, max(3, len(feature_names)*0.4)))
            ax.barh(feature_names, feature_importance, color="skyblue")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Feature importance (SHAP — mean absolute)")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
        else:
            st.info("SHAP explanation is available only for LightGBM and XGBoost in this app. Select one of those models for SHAP visuals.")
    except Exception as e:
        st.warning(f"SHAP explanation not available. Error: {e}")

# ----------------- CHATBOT (Single-turn + Multi-turn tabs) -----------------
# Hugging Face Inference client
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", None)
if HF_API_TOKEN is None:
    st.error("❌ Missing HF_API_TOKEN in Streamlit Secrets! Add it (key name: HF_API_TOKEN).")
    st.stop()

client = InferenceClient(token=HF_API_TOKEN)

# Preferred models (tries in order)
PREFERRED_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-2-13b-chat-hf",
    "tiiuae/falcon-7b-instruct"
]

def get_available_model():
    """Return first model from preferred list that supports chat_completion for this token."""
    for model in PREFERRED_MODELS:
        try:
            # small test call, short tokens
            client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5
            )
            return model
        except Exception:
            continue
    return None

# cache the working model in session
if "hf_chat_model" not in st.session_state:
    st.session_state.hf_chat_model = get_available_model()

st.markdown("---")
st.header("💬 Loan Advisor Chatbot")

if not st.session_state.hf_chat_model:
    st.error("No available HF chat models from the preferred list. Check token / model access.")
else:
    model_name = st.session_state.hf_chat_model
    st.info(f"Using chat model: **{model_name}**")

    tab1, tab2 = st.tabs(["💬 Single-Turn Chat", "🗨️ Multi-Turn Chat"])

    # ---------- Single-turn ----------
    with tab1:
        st.markdown("### Quick question (single-turn)")
        single_q = st.text_input("Ask a one-off question:", key="single_q")
        if st.button("Ask", key="single_ask"):
            if single_q.strip():
                try:
                    completion = client.chat_completion(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a friendly, detailed loan advisor. Explain steps clearly."},
                            {"role": "user", "content": single_q}
                        ],
                        max_tokens=300,
                        temperature=0.7
                    )
                    # safe access
                    bot_reply = completion.choices[0].message["content"] if hasattr(completion, "choices") and completion.choices else "No response."
                    st.markdown(f"**Bot:** {bot_reply}")
                except Exception as e:
                    st.error(f"Chat error: {type(e).__name__}: {e}")

    # ---------- Multi-turn ----------
    with tab2:
        st.markdown("### Conversation (multi-turn, session only)")

        # initialize conversation history with a system prompt
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "system", "content": "You are a helpful financial loan advisor. Give detailed, practical advice with clear steps."}
            ]

        # Display history excluding system message
        for msg in st.session_state.chat_history[1:]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**Bot:** {msg['content']}")

        # Input and send
        multi_input = st.text_input("You:", key="multi_input")
        if st.button("Send", key="send_multi"):
            if multi_input.strip():
                # append user message
                st.session_state.chat_history.append({"role": "user", "content": multi_input})
                try:
                    completion = client.chat_completion(
                        model=model_name,
                        messages=st.session_state.chat_history,
                        max_tokens=300,
                        temperature=0.7
                    )
                    bot_reply = completion.choices[0].message["content"] if hasattr(completion, "choices") and completion.choices else "No response."
                    # append assistant reply
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                    # rerender to show new messages (input will persist unless cleared explicitly)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Chat error: {type(e).__name__}: {e}")

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ by Team Numerixa</center>", unsafe_allow_html=True)
