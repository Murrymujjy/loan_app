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
st.markdown("An intelligent system to predict loan approval using Random Forest and LightGBM models.")

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

# ----------------- LOAD MODELS (ONLY RF + LGBM) -----------------
model_files = {
    "Random Forest": "random_forest_model.joblib",
    "LightGBM": "lightgbm_model.joblib"
}

models = {}
for name, path in model_files.items():
    try:
        models[name] = joblib.load(path)
    except Exception as e:
        st.warning(f"⚠️ Could not load model file for: {name} (expected: {path}). Error: {e}")

if not models:
    st.error("No models loaded. Put 'random_forest_model.joblib' and 'lightgbm_model.joblib' in the app folder and restart.")
    st.stop()

# Purpose mapping (you said you've been mapping already)
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
    "int.rate": st.sidebar.slider("Interest Rate", 0.0, 0.5, 0.12, step=0.001),
    "installment": st.sidebar.slider("Installment", 0.0, 5000.0, 250.0, step=1.0),
    "log.annual.inc": st.sidebar.slider("Log Annual Income", 0.0, 15.0, 10.0, step=0.01),
    "dti": st.sidebar.slider("Debt-to-Income Ratio", 0.0, 100.0, 18.0, step=0.1),
    "fico": st.sidebar.slider("FICO Score", 300, 850, 700, step=1),
    "days.with.cr.line": st.sidebar.slider("Days with Credit Line", 0, 20000, 4000, step=1),
    "revol.bal": st.sidebar.slider("Revolving Balance", 0, 1000000, 15000, step=1),
    "revol.util": st.sidebar.slider("Revolving Utilization (%)", 0.0, 100.0, 45.0, step=0.1),
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
        # handle models without predict_proba gracefully
        if hasattr(selected_model, "predict_proba"):
            proba = selected_model.predict_proba(input_df)[0][1]
        else:
            proba = None
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    st.subheader("🎯 Prediction Result")
    st.markdown(f"**Loan Decision:** {'Approved ✅' if prediction == 1 else 'Rejected ❌'}")
    if proba is not None:
        st.markdown(f"**Probability of Loan Approval:** {round(proba * 100, 2)}%")
    else:
        st.info("Model does not expose probability scores.")

    # Gauge Chart (show only if probability available)
    if proba is not None:
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

    # ----------------- EXPLANATIONS -----------------
    try:
        model_type = type(selected_model).__name__.lower()

        # ---- LightGBM: SHAP TreeExplainer (mean-abs bar) ----
        if "lightgbm" in model_type or "lgbm" in model_type:
            try:
                explainer = shap.TreeExplainer(selected_model)
                # TreeExplainer can accept DataFrame or numpy depending on model; try DataFrame first
                shap_values = explainer.shap_values(input_df)
                # shap_values may be list (binary)
                if isinstance(shap_values, list) and len(shap_values) >= 2:
                    shap_vals_pos = shap_values[1]
                else:
                    shap_vals_pos = shap_values

                # feature names try model first then input_df
                if hasattr(selected_model, "feature_names_in_"):
                    feature_names = np.array(selected_model.feature_names_in_)
                else:
                    feature_names = np.array(input_df.columns)

                feature_importance = np.abs(shap_vals_pos).mean(axis=0)
                order = np.argsort(feature_importance)[::-1]
                feature_importance = feature_importance[order]
                feature_names_ordered = feature_names[order]

                # Plot horizontal SHAP mean-abs bar
                fig, ax = plt.subplots(figsize=(8, max(3, len(feature_names_ordered) * 0.35)))
                ax.barh(feature_names_ordered, feature_importance, color="skyblue")
                ax.set_xlabel("Mean |SHAP value|")
                ax.set_title("Feature importance (SHAP — mean absolute) — LightGBM")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()
            except Exception as e_shap:
                st.warning(f"SHAP explanation not available for LightGBM. Error: {e_shap}")

        # ---- Random Forest: feature_importances_ bar (top N) ----
        elif "randomforest" in model_type or "forest" in model_type:
            try:
                if not hasattr(selected_model, "feature_importances_"):
                    raise AttributeError("Model has no attribute 'feature_importances_'")

                importances = np.array(selected_model.feature_importances_)
                # get feature names from model if present else from input_df (best-effort)
                if hasattr(selected_model, "feature_names_in_"):
                    feature_names = np.array(selected_model.feature_names_in_)
                else:
                    feature_names = np.array(input_df.columns)

                # if more features, show top N
                TOP_N = min(10, len(feature_names))
                order = np.argsort(importances)[::-1][:TOP_N]
                importances_sorted = importances[order]
                feature_names_sorted = feature_names[order]

                fig, ax = plt.subplots(figsize=(8, max(3, len(feature_names_sorted) * 0.35)))
                ax.barh(feature_names_sorted, importances_sorted, color="steelblue")
                ax.set_xlabel("Feature importance")
                ax.set_title(f"Feature importance (Random Forest) — top {len(feature_names_sorted)}")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()
            except Exception as e_rf:
                st.warning(f"Random Forest explanation not available. Error: {e_rf}")

        else:
            st.info("Explanations available for LightGBM (SHAP) and Random Forest (feature_importances_).")
    except Exception as e:
        st.warning(f"Explanation block failed: {e}")

# ----------------- CHATBOT (Single-turn + Multi-turn tabs) -----------------
# ----------------- CHATBOT (Zephyr / LLaMA / Falcon) -----------------
from huggingface_hub import InferenceClient

CHAT_MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",   # preferred
    "tiiuae/falcon-7b-instruct",
    "meta-llama/Llama-2-7b-chat-hf"
]

def get_available_model():
    """Return first available model from CHAT_MODELS"""
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN", None)
    if not hf_token:
        st.error("❌ Missing HUGGINGFACE_TOKEN in Streamlit Secrets!")
        st.stop()
        
        from huggingface_hub import InferenceClient

# Updated preferred models list (free / open ones included)
PREFERRED_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",   # good free instruct model
    "google/gemma-2b-it",                   # smaller, free, works with text generation
    "HuggingFaceH4/zephyr-7b-beta",         # keep it, but may need PRO
    "tiiuae/falcon-7b-instruct",  
    "meta-llama/Llama-2-7b-chat-hf"
]

def load_chat_model():
    for model_id in PREFERRED_MODELS:
        try:
            print(f"🔄 Trying {model_id}...")
            client = InferenceClient(model=model_id, token=st.secrets["HF_TOKEN"])

            # Test with a small prompt
            test = client.text_generation("Hello! This is a quick test.", max_new_tokens=10)
            print(f"✅ Model {model_id} is available")
            return client, model_id
        except Exception as e:
            print(f"⚠️ Model {model_id} failed: {e}")
            continue
    return None, None

chat_client, active_model = load_chat_model()

if chat_client is None:
    st.error("⚠️ No available HF chat models. Please check your Hugging Face token or model access.")
else:
    st.success(f"💬 Loan Advisor Chatbot is running on: {active_model}")

if "hf_chat_model" not in st.session_state:
    st.session_state.hf_chat_model = get_available_model()

st.markdown("---")
st.header("💬 Loan Advisor Chatbot")

if not st.session_state.hf_chat_model:
    st.error("⚠️ No available HF chat models. Please check your Hugging Face token or model access.")
else:
    model_name = st.session_state.hf_chat_model
    client = InferenceClient(model=model_name, token=st.secrets["HUGGINGFACE_TOKEN"])
    st.info(f"✅ Using chat model: **{model_name}**")

    tab1, tab2 = st.tabs(["💬 Single-Turn Chat", "🗨️ Multi-Turn Chat"])

    # ---------- Single-turn ----------
    with tab1:
        st.markdown("### Quick question (single-turn)")
        single_q = st.text_input("Ask a one-off question:", key="single_q")
        if st.button("Ask", key="single_ask"):
            if single_q.strip():
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": f"You are a helpful loan advisor. {single_q}"}],
                        max_tokens=300
                    )
                    bot_reply = response.choices[0].message["content"]
                    st.markdown(f"**Bot:** {bot_reply}")
                except Exception as e:
                    st.error(f"Chat error: {type(e).__name__}: {e}")

    # ---------- Multi-turn ----------
    with tab2:
        st.markdown("### Conversation (multi-turn, session only)")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**Bot:** {msg['content']}")

        multi_input = st.text_input("You:", key="multi_input")
        if st.button("Send", key="send_multi"):
            if multi_input.strip():
                st.session_state.chat_history.append({"role": "user", "content": multi_input})
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=st.session_state.chat_history,
                        max_tokens=300
                    )
                    bot_reply = response.choices[0].message["content"]
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                    st.rerun()
                except Exception as e:
                    st.error(f"Chat error: {type(e).__name__}: {e}")

st.markdown("---")
st.markdown("<center>Made with ❤️ by Team Numerixa</center>", unsafe_allow_html=True)

