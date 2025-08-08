import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Load trained models
logreg = joblib.load("logistic_regression_model.joblib")
dt = joblib.load("decision_tree_model.joblib")
rf = joblib.load("random_forest_model.joblib")
lgbm = joblib.load("lightgbm_model.joblib")
xgb = joblib.load("xgboost_model.joblib")

models = {
    "Logistic Regression": logreg,
    "Decision Tree": dt,
    "Random Forest": rf,
    "LightGBM": lgbm,
    "XGBoost": xgb
}

# App Title
st.title("📈 Credit Risk Prediction Dashboard")

# Sidebar Navigation
section = st.sidebar.radio("Select Mode", ["📊 Batch Predictions", "🤖 Embedded Single Prediction"])

# 1. 📊 BATCH PREDICTION
if section == "📊 Batch Predictions":
    st.header("Upload Test Dataset for Evaluation")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        if 'not.fully.paid' not in data.columns:
            st.error("Uploaded data must include 'not.fully.paid' as the target.")
        else:
            y_test = data['not.fully.paid']
            X_test = data.drop(['not.fully.paid'], axis=1)

            st.subheader("Model Evaluation Metrics")
            for name, model in models.items():
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.markdown(f"### 🔹 {name}")
                st.write(f"**Accuracy:** {acc:.3f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(ax=ax, cmap="Blues")
                plt.title(f"Confusion Matrix - {name}")
                st.pyplot(fig)

# 2. 🤖 EMBEDDED SINGLE PREDICTION
else:
    st.header("Predict Credit Risk for One Applicant")

    # Dummy feature list (replace with your real feature names & ranges)
    input_features = [
        "fico", "int.rate", "installment", "log.annual.inc", "dti",
        "revol.bal", "revol.util", "inq.last.6mths", "delinq.2yrs", "pub.rec"
    ]

    # Form for user input
    with st.form("prediction_form"):
        st.subheader("Applicant Features")
        inputs = []
        for feature in input_features:
            val = st.number_input(f"{feature}", format="%.4f", step=0.01)
            inputs.append(val)

        model_choice = st.selectbox("Select Prediction Model", list(models.keys()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        X_new = pd.DataFrame([inputs], columns=input_features)
        model = models[model_choice]
        prediction = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0][1] if hasattr(model, "predict_proba") else None

        st.markdown(f"## ✅ Prediction: {'Not Fully Paid (1)' if prediction == 1 else 'Fully Paid (0)'}")
        if prob is not None:
            st.write(f"Probability of default: **{prob:.2%}**")

        st.success("Prediction completed.")

