import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load saved models
logreg = joblib.load("models/logistic_regression_model.joblib")
dt = joblib.load("models/decision_tree_model.joblib")
rf = joblib.load("models/random_forest_model.joblib")
lgbm = joblib.load("models/lightgbm_model.joblib")
xgb = joblib.load("models/xgboost_model.joblib")

models = {
    "Logistic Regression": logreg,
    "Decision Tree": dt,
    "Random Forest": rf,
    "LightGBM": lgbm,
    "XGBoost": xgb
}

# Define top 10 features (replace with your actual feature list)
top_features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
                'feature6', 'feature7', 'feature8', 'feature9', 'feature10']

# App Title
st.title("📊 Loan Default Prediction Dashboard")

# Sidebar Navigation
section = st.sidebar.radio("Go to", ["Bulk Prediction", "Single Prediction"])

# Bulk Prediction Section
if section == "Bulk Prediction":
    st.subheader("📂 Bulk File Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file with 10 features", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview:", data.head())

        model_name = st.selectbox("Choose Model", list(models.keys()))
        if st.button("Predict"):
            model = models[model_name]
            preds = model.predict(data[top_features])
            data['Prediction'] = preds
            st.success("✅ Prediction Completed")
            st.write(data)
            csv = data.to_csv(index=False).encode()
            st.download_button("Download Results", csv, file_name="predictions.csv")

# Single Prediction (Embedded system-like)
elif section == "Single Prediction":
    st.subheader("🧠 Embedded System Prediction")

    st.markdown("Fill in the input values for a single prediction:")

    # Input form
    with st.form("input_form"):
        inputs = {}
        for feat in top_features:
            inputs[feat] = st.number_input(f"{feat}", value=0.0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([inputs])
        model_name = st.selectbox("Choose Model", list(models.keys()))
        model = models[model_name]

        # Prediction & Probability
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("### 🔍 Prediction Output")
        st.write(f"**Predicted Class:** {'Default' if prediction == 1 else 'Paid'}")
        st.write(f"**Probability of Default:** {probability:.2%}")

        # Progress bar like an embedded gauge
        st.progress(min(int(probability * 100), 100))

        # Insights Section
        st.markdown("---")
        st.markdown("### 📈 Prediction Insights")

        if probability > 0.5:
            st.warning("This customer has a high likelihood of not fully paying the loan.")
        else:
            st.success("This customer is likely to fully repay the loan.")

        st.markdown("Customize this section with SHAP plots or explanations in the next step.")

