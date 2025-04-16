# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Streamlit Page Config
st.set_page_config(page_title="CICFlowMeter ML Tester", layout="wide")

st.title("📊 CICFlowMeter ML Model Tester")

# Load dataset
file_path = "CICFlowMeter_Training_Balanced_added.csv"
data = pd.read_csv(file_path)

# Drop unnecessary columns
data = data.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], errors='ignore')

st.subheader("🔍 Dataset Preview")
st.dataframe(data.head())

# One-hot encoding for categorical features
X = pd.get_dummies(data.iloc[:, :-1])

# Replace inf and NaN values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Labels
y = data.iloc[:, -1]

# Load label encoder
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
y_encoded = label_encoder.transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model selection
model_choice = st.selectbox("🤖 Choose Model", ("XGBoost", "CatBoost", "Decision Tree", "Random Forest"))

if st.button("🚀 Predict on Test Data"):
    if model_choice == "XGBoost":
        model = pickle.load(open("models/xgboost_model.pkl", "rb"))
    elif model_choice == "CatBoost":
        model = pickle.load(open("models/catboost_model.pkl", "rb"))
    elif model_choice == "Decision Tree":
        model = pickle.load(open("models/decision_tree_model.pkl", "rb"))
    elif model_choice == "Random Forest":
        model = pickle.load(open("models/random_forest_model.pkl", "rb"))

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ {model_choice} Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    st.subheader("📑 Classification Report")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.subheader("📊 Confusion Matrix")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, ax=ax)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)

    # Feature Importance (if applicable)
    if hasattr(model, "feature_importances_"):
        st.subheader("📈 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(20)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax2)
        plt.title("Top 20 Important Features")
        st.pyplot(fig2)
    else:
        st.info(f"{model_choice} does not support feature importance.")

