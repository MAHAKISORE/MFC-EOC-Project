import streamlit as st
import sqlite3
import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Page Config
st.set_page_config(page_title="ML Model Selector", layout="centered")
st.title("🔍 Machine Learning Model Selector")

# Load data from SQLite
@st.cache_data
def load_data():
    conn = sqlite3.connect("database/data.db")
    df = pd.read_sql_query("SELECT * FROM sample_data", conn)
    conn.close()
    return df

df = load_data()
st.subheader("📊 Preview of Dataset")
st.dataframe(df)

# Model selection
model_choice = st.selectbox("Choose a Machine Learning Model", ["Random Forest", "XGBoost", "MLP"])

# Load model
def load_model(name):
    filename = f"models/{name}.pkl"
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None

model_map = {
    "Random Forest": "RandomForest",
    "XGBoost": "XGBoost",
    "MLP": "MLP"
}

model = load_model(model_map[model_choice])

# Predict and visualize
if st.button("Predict"):
    if model:
        X = df.drop("target", axis=1)
        y_true = df["target"]
        y_pred = model.predict(X)

        result_df = df.copy()
        result_df["Prediction"] = y_pred

        st.success(f"✅ Predictions using {model_choice}")
        st.dataframe(result_df)

        # Confusion Matrix
        st.subheader("📌 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # Prediction Distribution
        st.subheader("📈 Prediction Distribution")
        pred_count = result_df["Prediction"].value_counts()
        st.bar_chart(pred_count)

        # Classification Report
        st.subheader("📊 Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Download Predictions
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Predictions as CSV",
            data=csv,
            file_name=f"{model_map[model_choice]}_predictions.csv",
            mime="text/csv"
        )
    else:
        st.error("Model not found or not trained yet!")
