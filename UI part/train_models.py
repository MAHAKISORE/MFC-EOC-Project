import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Make sure the models folder exists
os.makedirs("models", exist_ok=True)

# Connect to the database and load the data
conn = sqlite3.connect("database/data.db")
df = pd.read_sql_query("SELECT * FROM sample_data", conn)
conn.close()

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
joblib.dump(rf, "models/RandomForest.pkl")

# Train and save XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
joblib.dump(xgb, "models/XGBoost.pkl")

# Train and save MLP
mlp = MLPClassifier(max_iter=500)
mlp.fit(X_train, y_train)
joblib.dump(mlp, "models/MLP.pkl")

print("✅ Models trained and saved to the 'models/' folder.")
