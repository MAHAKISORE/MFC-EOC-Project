# train_models.py
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "CICFlowMeter_Training_Balanced_added.csv"
data = pd.read_csv(file_path)

# Drop non-numeric columns
data = data.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], errors='ignore')

# Features & labels
X = pd.get_dummies(data.iloc[:, :-1])  # one-hot encode categorical features
y = data.iloc[:, -1]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# XGBoost
xgb_model = xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
pickle.dump(xgb_model, open("models/xgboost_model.pkl", "wb"))

# CatBoost
cat_model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=5, eval_metric='MultiClass', random_seed=42, verbose=0)
cat_model.fit(X_train, y_train)
pickle.dump(cat_model, open("models/catboost_model.pkl", "wb"))

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
pickle.dump(dt_model, open("models/decision_tree_model.pkl", "wb"))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
pickle.dump(rf_model, open("models/random_forest_model.pkl", "wb"))

# Save Label Encoder
pickle.dump(label_encoder, open("models/label_encoder.pkl", "wb"))

print("All models trained and saved successfully.")
