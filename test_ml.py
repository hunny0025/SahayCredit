import os
import lightgbm as lgb
import pandas as pd
import shap
import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
fraud_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "ml", "fraud_service", "fraud_model_ieee.txt"))
print("Model features:", fraud_model.feature_name())

# Test predict with mock features
features = {
    "TransactionAmt": 200000.0,
    "card1": 8500.0,
    "card2": 410.0,
    "addr1": 246.0,
    "dist1": 33.0,
    "C1": 8.0,
    "C2": 10.0,
    "C13": 16.0,
    "C14": 8.0,
    "D1": 0.0,
    "D15": 6.0
}

# The model might expect more features, let's see how many features it expects
# LightGBM booster will align features if they are passed as a pandas DataFrame matching model features.
# Let's check how main.py handles it:
# X = pd.DataFrame([txn.features])
# prob = float(fraud_model.predict(X)[0])
# Let's try to run this locally:
X = pd.DataFrame([features])
# Align features: fill missing ones with NaN if needed
for col in fraud_model.feature_name():
    if col not in X.columns:
        X[col] = np.nan
X = X[fraud_model.feature_name()]

print("Aligned columns:", X.columns.tolist()[:15])
try:
    prob = float(fraud_model.predict(X)[0])
    print("Predicted probability:", prob)
except Exception as e:
    print("Prediction failed:", e)

# Test SHAP explainer
try:
    explainer = shap.TreeExplainer(fraud_model)
    shap_values = explainer.shap_values(X)
    print("SHAP values calculated successfully. Shape:", np.shape(shap_values))
except Exception as e:
    print("SHAP failed:", e)
