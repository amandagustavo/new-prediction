import joblib
import pandas as pd

# Load model
model = joblib.load('Logistic_Regression.joblib')

# Print model features
if hasattr(model, 'feature_names_in_'):
    print("Model features:")
    features = model.feature_names_in_.tolist()
    for i, feature in enumerate(features, 1):
        print(f"{i}. {feature}")
else:
    print("Model does not have feature names stored") 