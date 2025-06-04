import joblib
import pandas as pd

# Load model
model = joblib.load('Logistic_Regression.joblib')

# Print feature names if available
if hasattr(model, 'feature_names_in_'):
    print("Feature names in model:")
    print("\n".join(model.feature_names_in_))
else:
    # Try to get feature names from a sample prediction
    # Load a sample from your dataset
    df = pd.read_csv('EmAt.csv')
    sample = df.iloc[0:1]
    
    # Process the sample similar to your training process
    # Drop unnecessary columns
    sample = sample.drop(['Employee ID', 'Attrition'], axis=1)
    
    # Get dummy variables
    categorical_cols = ['Gender', 'Job Role', 'Work-Life Balance', 'Job Satisfaction', 
                       'Performance Rating', 'Overtime', 'Education Level', 'Marital Status',
                       'Company Size', 'Remote Work', 'Leadership Opportunities', 
                       'Innovation Opportunities', 'Company Reputation', 'Employee Recognition']
    sample = pd.get_dummies(sample, columns=categorical_cols)
    
    print("Features expected by model:")
    print("\n".join(sample.columns)) 