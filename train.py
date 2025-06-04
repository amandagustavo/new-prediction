import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import os
import json

# Tạo thư mục models nếu chưa tồn tại
if not os.path.exists('models'):
    os.makedirs('models')

# Load data
print("Loading data...")
df = pd.read_csv('EmAt.csv')

# In ra tên các cột để kiểm tra
print("\nColumns in dataset:")
print(df.columns.tolist())

# Prepare features and target
X = df.drop(['Employee ID', 'Attrition'], axis=1)
y = (df['Attrition'] == 'Left').astype(int)

# Split categorical and numerical columns
numerical_cols = ['Age', 'Years at Company', 'Monthly Income', 'Number of Promotions',
                 'Distance from Home', 'Number of Dependents']
categorical_cols = ['Gender', 'Job Role', 'Work-Life Balance', 'Job Satisfaction', 
                   'Performance Rating', 'Overtime', 'Education Level', 'Marital Status',
                   'Job Level', 'Company Size', 'Remote Work', 'Leadership Opportunities', 
                   'Innovation Opportunities', 'Company Reputation', 'Employee Recognition']

print("\nNumerical columns:", numerical_cols)
print("\nCategorical columns:", categorical_cols)

# Create dummy variables for categorical columns
X = pd.get_dummies(X, columns=categorical_cols)

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Lưu danh sách các cột để sử dụng khi dự đoán
feature_columns = X.columns.tolist()
with open(os.path.join('models', 'feature_columns.json'), 'w') as f:
    json.dump(feature_columns, f)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining models...")

# 1. Logistic Regression với regularization để tránh overfitting
lr = LogisticRegression(
    random_state=42,
    max_iter=1000,
    C=0.1,  # Tăng regularization
    class_weight='balanced'  # Cân bằng classes
)
lr.fit(X_train, y_train)
joblib.dump(lr, os.path.join('models', 'Logistic_Regression.joblib'))
print("Logistic Regression accuracy:", lr.score(X_test, y_test))

# 2. Decision Tree với các tham số chống overfitting
dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,  # Giới hạn độ sâu
    min_samples_split=5,  # Yêu cầu tối thiểu mẫu để split
    min_samples_leaf=5,  # Yêu cầu tối thiểu mẫu ở leaf
    class_weight='balanced'  # Cân bằng classes
)
dt.fit(X_train, y_train)
joblib.dump(dt, os.path.join('models', 'Decision_Tree.joblib'))
print("Decision Tree accuracy:", dt.score(X_test, y_test))

# 3. XGBoost với các tham số chống overfitting
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_estimators=100,  # Số cây vừa phải
    max_depth=4,  # Giới hạn độ sâu
    learning_rate=0.1,  # Learning rate nhỏ
    subsample=0.8,  # Sử dụng 80% mẫu cho mỗi cây
    colsample_bytree=0.8,  # Sử dụng 80% features cho mỗi cây
    min_child_weight=3,  # Kiểm soát overfitting
    gamma=1,  # Minimum loss reduction
    scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)  # Cân bằng classes
)
xgb_model.fit(X_train, y_train)
# Lưu mô hình XGBoost bằng save_model
xgb_model.save_model(os.path.join('models', 'XGBoost.json'))
print("XGBoost accuracy:", xgb_model.score(X_test, y_test))

# 4. CatBoost với các tham số chống overfitting
cb = CatBoostClassifier(
    random_state=42,
    iterations=100,  # Số vòng lặp vừa phải
    depth=4,  # Giới hạn độ sâu
    learning_rate=0.1,  # Learning rate nhỏ
    l2_leaf_reg=3,  # L2 regularization
    subsample=0.8,  # Sử dụng 80% mẫu cho mỗi cây
    colsample_bylevel=0.8,  # Sử dụng 80% features cho mỗi level
    min_data_in_leaf=5,  # Số mẫu tối thiểu trong leaf
    class_weights={0: 1, 1: sum(y_train == 0) / sum(y_train == 1)},  # Cân bằng classes
    verbose=0  # Tắt output
)
cb.fit(X_train, y_train)
# Lưu mô hình CatBoost bằng save_model
cb.save_model(os.path.join('models', 'CatBoost.model'))
print("CatBoost accuracy:", cb.score(X_test, y_test))

# Lưu scaler
joblib.dump(scaler, os.path.join('models', 'scaler.joblib'))

print("\nTraining completed and models saved in 'models' directory!") 