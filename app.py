from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import traceback
import json
from datetime import datetime
import os
import joblib
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
import sys
from database import save_prediction, save_feedback, get_all_feedback

# Cấu hình logging chi tiết hơn
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Kiểm tra xem thư mục models có tồn tại không
if not os.path.exists('models'):
    logger.error("Directory 'models' not found!")
    raise FileNotFoundError("Directory 'models' not found!")

# Load các models
try:
    logger.info('Loading models...')
    # Load danh sách các cột
    feature_columns_path = os.path.join('models', 'feature_columns.json')
    if not os.path.exists(feature_columns_path):
        logger.error(f"Feature columns file not found at {feature_columns_path}")
        raise FileNotFoundError(f"Feature columns file not found at {feature_columns_path}")
        
    with open(feature_columns_path, 'r') as f:
        feature_columns = json.load(f)
        logger.info(f"Loaded {len(feature_columns)} feature columns")
    
    # Load các mô hình
    lr_path = os.path.join('models', 'Logistic_Regression.joblib')
    dt_path = os.path.join('models', 'Decision_Tree.joblib')
    xgb_path = os.path.join('models', 'XGBoost.json')
    cb_path = os.path.join('models', 'CatBoost.model')
    scaler_path = os.path.join('models', 'scaler.joblib')
    
    # Kiểm tra sự tồn tại của các file
    for path in [lr_path, dt_path, xgb_path, cb_path, scaler_path]:
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            raise FileNotFoundError(f"Model file not found: {path}")
    
    lr_model = joblib.load(lr_path)
    logger.info("Loaded Logistic Regression model")
    
    dt_model = joblib.load(dt_path)
    logger.info("Loaded Decision Tree model")
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)
    logger.info("Loaded XGBoost model")
    
    cb_model = CatBoostClassifier()
    cb_model.load_model(cb_path)
    logger.info("Loaded CatBoost model")
    
    scaler = joblib.load(scaler_path)
    logger.info("Loaded scaler")
    
    logger.info('All models loaded successfully')
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    logger.error(traceback.format_exc())
    raise

def preprocess_input(data):
    """Tiền xử lý dữ liệu đầu vào"""
    try:
        logger.info(f"Received input data: {data}")
        
        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame([data])
        logger.info(f"Input columns: {df.columns.tolist()}")
        
        # Map tên cột tiếng Anh sang tên trong dataset
        column_mapping = {
            'Age': 'Age',
            'Gender': 'Gender',
            'Years_at_Company': 'Years at Company',
            'Job_Role': 'Job Role',
            'Monthly_Income': 'Monthly Income',
            'Work_Life_Balance': 'Work-Life Balance',
            'Job_Satisfaction': 'Job Satisfaction',
            'Performance_Rating': 'Performance Rating',
            'Number_of_Promotions': 'Number of Promotions',
            'Overtime': 'Overtime',
            'Distance_from_Home': 'Distance from Home',
            'Education_Level': 'Education Level',
            'Marital_Status': 'Marital Status',
            'Number_of_Dependents': 'Number of Dependents',
            'Job_Level': 'Job Level',
            'Company_Size': 'Company Size',
            'Remote_Work': 'Remote Work',
            'Leadership_Opportunities': 'Leadership Opportunities',
            'Innovation_Opportunities': 'Innovation Opportunities',
            'Company_Reputation': 'Company Reputation',
            'Employee_Recognition': 'Employee Recognition'
        }
        
        # Đổi tên các cột trong DataFrame
        df.rename(columns=column_mapping, inplace=True)
        logger.info(f"Columns after mapping: {df.columns.tolist()}")
        
        # Xác định các cột số và cột phân loại
        numerical_cols = ['Age', 'Years at Company', 'Monthly Income', 'Number of Promotions',
                         'Distance from Home', 'Number of Dependents']
        categorical_cols = ['Gender', 'Job Role', 'Work-Life Balance', 'Job Satisfaction', 
                          'Performance Rating', 'Overtime', 'Education Level', 'Marital Status',
                          'Job Level', 'Company Size', 'Remote Work', 'Leadership Opportunities', 
                          'Innovation Opportunities', 'Company Reputation', 'Employee Recognition']
        
        # Kiểm tra xem có thiếu cột nào không
        missing_cols = [col for col in numerical_cols + categorical_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Thiếu các cột sau trong dữ liệu đầu vào: {', '.join(missing_cols)}")
        
        # Chuẩn hóa các cột số
        logger.info("Normalizing numerical columns...")
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        
        # One-hot encoding cho các cột phân loại
        logger.info("Performing one-hot encoding...")
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        logger.info(f"Columns after encoding: {df_encoded.columns.tolist()}")
        
        # Đảm bảo có đủ các cột như khi training
        logger.info("Ensuring all required columns are present...")
        missing_features = [col for col in feature_columns if col not in df_encoded.columns]
        if missing_features:
            logger.info(f"Adding missing feature columns: {missing_features}")
            for col in missing_features:
                df_encoded[col] = 0
        
        # Sắp xếp lại các cột theo thứ tự đúng
        logger.info("Reordering columns...")
        df_encoded = df_encoded[feature_columns]
        
        logger.info("Preprocessing completed successfully")
        return df_encoded
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        logger.error(f"Input data: {data}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Lỗi trong quá trình xử lý dữ liệu đầu vào: {str(e)}")

def get_risk_level(probability):
    """Xác định mức độ rủi ro dựa trên xác suất"""
    if probability < 0.3:
        return "Thấp"
    elif probability < 0.7:
        return "Trung bình"
    else:
        return "Cao"

@app.route('/')
def home():
    logger.info('Accessing home page')
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify server is running and models are loaded"""
    try:
        logger.info('Testing server connection')
        return jsonify({
            'status': 'ok',
            'message': 'Server is running',
            'models_loaded': {
                'logistic_regression': lr_model is not None,
                'decision_tree': dt_model is not None,
                'xgboost': xgb_model is not None,
                'catboost': cb_model is not None,
                'scaler': scaler is not None
            }
        })
    except Exception as e:
        logger.error(f"Error in server test: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'Server error',
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info('Received prediction request')
        
        if not request.is_json:
            logger.error('Request does not contain JSON data')
            return jsonify({
                'error': 'Dữ liệu gửi lên phải ở định dạng JSON'
            }), 400
            
        data = request.get_json()
        if not data:
            logger.error('No data received in request')
            return jsonify({
                'error': 'Không nhận được dữ liệu'
            }), 400
            
        logger.info(f"Received data: {data}")
        
        # Kiểm tra các trường bắt buộc
        required_fields = [
            'Age', 'Gender', 'Years_at_Company', 'Job_Role', 'Monthly_Income',
            'Work_Life_Balance', 'Job_Satisfaction', 'Performance_Rating',
            'Number_of_Promotions', 'Overtime', 'Distance_from_Home',
            'Education_Level', 'Marital_Status', 'Number_of_Dependents',
            'Job_Level', 'Company_Size', 'Remote_Work', 'Leadership_Opportunities',
            'Innovation_Opportunities', 'Company_Reputation', 'Employee_Recognition'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({
                'error': f"Thiếu các trường dữ liệu: {', '.join(missing_fields)}"
            }), 400
        
        # Tiền xử lý dữ liệu
        try:
            processed_data = preprocess_input(data)
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return jsonify({
                'error': f"Lỗi xử lý dữ liệu: {str(e)}"
            }), 400
        
        # Dự đoán với các mô hình
        try:
            predictions = [
                lr_model.predict_proba(processed_data)[0][1],
                dt_model.predict_proba(processed_data)[0][1],
                xgb_model.predict_proba(processed_data)[0][1],
                cb_model.predict_proba(processed_data)[0][1]
            ]
            
            # Tính trung bình của các dự đoán
            avg_probability = sum(predictions) / len(predictions)
            
            # Xác định mức độ rủi ro
            risk_level = get_risk_level(avg_probability)
            
            # Lưu kết quả vào database
            employee_id = save_prediction(data, avg_probability, risk_level)
            
            # Chuyển đổi xác suất thành phần trăm và làm tròn đến 1 chữ số thập phân
            attrition_percentage = round(avg_probability * 100, 1)
            
            response = {
                'prediction': f"Khả năng nghỉ việc của nhân viên là {attrition_percentage}%",
                'risk_level': risk_level,
                'employee_id': employee_id,
                'status': 'success'
            }
            
            logger.info(f'Prediction successful: {response}')
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            return jsonify({
                'error': f"Lỗi trong quá trình dự đoán: {str(e)}"
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Có lỗi xảy ra khi xử lý dự đoán.',
            'details': str(e)
        }), 500

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    try:
        logger.info('Received feedback submission')
        
        if not request.is_json:
            logger.error('Request does not contain JSON data')
            return jsonify({
                'error': 'Dữ liệu gửi lên phải ở định dạng JSON'
            }), 400
            
        data = request.get_json()
        logger.info(f"Received feedback data: {data}")
        
        if not data:
            logger.error('No data received in request')
            return jsonify({
                'error': 'Không nhận được dữ liệu'
            }), 400
        
        # Kiểm tra và chuyển đổi dữ liệu
        rating = data.get('rating')
        feedback_text = data.get('feedback')
        
        logger.info(f"Extracted data - rating: {rating}, feedback: {feedback_text}")
        
        # Kiểm tra từng trường dữ liệu
        if rating is None:
            return jsonify({'error': 'Thiếu đánh giá'}), 400
            
        if not feedback_text:
            return jsonify({'error': 'Thiếu nội dung phản hồi'}), 400
        
        # Kiểm tra giá trị đánh giá hợp lệ
        valid_ratings = ['Rất tệ', 'Tệ', 'Bình thường', 'Tốt', 'Rất tốt']
        if rating not in valid_ratings:
            logger.error(f'Invalid rating value: {rating}')
            return jsonify({
                'error': f'Đánh giá phải là một trong các giá trị: {", ".join(valid_ratings)}'
            }), 400
        
        if not isinstance(feedback_text, str) or not feedback_text.strip():
            logger.error('Invalid feedback text')
            return jsonify({
                'error': 'Nội dung phản hồi không hợp lệ'
            }), 400
        
        # Lưu feedback vào database với ID tự động
        try:
            employee_id = save_feedback(rating, feedback_text.strip())
            logger.info(f'Feedback saved successfully with employee ID: {employee_id}')
            return jsonify({
                'status': 'success',
                'message': 'Cảm ơn bạn đã gửi đánh giá!',
                'employee_id': employee_id
            })
        except Exception as e:
            logger.error(f'Error saving feedback to database: {str(e)}')
            return jsonify({
                'error': 'Có lỗi xảy ra khi lưu đánh giá vào cơ sở dữ liệu'
            }), 500
        
    except Exception as e:
        logger.error(f"Error in feedback submission: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Có lỗi xảy ra khi gửi đánh giá. Vui lòng thử lại sau.',
            'details': str(e)
        }), 500

@app.route('/view-feedback', methods=['GET'])
def view_feedback():
    try:
        # Lấy feedback từ database
        feedback_list = get_all_feedback()
        return render_template('view_feedback.html', feedback=feedback_list)
    except Exception as e:
        logger.error(f"Error viewing feedback: {str(e)}")
        return render_template('view_feedback.html', 
                             feedback=[], 
                             error='Có lỗi xảy ra khi tải đánh giá')

if __name__ == '__main__':
    logger.info('Starting Flask server in development mode...')
    logger.info(f'Python version: {sys.version}')
    logger.info(f'Working directory: {os.getcwd()}')
    logger.info(f'Server will run on: http://localhost:8000')
    app.run(host='0.0.0.0', port=8000, debug=True)
    except Exception as e:
        logger.error(f"Server startup error: {str(e)}", exc_info=True)
