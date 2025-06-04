from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import logging
import traceback
import os
import json
from datetime import datetime

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Đường dẫn file lưu feedback (nếu muốn dùng file đơn giản)
FEEDBACK_FILE = "feedback.json"

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    logger.info(f'Request: {request.method} {request.url}')
    logger.info(f'Response Status: {response.status}')
    return response

@app.route('/')
def home():
    logger.info('Accessing home page')
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok', 'message': 'Server is running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info('Received prediction request')
        # Nhận dữ liệu JSON từ JS (form gửi lên)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        logger.info(f"Received data: {data}")

        # Kiểm tra trường bắt buộc, trả lỗi nếu thiếu
        required_fields = ['Age', 'Gender', 'Years_at_Company', 'Job_Role', 'Monthly_Income']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ValueError(f"Thiếu các trường dữ liệu: {', '.join(missing_fields)}")

        # TODO: Gọi model thực tế ở đây, trả về kết quả dự đoán
        # Giả lập response
        # (Bạn nên thay bằng code gọi model thực tế của bạn)
        probability = 0.35  # Tỉ lệ nghỉ việc, ví dụ 35%
        result = {
            'prediction': f"{probability*100:.1f}%"  # Ví dụ "35.0%"
        }

        logger.info('Sending prediction response')
        return jsonify(result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        error_details = traceback.format_exc()
        return jsonify({
            'error': 'Có lỗi xảy ra khi xử lý dự đoán. Vui lòng thử lại sau.',
            'details': str(e) if app.debug else None
        }), 500

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        rating = data.get('rating', '')
        feedback_text = data.get('feedback', '').strip()
        if not rating or not feedback_text:
            raise ValueError("Bạn phải chọn mức độ đánh giá và nhập nội dung.")

        # Lưu feedback vào file (có thể thay bằng database nếu muốn)
        feedback_entry = {
            "rating": rating,
            "feedback": feedback_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedback_list = json.load(f)
        else:
            feedback_list = []
        feedback_list.append(feedback_entry)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(feedback_list, f, ensure_ascii=False, indent=2)
        logger.info(f"Received feedback: {feedback_entry}")
        return jsonify({"message": "Cảm ơn bạn đã gửi đánh giá!"})
    except Exception as e:
        logger.error(f"Error in submit-feedback: {str(e)}", exc_info=True)
        return jsonify({"error": "Có lỗi khi gửi đánh giá. Vui lòng thử lại sau."}), 500

@app.route('/view-feedback')
def view_feedback():
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedback_list = json.load(f)
        else:
            feedback_list = []
        return render_template('view_feedback.html', feedback=feedback_list)
    except Exception as e:
        logger.error(f"Error in view-feedback: {str(e)}", exc_info=True)
        return render_template('view_feedback.html', feedback=[], error="Không thể tải đánh giá.")

if __name__ == '__main__':
    logger.info('Starting Flask server...')
    try:
        port = int(os.environ.get("PORT", 8000))
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logger.error(f"Server startup error: {str(e)}", exc_info=True)