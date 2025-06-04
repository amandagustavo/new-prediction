from flask import Flask, render_template, request
import joblib
import os
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Ví dụ route xử lý dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    # Xử lý dữ liệu từ form, load model, v.v.
    # model = joblib.load('models/model.joblib')
    # Kết quả xử lý ...
    return render_template('index.html', result='Dự đoán: ...')

if __name__ == '__main__':
    app.run(debug=True)