# Prediction Turnover Web App

## Cài đặt
```bash
pip install -r requirements.txt
```

## Chạy local
```bash
python app.py
```

## Triển khai trên Render
1. Đảm bảo repo có Procfile, requirements.txt, app.py, thư mục templates/
2. Đẩy lên GitHub, kết nối với Render, chọn Python build.
3. Nếu model nặng, để link Google Drive tải về hoặc dùng Git LFS.

## Lưu ý
- Đặt các file HTML vào thư mục templates/
- Model lớn thì không up lên GitHub, chỉ để link.