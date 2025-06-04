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



# Tạo thư mục templates nếu chưa có
mkdir -p templates

# Di chuyển file index.html vào templates/
mv index.html templates/index.html

# Di chuyển file view_feedback.html vào templates/
mv view_feedback.html templates/view_feedback.html

# Thêm thay đổi vào git
git add templates/index.html templates/view_feedback.html

# Commit lại thay đổi
git commit -m "Move HTML files to templates/ for Flask compatibility"

# Đẩy lên GitHub
git push
