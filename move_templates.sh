#!/bin/bash
# Script tự động di chuyển file HTML vào thư mục templates/

mkdir -p templates
if [ -f index.html ]; then
    mv index.html templates/index.html
fi

if [ -f view_feedback.html ]; then
    mv view_feedback.html templates/view_feedback.html
fi

echo "Đã di chuyển các file HTML vào thư mục templates/"