import sqlite3
import logging
from datetime import datetime

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DATABASE_NAME = 'employee_prediction.db'

def init_db():
    """Khởi tạo cơ sở dữ liệu và các bảng"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # Tạo bảng employee_data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS employee_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tuoi INTEGER,
            gioi_tinh TEXT,
            so_nam_lam_viec INTEGER,
            vi_tri_cong_viec TEXT,
            thu_nhap_thang INTEGER,
            can_bang_cv_cs TEXT,
            muc_do_hai_long TEXT,
            danh_gia_hieu_suat TEXT,
            so_lan_thang_chuc INTEGER,
            lam_them_gio TEXT,
            khoang_cach_tn INTEGER,
            trinh_do_hoc_van TEXT,
            so_nguoi_phu_thuoc INTEGER,
            cap_bac_cong_viec TEXT,
            tinh_trang_hon_nhan TEXT,
            lam_viec_tu_xa TEXT,
            quy_mo_cong_ty TEXT,
            co_hoi_doi_moi TEXT,
            co_hoi_lanh_dao TEXT,
            danh_tieng_cong_ty TEXT,
            ghi_nhan_nhan_vien TEXT,
            ty_le_nghi_viec REAL,
            muc_do_rui_ro TEXT
        )
        ''')

        # Tạo bảng user_feedback
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER,
            danh_gia TEXT,
            noi_dung TEXT,
            thoi_gian DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employee_data(id)
        )
        ''')

        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        conn.close()

def save_prediction(employee_data, prediction_result, risk_level):
    """Lưu thông tin nhân viên và kết quả dự đoán"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        query = '''
        INSERT INTO employee_data (
            tuoi, gioi_tinh, so_nam_lam_viec, vi_tri_cong_viec, thu_nhap_thang,
            can_bang_cv_cs, muc_do_hai_long, danh_gia_hieu_suat, so_lan_thang_chuc,
            lam_them_gio, khoang_cach_tn, trinh_do_hoc_van, so_nguoi_phu_thuoc,
            cap_bac_cong_viec, tinh_trang_hon_nhan, lam_viec_tu_xa, quy_mo_cong_ty,
            co_hoi_doi_moi, co_hoi_lanh_dao, danh_tieng_cong_ty, ghi_nhan_nhan_vien,
            ty_le_nghi_viec, muc_do_rui_ro
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        values = (
            employee_data['Age'],
            employee_data['Gender'],
            employee_data['Years_at_Company'],
            employee_data['Job_Role'],
            employee_data['Monthly_Income'],
            employee_data['Work_Life_Balance'],
            employee_data['Job_Satisfaction'],
            employee_data['Performance_Rating'],
            employee_data['Number_of_Promotions'],
            employee_data['Overtime'],
            employee_data['Distance_from_Home'],
            employee_data['Education_Level'],
            employee_data['Number_of_Dependents'],
            employee_data['Job_Level'],
            employee_data['Marital_Status'],
            employee_data['Remote_Work'],
            employee_data['Company_Size'],
            employee_data['Innovation_Opportunities'],
            employee_data['Leadership_Opportunities'],
            employee_data['Company_Reputation'],
            employee_data['Employee_Recognition'],
            prediction_result,
            risk_level
        )

        cursor.execute(query, values)
        employee_id = cursor.lastrowid
        conn.commit()
        logger.info(f"Prediction saved successfully for employee ID: {employee_id}")
        return employee_id
    except Exception as e:
        logger.error(f"Error saving prediction: {str(e)}")
        raise
    finally:
        conn.close()

def save_feedback(rating, feedback_text):
    """Lưu phản hồi của người dùng với ID tự động tăng"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # Tạo employee_id mới
        cursor.execute("SELECT MAX(id) FROM employee_data")
        result = cursor.fetchone()
        employee_id = 1 if result[0] is None else result[0]

        query = '''
        INSERT INTO user_feedback (employee_id, danh_gia, noi_dung)
        VALUES (?, ?, ?)
        '''

        cursor.execute(query, (employee_id, rating, feedback_text))
        conn.commit()
        logger.info(f"Feedback saved successfully for employee ID: {employee_id}")
        return employee_id
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        raise
    finally:
        conn.close()

def get_all_feedback():
    """Lấy tất cả phản hồi từ cơ sở dữ liệu"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        query = '''
        SELECT f.*, e.ty_le_nghi_viec, e.muc_do_rui_ro
        FROM user_feedback f
        LEFT JOIN employee_data e ON f.employee_id = e.id
        ORDER BY f.thoi_gian DESC
        '''

        cursor.execute(query)
        feedback_list = cursor.fetchall()
        
        # Chuyển đổi kết quả thành list of dictionaries
        feedback_data = []
        for row in feedback_list:
            feedback_data.append({
                'id': row[0],
                'employee_id': row[1],
                'rating': row[2],
                'feedback': row[3],
                'timestamp': row[4],
                'prediction': row[5],
                'risk_level': row[6]
            })
        
        return feedback_data
    except Exception as e:
        logger.error(f"Error getting feedback: {str(e)}")
        return []
    finally:
        conn.close()

# Khởi tạo database khi import module
init_db() 