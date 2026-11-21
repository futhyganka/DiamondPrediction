# Import thư viện numpy để tính toán các phép toán số học như căn bậc hai
import numpy as np
# Import các hàm tính toán độ đo hồi quy từ scikit-learn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    # Tính sai số tuyệt đối trung bình (MAE) giữa giá trị thực tế và giá trị dự đoán
    mae = mean_absolute_error(y_true, y_pred)

    # Tính sai số bình phương trung bình (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Tính căn bậc hai của sai số bình phương trung bình (RMSE)
    rmse = np.sqrt(mse)

    # Tính hệ số xác định R2 (R-squared) để đo lường mức độ giải thích phương sai của mô hình
    r2 = r2_score(y_true, y_pred)

    # Trả về từ điển chứa đầy đủ 4 chỉ số đánh giá hiệu năng
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
