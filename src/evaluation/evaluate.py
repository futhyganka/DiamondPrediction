# Import thư viện pandas để chuyển đổi dữ liệu đánh giá thành DataFrame trực quan
import pandas as pd
# Import hàm tính các độ đo đánh giá hồi quy từ file metrics.py
from src.evaluation.metrics import calculate_metrics

def evaluate_model(model, X_test, y_test):
    # Sử dụng mô hình đã huấn luyện để dự đoán trên tập dữ liệu kiểm tra X_test
    y_pred = model.predict(X_test)

    # Tính toán và trả về các chỉ số đánh giá (MAE, MSE, RMSE, R2) dựa trên y_test thực tế và y_pred dự đoán
    return calculate_metrics(y_test, y_pred)

def compare_models(results):
    # Chuyển đổi từ điển kết quả của nhiều mô hình thành bảng DataFrame và chuyển vị ma trận (T) để mỗi hàng là một mô hình
    return pd.DataFrame(results).T
