# Import thư viện pandas để chuyển đổi dữ liệu đánh giá thành DataFrame trực quan
import pandas as pd
# Import hàm tính các độ đo đánh giá hồi quy từ file metrics.py
from src.evaluation.metrics import calculate_metrics

# Hàm đánh giá mô hình đã huấn luyện trên tập dữ liệu kiểm tra và trả về các chỉ số đánh giá
def evaluate_model(model, X_test, y_test):
    # Sử dụng mô hình đã huấn luyện để dự đoán trên tập dữ liệu kiểm tra X_test
    y_pred = model.predict(X_test)

    # Tính toán và trả về các chỉ số đánh giá (MAE, MSE, RMSE, R2) dựa trên y_test thực tế và y_pred dự đoán
    return calculate_metrics(y_test, y_pred)

# Hàm so sánh kết quả đánh giá của nhiều mô hình khác nhau và trả về bảng DataFrame trực quan
def compare_models(results):
    # Chuyển đổi từ điển kết quả của nhiều mô hình thành bảng DataFrame và chuyển vị ma trận (T) để mỗi hàng là một mô hình
    return pd.DataFrame(results).T
