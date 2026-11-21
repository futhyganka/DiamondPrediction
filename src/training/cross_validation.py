# Import thư viện numpy để tính toán các phép toán số học như giá trị trung bình
import numpy as np
# Import công cụ KFold từ scikit-learn để chia dữ liệu thành K phần (K-fold Cross Validation)
from sklearn.model_selection import KFold
# Import hàm clone từ scikit-learn để tạo bản sao mới của mô hình và scaler cho từng fold
from sklearn.base import clone
# Import hàm tính các chỉ số đánh giá hồi quy MAE, MSE, RMSE, R2
from src.evaluation.metrics import calculate_metrics

def cross_validation(model, X, y, n_splits=5, random_state=42, scaler=None):
    # Khởi tạo đối tượng KFold với 5 fold, xáo trộn dữ liệu (shuffle) và cố định seed ngẫu nhiên
    kf = KFold(
        n_splits=n_splits,          # Số lượng fold (mặc định là 5 theo yêu cầu)
        shuffle=True,               # Xáo trộn dữ liệu trước khi chia
        random_state=random_state   # Cố định trạng thái ngẫu nhiên
    )

    # Khởi tạo danh sách để lưu các chỉ số đánh giá của từng fold
    results = []
    # Khởi tạo biến lưu giá trị RMSE tốt nhất (nhỏ nhất) ban đầu bằng dương vô cùng
    best_rmse = float('inf')
    # Khởi tạo biến lưu mô hình có hiệu năng tốt nhất trong các fold
    best_model = None
    # Khởi tạo biến lưu bộ scaler tương ứng với fold tốt nhất
    best_scaler = None

    # Duyệt qua từng fold thông qua phương thức split của đối tượng kf
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):

        # Kiểm tra nếu dữ liệu là DataFrame/Series thì dùng .iloc, ngược lại dùng chỉ số mảng thông thường
        X_train = X.iloc[train_index] if hasattr(X, "iloc") else X[train_index]
        X_test = X.iloc[test_index] if hasattr(X, "iloc") else X[test_index]

        # Tương tự tách nhãn y cho tập huấn luyện và tập kiểm tra của fold hiện tại
        y_train = y.iloc[train_index] if hasattr(y, "iloc") else y[train_index]
        y_test = y.iloc[test_index] if hasattr(y, "iloc") else y[test_index]

        # Khởi tạo biến lưu scaler riêng của fold hiện tại
        fold_scaler = None
        # Nếu có truyền vào scaler (ví dụ MinMaxScaler), tiến hành chuẩn hóa trong từng fold như notebook thử nghiệm
        if scaler is not None:
            # Tạo bản sao mới của scaler để học độc lập trên tập train của fold hiện tại
            fold_scaler = clone(scaler)
            # Học tham số chuẩn hóa trên X_train và chuyển đổi X_train
            X_train = fold_scaler.fit_transform(X_train)
            # Chuyển đổi X_test dựa trên các tham số đã học từ X_train của fold hiện tại
            X_test = fold_scaler.transform(X_test)

        # Tạo bản sao mô hình độc lập cho fold hiện tại
        model_clone = clone(model)

        # Huấn luyện mô hình trên tập dữ liệu Train của fold hiện tại
        model_clone.fit(X_train, y_train)

        # Dự đoán kết quả trên tập kiểm tra Test của fold hiện tại
        y_pred = model_clone.predict(X_test)

        # Tính toán các chỉ số đánh giá (MAE, MSE, RMSE, R2) cho fold hiện tại
        metrics = calculate_metrics(y_test, y_pred)

        # Thêm kết quả của fold hiện tại vào danh sách kết quả tổng
        results.append(metrics)

        # In kết quả đánh giá chi tiết của fold hiện tại ra màn hình
        print(f"Fold {fold}: {metrics}")

        # Kiểm tra nếu RMSE của fold này nhỏ hơn RMSE tốt nhất trước đó thì lưu lại mô hình và scaler của fold này
        if metrics["RMSE"] < best_rmse:
            # Cập nhật mức RMSE thấp nhất
            best_rmse = metrics["RMSE"]
            # Lưu lại mô hình của fold đạt kết quả tốt nhất
            best_model = model_clone
            # Lưu lại scaler tương ứng của fold tốt nhất
            best_scaler = fold_scaler

    # Tính giá trị trung bình cộng của từng chỉ số đánh giá qua tất cả các fold
    mean_results = {
        metric: np.mean([result[metric] for result in results])
        for metric in results[0]
    }

    # Trả về kết quả các fold, kết quả trung bình, mô hình tốt nhất và scaler tương ứng
    return results, mean_results, best_model, best_scaler
