# Import mô hình hồi quy Support Vector Regression (SVR) từ thư viện scikit-learn
from sklearn.svm import SVR
# Import công cụ StandardScaler để chuẩn hóa các đặc trưng có trung bình bằng 0 và phương sai bằng 1
from sklearn.preprocessing import StandardScaler
# Import hàm make_pipeline để kết hợp bộ chuẩn hóa StandardScaler và mô hình SVR
from sklearn.pipeline import make_pipeline

# Hàm tạo model SVR với các tham số tối ưu theo yêu cầu
def create_svr(
    kernel='rbf',          # Dạng hàm nhân Kernel (mặc định là rbf theo yêu cầu)
    C=1000,                # Hệ số phạt điều chuẩn (mặc định C = 1000 theo yêu cầu)
    epsilon=0.05,          # Độ rộng vùng không phạt epsilon-tube (mặc định là 0.05 theo yêu cầu)
    gamma=0.1,             # Hệ số ảnh hưởng của hàm nhân rbf (mặc định là 0.1 theo yêu cầu)
    use_scaler=True        # Sử dụng StandardScaler để chuẩn hóa dữ liệu giúp SVR hội tụ tốt và chính xác
):
    # Khởi tạo mô hình hồi quy SVR với các siêu tham số đã định nghĩa
    svr_regressor = SVR(
        kernel=kernel,      # Thiết lập hàm nhân (kernel)
        C=C,                # Thiết lập hệ số C
        epsilon=epsilon,    # Thiết lập tham số epsilon
        gamma=gamma         # Thiết lập hệ số gamma
    )

    # Kiểm tra xem có sử dụng bộ chuẩn hóa dữ liệu hay không
    if use_scaler:
        # Tạo pipeline kết hợp bước chuẩn hóa StandardScaler trước rồi chuyển vào mô hình SVR
        model = make_pipeline(StandardScaler(), svr_regressor)
    else:
        # Nếu không cần chuẩn hóa thì trả về trực tiếp mô hình SVR
        model = svr_regressor

    # Trả về mô hình SVR (hoặc pipeline mô hình)
    return model
