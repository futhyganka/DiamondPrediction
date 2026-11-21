# Import mô hình RandomForest hồi quy từ thư viện scikit-learn
from sklearn.ensemble import RandomForestRegressor

# Hàm tạo model Random Forest dựa trên n_estimators, max_depth, max_features và min_samples_leaf
def create_random_forest(
    n_estimators=300,          # Số lượng cây quyết định trong rừng (mặc định là 300 theo yêu cầu)
    max_depth=30,              # Độ sâu tối đa của mỗi cây trong rừng (mặc định là 30 theo yêu cầu)
    max_features=0.6,          # Tỷ lệ số đặc trưng tối đa được xem xét tại mỗi lần phân chia (mặc định là 0.6 theo yêu cầu)
    min_samples_leaf=1,        # Số lượng mẫu tối thiểu tại mỗi nút lá (mặc định là 1 theo yêu cầu)
    min_samples_split=2,       # Số lượng mẫu tối thiểu cần có để phân chia một nút nội bộ
    random_state=42,           # Cố định hạt giống ngẫu nhiên để đảm bảo tính tái lập kết quả
    n_jobs=-1                  # Tận dụng tất cả các nhân CPU có sẵn để huấn luyện song song
):
    # Khởi tạo mô hình RandomForestRegressor với các tham số tương ứng
    model = RandomForestRegressor(
        n_estimators=n_estimators,          # Gán số lượng cây
        max_depth=max_depth,                # Gán độ sâu tối đa
        max_features=max_features,          # Gán tỷ lệ đặc trưng tối đa
        min_samples_leaf=min_samples_leaf,  # Gán số mẫu tối thiểu ở nút lá
        min_samples_split=min_samples_split,# Gán số mẫu tối thiểu để tách nút
        criterion="squared_error",          # Tiêu chí phân chia nút dựa trên sai số bình phương
        random_state=random_state,          # Gán seed ngẫu nhiên
        n_jobs=n_jobs                       # Gán số nhân CPU chạy song song
    )

    # Trả về đối tượng mô hình Random Forest
    return model
