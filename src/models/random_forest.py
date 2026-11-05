# Import mô hình RandomForest hồi quy
from sklearn.ensemble import RandomForestRegressor

# Hàm tạo model random forest dựa trên n_estimators, max_depth, min_samples_split và min_samples_leaf
def create_random_forest(
        n_estimators = 150,
        max_depth = 10,
        min_samples_leaf = 5,
        min_samples_split = 5
):
    # Tạo model
    model = RandomForestRegressor(
        n_estimators = n_estimators,            # Số lượng cây quyết định
        max_depth = max_depth,                  # Quy định độ sâu tối đa của cây
        min_samples_split = min_samples_split,  # Quy định số nút lá tối thiểu
        min_samples_leaf = min_samples_leaf,    # Quy định số mẫu tối thiểu tại mỗi nút lá
        criterion = "squared_error",            # Chọn cách chia nút dựa trên việc giảm tổng sai số bình phương
        random_state = 42,                      # Cố định giá trị ngẫu nhiên để kết quả có thể tái lập
        n_jobs = -1                             # Số lõi xử lý
    )

    # Trả về model
    return model