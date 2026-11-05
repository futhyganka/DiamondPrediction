# Import mô hình cây quyết định hồi quy
from sklearn.tree import DecisionTreeRegressor

# Hàm tạo model decision tree dựa trên max_depth, min_samples_split và min_samples_leaf
def create_decision_tree(
        max_depth = 10,
        min_samples_leaf = 5,
        min_samples_split = 5
):
    # Tạo model
    model = DecisionTreeRegressor(
            criterion="squared_error",          # Chọn cách chia nút dựa trên việc giảm tổng sai số bình phương
            random_state=42,                    # Cố định giá trị ngẫu nhiên để kết quả có thể tái lập
            max_depth=max_depth,                # Quy định độ sâu tối đa của cây
            min_samples_leaf=min_samples_leaf,  # Quy định số mẫu tối thiểu tại mỗi nút lá
            min_samples_split=min_samples_split # Quy định số nút lá tối thiểu
        )

    # Trả về model
    return model