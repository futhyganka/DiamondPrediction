# Import mô hình cây quyết định hồi quy từ thư viện scikit-learn
from sklearn.tree import DecisionTreeRegressor

# Hàm tạo model Decision Tree dựa trên max_depth, min_samples_leaf và min_samples_split
def create_decision_tree(
    max_depth=13,              # Độ sâu tối đa của cây quyết định (mặc định là 13 theo yêu cầu)
    min_samples_leaf=5,        # Số mẫu tối thiểu tại mỗi nút lá (mặc định là 5 theo yêu cầu)
    min_samples_split=17,      # Số mẫu tối thiểu cần có để tiếp tục phân chia một nút (mặc định là 17)
    random_state=42            # Cố định hạt giống ngẫu nhiên để đảm bảo tính tái lập của kết quả
):
    # Khởi tạo mô hình DecisionTreeRegressor với các tham số tối ưu
    model = DecisionTreeRegressor(
        criterion="squared_error",          # Tiêu chí phân chia nút dựa trên việc giảm phương sai (MSE)
        max_depth=max_depth,                # Gán độ sâu tối đa của cây
        min_samples_leaf=min_samples_leaf,  # Gán số mẫu tối thiểu ở nút lá
        min_samples_split=min_samples_split,# Gán số mẫu tối thiểu để tách nút
        random_state=random_state           # Cố định seed ngẫu nhiên
    )

    # Trả về đối tượng mô hình cây quyết định
    return model
