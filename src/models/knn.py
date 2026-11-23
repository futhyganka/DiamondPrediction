# Import mô hình KNN Regression từ thư viện scikit-learn
from sklearn.neighbors import KNeighborsRegressor

# Hàm tạo model KNN thuần 
def create_knn(
    n_neighbors=8,              # Số lượng láng giềng gần nhất (mặc định là 8 theo yêu cầu)
    weights='distance',         # Trọng số tính theo nghịch đảo khoảng cách
    metric='minkowski',         # Phương pháp đo khoảng cách Minkowski
    p=1                         # Bậc p=1 tương ứng với khoảng cách Manhattan
):
    # Khởi tạo đối tượng mô hình KNN Regressor thuần túy
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,  # Gán số lượng láng giềng
        weights=weights,          # Gán phương thức tính trọng số
        metric=metric,            # Gán độ đo khoảng cách
        p=p                       # Gán tham số bậc p
    )

    # Trả về mô hình KNN
    return model
