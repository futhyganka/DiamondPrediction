# Import mô hình KNN Regression từ thư viện scikit-learn
from sklearn.neighbors import KNeighborsRegressor

# Hàm tạo model knn dựa trên n_neighbors và p
def create_knn(
        n_neighbors = 5,
        p = 2               # default là Euclide
):
    # Tạo model
    model = KNeighborsRegressor(
            n_neighbors = n_neighbors,  # Số láng giềng
            metric = 'minkowski',       # Phương pháp đo khoảng cách
            p = p                       # Bậc khoảng cách Minkowski
        )

    # Trả về model
    return model