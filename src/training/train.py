# Import thư viện joblib để lưu và nạp các mô hình học máy đã huấn luyện
import joblib
# Import thư viện pandas để thao tác và hiển thị dữ liệu dạng bảng
import pandas as pd
# Import hàm chia tập dữ liệu train/test theo phương pháp Hold-out từ scikit-learn
from sklearn.model_selection import train_test_split
# Import công cụ chuẩn hóa MinMaxScaler để đưa dữ liệu về khoảng [0, 1] trong quá trình train KNN
from sklearn.preprocessing import MinMaxScaler
# Import hàm make_pipeline để ghép bộ chuẩn hóa scaler và mô hình thành một đối tượng duy nhất
from sklearn.pipeline import make_pipeline

# Import các đường dẫn thư mục và file từ file cấu hình config
from src.config import RAW_DATA, PROCESSED_DATA, MODELS_DIR

# Import hàm nạp dữ liệu từ thư mục src/data
from src.data.load_data import load_data
# Import hàm tiền xử lý dữ liệu và lưu vào thư mục processed
from src.data.preprocessing import preprocessing

# Import các hàm khởi tạo mô hình từ thư mục src/models
from src.models.knn import create_knn
from src.models.decision_tree import create_decision_tree
from src.models.random_forest import create_random_forest
from src.models.svr import create_svr

# Import hàm đánh giá K-Fold Cross Validation từ thư mục src/training
from src.training.cross_validation import cross_validation
# Import các hàm đánh giá mô hình và so sánh kết quả từ thư mục src/evaluation
from src.evaluation.evaluate import evaluate_model, compare_models

# Hàm lưu mô hình đã huấn luyện vào thư mục models dưới dạng file .pkl
def save_model(model, model_name):
    # Tự động tạo thư mục models nếu chưa tồn tại
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Tạo đường dẫn đầy đủ đến tệp tin .pkl của mô hình
    model_path = MODELS_DIR / f"{model_name}.pkl"

    # Lưu đối tượng mô hình đã huấn luyện vào file trên đĩa
    joblib.dump(model, model_path)

    # In thông báo xác nhận đã lưu mô hình thành công
    print(f"Đã lưu mô hình: {model_path}")

# Hàm huấn luyện tất cả các mô hình và lưu mô hình vào thư mục models/
def train_all_models(X, y):
    # Khởi tạo mô hình KNN thuần với các tham số tối ưu (không kèm scaler bên trong mô hình)
    knn = create_knn(
        n_neighbors=8,              # Số lượng láng giềng gần nhất 
        weights='distance',         # Trọng số tính theo nghịch đảo khoảng cách
        metric='minkowski',         # Phương pháp đo khoảng cách Minkowski
        p=1                         # Bậc p=1 tương ứng với khoảng cách Manhattan
    )

    # Khởi tạo mô hình Decision Tree với các tham số tối ưu
    decision_tree = create_decision_tree(
        max_depth=13,              # Độ sâu tối đa của cây quyết định
        min_samples_leaf=5,        # Số mẫu tối thiểu tại mỗi nút lá 
        min_samples_split=17,      # Số mẫu tối thiểu cần có để tiếp tục phân chia một nút
    )

    # Khởi tạo mô hình Random Forest với các tham số tối ưu
    random_forest = create_random_forest(
        n_estimators=300,          # Số lượng cây quyết định trong rừng
        max_depth=30,              # Độ sâu tối đa của mỗi cây trong rừng 
        max_features=0.6,          # Tỷ lệ số đặc trưng tối đa được xem xét tại mỗi lần phân chia
        min_samples_leaf=1,        # Số lượng mẫu tối thiểu tại mỗi nút lá
    )
    
    # Khởi tạo mô hình SVR với các tham số tối ưu
    svr = create_svr(
        kernel='rbf',          # Dạng hàm nhân Kernel
        C=1000,                # Hệ số phạt điều chuẩn
        epsilon=0.05,          # Độ rộng vùng không phạt epsilon-tube 
        gamma=0.1,             # Hệ số ảnh hưởng của hàm nhân rbf
        use_scaler=True        # Sử dụng StandardScaler để chuẩn hóa dữ liệu giúp SVR hội tụ tốt và chính xác
    )

    # 1. HUẤN LUYỆN VÀ ĐÁNH GIÁ KNN BẰNG K-FOLD CROSS VALIDATION (5-FOLD)
    # In tiêu đề phần đánh giá KNN
    print("\nMÔ HÌNH 1: KNN (5-FOLD CROSS VALIDATION)")

    # Khởi tạo công cụ MinMaxScaler để chuẩn hóa dữ liệu trực tiếp trong quá trình huấn luyện
    knn_scaler = MinMaxScaler()

    # Thực hiện 5-Fold Cross Validation trên X, y và lấy ra mô hình cùng scaler của fold tốt nhất
    fold_results, knn_results, best_knn, best_knn_scaler = cross_validation(knn, X, y, n_splits=5, scaler=knn_scaler)

    # In kết quả trung bình của 5 fold cho mô hình KNN
    print("\nKết quả trung bình 5-Fold của KNN:")
    print(knn_results)

    # Ghép scaler và mô hình của fold tốt nhất thành một pipeline duy nhất đã được chuẩn hóa
    knn_pipeline = make_pipeline(best_knn_scaler, best_knn)

    # Lưu duy nhất một file mô hình knn.pkl (đã tích hợp sẵn scaler bên trong) vào thư mục models
    save_model(knn_pipeline, "knn")

    # 2. CHIA TẬP DỮ LIỆU THEO PHƯƠNG PHÁP HOLDOUT (TỶ LỆ 8/2)
    # In tiêu đề phần chia dữ liệu Hold-out 8/2
    print("\nCHIA DỮ LIỆU HOLDOUT (80% TRAIN / 20% TEST)")

    # Chia dữ liệu X, y thành tập Train (80%) và tập Test (20%) với random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # In kích thước các tập dữ liệu sau khi chia
    print(f"Số mẫu tập Train: {X_train.shape[0]}, Số mẫu tập Test: {X_test.shape[0]}")

    # 3. HUẤN LUYỆN VÀ ĐÁNH GIÁ DECISION TREE (HOLDOUT 8/2)
    # In tiêu đề huấn luyện Decision Tree
    print("\nMÔ HÌNH 2: DECISION TREE (HOLDOUT 8/2)")

    # Huấn luyện cây quyết định trên tập Train
    decision_tree.fit(X_train, y_train)

    # Đánh giá hiệu năng mô hình trên tập Test
    dtree_results = evaluate_model(decision_tree, X_test, y_test)

    # In các chỉ số đánh giá của Decision Tree
    print("Kết quả đánh giá Decision Tree trên tập Test:")
    print(dtree_results)

    # Lưu mô hình Decision Tree vào thư mục models
    save_model(decision_tree, "decision_tree")

    # 4. HUẤN LUYỆN VÀ ĐÁNH GIÁ RANDOM FOREST (HOLDOUT 8/2)
    # In tiêu đề huấn luyện Random Forest
    print("\nMÔ HÌNH 3: RANDOM FOREST (HOLDOUT 8/2)")

    # Huấn luyện rừng ngẫu nhiên trên tập Train
    random_forest.fit(X_train, y_train)

    # Đánh giá hiệu năng của Random Forest trên tập Test
    rf_results = evaluate_model(random_forest, X_test, y_test)

    # In các chỉ số đánh giá của Random Forest
    print("Kết quả đánh giá Random Forest trên tập Test:")
    print(rf_results)

    # Lưu mô hình Random Forest vào thư mục models
    save_model(random_forest, "random_forest")

    # 5. HUẤN LUYỆN VÀ ĐÁNH GIÁ SVR (HOLDOUT 8/2)
    # In tiêu đề huấn luyện SVR
    print("\nMÔ HÌNH 4: SVR (HOLDOUT 8/2)")

    # Huấn luyện mô hình SVR trên tập Train
    svr.fit(X_train, y_train)

    # Đánh giá hiệu năng của SVR trên tập Test
    svr_results = evaluate_model(svr, X_test, y_test)

    # In các chỉ số đánh giá của SVR
    print("Kết quả đánh giá SVR trên tập Test:")
    print(svr_results)

    # Lưu mô hình SVR vào thư mục models
    save_model(svr, "svr")

    # 6. TỔNG HỢP VÀ SO SÁNH KẾT QUẢ TẤT CẢ CÁC MÔ HÌNH
    # Gom tất cả kết quả đánh giá vào một từ điển
    all_results = {
        "KNN (5-Fold CV)": knn_results,
        "Decision Tree (Holdout 8/2)": dtree_results,
        "Random Forest (Holdout 8/2)": rf_results,
        "SVR (Holdout 8/2)": svr_results
    }

    # Chuyển kết quả sang DataFrame để hiển thị bảng so sánh trực quan
    comparison_df = compare_models(all_results)

    # In bảng so sánh tổng hợp các chỉ số
    print("BẢNG TỔNG HỢP SO SÁNH KẾT QUẢ CÁC MÔ HÌNH")
    print(comparison_df)

    # Trả về từ điển kết quả và bảng so sánh
    return all_results, comparison_df

# Hàm chạy toàn bộ pipeline từ nạp dữ liệu, tiền xử lý, huấn luyện và lưu mô hình
def run_pipeline():

    # Bước 1: Nạp dữ liệu thô từ file raw data
    raw_df = load_data(RAW_DATA)

    # Bước 2: Tiền xử lý dữ liệu và tự động lưu vào file processed data
    processed_df = preprocessing(raw_df, save_path=PROCESSED_DATA)

    # Bước 3: Tách dữ liệu thành ma trận đặc trưng X và nhãn dự đoán y (cột price)
    X = processed_df.drop(columns=["price"])
    y = processed_df["price"]

    # Bước 4: Thực hiện huấn luyện toàn bộ các mô hình và lưu mô hình vào thư mục models/
    all_results, comparison_df = train_all_models(X, y)

    # Trả về kết quả đánh giá
    return all_results, comparison_df


# Kiểm tra nếu file được thực thi trực tiếp bằng python
if __name__ == "__main__":
    # Gọi hàm chạy toàn bộ quy trình
    run_pipeline()
