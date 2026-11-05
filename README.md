# Diamond Prediction

Dự án Machine Learning ứng dụng các thuật toán hồi quy để dự đoán giá
kim cương.

## Cấu trúc thư mục

``` text
DIAMOND_PREDICTION/
│
├── data/
│   ├── processed/                    # Dữ liệu sau tiền xử lý
│   └── raw/
│       └── Diamonds_Prices2022.csv   # Dataset ban đầu
│
├── experiments/                      # Notebook phân tích và thử nghiệm
│   ├── bayesian_ridge/
│   │   ├── BayesianRidge_Hold_out.ipynb
│   │   └── BayesianRidge_Kfold.ipynb
│   ├── data_analysis/
│   │   └── Data_analysis.ipynb
│   ├── decision_tree/
│   │   ├── DecisionTree_Hold_out.ipynb
│   │   └── DecisionTree_KFold.ipynb
│   ├── knn/
│   │   ├── KNN_Euclidean_Hold_out.ipynb
│   │   ├── KNN_Euclidean_KFold.ipynb
│   │   ├── KNN_Manhattan_Hold_out.ipynb
│   │   └── KNN_Manhattan_KFold.ipynb
│   └── random_forest/
│       ├── RandomForest_Hold_out.ipynb
│       └── RandomForest_KFold.ipynb
│
├── models/                           # Lưu mô hình đã huấn luyện
│
├── src/                              # Mã nguồn chính của dự án
│   ├── data/                         # Đọc và tiền xử lý dữ liệu
│   ├── evaluation/                   # Đánh giá mô hình
│   ├── models/                       # Định nghĩa thuật toán
│   ├── prediction/                   # Dự đoán giá kim cương
│   └── training/                     # Chia dữ liệu và huấn luyện
│
├── README.md                         # Tài liệu giới thiệu dự án
└── requirements.txt                  # Danh sách thư viện Python
```

## Mô tả thư mục

  -----------------------------------------------------------------------
  Thư mục                             Chức năng
  ----------------------------------- -----------------------------------
  `data/raw/`                         Chứa dữ liệu kim cương ban đầu.

  `data/processed/`                   Chứa dữ liệu sau khi làm sạch và
                                      tiền xử lý.

  `experiments/`                      Lưu notebook phân tích dữ liệu và
                                      thử nghiệm các mô hình bằng
                                      Hold-out, K-Fold.

  `models/`                           Lưu các mô hình đã huấn luyện, ví
                                      dụ `.pkl` hoặc `.joblib`.

  `src/data/`                         Chứa mã nguồn đọc và tiền xử lý dữ
                                      liệu.

  `src/models/`                       Chứa phần định nghĩa/cấu hình các
                                      thuật toán hồi quy.

  `src/training/`                     Chứa quy trình chia dữ liệu, huấn
                                      luyện và Cross Validation.

  `src/evaluation/`                   Chứa các hàm đánh giá hiệu quả mô
                                      hình.

  `src/prediction/`                   Chứa chức năng dự đoán giá cho kim
                                      cương mới.
  -----------------------------------------------------------------------

## Các thuật toán đang thử nghiệm

-   Decision Tree Regression
-   K-Nearest Neighbors (KNN) Regression với khoảng cách Euclidean và
    Manhattan
-   Random Forest Regression
-   Bayesian Ridge Regression

Các phương pháp chia và đánh giá dữ liệu:

-   Hold-out
-   K-Fold Cross Validation

## Cài đặt

Tạo môi trường ảo (khuyến nghị):

``` bash
python -m venv .venv
```

Kích hoạt trên Windows:

``` bash
.venv\\Scripts\\activate
```

Cài đặt thư viện:

``` bash
pip install -r requirements.txt
```

## Ghi chú

-   `experiments/` dùng để lưu các notebook phân tích và thử nghiệm;
    không nhất thiết là mã nguồn chạy ứng dụng cuối cùng.
-   `src/` là nơi tổ chức mã nguồn chính thức, có thể tái sử dụng cho xử
    lý dữ liệu, huấn luyện, đánh giá và dự đoán.
-   `models/` chỉ chứa các mô hình đã được huấn luyện và xuất ra tệp
    thực tế.
-   Các thư mục `processed/`, `models/` và thư mục con trong `src/` có
    thể đang trống; nội dung sẽ được bổ sung khi triển khai.
