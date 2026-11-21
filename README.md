# Diamond Price Prediction (Dự đoán giá kim cương)

Dự án Machine Learning ứng dụng các thuật toán học có giám sát (Hồi quy - Regression) để dự đoán giá kim cương dựa trên các đặc trưng vật lý và phẩm chất (carat, cut, color, clarity, depth, table, x, y, z).

---

## 1. Cấu trúc thư mục dự án

```text
diamond_prediction/
│
├── data/
│   ├── raw/
│   │   └── Diamonds Prices2022.csv       # Dataset thô ban đầu (53,943 dòng)
│   └── processed/
│       └── diamonds_processed.csv        # Dữ liệu sạch sau tiền xử lý và mã hóa
│
├── experiments/                          # Jupyter Notebooks phân tích và thử nghiệm
│   ├── data_analysis/                    # Phân tích khám phá dữ liệu (EDA)
│   ├── knn/                              # Thử nghiệm K-Nearest Neighbors
│   ├── decision_tree/                    # Thử nghiệm Decision Tree
│   ├── random_forest/                    # Thử nghiệm Random Forest
│   └── SVR/                              # Thử nghiệm Support Vector Regression
│
├── models/                               # Thư mục lưu các mô hình đã huấn luyện (.pkl)
│   ├── knn.pkl                           # Mô hình KNN (đã đóng gói kèm MinMaxScaler của fold tốt nhất)
│   ├── decision_tree.pkl                 # Mô hình Decision Tree
│   ├── svr.pkl                           # Mô hình SVR (kèm StandardScaler)
│   └── random_forest.pkl                 # Mô hình Random Forest
│
├── src/                                  # Mã nguồn chính của dự án
│   ├── config.py                         # Cấu hình đường dẫn thư mục và file
│   ├── data/
│   │   ├── load_data.py                  # Nạp và kiểm tra dữ liệu ban đầu
│   │   └── preprocessing.py              # Tiền xử lý, lọc nhiễu, mã hóa và lưu dữ liệu
│   ├── models/
│   │   ├── knn.py                        # Định nghĩa mô hình KNN thuần
│   │   ├── decision_tree.py              # Định nghĩa mô hình Decision Tree
│   │   ├── svr.py                        # Định nghĩa mô hình SVR
│   │   └── random_forest.py              # Định nghĩa mô hình Random Forest
│   ├── training/
│   │   ├── cross_validation.py           # Đánh giá 5-Fold CV và chọn fold tốt nhất
│   │   └── train.py                      # Kịch bản huấn luyện toàn bộ và lưu mô hình
│   ├── evaluation/
│   │   ├── metrics.py                    # Tính các chỉ số MAE, MSE, RMSE, R2
│   │   └── evaluate.py                   # Đánh giá trên tập test và so sánh kết quả
│   └── prediction/
│       └── predict.py                    # Giao diện dòng lệnh nhập liệu và dự đoán giá
│
├── requirements.txt                      # Danh sách các thư viện Python
└── README.md                             # Tài liệu hướng dẫn dự án
```

---

## 2. Quy trình tiền xử lý dữ liệu (Data Preprocessing)

Dữ liệu thô từ `data/raw/Diamonds Prices2022.csv` được làm sạch và biến đổi qua các bước:
1. **Loại bỏ thuộc tính không mang thông tin**: Xóa cột chỉ số thừa (`Unnamed: 0`).
2. **Loại bỏ dữ liệu nhiễu**: Lọc bỏ các dòng có kích thước không hợp lệ ($x \le 0$, $y \le 0$, hoặc $z \le 0$).
3. **Loại bỏ dữ liệu trùng lặp**: Xóa các bản ghi trùng lặp hoàn toàn (`drop_duplicates`).
4. **Mã hóa thứ tự (Ordinal Encoding)** cho các biến phân loại:
   - `cut`: `['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']` $\rightarrow$ giá trị từ `0` đến `4`.
   - `color`: `['J', 'I', 'H', 'G', 'F', 'E', 'D']` $\rightarrow$ giá trị từ `0` đến `6`.
   - `clarity`: `['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']` $\rightarrow$ giá trị từ `0` đến `7`.
5. **Lưu dữ liệu đã xử lý**: Tự động lưu thành file `data/processed/diamonds_processed.csv`.

---

## 3. Các mô hình và chiến lược huấn luyện

Dự án triển khai và so sánh 4 mô hình hồi quy với các siêu tham số tối ưu:

| Mô hình | Phương pháp chia dữ liệu | Siêu tham số tối ưu | Cơ chế đóng gói & Scaling |
| :--- | :--- | :--- | :--- |
| **K-Nearest Neighbors (KNN)** | **5-Fold Cross Validation** | `n_neighbors=8`<br>`weights='distance'`<br>`metric='minkowski', p=1` (Manhattan) | Chuẩn hóa `MinMaxScaler` độc lập trong từng fold. Mô hình và Scaler của **Fold có RMSE tốt nhất** được gom thành 1 `Pipeline` và lưu vào `models/knn.pkl`. |
| **Decision Tree** | **Hold-out 8/2** (80% Train / 20% Test) | `max_depth=13`<br>`min_samples_leaf=5`<br>`min_samples_split=17`<br>`random_state=42` | Huấn luyện trực tiếp trên tập train, lưu `models/decision_tree.pkl`. |
| **Support Vector Regression (SVR)** | **Hold-out 8/2** (80% Train / 20% Test) | `kernel='rbf'`<br>`C=1000`<br>`epsilon=0.05`<br>`gamma=0.1` | Đóng gói kèm `StandardScaler` trong `Pipeline`, lưu `models/svr.pkl`. |
| **Random Forest** | **Hold-out 8/2** (80% Train / 20% Test) | `n_estimators=300`<br>`max_depth=30`<br>`max_features=0.6`<br>`min_samples_leaf=1`<br>`random_state=42`, `n_jobs=-1` | Huấn luyện trực tiếp trên tập train, lưu `models/random_forest.pkl`. |

### Các chỉ số đánh giá (Evaluation Metrics)
* **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình.
* **MSE** (Mean Squared Error): Sai số bình phương trung bình.
* **RMSE** (Root Mean Squared Error): Căn bậc hai của sai số bình phương trung bình.
* **$R^2$ Score**: Hệ số xác định (đo lường tỷ lệ phương sai mà mô hình giải thích được, càng gần 1 càng tốt).

---

## 4. Cài đặt môi trường

1. Tạo môi trường ảo (khuyến nghị):
```bash
python -m venv .venv
```

2. Kích hoạt môi trường ảo:
- Trên Windows (PowerShell / CMD):
```powershell
.venv\Scripts\activate
```
- Trên Linux / macOS:
```bash
source .venv/bin/activate
```

3. Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

---

## 5. Hướng dẫn sử dụng

### 5.1. Chạy huấn luyện toàn bộ mô hình (Training Pipeline)

Từ thư mục gốc dự án, thực hiện lệnh:
```bash
python -m src.training.train
```

**Quy trình tự động thực hiện:**
1. Đọc dữ liệu thô từ `data/raw/Diamonds Prices2022.csv`.
2. Tiền xử lý dữ liệu và lưu ra `data/processed/diamonds_processed.csv`.
3. Đánh giá mô hình KNN qua 5-Fold Cross Validation (có chuẩn hóa `MinMaxScaler` trong từng fold) và lưu fold tốt nhất vào `models/knn.pkl`.
4. Chia Hold-out 8/2, huấn luyện và đánh giá trên tập kiểm tra độc lập cho Decision Tree, Random Forest, SVR.
5. In bảng tổng hợp so sánh các chỉ số (MAE, MSE, RMSE, $R^2$) giữa 4 mô hình.
6. Lưu 4 file mô hình `.pkl` vào thư mục `models/`.

---

### 5.2. Chạy chương trình dự đoán tương tác (Interactive Prediction)

Từ thư mục gốc dự án, thực hiện lệnh:
```bash
python -m src.prediction.predict
```

---

### 5.3. Sử dụng hàm dự đoán trong Code Python hoặc Jupyter Notebook

Bạn có thể gọi trực tiếp hàm `predict_price` trong bất kỳ script Python nào:

```python
from src.prediction.predict import predict_price

# Thông tin mẫu kim cương cần dự đoán
sample_diamond = {
    'carat': 0.7,
    'cut': 4,       # Ideal
    'color': 5,     # E
    'clarity': 3,   # VS2
    'depth': 61.5,
    'table': 56.0,
    'x': 5.72,
    'y': 5.75,
    'z': 3.53
}

# Dự đoán bằng mô hình mong muốn: 'knn', 'decision_tree', 'random_forest', hoặc 'svr'
price = predict_price(sample_diamond, model_name="knn")
print(f"Giá dự đoán: ${price:,.2f} USD")
```
