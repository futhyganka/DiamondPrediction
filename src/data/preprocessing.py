# Nạp công cụ OrdinalEncoder từ thư viện scikit-learn để mã hóa các biến phân loại có thứ tự
from sklearn.preprocessing import OrdinalEncoder
# Nạp hàm Path từ pathlib để xử lý đường dẫn thư mục và tệp tin
from pathlib import Path
# Nạp đường dẫn lưu trữ file tiền xử lý mặc định từ file cấu hình config
from src.config import PROCESSED_DATA, PROCESSED_DATA_DIR

def preprocessing(diamonds, save_path=PROCESSED_DATA):
    # Tạo bản sao dữ liệu để không làm ảnh hưởng trực tiếp đến DataFrame ban đầu
    df = diamonds.copy()

    # 1: Loại bỏ thuộc tính không mang thông tin
    # Kiểm tra xem cột số thứ tự thừa 'Unnamed: 0' có tồn tại trong DataFrame hay không
    if 'Unnamed: 0' in df.columns:
        # Nếu có, xóa bỏ cột này khỏi tập dữ liệu
        df = df.drop(columns=['Unnamed: 0'])

    # 2: Loại bỏ dữ liệu nhiễu (kích thước x, y, z <= 0)
    # Lấy tổng số dòng dữ liệu ban đầu trước khi lọc
    initial_rows = len(df)

    # Lọc giữ lại các dòng có cả 3 kích thước (x, y, z) đều lớn hơn 0 và tạo bản sao dữ liệu mới
    df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)].copy()

    # Tính số lượng dòng dữ liệu bị loại bỏ (bằng số dòng ban đầu trừ số dòng sau khi lọc)
    invalid_dimensions = initial_rows - len(df)

    # In số lượng dòng nhiễu đã bị loại bỏ ra màn hình
    print(f"\nĐã loại bỏ {invalid_dimensions} dòng dữ liệu nhiễu (x, y, z <= 0).")

    # 3: Loại bỏ các dòng trùng lặp hoàn toàn
    # Đếm tổng số dòng dữ liệu bị lặp lại hoàn toàn trong tập dữ liệu
    num_duplicates = df.duplicated().sum()

    # Xóa bỏ các dòng trùng lặp, chỉ giữ lại 1 bản ghi đại diện
    df = df.drop_duplicates()

    # In số lượng dòng trùng lặp đã xóa ra màn hình
    print(f"\nĐã loại bỏ {num_duplicates} dòng trùng lặp.")

    # 4: Mã hóa thứ tự (Ordinal Encoding) cho các thuộc tính phân loại
    # Định nghĩa danh sách các mức độ chất lượng của vết cắt 'cut' theo thứ tự tăng dần (Fair là thấp nhất, Ideal là cao nhất)
    cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']

    # Định nghĩa danh sách màu sắc 'color' theo thứ tự tăng dần từ kém nhất đến tốt nhất (J -> D)
    color_categories = ['J', 'I', 'H', 'G', 'F', 'E', 'D']

    # Định nghĩa danh sách độ tinh khiết 'clarity' theo thứ tự tăng dần từ kém nhất đến tốt nhất (I1 -> IF)
    clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

    # Khởi tạo đối tượng OrdinalEncoder với danh sách thứ tự đã thiết lập cho từng cột
    encoder = OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])

    # Chuyển đổi các cột dữ liệu dạng chữ (phân loại) thành dạng số dựa trên thứ tự đã định nghĩa
    df[['cut', 'color', 'clarity']] = encoder.fit_transform(df[['cut', 'color', 'clarity']])

    # 5: Kiểm tra dữ liệu sau tiền xử lý
    # In kích thước mới của tập dữ liệu (số dòng, số cột) sau khi đã làm sạch và chuyển đổi
    print(f"\nKích thước tập dữ liệu sau tiền xử lý: {df.shape}\n")

    # In tiêu đề thông báo hiển thị 5 dòng đầu
    print("--- 5 dòng dữ liệu đầu sau tiền xử lý ---")

    # Hiển thị 5 dòng dữ liệu đầu tiên sau tiền xử lý
    print(df.head())

    # 6: Lưu dữ liệu đã tiền xử lý vào file processed nếu có đường dẫn save_path
    if save_path is not None:
        # Chuyển save_path về đối tượng Path để thao tác
        save_file = Path(save_path)
        # Tự động tạo thư mục chứa nếu chưa tồn tại
        save_file.parent.mkdir(parents=True, exist_ok=True)
        # Lưu DataFrame ra tệp tin CSV, bỏ cột chỉ mục (index=False)
        df.to_csv(save_file, index=False)
        # In thông báo xác nhận đã lưu dữ liệu thành công
        print(f"\nĐã lưu dữ liệu sau tiền xử lý vào: {save_file}")

    # Trả về DataFrame sau khi tiền xử lý hoàn tất
    return df

# Tạo bí danh (alias) proprocessing để tương thích nếu có code cũ gọi hàm này
proprocessing = preprocessing
