# Nạp công cụ OrdinalEncoder từ thư viện scikit-learn để mã hóa các biến phân loại có thứ tự
from sklearn.preprocessing import OrdinalEncoder

def proprocessing(diamonds):
    # 1: Loại bỏ thuộc tính không mang thông tin
    # Kiểm tra xem cột số thứ tự thừa 'Unnamed: 0' có tồn tại trong DataFrame hay không
    if 'Unnamed: 0' in diamonds.columns:
        # Nếu có, xóa bỏ cột này khỏi tập dữ liệu
        diamonds = diamonds.drop(columns=['Unnamed: 0'])

    # 2: Loại bỏ dữ liệu nhiễu (kích thước x, y, z <= 0)
    # Lấy tổng số dòng dữ liệu ban đầu trước khi lọc
    initial_rows = len(diamonds)

    # Lọc giữ lại các dòng có cả 3 kích thước (x, y, z) đều lớn hơn 0 và tạo bản sao dữ liệu mới
    diamonds = diamonds[(diamonds['x'] > 0) & (diamonds['y'] > 0) & (diamonds['z'] > 0)].copy()

    # Tính số lượng dòng dữ liệu bị loại bỏ (bằng số dòng ban đầu trừ số dòng sau khi lọc)
    invalid_dimensions = initial_rows - len(diamonds)

    # In số lượng dòng nhiễu đã bị loại bỏ ra màn hình (thêm \n ở đầu để cách dòng trên)
    print(f"\nĐã loại bỏ {invalid_dimensions} dòng dữ liệu nhiễu (x, y, z <= 0).")

    # 3: Loại bỏ các dòng trùng lặp hoàn toàn
    # Đếm tổng số dòng dữ liệu bị lặp lại hoàn toàn trong tập dữ liệu
    num_duplicates = diamonds.duplicated().sum()

    # Xóa bỏ các dòng trùng lặp, chỉ giữ lại 1 bản ghi đại diện
    diamonds = diamonds.drop_duplicates()

    # In số lượng dòng trùng lặp đã xóa ra màn hình (thêm \n ở đầu để cách với dòng trên)
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

    # chuyển đổi các cột dữ liệu dạng chữ (phân loại) thành dạng số dựa trên thứ tự đã định nghĩa
    diamonds[['cut', 'color', 'clarity']] = encoder.fit_transform(diamonds[['cut', 'color', 'clarity']])

    # 5: Kiểm tra dữ liệu sau tiền xử lý
    # In kích thước mới của tập dữ liệu (số dòng, số cột) sau khi đã làm sạch và chuyển đổi
    print(f"\nKích thước tập dữ liệu sau tiền xử lý: {diamonds.shape}\n")

    # In tiêu đề thông báo hiển thị 5 dòng đầu
    print("--- 5 dòng dữ liệu đầu sau tiền xử lý ---")

    # Hiển thị 5 dòng dữ liệu đầu tiên sau tiền xử lý
    print(diamonds.head())

    # Trả về dataframe sau khi tiền xử lý
    return diamonds