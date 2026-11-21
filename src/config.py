# Nạp hàm Path từ thư viện pathlib để quản lý đường dẫn tệp tin và thư mục một cách linh hoạt
from pathlib import Path

# Xác định đường dẫn thư mục gốc của dự án (thư mục chứa src, data, models,...)
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn đến file dataset thô ban đầu (raw data)
RAW_DATA = BASE_DIR / "data" / "raw" / "Diamonds Prices2022.csv"

# Đường dẫn đến thư mục chứa dữ liệu sau khi đã tiền xử lý (processed data)
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Đường dẫn đến file dữ liệu đã được làm sạch và tiền xử lý
PROCESSED_DATA = PROCESSED_DATA_DIR / "diamonds_processed.csv"

# Đường dẫn đến thư mục lưu trữ các mô hình đã huấn luyện
MODELS_DIR = BASE_DIR / "models"
