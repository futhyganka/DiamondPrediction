# Nạp hàm Path từ thư viện pathlib
from pathlib import Path

# Đường dẫn thư mục gốc
BASE_DIR = Path(__file__).resolve().parent.parent
# Đường dẫn đến file dataset raw
RAW_DATA = BASE_DIR / "data" / "raw" / "Diamonds Prices2022.csv"