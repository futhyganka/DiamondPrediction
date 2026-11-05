# Nạp thư viện pandas và đặt tên viết tắt là pd — dùng để làm việc với dữ liệu dạng bảng (DataFrame), hỗ trợ đọc/ghi tệp CSV, Excel và tiền xử lý dữ liệu
import pandas as pd

def load_data(file_path):
    # Đọc tệp CSV với dấu phân cách các cột là dấu phẩy (,) và trả về dataframe
    return pd.read_csv(file_path, sep=',')

def check_data(dataframe):
    # In kích thước của tập dữ liệu (số dòng, số cột)
    print(f"Kích thước tập dữ liệu ban đầu: {dataframe.shape}\n")

    # Hiển thị thông tin tổng quan
    dataframe.info()

    # In tiêu đề thông báo cho phần hiển thị 5 dòng đầu
    print("\n--- 5 dòng đầu tiên ---")
    # Hiển thị 5 dòng dữ liệu đầu tiên
    print(dataframe.head())

    # In tiêu đề thông báo cho phần kiểm tra các giá trị phân loại
    print("\n--- Giá trị phân loại của các thuộc tính dạng chuỗi ---")
    # Lấy và in danh sách các giá trị duy nhất (không trùng lặp) của cột 'cut' (vết cắt)
    print(f"Cut: {dataframe['cut'].unique()}")
    # Lấy và in danh sách các giá trị duy nhất của cột 'color' (màu sắc)
    print(f"Color: {dataframe['color'].unique()}")
    # Lấy và in danh sách các giá trị duy nhất của cột 'clarity' (độ tinh khiết)
    print(f"Clarity: {dataframe['clarity'].unique()}")

    # In tiêu đề thông báo cho phần thống kê mô tả
    print("\n--- Thống kê mô tả ---")
    # Tính toán các chỉ số thống kê cơ bản cho các cột số
    print(dataframe.describe())