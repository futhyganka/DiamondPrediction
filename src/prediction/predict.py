# Import thư viện joblib để nạp file mô hình và scaler đã lưu
import joblib
# Import thư viện pandas để chuyển đổi dữ liệu đầu vào thành DataFrame
import pandas as pd
# Import đường dẫn thư mục models từ file cấu hình config
from src.config import MODELS_DIR

# Hàm nạp mô hình từ file .pkl dựa trên tên mô hình
def load_model(model_name):
    # Tạo đường dẫn đầy đủ đến file mô hình .pkl trong thư mục models
    model_path = MODELS_DIR / f"{model_name}.pkl"

    # Kiểm tra xem file mô hình có tồn tại hay không
    if not model_path.exists():
        # Nếu chưa có file mô hình thì báo lỗi và nhắc người dùng chạy train trước
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}. Hãy chạy train trước!")

    # Nạp và trả về đối tượng mô hình từ file
    return joblib.load(model_path)

# Hàm dự đoán giá kim cương dựa trên dữ liệu đầu vào và mô hình đã nạp
def predict_price(data, model_name="random_forest"):
    # Đọc mô hình tương ứng từ đĩa
    model = load_model(model_name)

    # Chuyển dữ liệu đầu vào (từ điển) thành bảng pandas DataFrame nếu đầu vào là dict
    if isinstance(data, dict):
        # Tạo DataFrame gồm 1 dòng từ dict dữ liệu
        input_data = pd.DataFrame([data])
    else:
        # Nếu đã là DataFrame thì giữ nguyên bản sao
        input_data = data.copy()

    # Kiểm tra xem có file scaler riêng biệt trong thư mục models hay không (ví dụ knn_scaler.pkl nếu có)
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    # Nếu tồn tại file scaler tương ứng thì nạp và chuyển đổi dữ liệu
    if scaler_path.exists():
        # Nạp scaler đã được lưu trong quá trình train
        scaler = joblib.load(scaler_path)
        # Chuẩn hóa dữ liệu đầu vào theo các tham số của scaler
        input_data = scaler.transform(input_data)

    # Sử dụng mô hình để dự đoán giá trị kim cương
    prediction = model.predict(input_data)

    # Trả về giá trị dự đoán đầu tiên
    return prediction[0]

# Hàm tương tác với người dùng để nhập thông số kim cương và dự đoán giá
def interactive_predict():
    # Vòng lặp chính cho phép người dùng nhập và dự đoán nhiều lần
    while True:

        # 1. Hiển thị danh sách các mô hình có sẵn để người dùng lựa chọn
        print("\nChọn model:")
        print("  1. knn          (K-Nearest Neighbors)")
        print("  2. decision_tree (Cây quyết định)")
        print("  3. random_forest (Rừng ngẫu nhiên)")
        print("  4. svr           (Support Vector Regression)")

        # Tạo từ điển ánh xạ giữa số thứ tự và tên file model
        model_map = {
            "1": "knn",
            "knn": "knn",
            "2": "decision_tree",
            "decision_tree": "decision_tree",
            "3": "random_forest",
            "random_forest": "random_forest",
            "4": "svr",
            "svr": "svr"
        }

        # Vòng lặp kiểm tra lựa chọn mô hình hợp lệ
        while True:
            # Nhận lựa chọn từ bàn phím và chuẩn hóa chuỗi nhập
            model_choice = input("Nhập tên hoặc số thứ tự mô hình muốn dùng: ").strip().lower()
            # Kiểm tra nếu lựa chọn hợp lệ trong danh sách
            if model_choice in model_map:
                # Gán tên mô hình chuẩn xác
                selected_model = model_map[model_choice]
                # Thoát vòng lặp chọn mô hình
                break
            else:
                # Thông báo nhập lại nếu chưa đúng
                print("Lựa chọn không hợp lệ! Vui lòng nhập 1, 2, 3, 4 hoặc đúng tên model.")

        # In thông báo mô hình đã được chọn
        print(f"\n>> Đã chọn mô hình: [{selected_model}] <<")
        print("\n--- NHẬP THÔNG SỐ KIM CƯƠNG CẦN DỰ ĐOÁN ---")

        # 2. Nhập các thông số đặc trưng của viên kim cương
        try:
            # Nhập trọng lượng carat
            carat = float(input("carat (ví dụ: 0.23, 2.4, 4.72): "))

            # Nhập chất lượng vết cắt cut kèm danh sách giá trị giải thích
            cut = float(input("cut [0: Fair, 1: Good, 2: Very Good, 3: Premium, 4: Ideal]: "))

            # Nhập màu sắc color kèm danh sách giá trị giải thích từ J đến D
            color = float(input("color [0: J, 1: I, 2: H, 3: G, 4: F, 5: E, 6: D]: "))

            # Nhập độ tinh khiết clarity kèm danh sách giá trị giải thích từ I1 đến IF
            clarity = float(input("clarity [0: I1, 1: SI2, 2: SI1, 3: VS2, 4: VS1, 5: VVS2, 6: VVS1, 7: IF]: "))

            # Nhập tỷ lệ chiều sâu depth
            depth = float(input("depth (ví dụ: 61.5): "))

            # Nhập độ rộng mặt bàn table
            table = float(input("table (ví dụ: 75.0): "))

            # Nhập kích thước chiều dài x tính bằng mm
            x = float(input("x - Chiều dài (mm, ví dụ: 3.95): "))

            # Nhập kích thước chiều rộng y tính bằng mm
            y = float(input("y - Chiều rộng (mm, ví dụ: 3.98): "))

            # Nhập kích thước chiều cao z tính bằng mm
            z = float(input("z - Chiều cao (mm, ví dụ: 2.43): "))

        # Bắt lỗi nếu người dùng nhập sai định dạng số
        except ValueError:
            # Bắt lỗi nếu người dùng nhập sai định dạng số
            print("\n[LỖI] Giá trị nhập phải là số hợp lệ! Vui lòng thử lại.")
            # Quay lại đầu vòng lặp
            continue

        # Đóng gói dữ liệu đầu vào thành một từ điển theo đúng cấu trúc đặc trưng
        diamond_data = {
            "carat": carat,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z
        }

        # 3. Tiến hành dự đoán và in kết quả
        try:
            # Gọi hàm dự đoán giá kim cương
            predicted_price = predict_price(diamond_data, model_name=selected_model)

            # In giá trị dự đoán định dạng tiền tệ đẹp mắt
            print(f"==> KẾT QUẢ DỰ ĐOÁN GIÁ KIM CƯƠNG [{selected_model}]: ${predicted_price:,.2f} USD")

        except Exception as e:
            # In thông báo lỗi nếu quá trình nạp model hoặc dự đoán gặp sự cố
            print(f"\n[LỖI DỰ ĐOÁN]: {e}")

        # 4. Hỏi người dùng có muốn tiếp tục dự đoán nữa không
        continue_choice = input("\nBạn có muốn tiếp tục không? (y/n): ").strip().lower()

        # Kiểm tra nếu người dùng không chọn 'y' thì dừng chương trình
        if continue_choice != "y":
            # In thông báo kết thúc chương trình
            print("\nCảm ơn bạn đã sử dụng chương trình dự đoán giá kim cương! Tạm biệt.")
            # Thoát vòng lặp
            break


# Kiểm tra nếu file được chạy trực tiếp từ dòng lệnh
if __name__ == "__main__":
    # Khởi động chương trình nhập liệu và dự đoán tương tác
    interactive_predict()
