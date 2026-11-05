# Nạp hàm BayesianRidge
from sklearn.linear_model import BayesianRidge

# Hàm tạo model bayesian ridge dựa trên max_iter và tol
def create_bayesian_ridge(
        max_iter = 300,
        tol = 1e-3
):
    # Tạo model
    model = BayesianRidge(
            max_iter = max_iter,    # Số vòng lặp tối đa
            tol = tol,              # Ngưỡng hội tụ
            alpha_1 = 1e-6,         # Tham số prior của alpha
            alpha_2 = 1e-6,         # Tham số prior của alpha
            lambda_1 = 1e-6,        # Tham số prior của lambda
            lambda_2 = 1e-6,        # Tham số prior của lambda
            fit_intercept = True    # Hệ số chặn
        )

    # Trả về model
    return model