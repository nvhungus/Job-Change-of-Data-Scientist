import numpy as np
import pickle
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

def save_model(model: Any, filepath: str) -> None:
    # Lưu một model đã huấn luyện vào file
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok = True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Lưu model thành công vào '{filepath}'.")
    except IOError as e:
        logger.error(f"Lỗi khi lưu model: {e}")

def load_model(filepath: str) -> Any:
    # Tải một model từ file
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Tải model thành công từ '{filepath}'.")
        return model
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file model tại '{filepath}'.")
        return None

class LogisticRegressionNumpy:
    # Cài đặt Logistic Regression từ đầu bằng NumPy
    # Sử dụng thuật toán Gradient Descent
    def __init__(self, lr: float = 0.01, n_iters: int = 1000, alpha: float = 0):
        self.lr = lr             # Learning rate
        self.n_iters = n_iters   # Số lần lặp
        self.alpha = alpha       # Tham số L2 regularization
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Tối ưu bằng Gradient Descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted_proba = self._sigmoid(linear_model)
            
            # Tính đạo hàm
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted_proba - y))
            db = (1 / n_samples) * np.sum(y_predicted_proba - y)
            
            # Thêm thành phần L2 regularization vào đạo hàm của weights
            dw += (self.alpha / n_samples) * self.weights

            # Cập nhật weights và bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Dự đoán xác suất
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        # Dự đoán nhãn (0 hoặc 1)
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Tính độ chính xác (accuracy)
    return np.mean(y_true == y_pred)

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Tính precision, recall, và F1-score
    tp = np.sum((y_pred == 1) & (y_true == 1)) # True Positives
    fp = np.sum((y_pred == 1) & (y_true == 0)) # False Positives
    fn = np.sum((y_pred == 0) & (y_true == 1)) # False Negatives
    
    epsilon = 1e-8 # Tránh lỗi chia cho 0
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

def k_fold_cross_validation(model_class: Any, X: np.ndarray, y: np.ndarray, k: int = 5, **model_params: Any) -> List[float]:
    # Thực hiện K-Fold Cross-Validation và trả về accuracy của mỗi fold
    fold_scores = []
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, k)
    
    for i in range(k):
        val_indices = fold_indices[i]
        train_indices = np.concatenate([fold_indices[j] for j in range(k) if j != i])
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        y_val_pred = model.predict(X_val)
        fold_scores.append(accuracy(y_val, y_val_pred))
        
    return fold_scores