import numpy as np


class Linear:
    """
    Полносвязный (линейный) слой: Y = XW + b
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Инициализация параметров слоя

        Xavier/Glorot инициализация для лучшей сходимости:
        N(0, sqrt(2 / (in_features + out_features)))
        """
        # Xavier инициализация весов
        std = np.sqrt(2.0 / (in_features + out_features))

        # Не под ReLu
        # std = np.sqrt(2.0 / in_features)

        self.W = np.random.randn(in_features, out_features).astype(np.float32) * std
        self.b = np.zeros(out_features, dtype=np.float32)


        # кеш для backward pass
        self.X = None

        # градиенты
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: Y = XW + b
        X: (N, in_features), W: (in_features, out_features), b: (out_features,)
        """
        self.X = X  # для backward
        return X @ self.W + self.b

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Backward pass через линейный слой.

        dL/dX = dL/dY · W^T   
        dL/dW = X^T · dL/dY   
        dL/db = Σ(dL/dY)
        """
        # Градиент для весов: X^T @ dY
        # (in_features, N) @ (N, out_features) -> (in_features, out_features)
        self.dW = self.X.T @ dY

        # Градиент для bias: сумма по batch dimension
        # (N, out_features) -> (out_features,)
        self.db = np.sum(dY, axis=0)

        # Градиент для входа: dY @ W^T
        # (N, out_features) @ (out_features, in_features) -> (N, in_features)
        dX = dY @ self.W.T

        return dX
    

class ReLU:
    """
    ReLU: f(x) = max(0, x)
    """

    def __init__(self):
        self.X = None  # Кеш входа для backward

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = max(0, x)
        """
        self.X = X
        return np.maximum(0, X)

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Backward pass

        y = max(0, x)
        dy/dx = { 1, если x > 0
                { 0, если x ≤ 0
        по chain rule:
        dL/dx = dL/dy · dy/dx = dL/dy · I(x > 0)
        """
        # 1, если X > 0, иначе 0
        mask = (self.X > 0).astype(np.float32)

        # Поэлементное умножение градиента на маску
        dX = dY * mask

        return dX
    

class Softmax:
    """
    Softmax активация
    """

    def __init__(self):
        # кеш выхода для backward
        self.Y = None


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Формула:
        y_i = exp(x_i - max(x)) / SUM_j exp(x_j - max(x))
        """
        # max для численной стабильности
        # keepdims=True сохраняет размерность для broadcasting
        X_shifted = X - np.max(X, axis=1, keepdims=True)

        # Экспонента
        exp_X = np.exp(X_shifted)

        # Нормализация
        self.Y = exp_X / np.sum(exp_X, axis=1, keepdims=True)

        return self.Y


    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Backward pass через Softmax

        Для одного:
        y_i = exp(x_i) / Z, где Z = SUM_j exp(x_j)
        Производная по x_k:
        dy_i/dx_k = d/dx_k [exp(x_i) / Z]
        По правилу частного:
        dy_i/dx_k = [Z · dexp(x_i)/dx_k - exp(x_i) · dZ/dx_k] / Z^2

        i = k
        d(x_i)/dx_i = exp(x_i)
        dZ/dx_i = exp(x_i)
        dy_i/dx_i = [Z · exp(x_i) - exp(x_i) · exp(x_i)] / Z^2
                  = [exp(x_i)/Z] · [Z - exp(x_i)] / Z
                  = y_i · (1 - y_i)

        i != k
        dexp(x_i)/dx_k = 0
        dZ/dx_k = exp(x_k)
        dy_i/dx_k = -exp(x_i) · exp(x_k) / Z^2 = -y_i · y_k

        по chain rule:
        dL/dx_k = SUM_i (dL/dy_i · dy_i/dx_k)
                = dL/dy_k · y_k(1 - y_k) + SUM_{i!=k} (dL/dy_i · (-y_i · y_k))
                = dL/dy_k · y_k - y_k · SUM_i (dL/dy_i · y_i)
                = y_k · (dL/dy_k - SUM_i (dL/dy_i · y_i))

        dL/dx = y * (dL/dy - (dL/dy · y) · 1)
        """
        # Для каждого вычисляем: SUM_j (dY_j · Y_j)
        # (N, C) * (N, C) -> sum по C -> (N, 1)
        sum_term = np.sum(dY * self.Y, axis=1, keepdims=True)

        # dX = Y * (dY - sum_term)
        # (N, C) * [(N, C) - (N, 1)] = (N, C)
        dX = self.Y * (dY - sum_term)

        return dX
    
