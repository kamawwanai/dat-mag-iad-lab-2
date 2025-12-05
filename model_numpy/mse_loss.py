import numpy as np


class MSELoss:
    """
    MSELoss для многоклассовой классификации

    (как в torch.nn.MSELoss(reduction='mean'))
    """

    def __init__(self):
        self.Y_pred = None  # кеш предсказаний
        self.Y_true = None  # кеш истинных значений

    def forward(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """
        Forward pass

        N  — размер batch
        C  — число классов (размер последней размерности)
        y  — предсказания (N, C)
        t  — целевые значения (N, C), one-hot

        Формула:
            L = (1 / (N * C)) · SUM (Y_pred - Y_true)^2
        """
        self.Y_pred = Y_pred
        self.Y_true = Y_true

        N, C = Y_pred.shape
        diff = Y_pred - Y_true
        loss = np.sum(diff ** 2) / (N * C)
        return float(loss)

    def backward(self) -> np.ndarray:
        """
        Backward pass

        Градиент:
        dL/dY_pred = (2 / (N * C)) · (Y_pred - Y_true)
        """
        N, C = self.Y_pred.shape
        dY_pred = (2.0 / (N * C)) * (self.Y_pred - self.Y_true)
        return dY_pred
    
    