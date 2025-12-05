import numpy as np
from tqdm.auto import tqdm

from .layers import Linear, ReLU, Softmax
from .mse_loss import MSELoss


class TwoLayerNet:
    """
    Input (N) -> Linear(N, hidden) -> ReLU -> Linear(hidden, out) -> Softmax
    """

    def __init__(self, input_size: int = 784, hidden_size: int = 128, output_size: int = 10):
        """
        Инициализация сети
        """
        # Слои
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)
        self.softmax = Softmax()

        # Функция потерь
        self.loss_fn = MSELoss()

        # История обучения
        self.train_losses = []
        self.val_accuracies = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass через сеть
        """
        out = self.fc1.forward(X)
        out = self.relu.forward(out)
        out = self.fc2.forward(out)
        out = self.softmax.forward(out)
        return out

    def backward(self) -> None:
        """
        Backward pass через всю сеть

        Последовательно backward для каждого слоя в обратном порядке
        MSELoss <- Softmax <- Linear2 <- ReLU <- Linear1
        """
        # Градиент от loss
        dout = self.loss_fn.backward()

        # Backprop через слои
        dout = self.softmax.backward(dout)
        dout = self.fc2.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.fc1.backward(dout)

    def update_parameters(self, learning_rate: float) -> None:
        """
        Обновление параметров методом градиентного спуска
        W_new = W_old - lr * dL/dW
        """
        # Обновляем веса и bias первого слоя
        self.fc1.W -= learning_rate * self.fc1.dW
        self.fc1.b -= learning_rate * self.fc1.db

        # Обновляем веса и bias второго слоя
        self.fc2.W -= learning_rate * self.fc2.dW
        self.fc2.b -= learning_rate * self.fc2.db

    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray, 
                   batch_size: int, learning_rate: float,
                   use_tqdm: bool = False) -> float:
        """
        Обучение на одной эпохе
        """
        N = X_train.shape[0]
        indices = np.random.permutation(N)
        epoch_loss = 0.0
        num_batches = 0

        batch_indices_iter = range(0, N, batch_size)
        if use_tqdm:
            batch_indices_iter = tqdm(batch_indices_iter, desc="Batches", leave=False)

        for i in batch_indices_iter:
            # Батч
            batch_idx = indices[i:i + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            # Forward
            y_pred = self.forward(X_batch)

            # Loss
            loss = self.loss_fn.forward(y_pred, y_batch)
            epoch_loss += loss
            num_batches += 1

            # Backward + обновление
            self.backward()
            self.update_parameters(learning_rate)

        return epoch_loss / num_batches

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Accuracy на данных
        """
        y_pred = self.forward(X)
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_labels == y_true)
        return accuracy
    

