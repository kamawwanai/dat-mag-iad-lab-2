from time import time
from tqdm.auto import trange
import numpy as np

from .model import TwoLayerNet


def train_model(
    X_train: np.ndarray,
    y_train_onehot: np.ndarray,
    y_train_labels: np.ndarray,
    X_val: np.ndarray,
    y_val_labels: np.ndarray,
    input_size: int = 784,
    hidden_size: int = 128,
    output_size: int = 10,
    batch_size: int = 64,
    num_epochs: int = 10,
    learning_rate: float = 0.1,
    experiment_name: str = ""
) -> TwoLayerNet:
    """
    Обучение двухслойной сети на заданных данных

    Args:
        X_train: (N_train, input_size) обучающие данные (нормализованные, уже в векторе)
        y_train_onehot: (N_train, output_size) one-hot метки для MSELoss
        y_train_labels: (N_train,) целочисленные метки
        X_val: (N_val, input_size) валидационные/тестовые данные
        y_val_labels: (N_val,) метки для валидации
        input_size: размер входа (для MNIST 784)
        hidden_size: размер скрытого слоя
        output_size: количество классов (для MNIST 10)
        batch_size: размер батча
        num_epochs: число эпох
        learning_rate: шаг градиентного спуска
    """
    model = TwoLayerNet(input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size)
    
    history = {
        "experiment": experiment_name,
        "train_losses": [],
        "val_accuracies": [],
        "params": {
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
        },
    }

    epoch_iter = trange(num_epochs, desc="Epochs")

    for epoch in epoch_iter:
        start_time = time()

        train_loss = model.train_epoch(
            X_train, y_train_onehot,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_tqdm=True,          # прогресс по батчам
        )

        val_acc = model.evaluate(X_val, y_val_labels)
        elapsed = time() - start_time

        model.train_losses.append(train_loss)
        model.val_accuracies.append(val_acc)

        history["train_losses"].append(train_loss)
        history["val_accuracies"].append(val_acc)

        epoch_iter.set_postfix({
            "loss": f"{train_loss:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "time": f"{elapsed:.2f}s",
        })

    print(f"\nФинальная точность на валидации: {val_acc:.4f} ({val_acc*100:.2f}%)")

    return model, history