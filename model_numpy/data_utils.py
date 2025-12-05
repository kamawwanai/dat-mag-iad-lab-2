import numpy as np
# from tensorflow.keras.datasets import mnist # для загрузки датасета

from torchvision import datasets

import matplotlib.pyplot as plt

# def load_mnist_tensorflow():
#     """
#     Загрузка MNIST через tf.keras.datasets.mnist.
#     """
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()

#     # Нормализация пикселей к [0, 1]
#     X_train = X_train.astype(np.float32) / 255.0
#     X_test  = X_test.astype(np.float32) / 255.0

#     # Развертка: (N, 28, 28) -> (N, 784) для подачи в linear слой
#     X_train = X_train.reshape(X_train.shape[0], -1)
#     X_test  = X_test.reshape(X_test.shape[0], -1)

#     return X_train, y_train, X_test, y_test


def load_mnist(data_dir="./data"):
    mnist_train = datasets.MNIST(root=data_dir, train=True,
                                 download=True, transform=None)
    mnist_test  = datasets.MNIST(root=data_dir, train=False,
                                 download=True, transform=None)

    X_train = mnist_train.data.numpy().astype(np.float32) / 255.0
    X_test  = mnist_test.data.numpy().astype(np.float32) / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

    y_train = mnist_train.targets.numpy()
    y_test  = mnist_test.targets.numpy()
    return X_train, y_train, X_test, y_test


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    One-hot кодирование меток для использования с MSELoss
    """
    N = labels.shape[0]
    one_hot = np.zeros((N, num_classes), dtype=np.float32)
    one_hot[np.arange(N), labels] = 1
    return one_hot


def plot_history(history):
    epochs = range(1, len(history["train_losses"]) + 1)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_losses"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Train loss")
    plt.grid(True)
    plt.xticks(list(epochs))

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_accuracies"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val accuracy")
    plt.title("Validation accuracy")
    plt.grid(True)
    plt.xticks(list(epochs))

    plt.tight_layout()
    plt.show()