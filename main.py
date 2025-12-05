from model_numpy.data_utils import load_mnist, one_hot_encode, plot_history
from model_numpy.train import train_model

if __name__ == "__main__":
    X_train, y_train_labels, X_test, y_test_labels = load_mnist()
    y_train_onehot = one_hot_encode(y_train_labels, num_classes=10)

    model, history = train_model(
        X_train, y_train_onehot, y_train_labels,
        X_test, y_test_labels,
        hidden_size=256,
        batch_size=64,
        num_epochs=10,
        learning_rate=0.1,
        experiment_name="final_run",
    )

    plot_history(history)