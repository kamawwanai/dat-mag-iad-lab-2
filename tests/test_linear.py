import numpy as np
import torch
import torch.nn as nn

from model_numpy.layers import Linear


def _run_linear_layer_test(N: int, in_features: int, out_features: int, desc: str = ""):
    """
    Запускает один тест Linear слоя с заданными параметрами размера.
    """
    print(f"\n\n{desc} (N={N}, in={in_features}, out={out_features})")

    np_layer = Linear(in_features, out_features)
    X_np = np.random.randn(N, in_features).astype(np.float32)

    # PyTorch версия
    torch_layer = nn.Linear(in_features, out_features)
    torch_layer.weight.data = torch.from_numpy(np_layer.W.T).float()
    torch_layer.bias.data = torch.from_numpy(np_layer.b).float()

    X_torch = torch.from_numpy(X_np).float()
    X_torch.requires_grad = True

    # Forward pass
    Y_np = np_layer.forward(X_np)
    Y_torch = torch_layer(X_torch)

    print(f"\nForward pass:")
    print(f"max difference: {np.max(np.abs(Y_np - Y_torch.detach().numpy())):.2e}")

    # Backward pass
    dY = np.random.randn(N, out_features).astype(np.float32)

    # NumPy backward
    dX_np = np_layer.backward(dY)

    # PyTorch backward
    Y_torch.backward(torch.from_numpy(dY).float())
    dX_torch = X_torch.grad.detach().cpu().numpy()
    dW_torch = torch_layer.weight.grad.detach().cpu().numpy().T  # Transpose!
    db_torch = torch_layer.bias.grad.detach().cpu().numpy()

    print(f"\nBackward pass:")
    print(f"dX max difference: {np.max(np.abs(dX_np - dX_torch)):.2e}")
    print(f"dW max difference: {np.max(np.abs(np_layer.dW - dW_torch)):.2e}")
    print(f"db max difference: {np.max(np.abs(np_layer.db - db_torch)):.2e}")


def test_linear_layer():
    print("\n\n TEST LINEAR LAYER")
    test_configs = [
        (2, 10, 4,  "case 1"),
        (4, 5, 3,   "case 2"),
        (3, 7, 1,   "case 3"),
    ]

    for N, in_features, out_features, desc in test_configs:
        _run_linear_layer_test(N, in_features, out_features, desc)