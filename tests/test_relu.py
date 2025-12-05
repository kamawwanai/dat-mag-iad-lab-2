import numpy as np
import torch
import torch.nn as nn


from model_numpy.layers import ReLU

def _run_relu_test(X_np: np.ndarray, desc: str = ""):
    """
    Запускает один тест ReLU для заданного входного массива X_np
    """
    print(f"\n\n{desc}  X shape={X_np.shape}")
    print("Input:")
    print(X_np)

    # NumPy версия
    np_relu = ReLU()
    Y_np = np_relu.forward(X_np)

    # PyTorch версия
    X_torch = torch.from_numpy(X_np).float()
    X_torch.requires_grad = True
    torch_relu = nn.ReLU()
    Y_torch = torch_relu(X_torch)

    print("\nForward pass:")
    print("NumPy:")
    print(Y_np)
    print("Torch:")
    print(Y_torch.detach().numpy())
    print(f"max difference: {np.max(np.abs(Y_np - Y_torch.detach().numpy())):.2e}")

    # Backward pass: dY = 1 (проверяем производную ReLU: 1 для x>0, 0 для x<=0)
    dY = np.ones_like(Y_np, dtype=np.float32)
    dX_np = np_relu.backward(dY)

    Y_torch.backward(torch.from_numpy(dY).float())
    dX_torch = X_torch.grad.detach().cpu().numpy()

    print("\nBackward pass:")
    print("dX NumPy:")
    print(dX_np)
    print("dX Torch:")
    print(dX_torch)
    print(f"max difference: {np.max(np.abs(dX_np - dX_torch)):.2e}")


def test_relu_activation():
    print("\n\nTEST ReLU ACTIVATION")

    test_inputs = [
        (
            np.array([
                [-1.0,  2.0, -0.5, 0.0],
                [ 3.0, -4.0,  1.0, -0.1],
                [ 0.0,  0.5, -2.0, 7.0],
            ], dtype=np.float32),
            "case 1"
        ),
        (
            np.array([
                [-10.0, 0.0, 1.0],
                [  0.5, 2.0, -3.0],
            ], dtype=np.float32),
            "case 2"
        ),
    ]

    for X_np, desc in test_inputs:
        _run_relu_test(X_np, desc)