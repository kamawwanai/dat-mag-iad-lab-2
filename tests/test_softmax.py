import numpy as np
import torch
import torch.nn as nn


from model_numpy.layers import Softmax


def _run_softmax_test(X_np: np.ndarray, desc: str = ""):
    """
    Запускает один тест Softmax для заданного входного массива X_np.
    """
    print(f"\n\n{desc}  X shape={X_np.shape}")
    print("Input:")
    print(X_np)

    # NumPy версия
    np_softmax = Softmax()
    Y_np = np_softmax.forward(X_np)

    # PyTorch версия
    X_torch = torch.from_numpy(X_np).float()
    X_torch.requires_grad = True
    torch_softmax = nn.Softmax(dim=1)
    Y_torch = torch_softmax(X_torch)

    print("\nForward pass:")
    print("NumPy:")
    print(Y_np)
    print("Torch:")
    print(Y_torch.detach().numpy())
    print("Sum over classes (must be 1):", np.sum(Y_np, axis=1))
    print(f"max diff: {np.max(np.abs(Y_np - Y_torch.detach().numpy())):.2e}")

    # Backward pass
    dY = np.random.randn(*Y_np.shape).astype(np.float32)
    dX_np = np_softmax.backward(dY)

    Y_torch.backward(torch.from_numpy(dY).float())
    dX_torch = X_torch.grad.detach().cpu().numpy()

    print("\nBackward pass:")
    print("dX NumPy shape:", dX_np.shape)
    print("dX Torch shape:", dX_torch.shape)
    print(f"max diff: {np.max(np.abs(dX_np - dX_torch)):.2e}")


def test_softmax_activation():
    print("\n\nTEST SOFTMAX ACTIVATION")

    test_inputs = [
        (
            np.array([
                [-1.0,  2.0, 0.5],   # разные 
                [ 5.0,  5.0, 5.0],   # равномерное распределение
            ], dtype=np.float32),
            "case 1"
        ),
        (
            np.array([
                [0.0,   0.0,   0.0],   # все нули
                [10.0, -10.0,  0.0],   # сильный перекос
            ], dtype=np.float32),
            "case 2"
        ),
    ]

    for X_np, desc in test_inputs:
        _run_softmax_test(X_np, desc)