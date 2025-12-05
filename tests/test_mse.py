import numpy as np
import torch
import torch.nn as nn

from model_numpy.mse_loss import MSELoss


def _run_mse_test(Y_pred_np: np.ndarray, Y_true_np: np.ndarray, desc: str = ""):
    """
    Запускает один тест MSELoss для заданных предсказаний и целей.
    """
    print(f"\n\n{desc}  shape={Y_pred_np.shape}")
    print("Y_pred:")
    print(Y_pred_np)
    print("Y_true:")
    print(Y_true_np)

    # NumPy версия
    np_loss = MSELoss()
    loss_np = np_loss.forward(Y_pred_np, Y_true_np)

    # PyTorch версия (nn.MSELoss(reduction='mean'): среднее по всем элементам)
    Y_pred_torch = torch.from_numpy(Y_pred_np).float()
    Y_pred_torch.requires_grad = True
    Y_true_torch = torch.from_numpy(Y_true_np).float()
    torch_loss_fn = nn.MSELoss()
    loss_torch = torch_loss_fn(Y_pred_torch, Y_true_torch)

    print("\nForward pass:")
    print(f"np loss:    {loss_np:.6f}")
    print(f"torch loss: {loss_torch.item():.6f}")
    print(f"diff:       {abs(loss_np - loss_torch.item()):.2e}")

    # Backward pass
    dY_pred_np = np_loss.backward()

    loss_torch.backward()
    dY_pred_torch = Y_pred_torch.grad.detach().cpu().numpy()

    print("\nBackward pass:")
    print("dY_pred NumPy:")
    print(dY_pred_np)
    print("dY_pred Torch:")
    print(dY_pred_torch)
    print(f"max diff: {np.max(np.abs(dY_pred_np - dY_pred_torch)):.2e}")


def test_mse_loss():
    print("\n\nTEST MSELoss")

    test_cases = [
        (
            np.array([
                [0.6, 0.2, 0.1, 0.1],    # почти правильно: класс 0
                [0.1, 0.7, 0.1, 0.1],    # почти правильно: класс 1
                [0.25, 0.25, 0.25, 0.25] # равномерно, цель класс 2
            ], dtype=np.float32),
            np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ], dtype=np.float32),
            "case 1"
        ),
        (
            np.array([
                [0.9, 0.1, 0.0, 0.0],    # почти идеальный one-hot
                [0.0, 0.0, 1.0, 0.0],    # точно one-hot
            ], dtype=np.float32),
            np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ], dtype=np.float32),
            "case 2"
        ),
    ]

    for Y_pred_np, Y_true_np, desc in test_cases:
        _run_mse_test(Y_pred_np, Y_true_np, desc)
