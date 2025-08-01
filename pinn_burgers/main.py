from model import FCN
from data import get_training_data
from train import trainPinn
from classic_nn import n_layer_nn
import torch


if __name__ == "__main__":
    layers = [2, 20, 20, 20, 20, 1]
    model = FCN(layers).cpu()

    X_u_train, u_train, X_f_train = get_training_data()
    X_np = X_u_train.cpu().numpy()
    u_np = u_train.cpu().numpy()

    W, b, y_pred, loss = n_layer_nn(X_np, u_np, N=4, lr=0.01, epochs=10000, hidden_size=20)
    print(f"Loss with backprop with X_u and u: {loss:.5e}\n")

    l2 = trainPinn(model, X_u_train, u_train, X_f_train, nu=0.01 / torch.pi, epochs=10000)
    print(f"Loss with PINN: {l2.item():.5e}")


    #plot
    import numpy as np
    import matplotlib.pyplot as plt
    x_test = np.linspace(-1, 1, 200).reshape(-1, 1)
    t_val = 0.25 * np.ones_like(x_test)

    X_test = np.hstack([t_val, x_test])

    def forward_nn(X, W, b):
        A = [X]
        for i in range(len(W) - 1):
            A.append(np.tanh(A[-1] @ W[i] + b[i]))
        return A[-1] @ W[-1] + b[-1]

    u_classic = forward_nn(X_test, W, b)

    with torch.no_grad():
        X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(next(model.parameters()).device)
        u_pinn = model(X_test_torch).cpu().numpy()

    from burger_solution import exact_burgers_solution
    u_exact = exact_burgers_solution(x_test[:, 0], t_val[0, 0])

    plt.figure(figsize=(8, 5))
    plt.plot(x_test, u_pinn, label='PINN', linewidth=2)
    plt.plot(x_test, u_classic, '--', label='Classic NN', linewidth=2)
    plt.plot(x_test, u_exact, ':', label='Exact', linewidth=2)
    plt.title("u(t=0.25, x): PINN vs Classic NN")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    plt.show()
