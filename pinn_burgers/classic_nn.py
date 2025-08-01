import numpy as np

def tanh(z):
    return np.tanh(z)

def tanh_d(z):
    return 1 - np.tanh(z) ** 2

def n_layer_nn(x, y, N, lr=0.01, epochs=5000, hidden_size=10):
    m, d_in = x.shape
    d_out = y.shape[1]

    sizes = [d_in] + [hidden_size] * (N - 2) + [d_out]
    W = [np.random.randn(sizes[i], sizes[i + 1]) * np.sqrt(1 / sizes[i])
         for i in range(len(sizes) - 1)]
    b = [np.zeros((1, sizes[i + 1])) for i in range(len(sizes) - 1)]

    for _ in range(epochs):
        Z, A = [], [x]
        for i in range(len(W) - 1):
            Z.append(A[-1] @ W[i] + b[i])
            A.append(tanh(Z[-1]))

        Z.append(A[-1] @ W[-1] + b[-1])
        y_pred = Z[-1]

        loss = np.mean((y_pred - y) ** 2)

        dL = 2 * (y_pred - y) / m
        dW, db = [None] * len(W), [None] * len(W)

        dW[-1] = A[-1].T @ dL
        db[-1] = np.sum(dL, axis=0, keepdims=True)
        dA = dL @ W[-1].T

        for i in reversed(range(len(W) - 1)):
            dZ = dA * tanh_d(Z[i])
            dW[i] = A[i].T @ dZ
            db[i] = np.sum(dZ, axis=0, keepdims=True)
            if i > 0:
                dA = dZ @ W[i].T

        for i in range(len(W)):
            W[i] -= lr * dW[i]
            b[i] -= lr * db[i]

    return W, b, y_pred, loss