from torchok.tensor import Tensor
import numpy as np
import matplotlib.pyplot as plt


X = Tensor(np.random.randn(10_000, 1), name="x")
W1 = Tensor(np.random.randn(1, 30), name="w1") / 1**0.5
b1 = Tensor(np.random.randn(30), name="b1") * 0.1
W2 = Tensor(np.random.randn(30, 1), name="w2") / 30**0.5
b2 = Tensor(np.random.randn(1), name="b2") * 0.1
y = X * 2 + 3

parameters = [W1, b1, W2, b2]

for parameter in parameters:
    parameter.requires_grad = True


lr = 0.000001

for epoch in range(50):
    h = X @ W1 + b1
    h.name = "h"
    out = h @ W2 + b2
    out.name = "out"
    loss = (out - y) ** 2
    loss.name = "Loss"
    for parameter in parameters:
        parameter.grad = np.zeros_like(parameter.items, dtype=np.float64)
    loss.backward()
    for parameter in parameters:
        parameter.items += -lr * parameter.grad
    print(loss.items.mean())

plt.scatter(X.items, y.items)
y_hat = (X @ W1 + b1) @ W2 + b2
plt.plot(X.items, y_hat.items, c="red")
plt.show()
