import torchok
from torchok import Tensor
from torchok.nn import Linear, ReLU, Softmax, CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np

# Regression Task

# Data Creation
X  =  torchok.randn(1_000, 1)
y = 0.1 * X**7 - 0.5 * X**6 + 3 * X**5 - 5 * X**4 + 0.3 * X**3 + 2 * X**2 - 4 * X + 10

# Preparation
lr  =  0.000001
loss_fn = MSELoss()
layers = [
	Linear(1, 30),
	ReLU(),
	Linear(30, 1)
]
parameters = []
for layer in layers:
	parameters.extend(layer._parameters)

# Training
for  epoch  in  range(10_000):
	h = X
	for layer in layers:
		h = layer(h)
	
	loss  = loss_fn(h, y)
	# Optimization
	for  parameter  in  parameters:
		parameter.grad =  np.zeros_like(parameter.items, dtype=np.float64)
	loss.backward()
	for  parameter  in  parameters:
		parameter.items +=  -lr  *  parameter.grad  # update
	print(loss.items.mean())


# Prediction
y_hat = X
for layer in layers:
	y_hat = layer(y_hat)

# Plot result
x_sorted = X.items.flatten()
y_hat_sorted = y_hat.items.flatten()
idx = np.argsort(x_sorted)

plt.scatter(X.items, y.items)
plt.plot(x_sorted[idx], y_hat_sorted[idx], c="red")
plt.show()
