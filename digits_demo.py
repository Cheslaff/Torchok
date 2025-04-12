from torchok import Tensor
from torchok.nn import Linear, ReLU, Softmax, CrossEntropyLoss

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np


X, y_ = load_digits(return_X_y=True)
X = X.reshape(X.shape[0], 64) / 16.0
X = Tensor(X)
y = np.zeros((y_.shape[0], 10))
y[np.arange(y.shape[0]), y_] =1
y = Tensor(y)


lr = 0.0003
loss_fn = CrossEntropyLoss()
net = [
    Linear(64, 30),
    ReLU(),
    Linear(30, 30),
    ReLU(),
    Linear(30, 10),
    Softmax()
]
parameters = []
for layer in net:
	parameters.extend(layer._parameters)


for epoch in range(1_000):
	h = X
	for layer in net:
		h = layer(h)
	
	loss = loss_fn(h, y)
	for parameter in parameters:
		parameter.grad = np.zeros_like(parameter.items, dtype=np.float64)
	loss.backward()
	for parameter in parameters:
		parameter.items += -lr * parameter.grad
	print(loss.items.mean())


predictions = []

X_testing = X[:5]

out = X_testing
for layer in net:
	out = layer(out)

for probas in out.items:
	predictions.append(probas.argmax())

for i in range(5):
	plt.title(f"Actual: {y_[i]} Prediction: {predictions[i]}")
	plt.imshow(X_testing.items[i].reshape(8, 8), cmap="gray")
	plt.show()
