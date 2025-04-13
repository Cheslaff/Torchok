from torchok import Tensor
from torchok.nn import Linear, ReLU, CEWithLogitsLoss, Module, BatchNorm1d
from torchok.optim import Adam, RMSprop, SGD

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np


X, y_ = load_digits(return_X_y=True)
X = X.reshape(X.shape[0], 64) / 16.0
X = Tensor(X)
y = np.zeros((y_.shape[0], 10))
y[np.arange(y.shape[0]), y_] =1
y = Tensor(y)


loss_fn = CEWithLogitsLoss()

class Net(Module):
	def __init__(self):
		super().__init__()
		self.l1 = Linear(64, 30)
		self.b1 = BatchNorm1d(30)
		self.l2 = ReLU()
		self.l3 = Linear(30, 30)
		self.b2 = BatchNorm1d(30)
		self.l4 = ReLU()
		self.l5 = Linear(30, 10)
	
	def forward(self, x):
		return self.l5(self.l4(self.b2(self.l3(self.l2(self.b1(self.l1(x)))))))

net = Net()
net.train()
optimizer = Adam(params=net.parameters(), lr=0.001)

for epoch in range(15_000):
	batch = np.random.randint(0, X.shape[0], size=(32,))
	X_batch = X[batch]
	y_batch = y[batch]
	
	out = net(X_batch)
	loss = loss_fn(out, y_batch)
	# optimization!
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	print(loss.items.mean())


net.eval()
predictions = []
X_testing = X[:5]
out = net(X_testing)

out = out.softmax()

for probas in out.items:
	predictions.append(probas.argmax())
for i in range(5):
	plt.title(f"Actual: {y_[i]} Prediction: {predictions[i]}")
	plt.imshow(X_testing.items[i].reshape(8, 8), cmap="gray")
	plt.show()
