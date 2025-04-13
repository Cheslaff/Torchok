<h1  align="center">Torchok</h1>
<h3  align="center">Development Logs</h3>
<p  align="center"><img  src="https://media.tenor.com/ZC3CDD-vvyEAAAAj/minecraft-lantern.gif" width=20%></p>

## 11.04.2025:
### Overview
Full rewrite of Tensor class, change in design philosophy, functional **autogradik** engine for autodifferentiation (yet not tested and may yield errors). Finally Functioning (Built Linear Regression model with it)
### Tensor
Oh boy... Tensor class was too problematic with centralized operations method and datatype diversity focus. Now `torchok.Tensor` is functioning correctly keeping track of gradients, previous Tensors in graph & calling operations from `torchok.autogradik.functions`<br>
API example:
```python
X  =  Tensor(np.random.randn(10_000, 1), name="x")
W1  =  Tensor(np.random.randn(1, 30), name="w1") /  1**0.5
b1  =  Tensor(np.random.randn(30), name="b1") *  0.1
W2  =  Tensor(np.random.randn(30, 1), name="w2") /  30**0.5
b2  =  Tensor(np.random.randn(1), name="b2") *  0.1
y  =  X  *  2  +  3
# Operations:
h  =  X  @  W1  +  b1
h.name  =  "h"
out  =  h  @  W2  +  b2
out.name =  "out"
loss  = (out  -  y) **  2
```
**Plan: Add nonlinearities**
### Autogradik/functions
Heart of automatic differentiation is finally working! Autogradik/functions includes all basic tensor operations with `forward` and `backward` methods. It finally makes autodiff possible.<br>
Example of `Mul` operation:
```python
class  Mul:
	def  forward(self, a, b):
		from  torchok.tensor  import  Tensor
		if  isinstance(b, (int, float)): b  =  Tensor(b)
			self.out  =  Tensor(a.items *  b.items)
		if  a.requires_grad or  b.requires_grad:
			self.out.prev  = (a, b)
			self.out.function  =  self
			self.out.requires_grad  =  True
		return  self.out

	def  backward(self):
		a, b  =  self.out.prev
		if  a.requires_grad:
			a.grad +=  b.items *  self.out.grad
		if  b.requires_grad:
			b.grad +=  a.items *  self.out.grad
			
	def  __repr__(self):
		return  "Mul"
```
### Example usage
```python
from  torchok.tensor  import  Tensor
import  numpy  as  np
import  matplotlib.pyplot  as  plt

X  =  Tensor(np.random.randn(10_000, 1), name="x")
W1  =  Tensor(np.random.randn(1, 30), name="w1") /  1**0.5
b1  =  Tensor(np.random.randn(30), name="b1") *  0.1
W2  =  Tensor(np.random.randn(30, 1), name="w2") /  30**0.5
b2  =  Tensor(np.random.randn(1), name="b2") *  0.1
y  =  X  *  2  +  3

parameters  = [W1, b1, W2, b2]

for  parameter  in  parameters:
	parameter.requires_grad =  True

lr  =  0.000001

for  epoch  in  range(50):
	h  =  X  @  W1  +  b1
	h.name  =  "h"
	out  =  h  @  W2  +  b2
	out.name =  "out"
	loss  = (out  -  y) **  2
	loss.name =  "Loss"
	for  parameter  in  parameters:
		parameter.grad =  np.zeros_like(parameter.items, dtype=np.float64)
	loss.backward()
	for  parameter  in  parameters:
		parameter.items +=  -lr  *  parameter.grad  # update
	print(loss.items.mean())

plt.scatter(X.items, y.items)
y_hat  = (X  @  W1  +  b1) @  W2  +  b2
plt.plot(X.items, y_hat.items, c="red")
plt.show()
```
Result:<br>
<img src="https://i.ibb.co/fVKKx56j/2025-04-11-185732.png" width=50%>
### Final Thoughts
Well... It was tough. 8 hours of non-stop programming, debugging and rick and morty playing in the background.<br>
Definitely worth it!<br>

## 12.04.2025:
### Overview
Non-linearities, `nn` lib, layers, loss functions and more!
### Non-linearities
Simple Non-linearities added to `autogradik.functions` and to `torchok.Tensor`.<br>
### `nn`
Neural Network library prototype. Module class in `nn.module`, layers `Linear`, `ReLU`, `Tanh`, `Sigmoid`, `Softmax`, `LReLU`.
```python
class Linear(nn.Module):
    def __init__(self, fan_in, fan_out, bias=True):
        super().__init__()
        self.W = torchok.randn(fan_in, fan_out, requires_grad=True) / (fan_in)**0.5
        self.b = torchok.randn(fan_out, requires_grad=True) * 0.1 if bias else None

    def forward(self, x):
        out = x @ self.W
        if self.b is not None:
            out += self.b
        return out
```
### Example usage now
```python
import torchok
from torchok.nn import Linear, ReLU, MSELoss
import matplotlib.pyplot as plt
import numpy as np

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

```
Result:<br>
<img src="https://i.ibb.co/ZpkmL7Vb/2025-04-12-133427.png" width=50%>

### Demo Digits
Succesfully works on `sklearn.load_digits()`.
See `digits_demo.py`<br>
<img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_lle_digits_thumb.png" widht=30%>

### Final Thoughts
It was just great!<br>

## 13.04.2025:
### Overview
Optimizers, BatchNorm, Module updates, Functions improved!
### `torchok.optim`
New branch of torchok project! `torchok.optim` finally delivered!<br>
Torchok is a small scale project, so it has only 3 essential optimizers:<br>

- `SGD` supports momentum; depends on batch size
- `RMSprop` smart learning rate scaling
- `Adam` mix of `RMSprop` and momentum `SGD`

Now instead of suffering, optimizing parameters by hand just call `optimizer.step()` (But don't forget to `optimizer.zero_grad()` before `loss.backward()`😆)

Example training loop:
```python
net = Net()  # inherits nn.Module
net.train()  # Model state controllers
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
```

### Batch Norm 1d
New layer for Batch normalization with automatic mean and std running average waiting for you in `torchok.nn.layers.py`.<br>
Network interspersed with BN1d layers:
```python
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
```
### `Module` tweaks
Now module has `.train()` and `.eval()` states.<br>
It affects `requires_grad` of parameters and `training` attribute of submodules (if one exists).<br>
Frankly speaking, it's a bit wrong, since `.eval()` should change only `training` state of submodules not touching `requires_grad`<br>

```python
net.eval()  # affects Batch Normalization
predictions = []
X_testing = X[:5]
out = net(X_testing)

out = out.softmax()
```

### `autogradik.functions` fixed
Now operators in autogradik resolve all sorts of shape mismatches becoming more robust.

## Attention!❤️‍🔥
With 660 lines of python and with only one dependency (numpy) **Torchok V1.0** releases!<br>
V1.0 covers minimal neural network pipeline with PyTorch-like syntax!<br>

Planning to drop on PyPi<br>
<img src="https://minecraft.wiki/images/Torch.gif?462d6" widht=30%>

### Final Thoughts
Unforgettable Project!<br>

---

### Why?
Torchok is an educational open-source pet project developed by [Cheslaff](https://github.com/Cheslaff).  It sets its goal to replicate pytorch-like API in the simplest possible way. It ommits complex abstractions leaving only necessary API components.<br>
Free to use, copy, steal, whatever.<br>
<p  align="center"><img src="https://media.tenor.com/Od2m5oBJlPkAAAAi/tf2-pyro.gif" width=10%></p>
