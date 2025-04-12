import numpy as np


class Add:
    def forward(self, a, b):
        from torchok.tensor import Tensor
        if isinstance(b, (int, float)): b = Tensor(b)
        self.a = a
        self.b = b
        self.out = Tensor(a.items + b.items)
        if a.requires_grad or b.requires_grad:
            self.out.prev = (a, b)
            self.out.function = self
            self.out.requires_grad = True
        return self.out
    
    def backward(self):
        if self.a.requires_grad:
            self.a.grad += self.out.grad
        if self.b.requires_grad:
            grad_b = self.out.grad
            while grad_b.ndim > self.b.grad.ndim:
                grad_b = grad_b.sum(axis=0)
            self.b.grad += grad_b
    
    def __repr__(self):
        return "Add"


class Mul:
    def forward(self, a, b):
        from torchok.tensor import Tensor
        if isinstance(b, (int, float)): b = Tensor(b)
        self.out = Tensor(a.items * b.items)
        if a.requires_grad or b.requires_grad:
            self.out.prev = (a, b)
            self.out.function = self
            self.out.requires_grad = True
        return self.out
    
    def backward(self):
        a, b = self.out.prev
        if a.requires_grad:
            a.grad += b.items * self.out.grad
        if b.requires_grad:
            b.grad += a.items * self.out.grad

    def __repr__(self):
        return "Mul"


class Sub:
    def forward(self, a, b):
        from torchok.tensor import Tensor
        if isinstance(b, (int, float)): b = Tensor(b)
        self.out = Tensor(a.items - b.items)
        if a.requires_grad or b.requires_grad:
            self.out.prev = (a, b)
            self.out.function = self
            self.out.requires_grad = True
        return self.out
    
    def backward(self):
        a, b = self.out.prev
        if a.requires_grad:
            a.grad += self.out.grad
        if b.requires_grad:
            b.grad -= self.out.grad

    def __repr__(self):
        return "Sub"


class Pow:
    def forward(self, a, b: int):
        from torchok.tensor import Tensor
        self.exp = b
        self.a = a
        self.out = Tensor(a.items ** b)
        if a.requires_grad:
            self.out.prev = (a,)
            self.out.function = self
            self.out.requires_grad = True
        return self.out

    def backward(self):
        if self.a.requires_grad:
            grad = self.exp * (self.a.items ** (self.exp - 1)) * self.out.grad
            self.a.grad += grad

    def __repr__(self):
        return "Pow"


class Div:
    def forward(self, a, b):
        from torchok.tensor import Tensor
        if isinstance(b, (int, float)): b = Tensor(b)
        self.out = Tensor(a.items / b.items)
        if a.requires_grad or b.requires_grad:
            self.out.prev = (a, b)
            self.out.function = self
            self.out.requires_grad = True
        return self.out
    
    def backward(self):
        a, b = self.out.prev
        grad = self.out.grad
        if a.requires_grad:
            a.grad += grad / b.items
        if b.requires_grad:
            b.grad -= (a.items / (b.items ** 2)) * grad

    def __repr__(self):
        return "Div"


class Matmul:
    def forward(self, a, b):
        from torchok.tensor import Tensor
        self.out = Tensor(a.items @ b.items)
        if a.requires_grad or b.requires_grad:
            self.out.prev = (a, b)
            self.out.function = self
            self.out.requires_grad = True
        return self.out
    
    def backward(self):
        a, b = self.out.prev
        grad = self.out.grad
        if a.requires_grad:
            a.grad += grad @ b.items.T
        if b.requires_grad:
            b.grad += a.items.T @ grad

    def __repr__(self):
        return "MatMul"


class Sum:
    def forward(self, a):
        from torchok.tensor import Tensor
        self.a = a
        self.out = Tensor(np.array(a.items.sum()))
        if a.requires_grad:
            self.out.prev = (a,)
            self.out.function = self
            self.out.requires_grad = True
        return self.out

    def backward(self):
        if self.a.requires_grad:
            self.a.grad += np.ones_like(self.a.items) * self.out.grad

    def __repr__(self):
        return "Sum"
    

class Log:
    def forward(self, a):
        from torchok.tensor import Tensor
        self.a = a
        self.out = Tensor(np.log(a.items))
        if a.requires_grad:
            self.out.prev = (a,)
            self.out.function = self
            self.out.requires_grad = True
        return self.out

    def backward(self):
        if self.a.requires_grad:
            self.a.grad += (1 / self.a.items) * self.out.grad

    def __repr__(self):
        return "Sum"
    

# Activation Functions
class ReLU:
    def forward(self, a):
        from torchok.tensor import Tensor
        self.a = a
        self.out = Tensor(np.maximum(a.items, 0))
        if a.requires_grad:
            self.out.prev = (a,)
            self.out.function = self
            self.out.requires_grad = True
        return self.out

    def backward(self):
        if self.a.requires_grad:
            grad = (self.a.items > 0) * self.out.grad
            self.a.grad += grad

    def __repr__(self):
        return "ReLU"
    

class LReLU:
    def forward(self, a):
        from torchok.tensor import Tensor
        self.a = a
        self.out = Tensor(np.where(a.items > 0, a.items, a.items * 0.01))
        if a.requires_grad:
            self.out.prev = (a,)
            self.out.function = self
            self.out.requires_grad = True
        return self.out

    def backward(self):
        if self.a.requires_grad:
            grad = np.where(self.a.items > 0, 1.0, 0.01) * self.out.grad
            self.a.grad += grad

    def __repr__(self):
        return "LeakyReLU"
    

class Sigmoid:
    def forward(self, a):
        from torchok.tensor import Tensor
        self.a = a
        self.out = Tensor(1 / (1 + np.exp(-a.items)))
        if a.requires_grad:
            self.out.prev = (a,)
            self.out.function = self
            self.out.requires_grad = True
        return self.out

    def backward(self):
        if self.a.requires_grad:
            grad = (self.out.items * (1 - self.out.items)) * self.out.grad
            self.a.grad += grad

    def __repr__(self):
        return "Sigmoid"


class Tanh:
    def forward(self, a):
        from torchok.tensor import Tensor
        self.a = a
        self.out = Tensor((np.exp(a.items) - np.exp(-a.items)) / (np.exp(a.items) + np.exp(-a.items)))
        if a.requires_grad:
            self.out.prev = (a,)
            self.out.function = self
            self.out.requires_grad = True
        return self.out

    def backward(self):
        if self.a.requires_grad:
            grad = (1 - self.out.items**2) * self.out.grad
            self.a.grad += grad

    def __repr__(self):
        return "Tanh"
    

class Softmax:
    def forward(self, a):
        from torchok.tensor import Tensor
        self.a = a
        exp_shifted = np.exp(a.items - np.max(a.items, axis=1, keepdims=True))
        softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        self.out = Tensor(softmax)
        if a.requires_grad:
            self.out.prev = (a,)
            self.out.function = self
            self.out.requires_grad = True
        return self.out

    def backward(self):
        if self.a.requires_grad:
            grad = np.zeros_like(self.a.items)

            for i in range(len(self.out.items)):  # for each sample
                s = self.out.items[i].reshape(-1, 1)
                jacobian = np.diagflat(s) - s @ s.T
                grad[i] = jacobian @ self.out.grad[i]

            self.a.grad += grad

    def __repr__(self):
        return "Softmax"
