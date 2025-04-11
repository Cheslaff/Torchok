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