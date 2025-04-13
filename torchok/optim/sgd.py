import numpy as np


class SGD:
    def __init__(self, params, lr=0.0001, momentum=0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.items, dtype=np.float64) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.items, dtype=np.float64)
    
    def step(self):
        for i, p in enumerate(self.params):
            velocity = self.velocities[i]
            velocity = velocity * self.momentum
            velocity = velocity + p.grad
            p.items = p.items - self.lr * velocity
