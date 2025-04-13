import numpy as np

class RMSprop:
    def __init__(self, params, lr=0.0001, alpha=0.9):
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.velocities = [np.zeros_like(p.items, dtype=np.float64) for p in self.params]
        self.eps = 1e-5

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.items, dtype=np.float64)
    
    def step(self):
        for i, p in enumerate(self.params):
            velocity = self.velocities[i]
            velocity *= self.alpha
            velocity += (1 - self.alpha) * p.grad**2
            p.items += -self.lr / (velocity**0.5 + self.eps) * p.grad