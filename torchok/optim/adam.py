import numpy as np

class Adam:
    def __init__(self, params, lr=0.0001, betas=(0.9, 0.999)):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.m = [np.zeros_like(p.items, dtype=np.float64) for p in self.params]
        self.v = [np.zeros_like(p.items, dtype=np.float64) for p in self.params]
        self.eps = 1e-5
        self.t = 1

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.items, dtype=np.float64)
    
    def step(self):
        for i, p in enumerate(self.params):
            # Moving avg
            m_i = self.m[i]
            m_i = m_i * self.betas[0]
            m_i = m_i + (1 - self.betas[0]) * p.grad

            v_i = self.v[i]
            v_i = v_i * self.betas[1]
            v_i = v_i + (1 - self.betas[1]) * p.grad**2

            # Bias Correction
            mc_i = m_i / (1 - self.betas[0]**self.t)
            vc_i = v_i / (1 - self.betas[1]**self.t)

            # Update
            p.items = p.items -self.lr / (vc_i**0.5 + self.eps) * mc_i
        self.t = self.t + 1  # Set it to next step