import numpy as np

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr # learning rate

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

    
class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum # 알파값
        self.v = None 
    def update(self,params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lf=0.01):
        self.lr = lf
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.item():
                self.h[key] = np.zero_like(val)
            for key in params.key():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + 1e-7))
                