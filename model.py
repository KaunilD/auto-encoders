import numpy as np
from chainer import Chain
from chainer import links as L
from chainer import functions as F
class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
           l1 = L.Linear(784, 260),
           l2 = L.Linear(260, 90),
           l3 = L.Linear(90, 10),
           l4 = L.Linear(10, 90),
           l5 = L.Linear(90, 260),
           l6 = L.Linear(260, 784),
        )
        self.train = True

    def __call__(self, x, t):
       # Forward pass
       h = self.l1(x)
       h = self.l2(h)
       h = self.l3(h)
       h = self.l4(h)
       h = self.l5(h)
       y = self.l6(h)
       if self.train:
           self.loss = F.mean_squared_error(y, t)
           return self.loss
       else:
           return y
