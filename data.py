import numpy as np

class Data:
    num_inputs = None
    ds_size = None

    def __init__(self, _numinputs, _size):
        self.num_inputs = _numinputs
        self.ds_size = _size

    def prepare(self):
        return np.random.rand(self.ds_size, self.num_inputs ).astype(np.float32)
