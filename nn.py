from Engine import Value
import numpy as np

class Neuron:
    def __init__(self, nin: int, initilization:str):
        self.weights = [Value(np.random.random() for _ in range (nin))]
        self.bias = Value(np.random.random())
        
    def __call__(self, x:list):
        return Value(sum((wi.data*xi for wi, xi in zip(self.weights, x))))