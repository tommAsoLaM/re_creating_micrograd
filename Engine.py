import math

class Value:

    def __init__(self, data:float, _op:str = '', label:str = '', children = ())->None:
        self.data = data
        self.grad = 0.0
        self._op = _op
        self.label = label
        self._prev = set(children)
        self._backward = lambda : None
        # self.requireGrad = True

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value (self.data + other.data, _op = '+', children = (self, other))
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _op = '*', children  = (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        out = Value(data = (math.exp(2*x) - 1)/(math.exp(2*x) + 1), children=(self, ), _op = 'tanh')
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        x = self.data
        out = Value (data = (1)/(1 + math.exp(-x)), _op = 'sigmoid', children= (self, ))
        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad
        out._backward = _backward
        return out

    
    def __repr__(self):
        if self.label != '':
            return f'{self.label}, Value.data = {self.data}'
        else:
            return f'Value.data = {self.data}'

    def __str__(self):
        if self.label != '':
            return f'{self.label}, Value.data = {self.data}'
        else:
            return f'Value.data = {self.data}'
        

    #we need some code to be able to calculate the gradients automaticcaly

    def backward(self):
        top_ordered = []
        visited = set()
        def buildTopo(val):
            if val not in visited:
                visited.add(val)
                for child in val._prev:
                    buildTopo(child)
                top_ordered.append(val)
        
        buildTopo(self)
        self.grad = 1.0
        for cell in reversed(top_ordered):
            cell._backward()
                    