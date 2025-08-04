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
    
    def __sub__(self,other):
        out = self + (-other)
        return out
    
    def __neg__(self):
        return self * -1

    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _op = '*', children = (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        #we are talking about e^x
        out = Value (data = math.exp(self.data), _op = 'exp', children=(self, ))
        def _backward():
            self.grad = self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        if not isinstance(exponent, (int,float)):
            raise ("exponent must be integer or float")
        out = Value(self.data**exponent, _op = 'exp', children = (self, ))
        def _backward():
            self.grad = exponent * self.data**(exponent-1) * out.grad
        out._backward = _backward
        return out
        
    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        x = self.data
        out = Value(data = (math.exp(2*x) - 1)/(math.exp(2*x) + 1), children = (self, ), _op = 'tanh')
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        x = self.data
        out = Value (data = (1)/(1 + math.exp(-x)), _op = 'sigmoid', children = (self, ))
        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value (data = max(self.data, 0), _op = 'relu', children = (self, ))
        def _backward():
            self.grad += (1 if self.data>=0 else 0) * out.grad
        out._backward = _backward
        return out
    
    def leaky_relu(self):
        alfa = 0.01
        out = Value(data = max(self.data, alfa * self.data), _op = 'leaky_relu', children = (self, ))
        def _backward():
            self.grad = (1 if self.data>= 0 else alfa) * out.grad
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
                    