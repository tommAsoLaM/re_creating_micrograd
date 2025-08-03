from re import match
from Engine import Value
import numpy as np

def softMax(x):
    return x
class Neuron:
    def __init__(self, nin: int, initilization:str = '', dropout:bool = False):
        
        self.bias = Value(np.random.random())
        match initilization:
            case '':
                self.weights = [Value(np.random.random()) for _ in range (nin)]
            case 'xavier':
                pass
        #the dropout case will be handled afterwads
        
    def __call__(self, x:np.ndarray):
        return Value(sum((wi.data*xi for wi, xi in zip(self.weights, x)), self.bias))
    
class Layer:
    def __init__(self, nin, nout, initilization:str = ''):
        self.neurons = [Neuron(nin,initilization) for _ in range(nout)]

    def __call__(self, x:np.ndarray):
        if x.size != len(self.neurons):
            raise("Imcompatible sizes of inputs and neuron number")
        result = [neuron(x) for neuron in self.neurons]
        return result[0] if len(result) == 1 else result
    
class MLP:
    def __init__(self, numNeurons:list, initilization:str = ''):
        #the input will be a list of at least two numbers (number of neurons in the input layer
        #and number of neurons in the output layer)
        self.layers=[]
        for i in range (len(numNeurons)-1):
            self.layers.append(Layer(numNeurons[i], numNeurons[i+1], initilization))
    
    def __call__(self,x):
        for i in range (len(self.layers)):
            output_layer = self.layers[i](x)
            x = output_layer
        return output_layer[0] if len(output_layer) == 1 else output_layer