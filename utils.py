import numpy as np
from Engine import Value
import math

def loss(ypreds:list, ytrue:list, criterion:str = 'sse'):
    # y_preds is a list of probabilities,it means I have to sum this for all numbers
    # practically speaking, for each prediction we have n values (n number of classes)
    if len(ypreds)!=len(ytrue):
        raise TypeError ("size of vectors must be the same")
    loss = 0
    if criterion == 'sse':
        for y_p, y_t in zip(ypreds, ytrue):
            loss = loss + (y_p - y_t)**2
    if criterion == 'mse':
        for y_p, y_t in zip(ypreds, ytrue):
            loss += (y_p - y_t)**2
        loss = loss/len(ypreds)

    if criterion == 'cross':
        loss = -sum(y_t * (y_p.log()) for y_t, y_p in zip(ytrue, ypreds))
    return loss

def softMax(x:list):
    tot = sum(xi.exp() for xi in x)
    out = [xi.exp()/(tot) for xi in x]
    return out
    
