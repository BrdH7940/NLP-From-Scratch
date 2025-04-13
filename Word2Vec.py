import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def initialize_neural_network(vocab_size: int, embedding_dimension: int = 10):
    return {
        'W1': np.random.randn(embedding_dimension, vocab_size),
        'W2': np.random.randn(vocab_size, embedding_dimension)
    }

def softmax(X):
    score = np.exp(X - np.max(X, axis = -1, keepdims = True))
    pred = score / np.sum(score, axis = -1, keepdims = True)
    return pred

def cross_entropy(z, y, N):
    return -np.sum(np.log(z) * y) / N

def forward(model: dict, X):
    cache = dict()
    
    cache['H'] = model['W1'] @ X.T
    cache['U'] = (model['W2'] @ cache['H']).T
    cache['P'] = softmax(cache['U'])

    return cache

def backward(model: dict, X, y, cache: dict, alpha: int = 0.001):
    N = X.shape[0]
    E = cache['P'] - y

    dw2 = (E.T @ cache['H'].T) / N
    dw1 = model['W2'].T @ E.T @ X

    model['W1'] -= alpha * dw1
    model['W2'] -= alpha * dw2

    return cross_entropy(cache['P'], y, N)