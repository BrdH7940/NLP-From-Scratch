import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import List, Dict, Any, Literal

def initialize_neural_network(
    layer_dims: List[int],
    init_method: Literal['random', 'xavier', 'he'] = 'random',
    include_bias: bool = True,
    random_scale: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Initializes parameters (weights W and biases b) for a multi-layer neural network.

    Args:
        layer_dims (List[int]): A list containing the number of units in each layer,
                                starting from the input layer.
                                Example: [input_size, hidden_size_1, ..., output_size]
        init_method (Literal['random', 'xavier', 'he']): The weight initialization method.
            - 'random': Initialize weights with small random values (scaled Gaussian).
            - 'xavier': Use Xavier/Glorot initialization. Good for tanh/sigmoid activations.
            - 'he': Use He initialization. Good for ReLU activations.
            Default is 'random'.
        include_bias (bool): If True, initializes bias terms 'b' for each layer (except input).
                             If False, only initializes weight terms 'W'. Default is True.
        random_scale (float): The scaling factor for 'random' initialization.
                              Ignored if init_method is 'xavier' or 'he'. Default is 0.01.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the initialized parameters.
                               Keys are "W1", "b1", "W2", "b2", ...
                               'Wl' has shape (layer_dims[l], layer_dims[l-1])
                               'bl' has shape (layer_dims[l], 1)

    Raises:
        ValueError: If an unknown init_method is provided.
    """
    parameters = {}
    L = len(layer_dims) # Number of layers in the network (including input)

    for l in range(1, L):
        n_in = layer_dims[l-1]
        n_out = layer_dims[l]

        if init_method == 'xavier':
            parameters['W' + str(l)] = np.random.randn(n_out, n_in) * np.sqrt(2. / (n_in + n_out))
        elif init_method == 'he':
            parameters['W' + str(l)] = np.random.randn(n_out, n_in) * np.sqrt(2. / n_in)
        elif init_method == 'random':
            parameters['W' + str(l)] = np.random.randn(n_out, n_in) * random_scale
        else:
            raise ValueError(f"Unknown initialization method: {init_method}. Choose 'random', 'xavier', or 'he'.")

        if include_bias:
            parameters['b' + str(l)] = np.zeros((n_out, 1))

    return parameters

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