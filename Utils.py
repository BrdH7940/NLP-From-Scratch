import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

### ============= Word2Vec ===============

def tokenize(text: str) -> list:
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def mapping(tokens: list[str]):
    word_to_id = {}
    id_to_word = {}
    
    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token
    
    return word_to_id, id_to_word

def encode_onehot(vocab_size: int, id: int):
    res = [0] * vocab_size
    res[id] = 1
    return res

def generate_input(tokens: list[str], word_to_id: dict, window_size: int = 5):
    n = len(tokens)
    m = len(word_to_id)
    X, y = [], []
    
    for i in range(n):
        for j in range(max(0, i - window_size), min(n - 1, i + window_size) + 1):
            if i == j:
                continue
            X.append(encode_onehot(m, word_to_id[tokens[i]]))
            y.append(encode_onehot(m, word_to_id[tokens[j]]))
    
    return np.asarray(X), np.asarray(y)