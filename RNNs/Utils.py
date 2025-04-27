import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import random
import sys
import time

# --- Utility Functions ---
def softmax(z):
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e_z / np.sum(e_z, axis=0, keepdims=True)

def tanh(a):
    return np.tanh(a)

def xavier_init(n_out, n_in):
    """ A common implementation of Xavier/Glorot initialization """
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_out, n_in))

# --- Word Data Preparation ---
def prepare_word_data_onehot(text, vocab_threshold=2, sequence_len=None):
    """
    Prepares word-level data: tokenization, vocab, one-hot sequences.

    Returns:
        tuple: (X_train_onehot, Y_train_onehot, word_to_ix, ix_to_word, vocab_size)
    """
    # --- Step 1: Tokenization ---
    print("Preparing word data (one-hot inputs)...")
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    print(f"Total words found: {len(words)}")

    # --- Step 2: Build Vocabulary ---
    word_counts = Counter(words) # Count the occurence of each word

    # If the number of occurences of a word does not satify the threshold, remove (Potential rare words)
    vocab = [word for word, count in word_counts.items() if count >= vocab_threshold] 

    # Represent any word encountered later that is not in our filtered vocabulary.
    vocab.append('<UNK>')
    vocab_size = len(vocab)
    print(f"Vocabulary size (threshold>={vocab_threshold}): {vocab_size}")

    # --- Step 3: Create Word-to-Index Mappings ---
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}
    unk_ix = word_to_ix['<UNK>']

    # --- Step 4: Convert Original Word Sequence to Index Sequence ---
    all_indices = [word_to_ix.get(word, unk_ix) for word in words]
    if not all_indices: return [], [], {}, {}, 0

    X_all_indices = all_indices[:-1]
    Y_all_indices = all_indices[1:]

    # --- Step 5: Create One-Hot Vectors (Or Sequences) ---
    X_train_onehot = []
    Y_train_onehot = []

    # Helper function to convert index to one-hot vector
    def to_one_hot(index, size):
        vec = np.zeros((size, 1))
        vec[index] = 1
        return vec

    if sequence_len:
        num_sequences = len(X_all_indices) // sequence_len
        print(f"Creating {num_sequences} sequences of length {sequence_len}")
        for i in range(num_sequences):
            start = i * sequence_len
            end = start + sequence_len

            x_idx_seq = X_all_indices[start:end]
            y_idx_seq = Y_all_indices[start:end]

            x_onehot_seq = [to_one_hot(idx, vocab_size) for idx in x_idx_seq]
            y_onehot_seq = [to_one_hot(idx, vocab_size) for idx in y_idx_seq]

            X_train_onehot.append(x_onehot_seq)
            Y_train_onehot.append(y_onehot_seq)
    else:
        print("Using entire text as one sequence.")
        x_onehot_seq = [to_one_hot(idx, vocab_size) for idx in X_all_indices]
        y_onehot_seq = [to_one_hot(idx, vocab_size) for idx in Y_all_indices]
        X_train_onehot.append(x_onehot_seq)
        Y_train_onehot.append(y_onehot_seq)

    print(f"Prepared {len(X_train_onehot)} training sequence(s).")
    return X_train_onehot, Y_train_onehot, word_to_ix, ix_to_word, vocab_size

class DataGenerator:
    """
    A class for generating input and output examples for a character-level language model.
    """
    
    def __init__(self, path):
        """
        Initializes a DataGenerator object.

        Args:
            path (str): The path to the text file containing the training data.
        """
        self.path = path
        
        # Read in data from file and convert to lowercase
        with open(path) as f:
            data = f.read().lower()
        
        # Create list of unique characters in the data
        self.chars = list(set(data))
        
        # Create dictionaries mapping characters to and from their index in the list of unique characters
        self.char_to_idx = {ch: i for (i, ch) in enumerate(self.chars)}
        self.idx_to_char = {i: ch for (i, ch) in enumerate(self.chars)}
        
        # Set the size of the vocabulary (i.e. number of unique characters)
        self.vocab_size = len(self.chars)
        
        # Read in examples from file and convert to lowercase, removing leading/trailing white space
        with open(path) as f:
            examples = f.readlines()
        self.examples = [x.lower().strip() for x in examples]
 
    def generate_example(self, idx):
        """
        Generates an input/output example for the language model based on the given index.

        Args:
            idx (int): The index of the example to generate.

        Returns:
            A tuple containing the input and output arrays for the example.
        """
        example_chars = self.examples[idx]
        
        # Convert the characters in the example to their corresponding indices in the list of unique characters
        example_char_idx = [self.char_to_idx[char] for char in example_chars]
        
        # Add newline character as the first character in the input array, and as the last character in the output array
        X = [self.char_to_idx['\n']] + example_char_idx
        Y = example_char_idx + [self.char_to_idx['\n']]
        
        return np.array(X), np.array(Y)
