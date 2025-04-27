from Utils import *

class RNN:
    def __init__(self, hidden_dim, vocab_size):
        """
        Initializes RNN for one-hot inputs.

        Args:
            hidden_dim (int): Number of units in the hidden layer (Dh).
            vocab_size (int): Size of the vocabulary (|V|).
        """
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # --- Parameters ---
        self.W_hx = xavier_init(hidden_dim, vocab_size)     # Shape Dh x |V|
        self.W_hh = xavier_init(hidden_dim, hidden_dim)     # Shape Dh x Dh
        self.W_hy = xavier_init(vocab_size, hidden_dim)     # Shape |V| x Dh
        self.b_h = np.zeros((hidden_dim, 1))                # Shape Dh x 1
        self.b_y = np.zeros((vocab_size, 1))                # Shape |V| x 1

        self.params = {
            'W_hx': self.W_hx, 'W_hh': self.W_hh,
            'W_hy': self.W_hy, 'b_h': self.b_h, 'b_y': self.b_y
        }
        
        # --- Mappings: For Sampling ---
        self.word_to_ix = None
        self.ix_to_word = None

    def forward(self, input_one_hots, h_prev):
        """
        Performs forward pass.

        Args:
            input_one_hots (list): List of one-hot vectors (shape |V| x 1), length T.
            h_prev (np.ndarray): Initial hidden state (shape Dh x 1).

        Returns:
            cache (dict): Stores 'one_hot_inputs', 'h_states', 'y_preds'.
        """
        # Cache
        X_one_hots_s, h_states, y_preds = {}, {}, {}

        h_states[-1] = np.copy(h_prev)

        for t in range(len(input_one_hots)):
            # Forward pass
            x_t = input_one_hots[t]
            h_prev_t = h_states[t - 1]
            a_t = self.W_hh @ h_prev_t + self.W_hx @ x_t + self.b_h
            h_t = tanh(a_t)
            y_pred = softmax(self.W_hy @ h_t + self.b_y)

            # Store necessary items for backward pass
            X_one_hots_s[t] = x_t
            h_states[t] = h_t
            y_preds[t] = y_pred

        cache = {'one_hot_inputs': X_one_hots_s, 'h_states': h_states, 'y_preds': y_preds}
        return cache

    def backward(self, targets, cache, clip_value=None):
        """
        Performs backward pass for one-hot input model.

        Args:
            targets (list): List of target one-hot vectors (shape |V| x 1), length T.
            cache (dict): Contains 'one_hot_inputs', 'h_states', 'y_preds'.
            clip_value (float, optional): Gradient clipping value.

        Returns:
            dict: Gradients.
        """
        # Cache Extraction
        X_one_hots = cache['one_hot_inputs']
        h_states = cache['h_states']
        y_preds = cache['y_preds']
        sequence_length = len(X_one_hots)

        # Initialization
        dW_hx = np.zeros_like(self.W_hx)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next_t = np.zeros_like(self.b_h)

        for t in reversed(range(sequence_length)):
            # --- Get values from cache ---
            x_t = X_one_hots[t]
            y_pred = y_preds[t]
            y = targets[t]
            h = h_states[t]
            h_prev = h_states[t - 1]

            # --- Gradient calculation ---
            dz = y_pred - y

            db_y += dz
            dW_hy += dz @ h.T

            h_next = h_states.get(t + 1, np.zeros_like(h))
            diag_term = np.diag(1 - h_next[:, 0] ** 2)
            dh = self.W_hy.T @ dz + self.W_hh.T @ diag_term @ dh_next_t
            da = dh * (1 - h ** 2)

            dW_hh += da @ h_prev.T
            db_h += da
            dW_hx += da @ x_t.T
            dh_next_t = dh

        # Returned updated grads
        grads = {'dW_hy': dW_hy, 'db_y': db_y, 'dW_hh': dW_hh, 'dW_hx': dW_hx, 'db_h': db_h}

        if clip_value is not None:
            for key in grads:
                np.clip(grads[key], -clip_value, clip_value, out=grads[key])

        return grads

    def train(self, X_one_hot_train, y_train, data_generator=None, lr=0.001, epochs=100, 
              clip_value=5.0, print_every=10, generated_examples=5):
        """
        Trains the RNN model using SGD with one-hot inputs and generates sample predictions
        during training to showcase the model's progress.
        """
        num_sequences = len(X_one_hot_train)
        loss_history = []
        
        # Store data_generator for sampling
        if data_generator:
            self.data_generator = data_generator
        
        print(f"Starting training (One-Hot Input) for {epochs} epochs...")
        print(f"Learning rate: {lr}, Clipping: {clip_value}")
        
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0.0
            
            for i in range(num_sequences):
                # --- Setting up variables ---
                x_seq = X_one_hot_train[i]  # List of one-hot vectors
                y_seq = y_train[i]
                sequence_length = len(x_seq)
                if sequence_length == 0: continue
                
                current_h0 = np.zeros((self.hidden_dim, 1))  # Reset per sequence
                
                # --- Forward ---
                cache = self.forward(x_seq, current_h0)
                y_preds = cache['y_preds']
                
                # --- Loss Calculation ---
                seq_loss = 0.0; epsilon = 1e-12
                for t in range(sequence_length):
                    y_pred = y_preds[t]; y = y_seq[t]
                    correct_class_prob = y_pred[np.argmax(y), 0]
                    seq_loss += -np.log(correct_class_prob + epsilon)
                avg_seq_loss = seq_loss / sequence_length
                epoch_loss += avg_seq_loss
                
                # --- Backward ---
                grads = self.backward(y_seq, cache, clip_value)
                
                # --- Parameter Update ---
                self.W_hy -= lr * grads['dW_hy']
                self.W_hh -= lr * grads['dW_hh']
                self.W_hx -= lr * grads['dW_hx']  
                self.b_h -= lr * grads['db_h']
                self.b_y -= lr * grads['db_y']
            
            # --- End of Epoch ---
            average_epoch_loss = epoch_loss / num_sequences
            loss_history.append(average_epoch_loss)
            end_time = time.time()
            
            if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {average_epoch_loss:.4f}, Time: {end_time - start_time:.2f}s")
                
                # --- Generate Examples ---
                if hasattr(self, 'word_to_ix') and hasattr(self, 'ix_to_word'):
                    print("\nGenerated examples:")
                    for _ in range(generated_examples):
                        # Generate a sequence of words, starting from "seed_word" (Chosen randomly)
                        seed_word = list(self.word_to_ix.keys())[0]  # Default to first word
                        try:
                            seed_word = random.choice(list(self.word_to_ix.keys()))
                        except:
                            pass
                        
                        h0_sample = np.zeros((self.hidden_dim, 1))
                        sampled_indices = self.sample_words(seed_word, h0_sample, n=10, sample_strategy='argmax')
                        
                        # Convert indices to words
                        sampled_text = ' '.join(self.ix_to_word.get(idx, "<UNK>") for idx in sampled_indices)
                        print(f"  {sampled_text}")
                
                print()
                sys.stdout.flush()
        
        print("Training Completed")
        return loss_history
    
    def sample_words(self, seed_word, h_prev, n=10, sample_strategy='random'):
        """
        Samples a sequence of words (Begins with seed_word) from RNN.
        
        Args:
            seed_word: Starting word or word index
            h_prev: Initial hidden state
            n: Number of words to generate (default: 10)
            sample_strategy: Strategy for sampling ('random' or 'argmax')
            
        Returns:
            List of indices representing the generated sequence of words
        """
        # Ensure having the integer index (seed_ix) corresponding to the starting word, 
        #                   regardless of whether a string or an index was provided.
        if isinstance(seed_word, str):
            if not hasattr(self, 'word_to_ix'):
                print("Warning: word_to_ix not found. Using random index.")
                seed_ix = np.random.randint(0, self.vocab_size)
            else:
                seed_ix = self.word_to_ix.get(seed_word.lower(), 0)  # Default to first word if not found
        else:
            seed_ix = seed_word
        
        # --- Intialization ---
        current_ix = seed_ix
        generated_indices = [current_ix]
        h = h_prev
        x = np.zeros((self.vocab_size, 1))
        x[current_ix] = 1
        
        # --- Generate n words ---
        for _ in range(n):
            # --- Forward pass ---
            a_t = self.W_hh @ h + self.W_hx @ x + self.b_h
            h = np.tanh(a_t)
            z_t = self.W_hy @ h + self.b_y
            y_hat_t = softmax(z_t)
            
            # --- Sampling Strategy ---
            p = y_hat_t[:, 0].copy()
            
            if sample_strategy == 'random':
                ix = np.random.choice(range(self.vocab_size), p=p)
            elif sample_strategy == 'argmax':
                ix = np.argmax(p)
            else:
                raise ValueError(f"Invalid sample_strategy: {sample_strategy}")
            
            # --- Output Handling ---
            current_ix = ix
            generated_indices.append(ix)
            
            # --- Prepare next input ---
            x = np.zeros((self.vocab_size, 1))
            x[current_ix] = 1
        
        return generated_indices
    
    def predict_words(self, start_text, n=10, sample_strategy='random'):
        """
        Generate a sequence of words starting with the given text.
        
        Args:
            start_text: Starting word or phrase
            n: Number of additional words to generate
            sample_strategy: Strategy for sampling ('random' or 'argmax')
            
        Returns:
            Generated text string including the start_text
        """
        if not hasattr(self, 'word_to_ix') or not hasattr(self, 'ix_to_word'):
            return "Error: Word mappings not available for prediction"
        
        # Initialize hidden state
        h = np.zeros((self.hidden_dim, 1))
        
        # --- Data Preprocessing ---

        # Split start_text into words and find their indices
        words = start_text.lower().split()
        indices = []
        
        for word in words:
            if word in self.word_to_ix:
                indices.append(self.word_to_ix[word])
            elif "<UNK>" in self.word_to_ix:
                indices.append(self.word_to_ix["<UNK>"])
        
        # If no valid words were found, use a default
        if not indices:
            if "<UNK>" in self.word_to_ix:
                indices = [self.word_to_ix["<UNK>"]]
            elif self.word_to_ix:
                indices = [list(self.word_to_ix.values())[0]]
            else:
                return "Error: Cannot find valid starting words"
        
        # Process each word to set up the hidden state
        for idx in indices:
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            
            # Update hidden state
            a_t = self.W_hh @ h + self.W_hx @ x + self.b_h
            h = np.tanh(a_t)
        
        # --- Sampling ---

        # Sample additional words
        last_idx = indices[-1]
        sampled_indices = self.sample_words(last_idx, h, n, sample_strategy)
        
        # Combine original words with generated words (skip the first as it's duplicated)
        all_indices = indices + sampled_indices[1:]
        generated_text = ' '.join(self.ix_to_word.get(idx, "<UNK>") for idx in all_indices)
        
        return generated_text