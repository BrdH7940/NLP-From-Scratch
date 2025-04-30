from Utils import *

class LSTM:
    def __init__(self, hidden_dim, vocab_size):
        """
        Initializes LSTM for one-hot inputs.

        Args:
            hidden_dim (int): Number of units in the hidden layer (Dh).
            vocab_size (int): Size of the vocabulary (|V|).
        """
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # --- Parameters ---
        self.W_x = xavier_init(4 * hidden_dim, vocab_size)     # Shape 4Dh x |V|
        self.W_h = xavier_init(4 * hidden_dim, hidden_dim)     # Shape 4Dh x Dh
        self.W_y = xavier_init(vocab_size, hidden_dim)     # Shape |V| x Dh

        self.b = np.zeros((4 * hidden_dim, 1))                # Shape Dh x 1
        self.b_y = np.zeros((vocab_size, 1))                # Shape |V| x 1

        self.params = {
            'W_x': self.W_x, 'W_h': self.W_h, 'b': self.b,
            'W_y': self.W_y, 'b_y': self.b_y
        }
        
        # --- Mappings: For Sampling ---
        self.word_to_ix = None
        self.ix_to_word = None

    def forward(self, input_one_hots, h_prev, c_prev):
        """
        Performs forward pass.

        Args:
            input_one_hots (list): List of one-hot vectors (shape |V| x 1), length T.
            h_prev (np.ndarray): Initial hidden state (shape Dh x 1).

        Returns:
            cache (dict): Stores 'one_hot_inputs', 'h_states', 'y_preds'.
        """
        # Cache
        x_states, h_states, c_states = {}, {}, {} # Input x, hidden state h, cell state c
        gates, y_preds = {}, {} # Gates (f,i,o,c_tilde), output predictions

        h_states[-1] = np.copy(h_prev)
        c_states[-1] = np.copy(c_prev)
        Dh = self.hidden_dim

        for t in range(len(input_one_hots)):
            # Get necessary variables
            x_t = input_one_hots[t]
            h_prev_t = h_states[t - 1]
            c_prev_t = c_states[t - 1]

            # Forward pass
            a_t = self.W_h @ h_prev_t + self.W_x @ x_t + self.b
            af_t = a_t[0 : Dh, :]
            ai_t = a_t[Dh : 2 * Dh, :]
            ac_t = a_t[2 * Dh : 3 * Dh, :]
            ao_t = a_t[3 * Dh : 4 * Dh, :]

            f_t = sigmoid(af_t)
            i_t = sigmoid(ai_t)
            ctilde_t = tanh(ac_t)
            o_t = sigmoid(ao_t)

            c_t = f_t * c_prev_t + i_t * ctilde_t
            h_t = o_t * tanh(c_t)

            z_t = self.W_y @ h_t + self.b_y
            y_pred_t = softmax(z_t)

            # Store cache
            x_states[t] = x_t
            h_states[t] = h_t
            c_states[t] = c_t
            gates[t] = {'f': f_t, 'i': i_t, 'o': o_t, 'ctilde': ctilde_t}
            y_preds[t] = y_pred_t

        cache = {'inputs': x_states, 'h_states': h_states, 'c_states': c_states, 
                 'gate_outputs': gates, 'y_preds': y_preds}

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
        x_states = cache['inputs']
        h_states = cache['h_states']
        c_states = cache['c_states']
        gates = cache['gate_outputs']
        y_preds = cache['y_preds']
        sequence_length = len(x_states)

        # Initialization
        dW_x = np.zeros_like(self.W_x)
        dW_h = np.zeros_like(self.W_h)
        dW_y = np.zeros_like(self.W_y)
        db = np.zeros_like(self.b)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_dim, 1))
        dc_next = np.zeros((self.hidden_dim, 1))

        for t in reversed(range(sequence_length)):
            # --- Get values from cache ---
            x_t = x_states[t]
            h_t = h_states[t]
            h_prev = h_states[t - 1]
            c_t = c_states[t]
            c_prev = c_states[t - 1]
            f_t, i_t, o_t, ctilde_t = gates[t]['f'], gates[t]['i'], gates[t]['o'], gates[t]['ctilde']
            y_pred_t = y_preds[t]
            y_t = targets[t]

            # --- Gradient calculation ---
            dz = y_pred_t - y_t

            dW_y += dz @ h_t.T
            db_y += dz

            dh_t = self.W_y.T @ dz + dh_next

            tmp = tanh(c_t)
            dc_t = dh_t * o_t * (1 - tmp ** 2) + dc_next

            do_t = dh_t * tmp
            df_t = dc_t * c_prev
            di_t = dc_t * ctilde_t
            dctilde_t = dc_t * i_t

            da_f = df_t * f_t * (1 - f_t)
            da_i = di_t * i_t * (1 - i_t)
            da_c = dctilde_t * (1 - ctilde_t ** 2)
            da_o = do_t * o_t * (1 - o_t)

            da_t = np.vstack((da_f, da_i, da_c, da_o))

            dW_x += da_t @ x_t.T
            dW_h += da_t @ h_prev.T
            db += da_t

            # --- Gradients for previous steps ---
            dh_next = self.W_h.T @ da_t
            dc_next = dc_t * f_t

        # Returned updated grads
        grads = {'dW_x': dW_x, 'dW_h': dW_h, 'dW_y': dW_y, 'db': db, 'db_y': db_y}

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
                current_c0 = np.zeros((self.hidden_dim, 1))
                
                # --- Forward ---
                cache = self.forward(x_seq, current_h0, current_c0)
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
                self.W_x -= lr * grads['dW_x']
                self.W_h -= lr * grads['dW_h']
                self.W_y -= lr * grads['dW_y']  
                self.b -= lr * grads['db']
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
                        c0_sample = np.zeros((self.hidden_dim, 1))
                        sampled_indices = self.sample_words(seed_word, h0_sample, c0_sample, n=10, sample_strategy='argmax')
                        
                        # Convert indices to words
                        sampled_text = ' '.join(self.ix_to_word.get(idx, "<UNK>") for idx in sampled_indices)
                        print(f"  {sampled_text}")
                
                print()
                sys.stdout.flush()
        
        print("Training Completed")
        return loss_history
    
    def sample_words(self, seed_word, h_prev, c_prev, n=10, sample_strategy='random'):
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
        c = c_prev
        x = np.zeros((self.vocab_size, 1))
        x[current_ix] = 1
        Dh = self.hidden_dim
        
        # --- Generate n words ---
        for _ in range(n):
            # --- Forward pass ---
            a_t = self.W_h @ h + self.W_x @ x + self.b
            af_t = a_t[0 : Dh, :]
            ai_t = a_t[Dh : 2 * Dh, :]
            ac_t = a_t[2 * Dh : 3 * Dh, :]
            ao_t = a_t[3 * Dh : 4 * Dh, :]

            f_t = sigmoid(af_t)
            i_t = sigmoid(ai_t)
            ctilde_t = tanh(ac_t)
            o_t = sigmoid(ao_t)

            c = f_t * c + i_t * ctilde_t
            h = o_t * tanh(c)

            z_t = self.W_y @ h + self.b_y
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
        c = np.zeros((self.hidden_dim, 1))
        Dh = self.hidden_dim
        
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
            a_t = self.W_h @ h + self.W_x @ x + self.b
            af_t = a_t[0 : Dh, :]
            ai_t = a_t[Dh : 2 * Dh, :]
            ac_t = a_t[2 * Dh : 3 * Dh, :]
            ao_t = a_t[3 * Dh : 4 * Dh, :]

            f_t = sigmoid(af_t)
            i_t = sigmoid(ai_t)
            ctilde_t = tanh(ac_t)
            o_t = sigmoid(ao_t)

            c = f_t * c + i_t * ctilde_t
            h = o_t * tanh(c)

        # --- Sampling ---

        # Sample additional words
        last_idx = indices[-1]
        sampled_indices = self.sample_words(last_idx, h, c, n, sample_strategy)
        
        # Combine original words with generated words (skip the first as it's duplicated)
        all_indices = indices + sampled_indices[1:]
        generated_text = ' '.join(self.ix_to_word.get(idx, "<UNK>") for idx in all_indices)
        
        return generated_text