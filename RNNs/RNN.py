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
        # W_xh now maps one-hot input (|V|) to hidden (Dh)
        # Renamed from W_hx for clarity (Input X -> Hidden H)
        self.W_xh = xavier_init(hidden_dim, vocab_size)     # Shape Dh x |V|
        self.W_hh = xavier_init(hidden_dim, hidden_dim)     # Shape Dh x Dh
        self.W_hy = xavier_init(vocab_size, hidden_dim)     # Shape |V| x Dh
        self.b_h = np.zeros((hidden_dim, 1))                # Shape Dh x 1
        self.b_y = np.zeros((vocab_size, 1))                # Shape |V| x 1

        self.params = {
            # 'W_embed' removed
            'W_xh': self.W_xh, 'W_hh': self.W_hh,
            'W_hy': self.W_hy, 'b_h': self.b_h, 'b_y': self.b_y
        }
        # Mappings set later
        self.word_to_ix = None
        self.ix_to_word = None


    # Modified forward to accept one-hot inputs
    def forward(self, input_one_hots, h_prev):
        """
        Performs forward pass with one-hot inputs.

        Args:
            input_one_hots (list): List of one-hot vectors (shape |V| x 1), length T.
            h_prev (np.ndarray): Initial hidden state (shape Dh x 1).

        Returns:
            cache (dict): Stores 'one_hot_inputs', 'h_states', 'y_preds'.
        """
        # Cache for necessary values
        X_one_hots_s, h_states, y_preds = {}, {}, {}
        h_states[-1] = np.copy(h_prev)

        for t in range(len(input_one_hots)):
            x_t = input_one_hots[t] # Get the one-hot vector directly
            h_prev_t = h_states[t - 1]

            # --- Forward Equations ---
            # Embedding lookup removed
            # Use W_xh with the one-hot vector x_t
            a_t = self.W_hh @ h_prev_t + self.W_xh @ x_t + self.b_h
            h = tanh(a_t)
            y_pred = softmax(self.W_hy @ h + self.b_y)

            # Store necessary items for backward pass
            X_one_hots_s[t] = x_t # Store the one-hot input
            h_states[t] = h
            y_preds[t] = y_pred

        # Cache includes one-hot inputs now
        cache = {'one_hot_inputs': X_one_hots_s, 'h_states': h_states, 'y_preds': y_preds}
        return cache

    # Modified backward to remove embedding calculations
    def backward(self, targets, cache, clip_value=None):
        """
        Performs backward pass for one-hot input model.

        Args:
            targets (list): List of target one-hot vectors (shape |V| x 1), length T.
            cache (dict): Contains 'one_hot_inputs', 'h_states', 'y_preds'.
            clip_value (float, optional): Gradient clipping value.

        Returns:
            dict: Gradients (excluding dW_embed).
        """
        # Cache Extraction
        X_one_hots = cache['one_hot_inputs']
        h_states = cache['h_states']
        y_preds = cache['y_preds']
        sequence_length = len(X_one_hots)

        # Initialization (dW_embed removed)
        dW_xh = np.zeros_like(self.W_xh) # Correct name and shape
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        delta_h_plus_1 = np.zeros_like(self.b_h)

        for t in reversed(range(sequence_length)):
            x_t = X_one_hots[t] # Get one-hot input
            y_pred = y_preds[t]
            y = targets[t]
            h = h_states[t]
            h_prev = h_states[t - 1]

            dz = y_pred - y
            db_y += dz
            dW_hy += dz @ h.T

            # delta_h_t calculation (using corrected verified rule)
            h_next = h_states.get(t + 1, np.zeros_like(h))
            diag_term = np.diag(1 - h_next[:, 0] ** 2)
            dh = self.W_hy.T @ dz + self.W_hh.T @ diag_term @ delta_h_plus_1
            da = dh * (1 - h ** 2)

            # Gradient w.r.t W_hh, b_h unchanged in formula
            dW_hh += da @ h_prev.T
            db_h += da

            # Gradient w.r.t W_xh uses one-hot x_t
            dW_xh += da @ x_t.T # Shape (Dhx1) @ (1x|V|) -> Dhx|V|

            # Gradient w.r.t embedding removed
            # delta_e_t calculation removed
            # np.add.at for dW_embed removed

            # Update delta_h_plus_1 for next iteration
            delta_h_plus_1 = dh

        # Returned grads updated
        grads = {'dW_hy': dW_hy, 'db_y': db_y, 'dW_hh': dW_hh, 'dW_xh': dW_xh, 'db_h': db_h}

        if clip_value is not None:
            for key in grads:
                np.clip(grads[key], -clip_value, clip_value, out=grads[key])

        return grads

    # Modified train to accept one-hot inputs X
    def train(self, X_one_hot_train, y_train, lr = 0.001, epochs = 100, clip_value = 5.0, print_every = 10):
        """
        Trains the RNN model using SGD with one-hot inputs.
        """
        num_sequences = len(X_one_hot_train)
        loss_history = []

        print(f"Starting training (One-Hot Input) for {epochs} epochs...")
        print(f"Learning rate: {lr}, Clipping: {clip_value}")

        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0.0
            h0_epoch = np.zeros((self.hidden_dim, 1))

            for i in range(num_sequences):
                x_seq = X_one_hot_train[i] # Now list of one-hot vectors
                y_seq = y_train[i]
                sequence_length = len(x_seq)
                if sequence_length == 0: continue

                current_h0 = np.zeros((self.hidden_dim, 1)) # Reset per sequence

                # Pass h0 to forward
                cache = self.forward(x_seq, current_h0)
                y_preds = cache['y_preds']

                # --- Loss Calculation (Unchanged logic) ---
                seq_loss = 0.0; epsilon = 1e-12
                for t in range(sequence_length):
                    y_pred = y_preds[t]; y = y_seq[t]
                    correct_class_prob = y_pred[np.argmax(y), 0]
                    seq_loss += -np.log(correct_class_prob + epsilon)
                avg_seq_loss = seq_loss / sequence_length
                epoch_loss += avg_seq_loss
                # -----------------------------------------

                grads = self.backward(y_seq, cache, clip_value)

                # --- Parameter Update Loop (No W_embed) ---
                for grad_key, grad_value in grads.items():
                    param_key = grad_key[1:]
                    if param_key in self.params:
                        self.params[param_key] -= lr * grad_value
                    else:
                        # This warning shouldn't trigger if grads dict is correct
                        print(f"Warning: Grad key {grad_key} mismatch")
                # ---------------------------------------------

            # --- End of Epoch ---
            average_epoch_loss = epoch_loss / num_sequences
            loss_history.append(average_epoch_loss)
            end_time = time.time()
            if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {average_epoch_loss:.4f}, Time: {end_time - start_time:.2f}s")
                sys.stdout.flush()

        print("Training Completed")
        return loss_history

    # Modified sample to use one-hot inputs
    def sample(self, seed_word, h_prev, n, sample_strategy='argmax'):
        """ Samples sequence using one-hot inputs. """
        if self.word_to_ix is None or not self.ix_to_word:
             print("Error: Word mapping not set."); return []

        seed_ix = self.word_to_ix.get(seed_word.lower(), self.word_to_ix['<UNK>'])
        current_ix = seed_ix
        generated_indices = [current_ix]
        h = h_prev

        # Create initial one-hot input from seed_ix
        x = np.zeros((self.vocab_size, 1))
        x[current_ix] = 1

        for _ in range(n):
            # --- Forward pass for one step ---
            # Input 'x' is the one-hot vector of the last chosen word
            a_t = self.W_hh @ h + self.W_xh @ x + self.b_h # Use x directly
            h = np.tanh(a_t)
            z_t = self.W_hy @ h + self.b_y
            y_hat_t = softmax(z_t)

            # --- Sampling Strategy (Unchanged logic, just uses y_hat_t) ---
            p = y_hat_t[:, 0].copy(); p[current_ix] = 0 # Prevent repetition
            p_sum = np.sum(p);
            if p_sum > 1e-9: p = p / p_sum
            else: p = np.ones(self.vocab_size) / self.vocab_size

            if sample_strategy == 'random': ix = np.random.choice(range(self.vocab_size), p=p)
            elif sample_strategy == 'argmax': ix = np.argmax(p)
            else: raise ValueError(f"Invalid sample_strategy")
            # --------------------------------------------------------------

            # Update current_ix and prepare next input x
            current_ix = ix
            generated_indices.append(ix)
            x = np.zeros((self.vocab_size, 1)) # Reset x
            x[current_ix] = 1 # Set the chosen index to 1 for next input

        return generated_indices