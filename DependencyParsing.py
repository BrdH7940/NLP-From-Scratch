import collections

# --- Configuration Class (optional, but helps clarity) ---
class Configuration:
    def __init__(self, sentence_len):
        self.stack = [0]  # Initially contains ROOT (index 0)
        self.buffer = list(range(1, sentence_len + 1)) # Word indices (1 to n)
        self.arcs = set() # Store arcs as (head_index, label, dependent_index)

    def __str__(self):
        # Helper to visualize the state
        stack_words = [sentence[i] for i in self.stack]
        buffer_words = [sentence[i] for i in self.buffer]
        return f"Stack: {stack_words}\nBuffer: {buffer_words}\nArcs: {self.arcs}\n"

# --- Transition Functions ---
def shift(config):
    """Move the first element from buffer to stack."""
    if not config.buffer:
        raise ValueError("Cannot SHIFT: Buffer is empty")
    w_i = config.buffer.pop(0)
    config.stack.append(w_i)
    print("Action: SHIFT")

def left_arc(config, label):
    """Add arc j -> i (head=j, dep=i) and remove i from stack."""
    if len(config.stack) < 2:
        raise ValueError("Cannot LEFT-ARC: Stack needs at least 2 elements")
    w_j = config.stack[-1] # Top element (potential head)
    w_i = config.stack[-2] # Second element (potential dependent)
    if w_i == 0: # Cannot make ROOT a dependent
         raise ValueError("Cannot LEFT-ARC: Cannot make ROOT the dependent (w_i)")

    config.arcs.add((w_j, label, w_i)) # head, label, dependent
    config.stack.pop(-2) # Remove w_i (second element)
    print(f"Action: LEFT-ARC ({label})")

def right_arc(config, label):
    """Add arc i -> j (head=i, dep=j) and remove j from stack."""
    if len(config.stack) < 2:
         raise ValueError("Cannot RIGHT-ARC: Stack needs at least 2 elements")
    w_j = config.stack.pop() # Top element (potential dependent)
    w_i = config.stack[-1] # New top element (potential head)

    config.arcs.add((w_i, label, w_j)) # head, label, dependent
    print(f"Action: RIGHT-ARC ({label})")

# --- Simple Oracle (Knows the correct parse beforehand) ---
# In a real system, this would be a trained classifier.
def oracle_predict(config, gold_arcs_dict):
    """
    Predicts the next transition based on the gold standard parse.
    This is a simplified oracle for demonstration.
    gold_arcs_dict: A dictionary mapping dependent_idx -> (head_idx, label)
    """
    s_len = len(config.stack)
    b_len = len(config.buffer)

    # Preconditions and Gold Standard Checks
    if s_len >= 2:
        w_j = config.stack[-1] # Top
        w_i = config.stack[-2] # Second top

        # Check for LEFT-ARC: Is w_i a dependent of w_j in gold, and have all dependents of w_i been found?
        if w_i != 0 and gold_arcs_dict.get(w_i) == (w_j, gold_arcs_dict.get(w_i, (None, None))[1]): # Check if head(w_i) == w_j
             # Check if w_i already has all its *own* dependents attached
             # (In arc-standard, this check ensures dependents are processed before the head is reduced)
             # Simplified check: assume if the head relation is correct, we can make the arc
             # A full oracle needs to check if all children of w_i are already created.
             # For this simple sentence, we can proceed if the head relation is correct.
             return "LEFT-ARC", gold_arcs_dict[w_i][1] # Return action and label

        # Check for RIGHT-ARC: Is w_j a dependent of w_i in gold, and have all dependents of w_j been found?
        if gold_arcs_dict.get(w_j) == (w_i, gold_arcs_dict.get(w_j, (None, None))[1]): # Check if head(w_j) == w_i
             # Check if w_j has all its dependents attached.
             # A full oracle is complex. Check if w_j has any remaining dependents in the buffer or stack.
             has_deps = False
             for dep, (head, _) in gold_arcs_dict.items():
                 if head == w_j and (dep in config.buffer or dep in config.stack[:-1]): # Don't count itself
                     has_deps = True
                     break
             if not has_deps: # If w_j has no more dependents waiting, we can attach it.
                 return "RIGHT-ARC", gold_arcs_dict[w_j][1] # Return action and label

    # Default action: SHIFT if buffer is not empty
    if b_len > 0:
        return "SHIFT", None

    # Fallback / Error condition (should ideally not be reached in a perfect oracle scenario)
    # If buffer is empty but stack has more than ROOT, might need a final RIGHT-ARC to ROOT
    if s_len == 2 and b_len == 0 and config.stack[0] == 0:
         w_j = config.stack[1]
         if gold_arcs_dict.get(w_j) == (0, gold_arcs_dict.get(w_j, (None, None))[1]):
             return "RIGHT-ARC", gold_arcs_dict[w_j][1]

    return "DONE", None # Should only happen when stack=[0], buffer=[]

# --- Main Parsing Function ---
def parse(sentence_tokens, gold_arcs_dict):
    """
    Performs transition-based parsing using the oracle.
    sentence_tokens: List of words, including ROOT at index 0.
    gold_arcs_dict: Dictionary mapping dependent_idx -> (head_idx, label)
    """
    config = Configuration(len(sentence_tokens) - 1) # Exclude ROOT for length
    step = 0

    print(f"Initial Configuration (Sentence: {' '.join(sentence_tokens[1:])})")
    print(config)

    while not (len(config.buffer) == 0 and len(config.stack) == 1 and config.stack[0] == 0):
        step += 1
        print(f"--- Step {step} ---")

        # Get the next transition from the oracle
        action, label = oracle_predict(config, gold_arcs_dict)

        if action == "SHIFT":
            shift(config)
        elif action == "LEFT-ARC":
            left_arc(config, label)
        elif action == "RIGHT-ARC":
            right_arc(config, label)
        elif action == "DONE":
            print("Parsing should be complete.")
            break # Safety break
        else:
             print(f"Error: Unknown action {action} or invalid state.")
             break # Error condition

        print(config)

        # Safety break to prevent infinite loops in case of oracle error
        if step > (2 * len(sentence_tokens)):
            print("Error: Parsing seems stuck, too many steps.")
            break

    print("--- Final State ---")
    print(config)
    print("Parsing Complete!")
    return config.arcs

# --- Example Usage ---
# Sentence: "She eats green apples"
# Add ROOT token
sentence = ["ROOT", "She", "eats", "green", "apples"]

# Gold standard parse represented as dependent -> (head, label)
# Indices: 0=ROOT, 1=She, 2=eats, 3=green, 4=apples
gold_arcs = {
    1: (2, 'nsubj'), # She <- eats (nsubj)
    2: (0, 'root'),   # eats <- ROOT (root)
    3: (4, 'amod'),  # green <- apples (amod)
    4: (2, 'dobj')   # apples <- eats (dobj)
}

# Run the parser
final_arcs = parse(sentence, gold_arcs)

# Print final arcs in a readable format
print("\n--- Final Arcs Found ---")
if not final_arcs:
    print("No arcs found.")
else:
    # Sort arcs for consistent output (optional)
    sorted_arcs = sorted(list(final_arcs))
    for head_idx, label, dep_idx in sorted_arcs:
        print(f"{sentence[head_idx]}({head_idx}) --{label}--> {sentence[dep_idx]}({dep_idx})")