import numpy as np

def build_sequences(X, seq_len=10):

    sequences = []

    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])

    return np.array(sequences)