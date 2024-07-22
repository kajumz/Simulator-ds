import numpy as np


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def softmax(x):
    #return np.exp(x) / np.sum(np.exp(x))
    max_vals = np.max(x, axis=-1, keepdims=True)
    exps = np.exp(x - max_vals)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def attention(q, k, v):  # [n_q, d_k], [n_k, d_k], [n_k, d_v] -> [n_q, d_v]
    sc = np.dot(q, k.T)
    sc /= np.sqrt(len(k))
    weights = softmax(sc)
    return np.dot(weights, v)


def self_attention(x, c_attn, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # QKV projections
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # Split into queries, keys, values
    q, k, v = np.split(x, 3, axis=-1)

    # Perform self-attention mechanism
    x = attention(q, k, v)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # Output projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

    return x