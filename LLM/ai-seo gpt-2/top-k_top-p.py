from typing import Callable
from typing import List

import numpy as np


def generate(
        llm: Callable[[List[int]], List[float]],
        prompt: List[str],
        n_tokens: int,
        vocab: List[str],
        top_k: int = 50,
        top_p: float = 0.75,
        temperature: float = 1.1,
        random_state: int = 0,
) -> List[str]:
    """Generate a sequence of tokens from a prompt using a language model."""
    np.random.seed(random_state)


    input_ids = [vocab.index(token) for token in prompt]# YOUR CODE HERE
    generated_tokens = []


    for _ in range(n_tokens):

        logits = llm(input_ids)# YOUR CODE HERE
        logits = np.asarray(logits) / temperature

        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities)

        # Apply top-k filtering and renormalize
        indices = np.argsort(probabilities)[::-1][:top_k]
        probabilities = probabilities[indices] / probabilities[indices].sum()

        # Sort token IDs by probability
        order = np.argsort(probabilities)[::-1]
        probabilities = probabilities[order]

        # Apply top-p filtering and renormalize
        cum_probs = np.cumsum(probabilities)
        mask = cum_probs <= np.min(cum_probs[cum_probs >= top_p])
        probabilities = probabilities[mask] / probabilities[mask].sum()
        indices = indices[mask]

        # Sample next token ID from the filtered distribution
        next_token_id = np.random.choice(indices, p=probabilities)

        generated_tokens.append(vocab[next_token_id])
        input_ids.append(int(next_token_id))


    return generated_tokens
