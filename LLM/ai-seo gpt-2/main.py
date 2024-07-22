from typing import Callable
from typing import List

import numpy as np
from gpt2 import gpt2
from util import load_encoder_hparams_and_weights


def generate(
    llm: Callable[[List[int]], List[float]],
    input_ids: List[int],
    n_tokens: int,
    top_k: int = 1,
    top_p: float = 0.75,
    temperature: float = 1.0,
    weights: dict = None,
    random_state: int = 0,
) -> List[int]:
    """Generate a sequence of tokens from a prompt using a language model."""
    np.random.seed(random_state)
    output_ids = []

    # Auto-regressive decode loop
    for _ in range(n_tokens):
        logits = llm(input_ids + output_ids, **weights)
        logits = np.array(logits)
        logits = logits / temperature
        probs = np.exp(logits) / np.exp(logits).sum()

        # Apply top-k filtering
        indices = np.argsort(probs)[-top_k:]
        probs = probs[indices] / probs[indices].sum()

        order = np.argsort(probs)[::-1]
        indices = indices[order]
        probs = probs[order]

        # Apply top-p filtering
        cum_probs = np.cumsum(probs)
        mask = cum_probs <= np.min(cum_probs[cum_probs >= top_p])
        probs = probs[mask] / probs[mask].sum()
        indices = indices[mask]

        next_id = int(np.random.choice(indices, p=probs))
        output_ids.append(next_id)

    return output_ids


# Main function
def main(
    prompt: str,
    n_tokens: int = 40,
    model_size: str = "124M",
    models_dir: str = "models",
):
    # Load encoder, hyperparameters and model weights
    encoder, hparams, weights = load_encoder_hparams_and_weights(model_size, models_dir)

    # Prepare input token IDs
    input_ids = [encoder.encode(prompt)]
    assert len(input_ids) + n_tokens < hparams["n_ctx"]

    # Generate tokens, this time we don't need to pass the vocab
    # because the 'generate' function will accept token IDs
    # as input and return token IDs as output
    output_ids = generate(
        llm=gpt2,
        input_ids=input_ids[0],
        n_tokens=n_tokens,
        top_k=50,
        top_p=0.8,
        temperature=0.75,
        random_state=0,
        weights=weights,
    )

    # Decode output token IDs using the encoder, it will convert
    # the token IDs to a string for us based on its vocabulary
    output = encoder.decode(output_ids)

    return output


if __name__ == "__main__":
    import fire

    fire.Fire(main)
