# LLM Token Sampling Interview Exercise

## Instructions

1. Clone the repository.
2. Run: `python -m pytest`
3. Implement `sample_next_token` in `sampling.py`.
4. Do not modify `test_sampling.py`.

## Problem Description

Implement:

```python
def sample_next_token(
    logits: list[float],
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    seed: int
) -> int:
```

### Behavior

- If `temperature == 0`:
    Return the index of the maximum logit (greedy decoding).

- If `temperature > 0`:
    Divide logits by temperature before softmax.

- Use numerically stable softmax.

- If `top_k` is not None:
    Keep only the top_k highest logits.
    Mask the rest.
    Renormalize probabilities.

- If `top_p` is not None:
    Apply nucleus sampling:
      - Sort tokens by probability descending.
      - Keep smallest prefix whose cumulative probability >= top_p.
      - Mask the rest.
      - Renormalize.

- Apply top_k first, then top_p.

- Sampling must be deterministic given `seed`.

Do not use numpy or torch.
