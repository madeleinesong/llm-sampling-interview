# Interviewer Notes

## Expected Approach

- Subtract max logit before exponentiation.
- Divide by temperature (not multiply).
- Mask logits by setting to -inf before softmax.
- Renormalize after masking.
- Use `random.Random(seed).random()`
- Sample via cumulative sum.

## Common Mistakes

- Softmax overflow
- Not renormalizing after top_k/top_p
- Incorrect top_p prefix logic
- Applying top_p before top_k
- Non-deterministic sampling

## Follow-up Discussion

- Add repetition penalty.
- Add `generate()` loop.
- Why numerical stability matters.
- Complexity analysis.
