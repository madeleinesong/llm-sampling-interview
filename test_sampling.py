import math
from sampling import sample_next_token


def test_greedy():
    logits = [1.0, 2.0, 3.0]
    assert sample_next_token(logits, temperature=0, top_k=None, top_p=None, seed=42) == 2


def test_top_k():
    logits = [0.0, 0.0, 10.0, 0.0]
    for seed in range(5):
        assert sample_next_token(logits, 1.0, top_k=1, top_p=None, seed=seed) == 2


def test_top_p_cutoff():
    logits = [math.log(0.6), math.log(0.25), math.log(0.1), math.log(0.05)]
    allowed = {0, 1}
    for seed in range(50):
        idx = sample_next_token(logits, 1.0, None, 0.8, seed)
        assert idx in allowed


def test_top_k_then_top_p_order():
    logits = [5.0, 4.0, 3.0, 2.0]
    # top_k=3 removes last token
    # top_p=0.6 should likely keep only first token after renorm
    idx = sample_next_token(logits, 1.0, 3, 0.6, seed=1)
    assert idx in {0}


def test_large_logits_stability():
    logits = [1000.0, 1001.0]
    idx = sample_next_token(logits, 1.0, None, None, seed=1)
    assert idx in {0, 1}
