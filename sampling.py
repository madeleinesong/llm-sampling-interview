from __future__ import annotations
from typing import Optional, List
import math
import random


def sample_next_token(
    logits: List[float],
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    seed: int,
) -> int:
    """
    Returns a sampled index from logits using temperature, top_k, and top_p.
    Deterministic with respect to seed.
    """
    # TODO: Implement
    raise NotImplementedError
