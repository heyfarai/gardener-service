import math
from typing import List

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    
    if not norm_a or not norm_b:
        return 0.0
    
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)
