def apply(
    sums: dict[str, float], count: int, probs: dict[str, float], sign: int
) -> tuple[dict[str, float], int]:
    """Add (sign=1) or subtract (sign=-1) one message; inputs are not mutated."""
    out = dict(sums)
    for label, p in probs.items():
        out[label] = out.get(label, 0.0) + sign * p
    return out, count + sign
