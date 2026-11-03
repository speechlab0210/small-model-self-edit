"""Bookkeeping for per-example gradient accumulation."""


def accumulation_divisor(index, total, accumulation):
    if not (0 <= index < total) or accumulation < 1:
        raise ValueError("invalid accumulation batch")
    start = (index // accumulation) * accumulation
    return min(accumulation, total - start)
