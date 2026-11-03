"""Strict paired comparisons shared by evaluation and the report (no ML dependencies)."""
import math


def indexed(records):
    result = {}
    for row in records:
        key = row["id"]
        if key in result:
            raise ValueError(f"duplicate problem id: {key}")
        if row["correct"] not in (0, 1):
            raise ValueError(f"non-binary correctness for {key}")
        result[key] = row["correct"]
    return result


def paired(base, new):
    """Require identical nonempty problem sets; never silently grade their intersection."""
    b, n = indexed(base), indexed(new)
    if not b or b.keys() != n.keys():
        raise ValueError("paired comparison requires identical, nonempty problem-id sets")
    diffs = [n[key] - b[key] for key in b]
    size = len(diffs)
    gain = sum(diffs) / size
    variance = sum((x - gain) ** 2 for x in diffs) / (size - 1) if size > 1 else 0.0
    se = math.sqrt(variance / size)
    # None represents an undefined zero-variance z score without non-standard JSON Infinity.
    z = gain / se if se else (0.0 if gain == 0 else None)
    return {"n": size, "gain": gain, "se": se, "z": z,
            "rescues": diffs.count(1), "losses": diffs.count(-1)}
