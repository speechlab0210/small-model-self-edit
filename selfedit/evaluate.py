"""Answering practice / test problems with the current weights, grading, summaries and paired comparisons."""
import gzip
import hashlib
import json
import math
import os
import time

from . import config as C
from . import mathqa as Q


def seed_of(pid, salt):
    """one fixed sampling seed per (problem, measurement): every model answering the same measurement shares it."""
    return int(hashlib.sha256(f"{salt}|{pid}".encode()).hexdigest()[:8], 16)


def params(thinking, max_tokens, seed):
    from vllm import SamplingParams
    return SamplingParams(n=1, max_tokens=max_tokens, seed=seed, **(C.THINK if thinking else C.NOTHINK))


def answer(engine, items, max_tokens, salt, thinking=True):
    """every item answered once with the current weights; returns (records, seconds). A record keeps the full text."""
    prompts = [engine.chat([{"role": "user", "content": Q.build_prompt(it["problem"])}], thinking) for it in items]
    ps = [params(thinking, max_tokens, seed_of(it["id"], salt)) for it in items]
    t0 = time.time()
    outs = engine.generate(prompts, ps)
    recs = []
    for it, o in zip(items, outs):
        pred, nbox = Q.extract(o["text"])
        ok = Q.is_correct(pred, it.get("answer"))
        recs.append({"id": it["id"], "source": it["source"], "correct": int(ok), "pred": pred[:120] if pred else None,
                     "boxes": nbox, "finish": o["finish"], "tokens": o["tokens"],
                     "closed_think": "</think>" in o["text"], "loop_end": Q.loop_end(o["text"]),
                     "text": o["text"], "ids": o["ids"], "prompt_ids": o["prompt_ids"]})
    return recs, time.time() - t0


def summary(recs):
    """accuracy, cut share, mean tokens and failure kinds, overall and per source."""
    def one(rs):
        n = len(rs)
        if not n:
            return {}
        return {"n": n, "acc": sum(r["correct"] for r in rs) / n,
                "cut_share": sum(r["finish"] == "length" for r in rs) / n,
                "mean_tokens": sum(r["tokens"] for r in rs) / n,
                "correct": sum(r["correct"] for r in rs),
                "cut": sum(r["finish"] == "length" for r in rs),
                "cut_looping": sum(r["finish"] == "length" and r["loop_end"] for r in rs),
                "wrong_answer_finished": sum(r["finish"] != "length" and not r["correct"] and r["pred"] is not None for r in rs),
                "no_answer_finished": sum(r["finish"] != "length" and r["pred"] is None for r in rs)}
    out = {"all": one(recs)}
    for s in sorted({r["source"] for r in recs}):
        out[s] = one([r for r in recs if r["source"] == s])
    return out


def paired(base, new):
    """gain of `new` over `base` on the same problems (mean of per-problem differences) with its standard error."""
    b = {r["id"]: r["correct"] for r in base}
    d = [r["correct"] - b[r["id"]] for r in new if r["id"] in b]
    n = len(d)
    if not n:
        return {}
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return {"n": n, "gain": m, "se": se, "z": (m / se) if se > 0 else (0.0 if m == 0 else float("inf")),
            "rescues": sum(x == 1 for x in d), "losses": sum(x == -1 for x in d)}


def save(recs, path, full_text=True):
    """records as gzip JSON lines; without full_text only the first 300 and last 1,500 characters are kept."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in recs:
            r = {k: v for k, v in r.items() if k not in ("ids", "prompt_ids")}
            if not full_text and len(r["text"]) > 1800:
                r["text"] = r["text"][:300] + " [...] " + r["text"][-1500:]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
