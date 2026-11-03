"""Test subsets and practice problems (PREREG.md section 2). `build()` writes everything to config.DATA and the chosen
ids to the repository's data/ folder, so that every run and every reader sees the same problems."""
import json
import os
import random
import re

from . import config as C
from . import mathqa as Q

MATH_TRAIN = ("DigitalLearningGmbH/MATH-lighteval", "0530c78699ea5e8eb5530600900e1f328b48acad")


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _gsm_answer(sol):
    return sol.split("####")[-1].strip().replace(",", "")


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build(repo_data_dir=None):
    from datasets import load_dataset
    os.makedirs(C.DATA, exist_ok=True)
    m500 = [{"id": r["unique_id"], "source": "math500", "problem": r["problem"], "answer": r["answer"]}
            for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    g = load_dataset("openai/gsm8k", "main")
    gsm_test = [{"id": f"gsm8k_test_{i}", "source": "gsm8k", "problem": r["question"], "answer": _gsm_answer(r["answer"])}
                for i, r in enumerate(g["test"])]
    gsm_train = [{"id": f"gsm8k_train_{i}", "source": "gsm8k", "problem": r["question"],
                  "answer": _gsm_answer(r["answer"]), "solution": r["answer"]} for i, r in enumerate(g["train"])]
    try:
        mt = load_dataset(MATH_TRAIN[0], "default", split="train", revision=MATH_TRAIN[1])
    except Exception:  # noqa: BLE001 - older/newer config naming
        mt = load_dataset(MATH_TRAIN[0], split="train", revision=MATH_TRAIN[1])
    seen = {_norm(r["problem"]) for r in m500}
    math_train, dropped = [], 0
    for i, r in enumerate(mt):
        gold = Q.gold_of_solution(r["solution"])
        if gold is None or _norm(r["problem"]) in seen:
            dropped += 1
            continue
        math_train.append({"id": f"math_train_{i}", "source": "math", "problem": r["problem"], "answer": gold,
                           "solution": r["solution"], "subject": r.get("type"), "level": r.get("level")})

    rng = random.Random(C.SUBSET_SEED)
    test = [m500[i] for i in sorted(rng.sample(range(len(m500)), C.N_TEST_EACH))]
    test += [gsm_test[i] for i in sorted(rng.sample(range(len(gsm_test)), C.N_TEST_EACH))]
    _write(os.path.join(C.DATA, "test.jsonl"), test)
    _write(os.path.join(C.DATA, "gsm8k_train.jsonl"), gsm_train)
    _write(os.path.join(C.DATA, "math_train.jsonl"), math_train)

    # disjoint practice pools for Arms B and C, half GSM8K and half MATH each (Arm A sees both whole files)
    rg = random.Random(C.SUBSET_SEED + 1)
    gs, ms = gsm_train[:], math_train[:]
    rg.shuffle(gs)
    rg.shuffle(ms)
    pools, gi, mi = {}, 0, 0

    def take(name, n):
        nonlocal gi, mi
        h = n // 2
        rows = gs[gi:gi + h] + ms[mi:mi + n - h]
        gi, mi = gi + h, mi + n - h
        pools[name] = [r["id"] for r in rows]

    take("b_diag", C.B_DIAG_N)
    for r in range(1, C.B_ROUNDS + 1):
        take(f"b_dev_{r}", C.B_DEV_N)
    for r in range(1, C.C_ROUNDS + 1):
        take(f"c_gen_{r}", C.C_GEN_N)
    meta = {"test_ids": [r["id"] for r in test], "pools": pools,
            "counts": {"math500": len(m500), "gsm8k_test": len(gsm_test), "gsm8k_train": len(gsm_train),
                       "math_train": len(math_train), "math_train_dropped": dropped}}
    json.dump(meta, open(os.path.join(C.DATA, "ids.json"), "w"), indent=1)
    if repo_data_dir:
        os.makedirs(repo_data_dir, exist_ok=True)
        json.dump(meta, open(os.path.join(repo_data_dir, "ids.json"), "w"), indent=1)
    return meta


def load_all():
    """(test items, practice dict id -> item, pools dict name -> list of items)."""
    test = read(os.path.join(C.DATA, "test.jsonl"))
    practice = {r["id"]: r for f in ("gsm8k_train.jsonl", "math_train.jsonl") for r in read(os.path.join(C.DATA, f))}
    meta = json.load(open(os.path.join(C.DATA, "ids.json")))
    pools = {k: [practice[i] for i in v] for k, v in meta["pools"].items()}
    return test, practice, pools
