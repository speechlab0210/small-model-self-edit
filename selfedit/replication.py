"""Replication of Arm C with the corrected training (PREREG.md section 7): the same two rules, new trajectories, more
rounds, and a second set of test problems nobody has looked at. Run by run_replication.py."""
import json
import os
import random
import re

import torch

from . import arm_c as AC
from . import config as C
from . import evaluate as E


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def practice_pools(practice, used, seeds, rounds, n, subset_seed):
    """new practice problems: one pool per (seed, round), half GSM8K and half MATH, all of them disjoint and none used
    by Arms B or C of the main run. Both rules of a seed share its pools, as C1 and C2 did in the main run."""
    rng = random.Random(subset_seed)
    gs = [r["id"] for r in practice.values() if r["source"] == "gsm8k" and r["id"] not in used]
    ms = [r["id"] for r in practice.values() if r["source"] == "math" and r["id"] not in used]
    rng.shuffle(gs)
    rng.shuffle(ms)
    pools, gi, mi, h = {}, 0, 0, n // 2
    for s in seeds:
        for r in range(1, rounds + 1):
            if gi + h > len(gs) or mi + n - h > len(ms):
                raise ValueError("not enough unused practice problems")
            pools[f"r{s}_gen_{r}"] = gs[gi:gi + h] + ms[mi:mi + n - h]
            gi, mi = gi + h, mi + n - h
    return pools


def fresh_test_items(m500, gsm_test, old_ids, blocked_texts, n_each, subset_seed):
    """test problems that no earlier measurement used and whose text is not among `blocked_texts` (practice and old test)."""
    rng = random.Random(subset_seed + 1)
    picked = []
    for rows in (m500, gsm_test):
        free = [r for r in rows if r["id"] not in old_ids and _norm(r["problem"]) not in blocked_texts]
        picked += [free[i] for i in sorted(rng.sample(range(len(free)), n_each))]
    return picked


def build_data(test, practice, used, m500, gsm_test, out_dir):
    """the pools and the fresh test problems, written once so that a resumed run and every reader see the same ones."""
    path = os.path.join(out_dir, "data_ids.json")
    if os.path.exists(path):
        meta = json.load(open(path))
    else:
        blocked = {_norm(r["problem"]) for r in test} | {_norm(r["problem"]) for r in practice.values()}
        fresh = fresh_test_items(m500, gsm_test, {r["id"] for r in test}, blocked, C.N_TEST_EACH, C.R_SUBSET_SEED)
        meta = {"subset_seed": C.R_SUBSET_SEED, "fresh_test": fresh,
                "pools": practice_pools(practice, used, C.R_SEEDS, C.R_ROUNDS, C.C_GEN_N, C.R_SUBSET_SEED)}
        os.makedirs(out_dir, exist_ok=True)
        json.dump(meta, open(path, "w"), indent=1)
    pools = {k: [practice[i] for i in v] for k, v in meta["pools"].items()}
    return meta["fresh_test"], pools


def distance(new, other):
    """how far the tensors `new` are from `other(name)`: entries that differ, L2 distance, its size relative to the L2
    norm of `other`, and the largest single change."""
    changed = total = 0
    sq = base = biggest = 0.0
    for name, t in new.items():
        a, b = t.float(), other(name).float()
        d = a - b
        changed += int((d != 0).sum())
        total += int(d.numel())
        sq += float(d.double().pow(2).sum())
        base += float(b.double().pow(2).sum())
        biggest = max(biggest, float(d.abs().max()))
    return {"entries_changed": changed, "entries_total": total, "l2": sq ** 0.5,
            "relative_l2": (sq / base) ** 0.5 if base else None, "max_abs": biggest}


def weight_change(engine, new, previous):
    """this round's weights against the original weights and against the previous round's."""
    return {"vs_original": distance(new, engine.original),
            "vs_previous_round": distance(new, lambda n: previous[n] if n in previous else engine.original(n))}


def answer_test(engine, items, salt, path, base):
    recs, secs = E.answer(engine, items, C.TEST_MAX_TOKENS, salt)
    E.save(recs, path, full_text=True)
    return recs, {"summary": E.summary(recs), "paired": E.paired(base, recs), "seconds": round(secs)}


def bug_check(engine, kept, train_seed, corrected, test, base, d, log):
    """what the audit's fix changed: from the same starting weights and the same kept answers, train again (1) with the
    main run's old divisor, (2) with the corrected code and another order of the answers, (3) with the corrected code
    and the same seed (no test run: are the weights identical?). Call it before the engine moves to `corrected`."""
    path = os.path.join(d, "bug_check.json")
    if os.path.exists(path):
        return
    out = {"kept": len(kept), "last_batch": (len(kept) - 1) % 8 + 1, "train_seed": train_seed}
    again, _ = AC.train(engine, kept, seed=train_seed, log=log)
    out["same_seed_again"] = {"difference_from_corrected": distance(again, lambda n: corrected[n])}
    del again
    for name, kw in (("old_divisor", {"seed": train_seed, "legacy_divisor": True}), ("other_order", {"seed": train_seed + 500})):
        new, _ = AC.train(engine, kept, log=log, **kw)
        row = {"difference_from_corrected": distance(new, lambda n: corrected[n])}
        with engine.using(new):
            _, row["test"] = answer_test(engine, test, "test", os.path.join(d, f"bug_check_{name}.jsonl.gz"), base)
        out[name] = row
        log(f"[R] bug check {name}: test gain {row['test']['paired']['gain']:+.3f} (se {row['test']['paired']['se']:.3f})")
    json.dump(out, open(path, "w"), indent=1)


def trajectory(engine, rule, seed, pools, test, base, fresh, fresh_base, out_dir, log):
    d = os.path.join(out_dir, f"{rule}_s{seed}")
    os.makedirs(d, exist_ok=True)
    spath = os.path.join(d, "state.pt")
    st = torch.load(spath) if os.path.exists(spath) else {"round": 0, "weights": {}, "rounds": []}
    for r in range(st["round"] + 1, C.R_ROUNDS + 1):
        engine.set_edits(st["weights"])
        name = f"r{seed}_gen_{r}"
        recs, secs = E.answer(engine, pools[name], C.DEV_MAX_TOKENS, name)
        E.save(recs, os.path.join(d, f"gen_round{r}.jsonl.gz"), full_text=True, keep_token_ids=True)
        kept = AC.keep(recs, rule)
        train_seed = 1000 * seed + r
        row = {"round": r, "generated": E.summary(recs)["all"], "kept": len(kept),
               "kept_correct": sum(x["correct"] for x in kept), "gen_seconds": round(secs),
               "training": {"seed": train_seed, "gradient_accumulation": 8, "partial_batch_divisor": "actual number of examples",
                            "kept_ids": [x["id"] for x in kept]}}
        if kept:
            stats = {}
            new, losses = AC.train(engine, kept, seed=train_seed, log=log, stats=stats)
            if seed == C.R_SEEDS[0] and r == 1:
                bug_check(engine, kept, train_seed, new, test, base, d, log)
            row["training"].update(stats)
            row["weight_change"] = weight_change(engine, new, st["weights"])
            row.update(train_loss_first=round(sum(losses[:8]) / min(8, len(losses)), 4),
                       train_loss_last=round(sum(losses[-8:]) / min(8, len(losses)), 4))
            st["weights"] = new
        else:
            row["note"] = "nothing kept: no training this round"
        engine.set_edits(st["weights"])
        _, row["test"] = answer_test(engine, test, "test", os.path.join(d, f"test_round{r}.jsonl.gz"), base)
        fresh_txt = ""
        if r in C.R_FRESH_AFTER:
            _, row["fresh_test"] = answer_test(engine, fresh, "fresh_test", os.path.join(d, f"fresh_test_round{r}.jsonl.gz"), fresh_base)
            fresh_txt = f", fresh gain {row['fresh_test']['paired']['gain']:+.3f} (se {row['fresh_test']['paired']['se']:.3f})"
        st["rounds"].append(row)
        st["round"] = r
        torch.save(st, spath)
        json.dump({"rule": rule, "seed": seed, "rounds": st["rounds"]}, open(os.path.join(d, "trajectory.json"), "w"), indent=1)
        log(f"[R] {rule} seed {seed} round {r}: kept {len(kept)} (correct {row['kept_correct']}), test gain "
            f"{row['test']['paired']['gain']:+.3f} (se {row['test']['paired']['se']:.3f}), acc "
            f"{row['test']['summary']['all']['acc']:.3f}, cut {row['test']['summary']['all']['cut_share']:.3f}{fresh_txt}")
    engine.set_edits({})
