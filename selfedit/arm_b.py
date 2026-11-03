"""Arm B (PREREG.md section 3): the model picks one of 8 ready-made edits of itself, 6 rounds in a row.

A fixed procedure (not the model) estimates a "looping direction" per layer once, from the unedited model's own
practice answers, and builds 8 candidate edits per round. A chooser (the current model itself, or a random number)
picks one; a referee (a paired check on new practice problems) keeps it or not. Edits accumulate."""
import json
import math
import os
import random
import re

import torch

from . import config as C
from . import evaluate as E

KINDS = ("loop_top64", "loop_full", "rand_top64", "head_off")


def down(L):
    return f"{C.LM_PREFIX}layers.{L}.mlp.down_proj.weight"


def oproj(L):
    return f"{C.LM_PREFIX}layers.{L}.self_attn.o_proj.weight"


# ---------------------------------------------------------------------------------------------- directions
def directions(engine, diag_items, out_dir, log=print):
    path = os.path.join(out_dir, "directions.pt")
    if os.path.exists(path):
        return torch.load(path)
    engine.set_edits({})
    recs, secs = E.answer(engine, diag_items, C.DEV_MAX_TOKENS, "b_diag")
    E.save(recs, os.path.join(out_dir, "diag_answers.jsonl.gz"), full_text=False)
    end_think = engine.tok.convert_tokens_to_ids("</think>")
    looping = [r for r in recs if r["finish"] == "length" and r["loop_end"]][:C.B_DIR_TEXTS]
    prod = []
    for r in recs:
        if r["finish"] == "stop" and end_think in r["ids"]:
            k = r["ids"].index(end_think)
            if k >= 64:
                prod.append((r, k))
    prod = prod[:C.B_DIR_TEXTS]
    fallback = len(prod) < 8
    seq_l = [(r["prompt_ids"] + r["ids"], len(r["prompt_ids"]) + len(r["ids"]) - C.B_WINDOW,
              len(r["prompt_ids"]) + len(r["ids"])) for r in looping]
    if fallback:   # too few finished answers: the first 1,024 thinking tokens of the looping answers instead
        seq_p = [(r["prompt_ids"] + r["ids"], len(r["prompt_ids"]), len(r["prompt_ids"]) + C.B_WINDOW) for r in looping]
    else:
        seq_p = [(r["prompt_ids"] + r["ids"], len(r["prompt_ids"]) + max(0, k - C.B_WINDOW), len(r["prompt_ids"]) + k)
                 for r, k in prod]
    layers = list(range(1, C.N_LAYERS + 1))
    hl = engine.hidden_means(seq_l, layers).mean(0)
    hp = engine.hidden_means(seq_p, layers).mean(0)
    d = hl - hp
    u = d / d.norm(dim=-1, keepdim=True)     # u[L] = direction right after layer L (0-based) = hidden_states[L + 1]
    info = {"looping_texts": len(seq_l), "productive_texts": len(seq_p), "productive_fallback": fallback,
            "diag_summary": E.summary(recs), "diag_seconds": round(secs),
            "diff_norm_by_layer": [round(float(x), 4) for x in d.norm(dim=-1)],
            "cos_neighbouring_layers": [round(float(u[i] @ u[i + 1]), 4) for i in range(C.N_LAYERS - 1)]}
    res = {"u": u, "info": info, "diag_records": [{k: v for k, v in r.items() if k not in ("ids", "prompt_ids")} for r in recs]}
    torch.save(res, path)
    json.dump(info, open(os.path.join(out_dir, "directions_info.json"), "w"), indent=1)
    log(f"[B] directions from {len(seq_l)} looping / {len(seq_p)} productive texts (fallback={fallback})")
    return res


# ---------------------------------------------------------------------------------------------- the menu
def menu(round_no):
    """the 8 candidates of a round; the same for every trajectory (seeded by the round only)."""
    rng = random.Random(C.SUBSET_SEED * 100 + round_no)
    layers = rng.sample(C.B_LAYERS, 7)
    cands = [{"kind": "loop_top64", "layer": layers[i], "alpha": rng.choice([2, 4, 6])} for i in range(3)]
    cands += [{"kind": "loop_full", "layer": layers[i], "alpha": rng.choice([0.5, 1.0])} for i in range(3, 5)]
    cands += [{"kind": "rand_top64", "layer": layers[i], "alpha": rng.choice([2, 4, 6]), "seed": rng.randrange(1 << 30)}
              for i in range(5, 7)]
    cands.append({"kind": "head_off", "layer": rng.choice(C.FULL_ATTN_LAYERS), "head": rng.randrange(C.N_HEADS)})
    rng.shuffle(cands)
    for j, c in enumerate(cands):
        c["cid"] = f"r{round_no}c{j}"
    return cands


def describe(c):
    L = c["layer"]
    if c["kind"] == "loop_top64":
        return (f"Layer {L}, MLP output: find the 64 neurons that push hardest in the 'going in circles' direction "
                f"(found by comparing your own answers that kept repeating themselves with your answers that finished) "
                f"and make them push the opposite way (strength {c['alpha']}).")
    if c["kind"] == "loop_full":
        return (f"Layer {L}, MLP output: remove the 'going in circles' direction (the same kind of direction) from "
                f"everything this layer writes (strength {c['alpha']}).")
    if c["kind"] == "rand_top64":
        return (f"Layer {L}, MLP output: find the 64 neurons that push hardest along a randomly drawn direction and "
                f"make them push the opposite way (strength {c['alpha']}).")
    return f"Layer {L}, attention head {c['head']}: switch this attention head off completely."


def apply(engine, edits, c, u):
    """the edit set after applying candidate c on top of `edits` (float32 tensors)."""
    new = dict(edits)
    if c["kind"] == "head_off":
        n = oproj(c["layer"])
        W = (edits[n] if n in edits else engine.original(n)).float().clone()
        h = c["head"]
        W[:, h * C.HEAD_DIM:(h + 1) * C.HEAD_DIM] = 0
    else:
        n = down(c["layer"])
        W = (edits[n] if n in edits else engine.original(n)).float().clone()
        if c["kind"] == "rand_top64":
            g = torch.Generator().manual_seed(c["seed"])
            v = torch.randn(C.HIDDEN, generator=g)
            v = v / v.norm()
        else:
            v = u[c["layer"]].float()
        s = v @ W
        if c["kind"].endswith("top64"):
            mask = torch.zeros_like(s)
            mask[torch.topk(s, 64).indices] = 1.0
            W = W - c["alpha"] * torch.outer(v, s * mask)
        else:
            W = W - c["alpha"] * torch.outer(v, s)
    new[n] = W
    return new


# ---------------------------------------------------------------------------------------------- choosing
def report_text(recs, cap=C.DEV_MAX_TOKENS):
    s = E.summary(recs)["all"]
    t = (f"How you did on your last {s['n']} practice math problems: {s['correct']} correct; {s['cut']} ran out of "
         f"space ({cap:,} tokens) before finishing ({s['cut_looping']} of them while repeating themselves); "
         f"{s['wrong_answer_finished']} finished with a wrong answer; {s['no_answer_finished']} finished without an answer.")
    tail = next((r["text"][-400:] for r in recs if r["finish"] == "length" and r["loop_end"]), None)
    if tail:
        t += f'\nThe end of one of your answers that ran out of space:\n"...{tail.strip()}"'
    return t


def choice_prompt(report, ordered):
    opts = "\n".join(f"{i + 1}. {describe(c)}" for i, c in enumerate(ordered))
    return ("You are a language model (Qwen3.5-0.8B). Your own weights are about to be changed, and you choose the "
            f"change.\n\n{report}\n\nPossible changes (exactly one will be applied to you; then you will be tested again "
            f"on new math problems):\n{opts}\n\nWhich ONE change will make you better at solving math problems? Reply "
            "with the number of the change only.")


def self_choose(engine, report, cands, n, salt):
    """n samples (thinking off) with shuffled option orders; majority vote over valid answers (ties: earliest)."""
    orders, prompts, ps = [], [], []
    for s in range(n):
        order = list(range(len(cands)))
        random.Random(f"{salt}|{s}").shuffle(order)
        orders.append(order)
        prompts.append(engine.chat([{"role": "user", "content": choice_prompt(report, [cands[i] for i in order])}], False))
        ps.append(E.params(False, 256, E.seed_of(str(s), salt)))
    outs = engine.generate(prompts, ps)
    samples = []
    for order, o in zip(orders, outs):
        final = o["text"].split("</think>")[-1]
        m = (re.match(r"\s*[*#(\[]*\s*(?:(?:option|change|number)\s*)?([1-9])(?!\d)", final, re.I)
             or re.search(r"(?<!\d)([1-9])(?!\d)", final))
        num = int(m.group(1)) if m else None
        pick = order[num - 1] if num is not None and 1 <= num <= len(cands) else None
        samples.append({"order": order, "reply": o["text"][:400], "number": num, "pick": pick})
    valid = [x["pick"] for x in samples if x["pick"] is not None]
    if not valid:
        return None, samples
    counts = {p: valid.count(p) for p in valid}
    best = max(counts.values())
    choice = next(p for p in valid if counts[p] == best)
    return choice, samples


def accept_rule(p):
    net = p["rescues"] - p["losses"]
    return net >= 2 and net >= math.sqrt(p["rescues"] + p["losses"])


# ---------------------------------------------------------------------------------------------- round 1 table
def round1_table(engine, u, dev1, diag_recs, out_dir, log=print):
    path = os.path.join(out_dir, "round1_table.json")
    if os.path.exists(path):
        return json.load(open(path))
    cands = menu(1)
    engine.set_edits({})
    base, _ = E.answer(engine, dev1, C.DEV_MAX_TOKENS, "b_dev_1")
    E.save(base, os.path.join(out_dir, "round1", "base.jsonl.gz"), full_text=False)
    rows = []
    for c in cands:
        with engine.using(apply(engine, {}, c, u)):
            recs, _ = E.answer(engine, dev1, C.DEV_MAX_TOKENS, "b_dev_1")
        E.save(recs, os.path.join(out_dir, "round1", f"{c['cid']}.jsonl.gz"), full_text=False)
        p = E.paired(base, recs)
        rows.append({**c, "paired": p, "summary": E.summary(recs)["all"], "accept": accept_rule(p)})
        log(f"[B] round-1 table {c['cid']} {c['kind']} L{c['layer']}: gain {p['gain']:+.3f} (se {p['se']:.3f})")
    choice, samples = self_choose(engine, report_text(diag_recs), cands, C.B_TABLE_SAMPLES, "b_table")
    res = {"candidates": rows, "base_summary": E.summary(base)["all"], "self_samples": samples, "self_majority": choice}
    json.dump(res, open(path, "w"), indent=1)
    return res


# ---------------------------------------------------------------------------------------------- trajectories
def trajectory(engine, name, chooser, use_referee, u, pools, table, diag_recs, test_items, base_test, out_dir, log=print):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    spath = os.path.join(d, "state.pt")
    st = torch.load(spath) if os.path.exists(spath) else {"round": 0, "edits": {}, "rounds": [], "report_recs": diag_recs,
                                                            "tests": {}}
    for r in range(st["round"] + 1, C.B_ROUNDS + 1):
        cands = menu(r)
        engine.set_edits(st["edits"])
        if chooser == "self":
            choice, samples = self_choose(engine, report_text(st["report_recs"]), cands, C.B_CHOICE_SAMPLES, f"{name}|{r}")
        else:
            choice, samples = random.Random(f"{name}|{r}").randrange(len(cands)), None
        row = {"round": r, "menu": cands, "choice": choice, "samples": samples}
        if choice is None:
            row["result"] = "no valid choice: nothing changed"
        else:
            c = cands[choice]
            new = apply(engine, st["edits"], c, u)
            row["chosen"] = c
            dev = pools[f"b_dev_{r}"]
            if r == 1:     # round 1 starts from the unedited model for everyone: use the measured table
                trow = next(x for x in table["candidates"] if x["cid"] == c["cid"])
                p = trow["paired"]
                cur_recs = E.load(os.path.join(out_dir, "round1", "base.jsonl.gz"))
                new_recs = E.load(os.path.join(out_dir, "round1", f"{c['cid']}.jsonl.gz"))
            elif use_referee or chooser == "self":
                cur_recs, new_recs, p = None, None, None
                if use_referee:
                    cur_recs, _ = E.answer(engine, dev, C.DEV_MAX_TOKENS, f"b_dev_{r}")
                with engine.using(new):
                    new_recs, _ = E.answer(engine, dev, C.DEV_MAX_TOKENS, f"b_dev_{r}")
                if cur_recs is not None:
                    p = E.paired(cur_recs, new_recs)
            else:
                cur_recs = new_recs = p = None
            if use_referee:
                ok = accept_rule(p)
                row.update(paired=p, accepted=ok)
                if ok:
                    st["edits"], st["report_recs"] = new, new_recs
                else:
                    st["report_recs"] = cur_recs
            else:
                row.update(accepted=True, paired_if_measured=p)
                st["edits"] = new
                if new_recs is not None:
                    st["report_recs"] = new_recs
            if new_recs is not None:
                E.save(new_recs, os.path.join(d, f"dev_round{r}_candidate.jsonl.gz"), full_text=False)
        st["rounds"].append(row)
        if r in C.B_TEST_AFTER:
            key = json.dumps(sorted((n, float(t.double().sum())) for n, t in st["edits"].items()))
            prev = next((v for v in st["tests"].values() if v["key"] == key), None)
            if not st["edits"]:
                tsum, tp = E.summary(base_test), E.paired(base_test, base_test)
                st["tests"][str(r)] = {"key": key, "same_as": "unedited", "summary": tsum, "paired": tp}
            elif prev is not None:
                st["tests"][str(r)] = dict(prev, same_as=f"round {prev.get('round')} (no change since)")
            else:
                with engine.using(st["edits"]):
                    recs, secs = E.answer(engine, test_items, C.TEST_MAX_TOKENS, "test")
                E.save(recs, os.path.join(d, f"test_round{r}.jsonl.gz"), full_text=True)
                st["tests"][str(r)] = {"key": key, "round": r, "summary": E.summary(recs), "paired": E.paired(base_test, recs),
                                       "seconds": round(secs)}
            t = st["tests"][str(r)]
            log(f"[B] {name} test after round {r}: gain {t['paired']['gain']:+.3f} (se {t['paired']['se']:.3f})")
        st["round"] = r
        torch.save(st, spath)
        json.dump({"name": name, "chooser": chooser, "referee": use_referee,
                   "rounds": [{k: v for k, v in x.items() if k != "menu"} | {"menu": [describe(c) for c in x["menu"]]}
                              for x in st["rounds"]],
                   "tests": st["tests"],
                   "final_edits": {n: int((t.to(engine.original(n).dtype) != engine.original(n)).sum()) for n, t in st["edits"].items()}},
                  open(os.path.join(d, "trajectory.json"), "w"), indent=1, default=str)
        log(f"[B] {name} round {r}: choice={choice} accepted={row.get('accepted')} edits={len(st['edits'])}")
    engine.set_edits({})
    return st
