"""Arm C (PREREG.md section 3): self-training, the conventional route. Each round the current model answers 256 new
practice problems; C1 keeps the answers that are correct by the answer key, C2 keeps every answer that finished with a
\\boxed{} answer (no answer key); one epoch of full fine-tuning on the kept answers (loss on the answer only, AdamW,
lr 1e-5, float32 master weights); the result is the next round's model. Test set after every round."""
import copy
import json
import os
import random

import torch

from . import config as C
from . import evaluate as E
from .training_utils import accumulation_divisor


def keep(recs, rule):
    if rule == "C1_answer_key":
        return [r for r in recs if r["correct"]]
    if rule == "C2_finished_only":
        return [r for r in recs if r["finish"] == "stop" and r["pred"] is not None]
    raise ValueError(f"unknown retention rule: {rule}")


def train(engine, examples, seed, log=print, legacy_divisor=False, stats=None):
    """one epoch over `examples` (records with prompt_ids and ids); returns the new language-model tensors (bf16).
    legacy_divisor: the main run's bug (a last batch of fewer than 8 examples is still divided by 8), kept only so that
    the replication can measure what it did. stats: a dict that receives the gradient norms before clipping and how
    many weights the float32 update moved, before and after rounding to the bfloat16 the model runs in."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = copy.deepcopy(engine.hf).float()
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        # non-reentrant: the embedding is frozen, and reentrant checkpointing would then pass no gradient to the layers
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm = getattr(getattr(model, "model", model), "language_model", model)
    names = {n: p for n, p in model.named_parameters()}
    train_names = [engine._hf_name[n] for n in engine.names if n not in C.FROZEN and n in engine._hf_name]
    for n, p in names.items():
        p.requires_grad_(n in set(train_names))
    opt = torch.optim.AdamW([names[n] for n in train_names], lr=C.C_LR, weight_decay=0.0)
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    accum, losses = 8, []
    opt.zero_grad(set_to_none=True)
    W = lm.embed_tokens.weight.detach()
    start = {n: names[n].detach().clone() for n in train_names} if stats is not None else None
    grad_norms = []
    for k, i in enumerate(order):
        r = examples[i]
        ids = torch.tensor([r["prompt_ids"] + r["ids"]], device="cuda")
        y = ids[0, 1:]
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[len(r["prompt_ids"]) - 1:] = True          # predict the answer tokens only
        h = lm(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
        hd = h.detach().requires_grad_(True)            # chunked cross-entropy: never hold all logits at once
        total, ntok = 0.0, int(mask.sum())
        for s in range(0, y.shape[0], 1024):
            m = mask[s:s + 1024]
            if not m.any():
                continue
            lg = hd[s:s + 1024][m] @ W.T
            l_ = torch.nn.functional.cross_entropy(lg.float(), y[s:s + 1024][m], reduction="sum")
            l_.backward()
            total += float(l_)
        divisor = accum if legacy_divisor else accumulation_divisor(k, len(order), accum)
        h.backward(hd.grad / ntok / divisor)
        losses.append(total / ntok)
        if (k + 1) % accum == 0 or k + 1 == len(order):
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_([names[n] for n in train_names], 1.0)))
            opt.step()
            opt.zero_grad(set_to_none=True)
    new = {}
    inv = {v: k for k, v in engine._hf_name.items()}
    with torch.no_grad():
        for n in train_names:
            new[inv[n]] = names[n].detach().to("cpu", torch.bfloat16)
        if stats is not None:
            stats.update(optimizer_steps=len(grad_norms), grad_norms_before_clipping=grad_norms,
                         weights_total=sum(int(start[n].numel()) for n in train_names),
                         moved_by_float32_update=sum(int((names[n] != start[n]).sum()) for n in train_names),
                         moved_after_rounding_to_bfloat16=sum(int((names[n].to(torch.bfloat16).float() != start[n]).sum())
                                                              for n in train_names))
    del model, opt, names, start
    torch.cuda.empty_cache()
    return new, losses


def run(engine, pools, test, base_test, out_dir, log, mark, done):
    os.makedirs(out_dir, exist_ok=True)
    for rule in ("C1_answer_key", "C2_finished_only"):
        d = os.path.join(out_dir, rule)
        os.makedirs(d, exist_ok=True)
        spath = os.path.join(d, "state.pt")
        st = torch.load(spath) if os.path.exists(spath) else {"round": 0, "weights": {}, "rounds": []}
        for r in range(st["round"] + 1, C.C_ROUNDS + 1):
            engine.set_edits(st["weights"])
            recs, secs = E.answer(engine, pools[f"c_gen_{r}"], C.DEV_MAX_TOKENS, f"c_gen_{r}")
            E.save(recs, os.path.join(d, f"gen_round{r}.jsonl.gz"), full_text=True, keep_token_ids=True)
            kept = keep(recs, rule)
            row = {"round": r, "generated": E.summary(recs)["all"], "kept": len(kept),
                   "kept_correct": sum(x["correct"] for x in kept), "gen_seconds": round(secs),
                   "training": {"seed": r, "gradient_accumulation": 8,
                                "partial_batch_divisor": "actual number of examples",
                                "kept_ids": [x["id"] for x in kept],
                                "optimizer_steps": (len(kept) + 7) // 8}}
            if kept:
                new, losses = train(engine, kept, seed=r, log=log)
                st["weights"] = new
                row.update(train_loss_first=round(sum(losses[:8]) / min(8, len(losses)), 4),
                           train_loss_last=round(sum(losses[-8:]) / min(8, len(losses)), 4))
            else:
                row["note"] = "nothing kept: no training this round"
            engine.set_edits(st["weights"])
            trecs, tsecs = E.answer(engine, test, C.TEST_MAX_TOKENS, "test")
            E.save(trecs, os.path.join(d, f"test_round{r}.jsonl.gz"), full_text=True)
            row["test"] = {"summary": E.summary(trecs), "paired": E.paired(base_test, trecs), "seconds": round(tsecs)}
            st["rounds"].append(row)
            st["round"] = r
            torch.save(st, spath)
            json.dump({"rule": rule, "rounds": st["rounds"]}, open(os.path.join(d, "trajectory.json"), "w"), indent=1)
            log(f"[C] {rule} round {r}: kept {len(kept)} (correct {row['kept_correct']}), test gain "
                f"{row['test']['paired']['gain']:+.3f} (se {row['test']['paired']['se']:.3f})")
        engine.set_edits({})
        mark(f"arm_c_{rule}")
