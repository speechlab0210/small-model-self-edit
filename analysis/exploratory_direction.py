"""EXPLORATORY, NOT PRE-REGISTERED (written after the round-1 table of the main run showed that none of Arm B's
ready-made edits helped): why did they not help, when Astra's edit of the same kind did?

1. How close is the looping direction this experiment estimated (arm_b/directions.pt) to the direction Astra's edit
   subtracts? Astra's change of each tensor has rank one, W_astra - W0 = -a * outer(u, m * (u^T W0)), so its u is the
   top left singular vector of the change.
2. On the 64 practice problems of the round-1 table (same seeds, paired with the unedited model's answers there):
   Astra's edit itself, and Astra's recipe (layers 15 and 19, the 64 strongest columns, strength 6) with this
   experiment's direction in place of Astra's.

Run on the machine of the main run, after it finished:  python analysis/exploratory_direction.py
Writes <out>/exploratory/direction_check.json and the answers.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from selfedit import arm_b as B  # noqa: E402
from selfedit import config as C  # noqa: E402
from selfedit import data as D  # noqa: E402
from selfedit import evaluate as E  # noqa: E402
from selfedit.engine import Engine  # noqa: E402

out = os.path.join(C.OUT, "exploratory")
os.makedirs(out, exist_ok=True)
res = {"note": "exploratory, added after the pre-registered run; not one of the pre-registered tests"}
u_ours = torch.load(os.path.join(C.OUT, "arm_b", "directions.pt"))["u"].float()   # u[L]: right after layer L (0-based)
astra = torch.load(os.path.join(C.OUT, "astra_edit.pt"))
engine = Engine()

sim = {}
for n, W1 in sorted(astra.items()):
    L = int(n.split("layers.")[1].split(".")[0])
    W0 = engine.original(n).float()
    dW = W1.float() - W0
    U, S, _ = torch.linalg.svd(dW, full_matrices=False)
    ua = U[:, 0]
    cs = [float(abs(ua @ u_ours[k])) for k in range(C.N_LAYERS)]
    sim[str(L)] = {"changed_columns": int((dW != 0).any(0).sum()),
                   "second_to_first_singular_value": float(S[1] / S[0]),
                   "abs_cos_with_our_direction_same_layer": cs[L],
                   "abs_cos_with_our_direction_by_layer": [round(x, 4) for x in cs],
                   "best_matching_layer": max(range(C.N_LAYERS), key=lambda k: cs[k])}
res["direction_similarity"] = sim

_, _, pools = D.load_all()
dev = pools["b_dev_1"]
base = E.load(os.path.join(C.OUT, "arm_b", "round1", "base.jsonl.gz"))
recipe = {}
for L in (15, 19):
    recipe = B.apply(engine, recipe, {"kind": "loop_top64", "layer": L, "alpha": 6}, u_ours)
res["dev_check"] = {}
for name, edits in (("astra_edit", astra), ("astra_recipe_with_our_direction", recipe)):
    with engine.using(edits):
        recs, secs = E.answer(engine, dev, C.DEV_MAX_TOKENS, "b_dev_1")
    E.save(recs, os.path.join(out, f"dev_{name}.jsonl.gz"), full_text=False)
    res["dev_check"][name] = {"summary": E.summary(recs)["all"], "paired_vs_unedited": E.paired(base, recs),
                              "seconds": round(secs)}
res["dev_check"]["unedited"] = {"summary": E.summary(base)["all"]}
json.dump(res, open(os.path.join(out, "direction_check.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "direction_similarity"}, indent=1))
print({L: {k: v for k, v in d.items() if k != "abs_cos_with_our_direction_by_layer"} for L, d in sim.items()})
