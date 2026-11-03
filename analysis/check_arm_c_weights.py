"""A check added after the main run (not pre-registered): does the generation engine (vLLM) hold the same weights as
the model Arm C trained (transformers)? Every tensor of the language model changes in Arm C, while the sanity check of
the main run changed one tensor of each kind at a time. Each rule's final weights (arm_c/<rule>/state.pt) are loaded
into both, and the token log-probs of one text are compared as in the sanity check; it also counts how many weights
the training changed.

Run on the machine of the main run, after it finished:  python analysis/check_arm_c_weights.py
Writes <out>/exploratory/arm_c_weight_check.json.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from selfedit import config as C  # noqa: E402
from selfedit.engine import Engine  # noqa: E402

from vllm import SamplingParams  # noqa: E402

engine = Engine()
text = (engine.chat([{"role": "user", "content": "Compute 3 + 4 * 2."}], False)
        + "We multiply first: 4 * 2 = 8, then 3 + 8 = 11. The answer is 11.")
ids = engine.tok(text, add_special_tokens=False).input_ids


def compare():
    hf = engine.token_logprobs(ids)
    o = engine.llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)[0]
    vl = torch.tensor([d[t].logprob for d, t in zip(o.prompt_logprobs[1:], ids[1:])])
    return {"max_abs_logprob_diff": float((hf - vl).abs().max()), "mean_abs_logprob_diff": float((hf - vl).abs().mean()),
            "mean_logprob_vllm": float(vl.mean()), "mean_logprob_transformers": float(hf.mean())}


res = {"note": "added after the main run; not pre-registered", "original": compare()}
total = sum(engine.original(n).numel() for n in engine.names if n not in C.FROZEN)
for rule in ("C1_answer_key", "C2_finished_only"):
    p = os.path.join(C.OUT, "arm_c", rule, "state.pt")
    if not os.path.exists(p):
        continue
    st = torch.load(p)
    engine.set_edits(st["weights"])
    ch = engine.changed_entries()
    res[rule] = {"after_round": st["round"], "tensors_changed": len(ch), "weights_changed": sum(ch.values()),
                 "weights_total": total, "engines_agree": compare()}
    engine.set_edits({})
res["restored"] = compare()
out = os.path.join(C.OUT, "exploratory")
os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, "arm_c_weight_check.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
