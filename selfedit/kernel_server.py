"""The editor's persistent Python kernel (Arm A), run as an unprivileged user: variables survive between code blocks,
like a notebook. Requests come as JSON lines on stdin ({"code": ...}); each answer is one JSON line written to the
file descriptor given in KERNEL_FD ({"output": ..., "error": ...}), so that code printing to stdout cannot break it.
The helpers below are the ones the blog post's task message lists: NAMES, load, load_original, save_edit, drop_edit,
edited, tok (the tokenizer). save_edit refuses what the authors' save_edit refused (the final norm, the embedding /
output layer, an edit that only multiplies a whole tensor by a constant) and, in addition, non-finite values."""
import ast
import contextlib
import io
import json
import os
import sys
import traceback

SETUP = r'''
import os, json, math
import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
torch.set_grad_enabled(False)
_F = safe_open("/model/model.safetensors", framework="pt")
NAMES = sorted(n for n in _F.keys() if n.startswith("model.language_model."))
_FROZEN = ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight")
EDIT_DIR = "/work/edited"
os.makedirs(EDIT_DIR, exist_ok=True)
tok = Tokenizer.from_file("/model/tokenizer.json")

def load_original(name):
    """the original tensor as float32"""
    if name not in NAMES:
        raise KeyError(f"unknown tensor name {name!r} (see NAMES)")
    return _F.get_tensor(name).float()

def load(name):
    """your current edit of the tensor if there is one, else the original (float32)"""
    p = os.path.join(EDIT_DIR, name + ".npy")
    if os.path.exists(p):
        return torch.from_numpy(np.load(p)).float()
    return load_original(name)

def save_edit(name, tensor):
    """make `tensor` the new value of weight `name` (takes effect on the model right after this code block)"""
    if name not in NAMES:
        raise KeyError(f"unknown tensor name {name!r} (see NAMES)")
    if name in _FROZEN:
        raise PermissionError(f"refused: {name} is out of scope (final norm / embedding / output layer)")
    t = torch.as_tensor(tensor).detach().float().cpu()
    o = load_original(name)
    if tuple(t.shape) != tuple(o.shape):
        raise ValueError(f"refused: shape {tuple(t.shape)} != {tuple(o.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError("refused: the tensor has NaN or infinite values")
    oo = float((o * o).sum())
    c = float((t * o).sum()) / oo if oo > 0 else 0.0
    if oo > 0 and abs(c - 1.0) > 1e-6 and float(((t - c * o) ** 2).sum()) <= 1e-12 * max(float((t * t).sum()), 1e-30):
        raise PermissionError("refused: this edit only multiplies the whole tensor by a constant")
    np.save(os.path.join(EDIT_DIR, name + ".npy"), t.numpy())
    return f"saved edit of {name}: {int((t != o).sum())} of {t.numel()} values differ from the original"

def drop_edit(name):
    """remove your edit of `name` (back to the original)"""
    p = os.path.join(EDIT_DIR, name + ".npy")
    if os.path.exists(p):
        os.remove(p)
        return f"dropped edit of {name}"
    return f"no edit of {name}"

def edited():
    """names of the tensors you have edited"""
    return sorted(f[:-4] for f in os.listdir(EDIT_DIR) if f.endswith(".npy"))
'''


def main():
    out = os.fdopen(int(os.environ["KERNEL_FD"]), "w", buffering=1)
    g = {"__name__": "__main__"}
    try:
        exec(SETUP, g)
        out.write(json.dumps({"ready": True}) + "\n")
    except BaseException:  # noqa: BLE001
        out.write(json.dumps({"ready": False, "error": traceback.format_exc()}) + "\n")
        return
    for line in sys.stdin:
        code = json.loads(line)["code"]
        buf, err = io.StringIO(), None
        try:
            tree = ast.parse(code, "<cell>", "exec")
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(tree, "<cell>", "exec"), g)
                if last is not None:
                    val = eval(compile(ast.Expression(last.value), "<cell>", "eval"), g)
                    if val is not None:
                        print(repr(val))
        except BaseException:  # noqa: BLE001
            err = traceback.format_exc(limit=6)
        out.write(json.dumps({"output": buf.getvalue(), "error": err}) + "\n")


if __name__ == "__main__":
    main()
