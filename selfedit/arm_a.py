"""Arm A (PREREG.md section 3): the small model gets the blog post's tools and edits its own weights.

The editor IS the edited model: after every code block, whatever is in /work/edited is loaded into the engine, so
the next reply is written by the edited weights. The kernel runs as the unprivileged user `editor`, which can read
/model and /practice and write only /work."""
import json
import os
import pwd
import re
import select
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from . import config as C
from . import evaluate as E
from . import mathqa as Q

AGENT_USER = "editor"
SYSTEM = "Answer the user's request."
BLOCK = re.compile(r"```(python|forward)[ \t]*\n(.*?)```", re.S)
HERE = os.path.dirname(os.path.abspath(__file__))
TASK_TXT = os.path.join(os.path.dirname(HERE), "prompts", "arm_a_task.txt")
MODEL_FILES = ("model.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json", "chat_template.jinja",
               "generation_config.json", "vocab.json", "merges.txt")


def _ids(user):
    p = pwd.getpwnam(user)
    return p.pw_uid, p.pw_gid


def setup_fs():
    """the editor's view: /model (read-only links to the model files), /practice (copies), an unprivileged user."""
    try:
        pwd.getpwnam(AGENT_USER)
    except KeyError:
        subprocess.run(["useradd", "-M", "-s", "/bin/false", AGENT_USER], check=True)
    os.makedirs("/model", exist_ok=True)
    for f in MODEL_FILES:
        src, dst = os.path.join(C.MODEL_DIR, f), os.path.join("/model", f)
        if f == "model.safetensors":
            src = C.weights_file()
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    os.makedirs("/practice", exist_ok=True)
    for f in ("gsm8k_train.jsonl", "math_train.jsonl"):
        shutil.copy(os.path.join(C.DATA, f), os.path.join("/practice", f))
    os.chmod(C.WORK, 0o711)
    for d in (C.OUT, C.DATA):
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o700)
    subprocess.run(["chmod", "-R", "a+rX", C.MODEL_DIR, "/model", "/practice", os.path.dirname(HERE)], check=False)


def fresh_work():
    if os.path.exists("/work"):
        shutil.rmtree("/work")
    os.makedirs("/work/edited")
    uid, gid = _ids(AGENT_USER)
    for d in ("/work", "/work/edited"):
        os.chown(d, uid, gid)


def chown_tree(path):
    uid, gid = _ids(AGENT_USER)
    for root, dirs, files in os.walk(path):
        for n in dirs + files:
            try:
                os.chown(os.path.join(root, n), uid, gid)
            except OSError:
                pass
    os.chown(path, uid, gid)


class Kernel:
    def __init__(self):
        self.p, self.r, self.buf = None, None, b""

    def start(self):
        r, w = os.pipe()
        uid, gid = _ids(AGENT_USER)

        def demote():
            os.setsid()
            os.setgid(gid)
            os.setuid(uid)

        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/work", "KERNEL_FD": str(w),
               "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1"}
        self.p = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "kernel_server.py")], stdin=subprocess.PIPE,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd="/work", env=env,
                                  pass_fds=(w,), preexec_fn=demote)
        os.close(w)
        self.r, self.buf = r, b""
        msg = self._readline(180)
        if not msg or not msg.get("ready"):
            raise RuntimeError(f"kernel did not start: {msg}")

    def _readline(self, timeout):
        end = time.time() + timeout
        while b"\n" not in self.buf:
            left = end - time.time()
            if left <= 0:
                return None
            ready, _, _ = select.select([self.r], [], [], left)
            if not ready:
                return None
            chunk = os.read(self.r, 1 << 20)
            if not chunk:
                return {"eof": True}
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)

    def stop(self):
        if self.p is not None:
            try:
                os.killpg(self.p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.p.wait()
            os.close(self.r)
            self.p = None

    def run(self, code, timeout):
        try:
            self.p.stdin.write((json.dumps({"code": code}) + "\n").encode())
            self.p.stdin.flush()
            msg = self._readline(timeout)
        except BrokenPipeError:
            msg = {"eof": True}
        if msg is None or msg.get("eof"):
            why = (f"the code ran longer than {timeout} seconds" if msg is None else "the kernel process died")
            self.stop()
            self.start()
            return {"output": "", "error": f"{why}; the kernel was restarted and its variables are lost", "restarted": True}
        return msg


class EditSync:
    """loads /work/edited into the engine after every code block (cached by file mtime and size)."""

    def __init__(self, engine):
        self.engine, self.cache = engine, {}

    def __call__(self):
        new, refused = {}, []
        d = "/work/edited"
        names = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npy")) if os.path.isdir(d) else []
        for name in names:
            path = os.path.join(d, name + ".npy")
            st = os.stat(path)
            key = (st.st_mtime_ns, st.st_size)
            if self.cache.get(name, (None,))[0] == key:
                new[name] = self.cache[name][1]
                continue
            why = None
            if name not in self.engine.names or name in C.FROZEN:
                why = "not an editable tensor name"
            else:
                try:
                    t = torch.from_numpy(np.load(path, allow_pickle=False)).float()
                except Exception as e:  # noqa: BLE001
                    why = f"unreadable ({type(e).__name__})"
                else:
                    o = self.engine.original(name).float()
                    if tuple(t.shape) != tuple(o.shape):
                        why = f"shape {tuple(t.shape)} != {tuple(o.shape)}"
                    elif not torch.isfinite(t).all():
                        why = "NaN or infinite values"
                    else:
                        oo = float((o * o).sum())
                        c = float((t * o).sum()) / oo if oo > 0 else 0.0
                        if oo > 0 and abs(c - 1) > 1e-6 and float(((t - c * o) ** 2).sum()) <= 1e-12 * max(float((t * t).sum()), 1e-30):
                            why = "only multiplies the whole tensor by a constant"
            if why:
                os.makedirs("/work/refused", exist_ok=True)
                shutil.move(path, os.path.join("/work/refused", name + ".npy"))
                refused.append(f"{name}: {why}")
                self.cache.pop(name, None)
                continue
            self.cache[name] = (key, t)
            new[name] = t
        for name in list(self.cache):
            if name not in new:
                self.cache.pop(name)
        changed = self.engine.set_edits(new)
        return {"changed": changed, "refused": refused}


def _status(state, turn):
    ed = sorted(state["engine"].edits)
    return (f"\n[Python runs left: {C.A_PY_RUNS - state['py']}. Forward passes left: {C.A_PASSES - state['passes']}. "
            f"Turns left: {C.A_TURNS - turn}. Tensors you have edited (now in effect on you): "
            f"{', '.join(ed) if ed else 'none'}.]")


def _clip(s, n):
    return s if len(s) <= n else s[:n] + f"\n... [cut: {len(s) - n} more characters]"


def _gen_pass(engine, req, practice, k):
    model = req.get("model", "edited")
    if model not in ("original", "edited", "both"):
        raise ValueError('"model" must be "original", "edited" or "both"')
    ids, prompts = req.get("problems") or [], req.get("prompts") or []
    if not isinstance(ids, list) or not isinstance(prompts, list):
        raise ValueError('"problems" and "prompts" must be lists')
    unknown = [i for i in ids if i not in practice]
    if unknown:
        raise ValueError(f"unknown practice ids: {unknown[:5]}")
    n = int(req.get("n", 1))
    mt = int(req.get("max_tokens", C.DEV_MAX_TOKENS))
    if n < 1 or mt < 1 or mt > C.DEV_MAX_TOKENS:
        raise ValueError(f'"n" must be >= 1 and "max_tokens" between 1 and {C.DEV_MAX_TOKENS}')
    items = [practice[i] for i in ids] + [{"id": f"prompt_{j}", "source": "own_prompt", "prompt": str(p)}
                                          for j, p in enumerate(prompts)]
    limit = C.A_GEN_MAX // 2 if model == "both" else C.A_GEN_MAX
    if not items or len(items) * n > limit:
        raise ValueError(f"items x n must be between 1 and {limit} (got {len(items) * n})")

    def run():
        prompts_, ps, meta = [], [], []
        for it in items:
            user = it["prompt"] if "prompt" in it else Q.build_prompt(it["problem"])
            for s in range(n):
                prompts_.append(engine.chat([{"role": "user", "content": user}], True))
                ps.append(E.params(True, mt, E.seed_of(f"{it['id']}#{s}", "arm_a_gen")))
                meta.append((it, s))
        outs = engine.generate(prompts_, ps)
        recs = []
        for (it, s), o in zip(meta, outs):
            pred, _ = Q.extract(o["text"])
            graded = "answer" in it
            recs.append({"id": it["id"], "source": it["source"], "sample": s, "pred": pred,
                         "correct": int(Q.is_correct(pred, it["answer"])) if graded else None,
                         "finish": o["finish"], "tokens": o["tokens"], "text": o["text"]})
        return recs

    def summ(recs):
        g = [r for r in recs if r["correct"] is not None]
        by = {}
        for s in sorted({r["source"] for r in g}):
            rs = [r for r in g if r["source"] == s]
            by[s] = round(sum(r["correct"] for r in rs) / len(rs), 4)
        return {"answers": len(recs), "accuracy": round(sum(r["correct"] for r in g) / len(g), 4) if g else None,
                "accuracy_by_source": by, "mean_tokens": round(sum(r["tokens"] for r in recs) / len(recs), 1),
                "share_cut_at_max_tokens": round(sum(r["finish"] == "length" for r in recs) / len(recs), 4),
                "failures": {"cut_at_max_tokens": sum(r["finish"] == "length" for r in g),
                             "wrong_answer_finished": sum(r["finish"] != "length" and r["pred"] is not None and not r["correct"] for r in g),
                             "no_answer_finished": sum(r["finish"] != "length" and r["pred"] is None for r in g)}}

    def dump(recs, path):
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    res = {"pass": k, "type": "generate", "model": model}
    if model in ("original", "both"):
        with engine.using({}):
            ro = run()
    if model in ("edited", "both"):
        re_ = run()
    if model == "both":
        dump(ro, f"/work/gen_{k}_original.jsonl")
        dump(re_, f"/work/gen_{k}_edited.jsonl")
        po = E.paired([{"id": f"{r['id']}#{r['sample']}", "correct": r["correct"]} for r in ro if r["correct"] is not None],
                      [{"id": f"{r['id']}#{r['sample']}", "correct": r["correct"]} for r in re_ if r["correct"] is not None])
        res.update(original=summ(ro), edited=summ(re_),
                   paired={"difference_edited_minus_original": round(po.get("gain", 0), 4), "se": round(po.get("se", 0), 4),
                           "wrong_to_right": po.get("rescues", 0), "right_to_wrong": po.get("losses", 0)},
                   files=[f"/work/gen_{k}_original.jsonl", f"/work/gen_{k}_edited.jsonl"])
    else:
        recs = ro if model == "original" else re_
        dump(recs, f"/work/gen_{k}.jsonl")
        res.update(summ(recs), files=[f"/work/gen_{k}.jsonl"])
    return res


def _tf_pass(engine, req, k):
    model = req.get("model", "edited")
    if model not in ("original", "edited"):
        raise ValueError('"model" must be "original" or "edited"')
    texts, chats = req.get("texts"), req.get("chats")
    if chats:
        texts = [engine.tok.apply_chat_template(c, tokenize=False, enable_thinking=bool(req.get("enable_thinking", False)))
                 for c in chats]
    if not texts or not isinstance(texts, list) or len(texts) > C.A_TF_MAX_TEXTS:
        raise ValueError(f'give 1 to {C.A_TF_MAX_TEXTS} "texts" or "chats"')
    ret = req.get("return") or {}
    want_lp, layers = bool(ret.get("token_logprobs")), ret.get("hidden_states") or []
    if not isinstance(layers, list) or any((not isinstance(l, int)) or l < 0 or l > C.N_LAYERS for l in layers):
        raise ValueError('"hidden_states" must be a list of layer numbers 0..24')
    if not want_lp and not layers:
        raise ValueError('ask for "token_logprobs" and/or "hidden_states" in "return"')
    ids = [engine.tok(str(t), add_special_tokens=False).input_ids for t in texts]
    long_ = [i for i, x in enumerate(ids) if len(x) > C.A_TF_MAX_TOKENS or len(x) < 2]
    if long_:
        raise ValueError(f"texts {long_[:5]} are longer than {C.A_TF_MAX_TOKENS} tokens or shorter than 2")
    d = f"/work/fwd_{k}"
    os.makedirs(d, exist_ok=True)
    out = {"pass": k, "type": "teacher_forced", "model": model, "texts": len(ids), "tokens": [len(x) for x in ids],
           "folder": d}
    unsupported = sorted(set(ret) - {"token_logprobs", "hidden_states"})
    if unsupported:
        out["not_available_in_this_harness"] = unsupported
    with engine.using({} if model == "original" else dict(engine.edits)):
        if layers:
            hm = engine.hidden_means([(x, 0, len(x)) for x in ids], layers)
            np.save(os.path.join(d, "hidden_mean.npy"), hm.numpy())
        if want_lp:
            means = []
            for i, x in enumerate(ids):
                lp = engine.token_logprobs(x)
                np.save(os.path.join(d, f"token_logprobs_{i}.npy"), lp.numpy())
                means.append(round(float(lp.mean()), 4))
            out["mean_token_logprob_per_text"] = means
    json.dump({"texts": len(ids), "tokens": out["tokens"], "hidden_state_layers": layers}, open(os.path.join(d, "meta.json"), "w"))
    return out


def _conversation(engine, task, msgs, notes):
    """system + task + as many of the latest exchanges as fit in the prompt budget."""
    keep = len(msgs)
    while True:
        conv = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
        tail = msgs[len(msgs) - keep:]
        if keep < len(msgs) and tail and tail[0]["role"] == "user":
            tail = tail[1:]
        if keep < len(msgs):
            head = "(Older turns were dropped to save space.)"
            if notes:
                head += " Your /work/notes.md:\n" + notes[:2000]
            conv.append({"role": "assistant", "content": "(earlier work)"})
            conv.append({"role": "user", "content": head})
        conv += tail
        p = engine.chat(conv, False)
        if len(engine.tok(p, add_special_tokens=False).input_ids) <= C.A_PROMPT_BUDGET or keep <= 2:
            return conv, keep < len(msgs)
        keep -= 2


def harness_check(engine, practice, out_dir, log=print):
    """a scripted editor (not the model) uses every tool once, so that the record shows the tools work: an edit saved
    in the kernel reaches the engine, a frozen tensor is refused, the kernel cannot read the experiment's files, both
    kinds of forward pass run, and dropping the edit restores the original."""
    os.makedirs(out_dir, exist_ok=True)
    engine.set_edits({})
    fresh_work()
    kern = Kernel()
    kern.start()
    sync = EditSync(engine)
    name = f"{C.LM_PREFIX}layers.15.mlp.down_proj.weight"
    res = {}
    r = kern.run(f"W = load({name!r})\nW[:, :8] = 0\nprint(save_edit({name!r}, W))\nprint(len(NAMES), edited())\n"
                 "print(tok.encode('2 + 2 = 4').ids)", C.A_PY_TIMEOUT)
    s = sync()
    res["save_edit"] = {"output": r.get("output"), "error": r.get("error"), "changed": s["changed"],
                        "refused": s["refused"], "engine_changed_entries": engine.changed_entries()}
    r = kern.run("try:\n    save_edit('model.language_model.norm.weight', load('model.language_model.norm.weight') + 1)\n"
                 "except Exception as e:\n    print(type(e).__name__, e)", C.A_PY_TIMEOUT)
    res["frozen_refused"] = {"output": r.get("output"), "error": r.get("error")}
    r = kern.run(f"try:\n    print(os.listdir({C.OUT!r}))\nexcept Exception as e:\n    print('blocked:', type(e).__name__)",
                 C.A_PY_TIMEOUT)
    res["sandbox"] = {"output": r.get("output"), "error": r.get("error")}
    ids = sorted(practice)[:2]
    res["generate"] = _gen_pass(engine, {"type": "generate", "model": "both", "problems": ids, "n": 1, "max_tokens": 512},
                                practice, 1)
    res["teacher_forced"] = _tf_pass(engine, {"type": "teacher_forced", "model": "edited", "texts": ["2 + 2 = 4", "The cat sat."],
                                              "return": {"token_logprobs": True, "hidden_states": [0, 12, 24]}}, 2)
    r = kern.run(f"print(drop_edit({name!r}))", C.A_PY_TIMEOUT)
    s2 = sync()
    res["drop_edit"] = {"output": r.get("output"), "error": r.get("error"), "changed": s2["changed"],
                        "engine_edits_after": sorted(engine.edits)}
    kern.stop()
    engine.set_edits({})
    res["ok"] = bool(not res["save_edit"]["error"] and res["save_edit"]["changed"] == [name]
                     and "PermissionError" in (res["frozen_refused"]["output"] or "")
                     and "blocked" in (res["sandbox"]["output"] or "")
                     and res["generate"]["edited"]["answers"] == 2 and res["teacher_forced"]["texts"] == 2
                     and s2["changed"] == [name] and not res["drop_edit"]["engine_edits_after"])
    json.dump(res, open(os.path.join(out_dir, "harness_check.json"), "w"), indent=1, default=str)
    log(f"[A] harness check (a scripted editor, not the model): ok={res['ok']}")
    return res


def run_one(engine, variant, seed, practice, out_dir, log=print):
    thinking = variant == "think"
    os.makedirs(out_dir, exist_ok=True)
    task = open(TASK_TXT, encoding="utf-8").read()
    engine.set_edits({})
    fresh_work()
    kern = Kernel()
    kern.start()
    sync = EditSync(engine)
    state = {"py": 0, "passes": 0, "engine": engine}
    msgs, turns = [], []
    tpath = os.path.join(out_dir, "turns.jsonl")
    open(tpath, "w").close()
    t_start = time.time()
    for t in range(1, C.A_TURNS + 1):
        notes = open("/work/notes.md", errors="replace").read() if os.path.exists("/work/notes.md") else ""
        conv, dropped = _conversation(engine, task, msgs, notes)
        prompt = engine.chat(conv, thinking)
        sp = E.params(thinking, C.A_TURN_TOKENS[variant], E.seed_of(f"{variant}|{seed}|{t}", "arm_a_turn"))
        t0 = time.time()
        o = engine.generate([prompt], [sp])[0]
        text = o["text"]
        if thinking:
            final = text.split("</think>", 1)[1] if "</think>" in text else ""
        else:
            final = text.split("</think>")[-1]
        rec = {"turn": t, "prompt_tokens": len(o["prompt_ids"]), "reply_tokens": o["tokens"], "finish": o["finish"],
               "dropped_old_turns": dropped, "reply": text, "block": None, "gen_seconds": round(time.time() - t0, 1)}
        m = BLOCK.search(final)
        if m is None:
            if thinking and "</think>" not in text:
                fb = (f"Your reply was cut off while you were still thinking (a reply can be at most "
                      f"{C.A_TURN_TOKENS[variant]:,} tokens), so no block was found and nothing was run.")
            else:
                fb = "Your reply contained no ```python or ```forward block, so nothing was run."
            fb += " To act, reply with exactly one ```python block or one ```forward block."
        elif m.group(1) == "python":
            code = m.group(2)
            rec.update(block="python", code=code)
            if state["py"] >= C.A_PY_RUNS:
                fb = "You have no Python runs left; the code was not run."
            else:
                state["py"] += 1
                t1 = time.time()
                r = kern.run(code, C.A_PY_TIMEOUT)
                s = sync()
                rec.update(output=r.get("output", ""), error=r.get("error"), restarted=bool(r.get("restarted")),
                           run_seconds=round(time.time() - t1, 1), edits_changed=s["changed"], edits_refused=s["refused"])
                fb = "Output:\n" + (_clip(r.get("output", ""), C.A_OUTPUT_CHARS) or "(nothing printed)")
                if r.get("error"):
                    fb += "\nError:\n" + r["error"][-2000:]
                if s["changed"]:
                    fb += f"\nYour edits of {len(s['changed'])} tensor(s) are now in effect on you."
                if s["refused"]:
                    fb += "\nRefused files in /work/edited (moved to /work/refused): " + "; ".join(s["refused"])
        else:
            rec.update(block="forward", forward_text=m.group(2))
            try:
                req = json.loads(m.group(2))
                if not isinstance(req, dict):
                    raise ValueError("the block must hold one JSON object")
            except ValueError as e:
                fb = f"Could not read the JSON of your forward block ({e}); not counted."
                rec["forward_error"] = str(e)
            else:
                if state["passes"] >= C.A_PASSES:
                    fb = "You have no forward passes left; not run."
                else:
                    k = state["passes"] + 1
                    try:
                        t1 = time.time()
                        if req.get("type") == "generate":
                            res = _gen_pass(engine, req, practice, k)
                        elif req.get("type") == "teacher_forced":
                            res = _tf_pass(engine, req, k)
                        else:
                            raise ValueError('"type" must be "generate" or "teacher_forced"')
                    except Exception as e:  # noqa: BLE001 - a bad request must not end the run
                        fb = f"Forward request refused, not counted: {type(e).__name__}: {e}"
                        rec["forward_error"] = f"{type(e).__name__}: {e}"
                    else:
                        state["passes"] = k
                        chown_tree("/work")
                        rec.update(forward_result=res, forward_seconds=round(time.time() - t1, 1))
                        fb = f"Forward pass {k} done:\n" + json.dumps(res)
        fb += _status(state, t)
        rec["feedback"] = fb
        rec["edits_now"] = engine.changed_entries()
        msgs.append({"role": "assistant", "content": final.strip() or "(this reply was cut off while thinking)"})
        msgs.append({"role": "user", "content": fb})
        turns.append(rec)
        with open(tpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"[A {variant} s{seed}] turn {t}: {rec['reply_tokens']} tok, finish={rec['finish']}, block={rec['block']}, "
            f"edits={len(engine.edits)}")
    kern.stop()
    final_edits = dict(engine.edits)
    if os.path.isdir("/work"):
        shutil.copytree("/work", os.path.join(out_dir, "work"), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.npy") if sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk("/work") for f in fs) > 3e8 else None)
    blocks = [r["block"] for r in turns]
    py_ok = sum(1 for r in turns if r["block"] == "python" and not r.get("error"))
    fw_ok = sum(1 for r in turns if r["block"] == "forward" and "forward_result" in r)
    summary = {"variant": variant, "seed": seed, "turns": len(turns), "minutes": round((time.time() - t_start) / 60, 1),
               "python_blocks": blocks.count("python"), "python_ok": py_ok, "forward_blocks": blocks.count("forward"),
               "forward_ok": fw_ok, "no_block": blocks.count(None),
               "cut_while_thinking": sum(1 for r in turns if thinking and "</think>" not in r["reply"]),
               "valid_action_share": round((py_ok + fw_ok) / len(turns), 4),
               "save_edit_calls": sum(r.get("code", "").count("save_edit(") for r in turns),
               "final_edits": {n: int((t != engine.original(n)).sum()) for n, t in final_edits.items()},
               "final_changed_weights": sum(int((t != engine.original(n)).sum()) for n, t in final_edits.items())}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    if final_edits:
        torch.save({n: t.clone() for n, t in final_edits.items()}, os.path.join(out_dir, "final_edit.pt"))
    return summary, final_edits
