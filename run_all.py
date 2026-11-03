"""Runs the whole experiment in the pre-registered order (PREREG.md section 4). Resumable: every finished stage leaves
a marker in OUT/done/, and a restarted run skips it.

  python run_all.py            # the real run
  python run_all.py --pilot    # tiny sizes, only to find bugs (its numbers are not results)
"""
import argparse
import json
import os
import sys
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--pilot", action="store_true")
ap.add_argument("--only", default="", help="comma-separated stages to run (default: all)")
ap.add_argument("--with-c", action="store_true", help="also run Arm C (self-training)")
ARGS = ap.parse_args()
if ARGS.pilot:
    os.environ["SELFEDIT_WORK"] = os.environ.get("SELFEDIT_WORK_PILOT", "/content/selfedit_pilot")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from selfedit import config as C  # noqa: E402

if ARGS.pilot:
    C.N_TEST_EACH, C.TEST_MAX_TOKENS, C.DEV_MAX_TOKENS = 6, 3072, 2048
    C.B_DIAG_N, C.B_DEV_N, C.B_ROUNDS, C.B_TEST_AFTER, C.B_DIR_TEXTS = 24, 8, 2, (2,), 12
    C.B_CHOICE_SAMPLES, C.B_TABLE_SAMPLES, C.B_WINDOW = 3, 4, 256
    C.A_SEEDS, C.A_TURNS, C.A_PASSES, C.A_GEN_MAX = (1,), 4, 2, 8
    C.A_TURN_TOKENS = {"think": 1024, "nothink": 1024}
    C.C_ROUNDS, C.C_GEN_N = 1, 16

from selfedit import data as D  # noqa: E402
from selfedit import evaluate as E  # noqa: E402

os.makedirs(C.OUT, exist_ok=True)
STAGES = [s for s in ARGS.only.split(",") if s]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(C.OUT, "run.log"), "a") as f:
        f.write(line + "\n")
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "last": msg}, open(os.path.join(C.OUT, "progress.json"), "w"))


def done(stage):
    return os.path.exists(os.path.join(C.OUT, "done", stage))


def mark(stage, info=None):
    os.makedirs(os.path.join(C.OUT, "done"), exist_ok=True)
    json.dump(info or {}, open(os.path.join(C.OUT, "done", stage), "w"), indent=1)
    log(f"stage done: {stage}")


def want(stage):
    return not STAGES or stage in STAGES


def setup():
    from huggingface_hub import hf_hub_download, snapshot_download
    try:
        C.weights_file()
    except (OSError, AssertionError):
        snapshot_download(C.MODEL_ID, revision=C.MODEL_REV, local_dir=C.MODEL_DIR)
    if not os.path.exists(os.path.join(C.DATA, "ids.json")):
        meta = D.build(os.path.join(HERE, "data"))
        json.dump(meta, open(os.path.join(C.OUT, "data_ids.json"), "w"), indent=1)
    apath = os.path.join(C.OUT, "astra_edit.pt")
    if not os.path.exists(apath):
        import torch
        from safetensors import safe_open
        from huggingface_hub import list_repo_files
        st_files = sorted(x for x in list_repo_files(C.ASTRA_ID, revision=C.ASTRA_REV) if x.endswith(".safetensors"))
        f = hf_hub_download(C.ASTRA_ID, st_files[0], revision=C.ASTRA_REV)   # named model.safetensors-00001-of-00001.safetensors
        base = safe_open(C.weights_file(), framework="pt")
        ed = safe_open(f, framework="pt")
        diff = {}
        for n in base.keys():
            if n in ed.keys() and not torch.equal(base.get_tensor(n), ed.get_tensor(n)):
                diff[n] = int((base.get_tensor(n) != ed.get_tensor(n)).sum())
        edit = {n: ed.get_tensor(n) for n in C.ASTRA_TENSORS}
        torch.save(edit, apath)
        json.dump({"tensors_that_differ": diff, "changed_entries": sum(diff.values())},
                  open(os.path.join(C.OUT, "astra_edit_check.json"), "w"), indent=1)
        log(f"Astra edit: {len(diff)} tensors differ from the base, {sum(diff.values())} entries")


def sanity(engine):
    """the weight updates reach vLLM and transformers alike, and restoring the original restores the outputs."""
    import torch
    from vllm import SamplingParams
    out = {}
    p = engine.chat([{"role": "user", "content": "What is 17 * 23? Answer with the number only."}], False)
    g = SamplingParams(temperature=0.0, max_tokens=24)
    t0 = engine.generate([p], [g])[0]["text"]
    name = f"{C.LM_PREFIX}layers.15.mlp.down_proj.weight"
    engine.set_edits({name: torch.zeros_like(engine.original(name), dtype=torch.float32)})
    t1 = engine.generate([p], [g])[0]["text"]
    sums = engine.vllm_sums([name])
    engine.set_edits({})
    t2 = engine.generate([p], [g])[0]["text"]
    out["greedy"] = {"original": t0, "layer15_mlp_zeroed": t1, "restored": t2, "restored_equals_original": t0 == t2,
                     "zeroed_changes_output": t0 != t1, "vllm_sum_after_zeroing": sums}
    # transformers and vLLM must read the same weights: compare token log-probs of one text, per tensor type
    text = engine.chat([{"role": "user", "content": "Compute 3 + 4 * 2."}], False) + "We multiply first: 4 * 2 = 8, then 3 + 8 = 11. The answer is 11."
    ids = engine.tok(text, add_special_tokens=False).input_ids

    def compare():
        hf = engine.token_logprobs(ids)
        o = engine.llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)[0]
        vl = torch.tensor([d[t].logprob for d, t in zip(o.prompt_logprobs[1:], ids[1:])])
        return float((hf - vl).abs().max()), float((hf - vl).abs().mean()), float(vl.mean())

    out["hf_vs_vllm_original"] = compare()
    gen = torch.Generator().manual_seed(0)
    per_type = {}
    for n in engine.names:
        kind = n.split("layers.")[-1].split(".", 1)[-1] if "layers." in n else n
        if kind in per_type or n in C.FROZEN:
            continue
        o_ = engine.original(n).float()
        noise = torch.randn(o_.shape, generator=gen) * (o_.std() if o_.numel() > 1 else o_.abs() + 1e-3) * 0.05
        engine.set_edits({n: o_ + noise})
        per_type[kind] = {"tensor": n, "max_abs_logprob_diff": compare()}
        engine.set_edits({})
    out["hf_vs_vllm_after_noise_by_tensor_type"] = per_type
    out["hf_vs_vllm_restored"] = compare()
    json.dump(out, open(os.path.join(C.OUT, "sanity.json"), "w"), indent=1)
    bad = {k: v for k, v in per_type.items() if v["max_abs_logprob_diff"][0] > 0.25}
    log(f"sanity: restored={t0 == t2} zeroed_changes={t0 != t1} hf/vllm max diff original={out['hf_vs_vllm_original'][0]:.3f}; "
        f"tensor types with mismatch: {sorted(bad)}")
    return out, bad


def test_eval(engine, name, edits, test):
    path = os.path.join(C.OUT, "test", f"{name}.jsonl.gz")
    if os.path.exists(path):
        return E.load(path)
    with engine.using(edits):
        recs, secs = E.answer(engine, test, C.TEST_MAX_TOKENS, "test")
    E.save(recs, path, full_text=True)
    s = E.summary(recs)
    json.dump({"summary": s, "seconds": round(secs), "edits": sorted(edits)},
              open(os.path.join(C.OUT, "test", f"{name}.summary.json"), "w"), indent=1)
    log(f"test {name}: acc {s['all']['acc']:.3f} cut {s['all']['cut_share']:.3f} ({secs / 60:.1f} min)")
    return recs


def main():
    import torch
    log(f"start (pilot={ARGS.pilot}) work={C.WORK}")
    import platform
    import subprocess
    import vllm
    import transformers
    env = {"python": platform.python_version(), "torch": torch.__version__, "vllm": vllm.__version__,
           "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
           "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
           "code_commit": subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "pilot": ARGS.pilot, "args": sys.argv[1:]}
    json.dump(env, open(os.path.join(C.OUT, f"env_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
    log(f"env: {env}")
    setup()
    test, practice, pools = D.load_all()
    from selfedit.engine import Engine
    engine = Engine()
    log(f"engine ready: {len(engine.names)} language-model tensors")
    if not done("sanity"):
        out, bad = sanity(engine)
        if not out["greedy"]["restored_equals_original"] or not out["greedy"]["zeroed_changes_output"]:
            log("STOP: weight updates do not reach the generation engine")
            return
        mark("sanity", {"mismatched_tensor_types": sorted(bad)})

    # 1. unedited model and positive control; stop rule
    base = test_eval(engine, "unedited", {}, test)
    astra = test_eval(engine, "astra_positive_control", torch.load(os.path.join(C.OUT, "astra_edit.pt")), test)
    pc = E.paired(base, astra)
    json.dump({"unedited": E.summary(base), "astra": E.summary(astra), "paired": pc},
              open(os.path.join(C.OUT, "test", "positive_control.json"), "w"), indent=1)
    log(f"positive control gain {pc['gain']:+.3f} (se {pc['se']:.3f})")
    if not ARGS.pilot and pc["gain"] < 0.15:
        log("STOP RULE: the positive control gained less than 15 points; the measurement is broken")
        return
    mark("tests_base", {"positive_control": pc})

    # 2. Arm B
    from selfedit import arm_b as B
    bdir = os.path.join(C.OUT, "arm_b")
    os.makedirs(bdir, exist_ok=True)
    if want("B") and not done("arm_b"):
        dres = B.directions(engine, pools["b_diag"], bdir, log)
        table = B.round1_table(engine, dres["u"], pools["b_dev_1"], dres["diag_records"], bdir, log)
        for name, chooser, ref in (("B1_self_referee", "self", True), ("B2_random_referee", "random", True),
                                   ("B3_self_no_referee", "self", False), ("B4_random_no_referee", "random", False)):
            B.trajectory(engine, name, chooser, ref, dres["u"], pools, table, dres["diag_records"], test, base, bdir, log)
        mark("arm_b")

    # 3. Arm A
    from selfedit import arm_a as A
    if want("A"):
        A.setup_fs()
        for variant in C.A_VARIANTS:
            for seed in C.A_SEEDS:
                stage = f"arm_a_{variant}_s{seed}"
                if done(stage):
                    continue
                d = os.path.join(C.OUT, "arm_a", f"{variant}_s{seed}")
                try:
                    summ, edits = A.run_one(engine, variant, seed, practice, d, log)
                except Exception:  # noqa: BLE001
                    log(f"{stage} crashed:\n{traceback.format_exc()}")
                    engine.set_edits({})
                    continue
                if edits:
                    recs = test_eval(engine, f"arm_a_{variant}_s{seed}", edits, test)
                    summ["test"] = {"summary": E.summary(recs), "paired": E.paired(base, recs)}
                else:
                    summ["test"] = {"same_as": "unedited", "summary": E.summary(base), "paired": E.paired(base, base)}
                json.dump(summ, open(os.path.join(d, "summary.json"), "w"), indent=1)
                engine.set_edits({})
                mark(stage, summ)

    # 4. Arm C (optional)
    if ARGS.with_c and want("C"):
        from selfedit import arm_c as AC
        AC.run(engine, pools, test, base, os.path.join(C.OUT, "arm_c"), log, mark, done)
    log("all stages finished")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        log("FATAL\n" + traceback.format_exc())
        raise
