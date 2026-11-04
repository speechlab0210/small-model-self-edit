"""Replication of Arm C with the corrected training (PREREG.md section 7). Resumable like run_all.py. Use a new work
folder: it never touches the main run's results.

  python run_replication.py            # the real run
  python run_replication.py --pilot    # tiny sizes, only to find bugs (its numbers are not results)
"""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "colab"))

import run_all as RA  # noqa: E402  (reads --pilot; shares the setup, the sanity check, the test evaluation and the log)
from packing import pack_results  # noqa: E402
from selfedit import replication as R  # noqa: E402

C, D, E = RA.C, RA.D, RA.E
if RA.ARGS.pilot:
    C.R_SEEDS, C.R_ROUNDS, C.R_FRESH_AFTER = (1,), 2, (2,)
ARCHIVE = os.environ.get("SELFEDIT_ARCHIVE", "/content/selfedit_replication_results.tar.gz")


def pack():
    """a current archive after every stage: if the machine is lost, what has finished can still be downloaded."""
    tmp = ARCHIVE + ".tmp"
    pack_results(C.WORK, tmp)
    os.replace(tmp, ARCHIVE)
    RA.log(f"packed {ARCHIVE}: {os.path.getsize(ARCHIVE) / 1e6:.1f} MB")


def main():
    import torch
    from selfedit.run_protocol import check_protocol
    check_protocol(C.OUT)
    RA.log(f"replication start (pilot={RA.ARGS.pilot}) work={C.WORK}")
    RA.write_env()
    RA.setup()
    test, practice, _ = D.load_all()
    meta = json.load(open(os.path.join(C.DATA, "ids.json")))
    if not RA.ARGS.pilot:
        old = json.load(open(os.path.join(HERE, "results", "raw", "data_ids.json")))
        if meta["test_ids"] != old["test_ids"] or meta["pools"] != old["pools"]:
            raise RuntimeError("STOP: the rebuilt test problems or practice pools differ from the main run's")
    used = {i for ids in meta["pools"].values() for i in ids}
    out_dir = os.path.join(C.OUT, "replication")
    fresh, pools = R.build_data(test, practice, used, *D.test_sources(), out_dir)

    from selfedit.engine import Engine
    engine = Engine()
    RA.log(f"engine ready: {len(engine.names)} language-model tensors")
    if not RA.done("sanity"):
        out, bad = RA.sanity(engine)
        if not out["greedy"]["restored_equals_original"] or not out["greedy"]["zeroed_changes_output"]:
            raise RuntimeError("STOP: weight updates do not reach the generation engine")
        if bad:
            raise RuntimeError(f"STOP: engine log-prob mismatch exceeds 0.25 for {sorted(bad)}")
        RA.mark("sanity", {"mismatched_tensor_types": sorted(bad)})

    # 1. unedited model and positive control on both sets of test problems; the same stop rule as the main run
    astra_edit = torch.load(os.path.join(C.OUT, "astra_edit.pt"))
    base = RA.test_eval(engine, "unedited", {}, test)
    astra = RA.test_eval(engine, "astra_positive_control", astra_edit, test)
    pc = E.paired(base, astra)
    if not RA.ARGS.pilot and pc["gain"] < 0.15:
        raise RuntimeError("STOP RULE: positive control gained less than 15 points; investigate before continuing")
    fresh_dir = os.path.join(out_dir, "fresh_test")
    os.makedirs(fresh_dir, exist_ok=True)
    fresh_runs = {}
    for name, edits in (("unedited", {}), ("astra_positive_control", astra_edit)):
        path = os.path.join(fresh_dir, f"{name}.jsonl.gz")
        if not os.path.exists(path):
            with engine.using(edits):
                recs, secs = E.answer(engine, fresh, C.TEST_MAX_TOKENS, "fresh_test")
            E.save(recs, path, full_text=True)
            s = E.summary(recs)["all"]
            RA.log(f"fresh test {name}: acc {s['acc']:.3f} cut {s['cut_share']:.3f} ({secs / 60:.1f} min)")
        fresh_runs[name] = E.load(path)
    if not RA.done("r_base"):
        RA.mark("r_base", {"positive_control": pc, "fresh_positive_control": E.paired(fresh_runs["unedited"], fresh_runs["astra_positive_control"])})
        pack()

    # 2. the trajectories, every seed of C2 first
    for rule in C.R_RULES:
        for seed in C.R_SEEDS:
            stage = f"r_{rule}_s{seed}"
            if RA.done(stage):
                continue
            R.trajectory(engine, rule, seed, pools, test, base, fresh, fresh_runs["unedited"], out_dir, RA.log)
            RA.mark(stage)
            pack()
    RA.log("all replication stages finished")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        RA.log("FATAL\n" + traceback.format_exc())
        raise
