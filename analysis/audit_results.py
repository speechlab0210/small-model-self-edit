"""Audit the released records without running a language model.

python analysis/audit_results.py --output results/audit/checks.json
Add --regrade to check full test answers against pinned public dataset snapshots.
Regrading needs datasets and math-verify[antlr4_13_2]; basic checks use the standard library.
Original run did not archive test golds or revisions; reconstructed golds are labelled as such.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selfedit import mathqa
from selfedit.metrics import indexed, paired


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results/raw")
    ap.add_argument("--output", type=Path, default=ROOT / "results/audit/checks.json")
    ap.add_argument("--regrade", action="store_true")
    args = ap.parse_args()
    raw = args.raw
    ids = json.loads((raw / "data_ids.json").read_text())
    expected = set(ids["test_ids"])
    assert len(expected) == len(ids["test_ids"]) == 200
    pools = ids["pools"]
    flat = [pid for pool in pools.values() for pid in pool]
    assert len(set(flat)) == len(flat), "practice pools overlap"
    assert not expected.intersection(flat), "test/practice ids overlap"
    paths = sorted((raw / "test").glob("*.jsonl.gz"))
    paths += sorted((raw / "arm_b").glob("B*/test_round*.jsonl.gz"))
    paths += sorted((raw / "arm_c").glob("C*/test_round*.jsonl.gz"))
    all_rows = {p.relative_to(raw).as_posix(): read(p) for p in paths}
    base = all_rows["test/unedited.jsonl.gz"]
    failures, extracted = [], 0
    comparisons = {}
    for path, rows in all_rows.items():
        assert indexed(rows).keys() == expected, f"test set mismatch: {path}"
        for row in rows:
            assert row["finish"] in ("stop", "length")
            assert 0 < row["tokens"] <= 16384
            pred, count = mathqa.extract(row["text"])
            extracted += 1
            if row["pred"] != (pred[:120] if pred else None) or row["boxes"] != count:
                failures.append({"file": path, "id": row["id"], "kind": "extraction"})
        comparisons[path] = {"n": len(rows), "correct": sum(x["correct"] for x in rows),
                             "paired_vs_unedited": paired(base, rows)}
    summary = json.loads((ROOT / "results/summary.json").read_text())
    for path, row in comparisons.items():
        parts = Path(path).parts
        name = Path(path).name.removesuffix(".jsonl.gz") if parts[0] == "test" else parts[1] + "_after_round" + Path(path).name[len("test_round"):-len(".jsonl.gz")]
        published = summary["tests"][name]
        assert math.isclose(published["all"], row["correct"] / row["n"], abs_tol=1e-12)
        for key, value in row["paired_vs_unedited"].items():
            assert math.isclose(published["paired"][key], value, abs_tol=1e-12), (name, key)
    a_turns = []
    for path in sorted((raw / "arm_a").glob("*/turns.jsonl")):
        turns = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(turns) == 30
        a_turns += turns
    assert len(a_turns) == 180
    assert all(not turn["edits_now"] for turn in a_turns)
    c_adjacent = {}
    for rule in ("C1_answer_key", "C2_finished_only"):
        previous = base
        for rnd in range(1, 4):
            rows = all_rows[f"arm_c/{rule}/test_round{rnd}.jsonl.gz"]
            delta = paired(previous, rows)
            delta["normal_95pct_interval"] = [delta["gain"] - 1.96 * delta["se"], delta["gain"] + 1.96 * delta["se"]]
            c_adjacent[f"{rule}:round{rnd-1}_to_{rnd}"] = delta
            previous = rows
    truncated = {}
    for path in sorted(raw.rglob("*.jsonl.gz")):
        rows = read(path)
        n = sum(" [...] " in row.get("text", "") for row in rows)
        if n:
            truncated[path.relative_to(raw).as_posix()] = n
    result = {
        "scope": "Audit of released historical data, not a corrected-model rerun",
        "test_files": len(paths), "test_records": extracted, "arm_a_turns": len(a_turns),
        "practice_pool_ids_disjoint": len(flat),
        "published_accuracy_and_paired_statistics_match": True,
        "test_comparisons": comparisons, "adjacent_round_comparisons_exploratory": c_adjacent,
        "abbreviated_practice_records": truncated, "failures": failures,
        "raw_sha256": {p.relative_to(raw).as_posix(): file_hash(p) for p in sorted(raw.rglob("*")) if p.is_file()},
    }
    if args.regrade:
        from datasets import load_dataset
        from selfedit.data import MATH_TEST, GSM, MATH_TRAIN, _norm
        import importlib.metadata
        math_data = load_dataset(MATH_TEST[0], split="test", revision=MATH_TEST[1])
        gsm_data = load_dataset(GSM[0], "main", split="test", revision=GSM[1])
        gsm_train = load_dataset(GSM[0], "main", split="train", revision=GSM[1])
        math_train = load_dataset(MATH_TRAIN[0], split="train", revision=MATH_TRAIN[1])
        math_test_texts = {_norm(row["problem"]) for row in math_data}
        test_texts = math_test_texts | {_norm(row["question"]) for i, row in enumerate(gsm_data) if f"gsm8k_test_{i}" in expected}
        practice = {f"gsm8k_train_{i}": _norm(row["question"]) for i, row in enumerate(gsm_train)}
        practice.update({f"math_train_{i}": _norm(row["problem"]) for i, row in enumerate(math_train)
                         if mathqa.gold_of_solution(row["solution"]) is not None and _norm(row["problem"]) not in math_test_texts})
        overlaps = sorted(pid for pid, problem in practice.items() if problem in test_texts)
        assert not overlaps, f"practice/test exact-text overlap: {overlaps}"
        assert set(flat).issubset(practice), "saved practice ids missing in pinned sources"
        gold = {row["unique_id"]: row["answer"] for row in math_data}
        gold.update({f"gsm8k_test_{i}": row["answer"].split("####")[-1].strip().replace(",", "") for i, row in enumerate(gsm_data)})
        mismatches = []
        for path, rows in all_rows.items():
            for row in rows:
                pred, _ = mathqa.extract(row["text"])
                grade = int(mathqa.is_correct(pred, gold[row["id"]]))
                if grade != row["correct"]:
                    mismatches.append({"file": path, "id": row["id"], "stored": row["correct"], "regraded": grade, "pred": pred, "gold": gold[row["id"]]})
        result["regrade"] = {
            "note": "Gold answers reconstructed at audit time; original run did not archive golds or these two dataset revisions.",
            "dataset_revisions": dict([MATH_TEST, GSM, MATH_TRAIN]), "math_verify_version": importlib.metadata.version("math-verify"),
            "exact_text_practice_test_overlaps": overlaps, "practice_records_checked": len(practice),
            "records": extracted, "mismatches": mismatches,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"test_files": len(paths), "test_records": extracted, "extraction_failures": len(failures),
                      "regrade_mismatches": len(result.get("regrade", {}).get("mismatches", [])) if args.regrade else "not run"}))
    return bool(failures or result.get("regrade", {}).get("mismatches"))


if __name__ == "__main__":
    raise SystemExit(main())
