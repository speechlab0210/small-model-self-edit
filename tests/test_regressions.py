import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selfedit import evaluate, mathqa
from selfedit.metrics import paired
from selfedit.run_protocol import check_protocol
from selfedit.training_utils import accumulation_divisor
from colab.packing import pack_results


class DataIntegrityTests(unittest.TestCase):
    def test_pairing_rejects_missing_and_duplicate_ids(self):
        rows = [{"id": "a", "correct": 0}, {"id": "b", "correct": 1}]
        for other in (rows[:1], rows + rows[:1], []):
            with self.assertRaises(ValueError):
                paired(rows, other)

    def test_paired_changes(self):
        base = [{"id": str(i), "correct": c} for i, c in enumerate([0, 0, 1, 1])]
        new = [{"id": str(i), "correct": c} for i, c in enumerate([1, 1, 0, 1])]
        p = paired(base, list(reversed(new)))
        self.assertEqual((p["n"], p["gain"], p["rescues"], p["losses"]), (4, .25, 2, 1))
        self.assertEqual(paired(base, base)["z"], 0)

    def test_zero_variance_nonzero_gain_is_explicitly_undefined(self):
        self.assertIsNone(paired([{"id": "a", "correct": 0}], [{"id": "a", "correct": 1}])["z"])

    def test_complete_training_output_and_ids_survive(self):
        row = {"id": "a", "text": "x" * 9000, "ids": [1, 2], "prompt_ids": [3]}
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "records.gz")
            evaluate.save([row], path, keep_token_ids=True)
            loaded = evaluate.load(path)[0]
            for key in row:
                self.assertEqual(loaded[key], row[key])
            self.assertFalse(loaded["text_truncated"])

    def test_reduced_export_is_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "records.gz")
            evaluate.save([{"text": "x" * 9000}], path, full_text=False)
            self.assertTrue(evaluate.load(path)[0]["text_truncated"])

    def test_archive_preserves_data_and_discloses_omitted_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            (work / "out").mkdir(parents=True)
            (work / "data").mkdir()
            (work / "out/result.json").write_text('{"ok":true}')
            (work / "out/state.pt").write_bytes(b"checkpoint")
            (work / "data/test.jsonl").write_text('{"id":"a","answer":"4"}')
            archive = Path(tmp) / "results.tar.gz"
            pack_results(work, archive)
            with tarfile.open(archive) as tf:
                self.assertIn("data/test.jsonl", tf.getnames())
                self.assertIn("out/result.json", tf.getnames())
                self.assertNotIn("out/state.pt", tf.getnames())
                manifest = json.load(tf.extractfile("MANIFEST.json"))
            entry = next(x for x in manifest["files"] if x["path"] == "out/state.pt")
            self.assertFalse(entry["included"])
            self.assertEqual(entry["sha256"], hashlib.sha256(b"checkpoint").hexdigest())

    def test_old_runs_cannot_silently_resume_under_fixed_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "old_result.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                check_protocol(p)
        with tempfile.TemporaryDirectory() as tmp:
            check_protocol(tmp)
            check_protocol(tmp)

    def test_grader_disables_unavailable_posix_alarm(self):
        with patch.object(mathqa, "signal", types.SimpleNamespace()):
            self.assertIsNone(mathqa._timeouts())

    def test_launcher_rejects_invalid_mode_before_setup(self):
        proc = subprocess.run([sys.executable, str(ROOT / "colab/run_in_colab.py"), "typo"], capture_output=True, timeout=15)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"mode must be", proc.stderr)


class TorchRegressionTests(unittest.TestCase):
    def test_partial_batch_gradient_matches_direct_batch_mean(self):
        import torch
        for total in (3, 8, 11):
            for start in range(0, total, 8):
                actual = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
                batch = list(range(start, min(start + 8, total)))
                for i in batch:
                    ((actual - (i + 1)) ** 2 / accumulation_divisor(i, total, 8)).backward()
                expected = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
                ((expected - torch.tensor([i + 1 for i in batch], dtype=torch.float64)) ** 2).mean().backward()
                torch.testing.assert_close(actual.grad, expected.grad)

    def test_weight_identity_distinguishes_equal_sums(self):
        import torch
        from selfedit.weights import fingerprint
        engine = types.SimpleNamespace(original=lambda name: torch.zeros(2, dtype=torch.bfloat16))
        a = {"weight": torch.tensor([1., 2.])}
        b = {"weight": torch.tensor([2., 1.])}
        self.assertNotEqual(fingerprint(engine, a), fingerprint(engine, b))
        self.assertEqual(fingerprint(engine, a), fingerprint(engine, a))

    def test_c2_selection_does_not_read_correctness(self):
        from selfedit.arm_c import keep
        rows = [{"id": "a", "finish": "stop", "pred": "4", "correct": 1},
                {"id": "b", "finish": "length", "pred": "4", "correct": 1},
                {"id": "c", "finish": "stop", "pred": None, "correct": 0}]
        flipped = [dict(r, correct=1-r["correct"]) for r in rows]
        self.assertEqual([r["id"] for r in keep(rows, "C2_finished_only")], [r["id"] for r in keep(flipped, "C2_finished_only")])
        with self.assertRaises(ValueError):
            keep(rows, "typo")


if __name__ == "__main__":
    unittest.main()
