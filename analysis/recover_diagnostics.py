"""Export the diagnostic data that survived in the original downloaded directions.pt.

python analysis/recover_diagnostics.py PATH_TO_ORIGINAL_DIRECTIONS_PT results/recovered
Does not run a model or reconstruct missing Arm C training answers.
"""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import torch

source, destination = Path(sys.argv[1]), Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
saved = torch.load(source, map_location="cpu", weights_only=True)
assert tuple(saved["u"].shape) == (24, 1024)
assert len(saved["diag_records"]) == 128
(destination / "directions.json").write_text(json.dumps({"u": saved["u"].tolist(), "info": saved["info"]}) + "\n")
with gzip.open(destination / "diagnostic_answers_full.jsonl.gz", "wt", encoding="utf-8") as stream:
    for record in saved["diag_records"]:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
manifest = {
    "source": "original full-run download: out/arm_b/directions.pt",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "records": len(saved["diag_records"]), "shape": list(saved["u"].shape),
    "note": "Recovered original values, not regenerated. Token ids were not retained in this object. Original abbreviated files remain unchanged.",
}
(destination / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest))
