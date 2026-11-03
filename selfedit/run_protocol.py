"""Do not resume historical results under a changed training/recording protocol."""
import json
from pathlib import Path

VERSION = "audit-v2-20261104"


def check_protocol(out):
    out = Path(out)
    path = out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text())["version"] != VERSION:
            raise RuntimeError("Different experiment protocol: use a new SELFEDIT_WORK directory")
    elif any(out.iterdir()):
        raise RuntimeError("Unversioned historical results exist: use a new SELFEDIT_WORK directory; do not mix runs")
    else:
        path.write_text(json.dumps({"version": VERSION}) + "\n", encoding="utf-8")
