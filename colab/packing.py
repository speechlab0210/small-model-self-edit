"""Archive records and data; explicitly inventory every omitted large checkpoint."""
import hashlib
import io
import json
from pathlib import Path
import tarfile


def pack_results(work, destination):
    work, destination = Path(work), Path(destination)
    files = sorted(p for folder in (work / "out", work / "data") if folder.exists() for p in folder.rglob("*") if p.is_file())
    if (work / "console.log").exists():
        files.append(work / "console.log")
    manifest = []
    with tarfile.open(destination, "w:gz") as archive:
        for path in files:
            name, size = path.relative_to(work).as_posix(), path.stat().st_size
            sha = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    sha.update(block)
            omitted = path.name == "state.pt" or size > 50_000_000
            manifest.append({"path": name, "bytes": size, "sha256": sha.hexdigest(), "included": not omitted,
                             "reason": "resumable checkpoint or file larger than 50 MB; retain separately" if omitted else None})
            if not omitted:
                archive.add(path, arcname=name, recursive=False)
        encoded = (json.dumps({"files": manifest}, indent=2) + "\n").encode()
        item = tarfile.TarInfo("MANIFEST.json")
        item.size = len(encoded)
        archive.addfile(item, io.BytesIO(encoded))
    (work / "archive_manifest.json").write_bytes(encoded)
    return manifest
