"""Content identity for effective edited tensors, rather than a collision-prone sum."""
import hashlib
import json


def fingerprint(engine, edits):
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(edits.items()):
        value = tensor.detach().to(device="cpu", dtype=engine.original(name).dtype).contiguous()
        h.update(json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":")).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
