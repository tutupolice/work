"""Restore verified public-source fold CSV checkpoints without overwriting differences."""
from pathlib import Path
import hashlib
import json
import tarfile
from checkpoint_archive import open_archive

ROOT = Path(__file__).resolve().parents[1] / "source"

def main():
    records = json.loads((ROOT / "cache/BENCHMARK_CACHE_MANIFEST.json").read_text())
    expected = {r["path"]: r for r in records}
    count = 0
    with open_archive(ROOT) as tf:
        for member in tf:
            if member.name not in expected or not member.isfile():
                raise ValueError("Unexpected archive entry")
            dest = ROOT / member.name
            if not dest.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError("Unsafe archive path")
            data = tf.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected[member.name]["sha256"]:
                raise ValueError("Archive hash mismatch")
            if dest.exists() and dest.read_bytes() != data:
                raise FileExistsError(f"Refusing to replace changed result: {dest}")
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            count += 1
    if count != 4800 or count != len(expected):
        raise AssertionError("Incomplete fold cache")
    print(f"Verified/restored {count} public-source fold predictions.")

if __name__ == "__main__":
    main()
