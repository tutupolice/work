"""Read the transport-split source archive without permanently joining files."""
from contextlib import contextmanager
import hashlib
import json
import shutil
import tarfile
import tempfile

@contextmanager
def open_archive(source_root):
    info = json.loads((source_root / "cache/ARCHIVE_PARTS.json").read_text())
    with tempfile.TemporaryFile() as joined:
        digest = hashlib.sha256()
        for row in info["parts"]:
            path = source_root.parent / row["path"]
            data = path.read_bytes()
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Archive part checksum mismatch")
            digest.update(data)
            joined.write(data)
        if joined.tell() != info["archive_bytes"] or digest.hexdigest() != info["archive_sha256"]:
            raise ValueError("Combined archive checksum mismatch")
        joined.seek(0)
        with tarfile.open(fileobj=joined, mode="r:gz") as tf:
            yield tf
