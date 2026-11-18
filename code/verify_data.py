"""Verify the frozen dataset against its recorded sizes and SHA-256 hashes."""
import hashlib
import json
from pathlib import Path


def main():
    data = Path(__file__).resolve().parents[1] / "data"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = data / record["path"]
        if path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Unexpected file size: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise RuntimeError(f"Unexpected SHA-256: {path}")
    print(f"Verified {len(manifest['files'])} frozen data files.")


if __name__ == "__main__":
    main()
