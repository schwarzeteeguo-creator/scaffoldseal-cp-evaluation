"""Refresh tracked internal checksums while preserving the external vault hash.

The external vault path is never opened or hashed by this script.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKSUMS = ROOT / "SHA256SUMS"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    external_lines = []
    if CHECKSUMS.exists():
        for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
            if "  external:" in line:
                external_lines.append(line)
    if not external_lines:
        raise RuntimeError("Expected the pre-existing external vault checksum entry")
    internal = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative == "SHA256SUMS" or "__pycache__" in path.parts:
            continue
        internal.append(f"{sha256_file(path)}  {relative}")
    payload = "\n".join(internal + sorted(external_lines)) + "\n"
    CHECKSUMS.write_text(payload, encoding="utf-8", newline="\n")
    print(f"refreshed {len(internal)} internal entries; preserved external entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
