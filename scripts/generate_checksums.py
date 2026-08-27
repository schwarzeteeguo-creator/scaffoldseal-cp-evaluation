"""Generate or verify SHA-256 checksums for this release candidate."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = ROOT / "SHA256SUMS"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}
LEGACY_RELEASE_PATHS = {
    "paper/manuscript_v0.6.md",
    "paper/supplementary/supporting_information_v0.5.md",
    "paper/supplementary/supporting_information_v0.6.md",
    "docs/RELEASE_INVENTORY_V0.6.md",
}


def release_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == CHECKSUM_FILE:
            continue
        relative = path.relative_to(ROOT)
        relative_posix = relative.as_posix()
        if relative_posix in LEGACY_RELEASE_PATHS:
            continue
        if relative_posix.startswith("paper/pdf/") and relative.name.endswith("v0.6.pdf"):
            continue
        if relative_posix.startswith("paper/figures/output/") and relative.name.startswith(("figure1_evaluation_schematic.", "figure2_main_results.")):
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in {".pyc", ".tif", ".tiff"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_lines() -> list[str]:
    return [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in release_files()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="verify SHA256SUMS instead of rewriting it"
    )
    args = parser.parse_args()
    expected = expected_lines()

    if args.check:
        if not CHECKSUM_FILE.exists():
            print("FAIL: SHA256SUMS does not exist")
            return 1
        actual = CHECKSUM_FILE.read_text(encoding="utf-8").splitlines()
        if actual != expected:
            print("FAIL: SHA256SUMS is missing, stale, or does not match the package")
            return 1
        print(f"PASS: verified {len(expected)} files against SHA256SUMS")
        return 0

    CHECKSUM_FILE.write_text("\n".join(expected) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote SHA256SUMS for {len(expected)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
