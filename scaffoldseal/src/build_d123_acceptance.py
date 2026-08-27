"""Create an external D123 acceptance record that is pending by default."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    plan_path = root / "d123_plan_candidate.json"
    runner_path = root / "src/d123_runner.py"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return {
        "schema_version": "scaffoldseal-d123-acceptance-v1",
        "status": "PENDING_INDEPENDENT_PREFIT_REVIEW",
        "accepted": False,
        "execution_authorized": False,
        "gpu_training_allowed": False,
        "plan_relative_path": "d123_plan_candidate.json",
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": stream_sha256(plan_path),
        "protocol_lock_sha256": plan["protocol_lock_sha256"],
        "runner_relative_path": "src/d123_runner.py",
        "runner_file_sha256": stream_sha256(runner_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("Refusing to overwrite D123 external acceptance")
    output.write_text(
        json.dumps(build(args.project_root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
