"""Governance-only resume entry point that removes the cumulative 72-hour stop.

This module does not change the accepted D0 plan, data, folds, model, seeds,
metrics, thresholds, stage identities, or sealing policy.  It can be used only
after an independently reviewed amendment record authorizes this exact wrapper.
The already-running legacy process must not be interrupted merely to activate
this amendment; it applies on the next safe resume.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import d0_full_runner as frozen


SCHEMA_VERSION = "scaffoldseal-d0-runtime-cap-amendment-v1"
AMENDMENT_NAME = "d0_runtime_cap_amendment.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_authorized_amendment(project_root: Path) -> dict[str, object]:
    amendment_path = (project_root / AMENDMENT_NAME).resolve()
    if amendment_path.parent != project_root.resolve():
        raise RuntimeError("Runtime-cap amendment escapes the project root")
    try:
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Runtime-cap amendment is unreadable") from error
    if amendment.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Runtime-cap amendment schema drift")
    if amendment.get("accepted") is not True:
        raise RuntimeError("Runtime-cap amendment lacks independent acceptance")
    if amendment.get("user_authorized") is not True:
        raise RuntimeError("Runtime-cap removal lacks user authorization")
    if amendment.get("scope") != "remove_cumulative_runtime_stop_only":
        raise RuntimeError("Runtime-cap amendment scope drift")

    runner_path = (project_root / "src" / "d0_full_runner.py").resolve()
    wrapper_path = Path(__file__).resolve()
    manifest_path = (project_root / frozen.DEFAULT_MANIFEST).resolve()
    acceptance_path = (project_root / frozen.DEFAULT_ACCEPTANCE).resolve()
    if _sha256(runner_path) != amendment.get("frozen_runner_sha256"):
        raise RuntimeError("Frozen runner differs from the independently reviewed anchor")
    if _sha256(wrapper_path) != amendment.get("resume_wrapper_sha256"):
        raise RuntimeError("Resume wrapper differs from the independently reviewed anchor")
    if _sha256(manifest_path) != amendment.get("manifest_sha256"):
        raise RuntimeError("Candidate manifest differs from the amendment anchor")
    if _sha256(acceptance_path) != amendment.get("acceptance_sha256"):
        raise RuntimeError("External acceptance differs from the amendment anchor")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("plan_sha256") != amendment.get("plan_sha256"):
        raise RuntimeError("Candidate plan differs from the amendment anchor")
    ledger_relative = str(manifest.get("scheduler", {}).get("ledger_relative_path", ""))
    if ledger_relative != amendment.get("ledger_relative_path"):
        raise RuntimeError("Ledger binding differs from the amendment anchor")
    ledger_path = (project_root / ledger_relative).resolve()
    try:
        ledger_path.relative_to(project_root.resolve())
    except ValueError as error:
        raise RuntimeError("Bound ledger escapes the project root") from error
    return amendment


def _command_project_root() -> Path:
    values: list[str] = []
    for index, argument in enumerate(sys.argv):
        if argument == "--project-root" and index + 1 < len(sys.argv):
            values.append(sys.argv[index + 1])
        elif argument.startswith("--project-root="):
            values.append(argument.split("=", 1)[1])
    if len(values) != 1:
        raise RuntimeError("Exactly one explicit --project-root is required")
    return Path(values[0]).resolve()


def _reject_active_legacy_executor() -> None:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.ProcessId -ne "
        f"{os.getpid()} -and $_.CommandLine -match 'd0_full_runner\\.py' "
        "-and $_.CommandLine -match '--execute' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError("Cannot reliably audit active legacy executors")
    active = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if active:
        raise RuntimeError(
            "A legacy D0 executor is still active; safe concurrent resume is forbidden"
        )


def _finite_accounting_only(self: frozen.D0FullExecutor) -> None:
    """Retain strict ledger accounting while removing only the elapsed-time stop."""
    total = self._accepted_gpu_seconds()
    if not math.isfinite(total) or total < 0.0:
        raise RuntimeError("Cumulative D0 GPU runtime accounting is invalid")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if _command_project_root() != project_root.resolve():
        raise RuntimeError("Command project root differs from the amendment project")
    _load_authorized_amendment(project_root)
    _reject_active_legacy_executor()
    frozen.D0FullExecutor._check_gpu_budget = _finite_accounting_only
    frozen.main()


if __name__ == "__main__":
    main()
