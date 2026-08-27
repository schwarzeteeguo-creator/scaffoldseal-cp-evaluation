"""Reviewed D123 resume boundary for five frozen external D0 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import d123_runner as runner
import d123_runner_governance as governance
from d123_locked_adapter import D123LockedAdapter as _LockedAdapter
from d123_dmpnn_integration import (
    build_ordered_fit_dataset,
    build_ordered_role_dataset,
)


EXPECTED_PLAN_SHA256 = "6b73097ee83c000a48ec4dfc646ddbcad3a561d3276c33e3661d3bf8b9d8f5c5"
EXPECTED_PLAN_FILE_SHA256 = "ccf3afa818cc6ee453a720f38a8342484986df6f5dbb4d7fffbaf7fd146c0a90"
EXPECTED_RUNNER_SHA256 = "2d5d7c3de8a447164bce016feee0adf8b2e0c88d2fe3e5bca9a32f6f1ec35ba8"
EXPECTED_GOVERNANCE_SHA256 = "c536f55a78ac46c5d09d971db91c56efd85fb5a6e1641c0bddf430fd322766f4"
AMENDMENT_NAME = "d123_external_dependency_resume_amendment.json"
GROUP_METADATA = (
    "artifacts/d123_inputs_v1/group_metadata.csv",
    536926,
    "fdfda5c9bf714fed03adf56272fa63f78c2a6ce4060942ccf4bfb6205008205a",
)
RAW_DESCRIPTORS = (
    "artifacts/d123_inputs_v1/raw_descriptors.csv",
    1245171,
    "2c0e0e8422f0b27b8a11e49c5c4ccb70aee197e7ec229df4ec1d654e96fb7ca3",
)


class ResumeSafeD123LockedAdapter(_LockedAdapter):
    """Keep locked label-free tables outside adapter state until fit calls."""

    runtime_root: Path | None = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        del self.group_metadata
        del self.raw_descriptors
        if self.runtime_root is None:
            raise RuntimeError("D123 resume adapter runtime root is unset")
        self.input_root = str(self.runtime_root.resolve())

    def _locked_frame(self, record):
        import pandas as pd

        relative, size_bytes, sha256 = record
        root = Path(self.input_root).resolve()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("D123 resume input escapes runtime root") from error
        if (
            not path.is_file()
            or path.stat().st_size != size_bytes
            or runner.stream_sha256(path) != sha256
        ):
            raise RuntimeError("D123 resume input bytes drifted")
        return pd.read_csv(path)

    def _fit_dataset(self, frame, target_column):
        import deepchem as dc

        return build_ordered_fit_dataset(
            variant=self.variant,
            fit_frame=frame,
            group_metadata=self._locked_frame(GROUP_METADATA),
            raw_descriptors=self._locked_frame(RAW_DESCRIPTORS),
            target_column=target_column,
            featurizer=dc.feat.DMPNNFeaturizer(),
        )

    def _role_dataset(self, frame, fit_payload, target_column):
        import deepchem as dc

        return build_ordered_role_dataset(
            frame=frame,
            fit_payload=fit_payload,
            raw_descriptors=self._locked_frame(RAW_DESCRIPTORS),
            featurizer=dc.feat.DMPNNFeaturizer(),
            target_column=target_column,
        )


def frozen_external_identities(plan: Mapping[str, object]) -> frozenset[str]:
    receipts = plan["scientific_lock"]["d1_accepted_checkpoint_receipts"]
    if not isinstance(receipts, Mapping) or set(receipts) != {"0", "1", "2", "3", "4"}:
        raise RuntimeError("D123 frozen D1 receipt geometry drifted")
    identities = frozenset(
        str(receipt["scientific_identity_sha256"])
        for receipt in receipts.values()
    )
    if len(identities) != 5:
        raise RuntimeError("D123 frozen D1 checkpoint identities are not unique")
    return identities


def next_stage_with_frozen_external_receipts(plan, ledger, project_root: Path):
    """Treat only the five plan-locked D0 receipts as already satisfied."""

    by_id = governance._by_id(plan)
    external = frozen_external_identities(plan)
    for stage in plan["stages"]:
        if governance.accepted_output_if_valid(stage, ledger, project_root) is not None:
            continue
        dependencies = tuple(map(str, stage["dependencies"]))
        unknown = set(dependencies) - set(by_id) - set(external)
        if unknown:
            raise RuntimeError("D123 stage dependency is neither internal nor frozen D0")
        ready = all(
            identity in external
            or governance.accepted_output_if_valid(
                by_id[identity], ledger, project_root
            )
            is not None
            for identity in dependencies
        )
        if ready:
            return stage
    incomplete = [
        stage
        for stage in plan["stages"]
        if governance.accepted_output_if_valid(stage, ledger, project_root) is None
    ]
    if incomplete:
        raise RuntimeError("D123 has incomplete stages but no dependency-ready stage")
    return None


def validate_resume_boundary(project_root: Path) -> tuple[dict, dict]:
    root = project_root.resolve()
    plan_path = root / runner.DEFAULT_PLAN
    acceptance_path = root / runner.DEFAULT_ACCEPTANCE
    amendment_path = root / AMENDMENT_NAME
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    runner_path = root / "src/d123_runner.py"
    governance_path = root / "src/d123_runner_governance.py"
    wrapper_path = Path(__file__).resolve()
    if plan.get("plan_sha256") != EXPECTED_PLAN_SHA256:
        raise RuntimeError("D123 resume wrapper identifies another plan")
    exact = (
        runner.stream_sha256(plan_path) == EXPECTED_PLAN_FILE_SHA256
        and runner.stream_sha256(runner_path) == EXPECTED_RUNNER_SHA256
        and runner.stream_sha256(governance_path) == EXPECTED_GOVERNANCE_SHA256
    )
    if not exact:
        raise RuntimeError("D123 resume wrapper exact byte binding failed")
    runner.validate_plan(plan, root)
    runner.assert_execution_authorized(
        plan, acceptance, plan_path, runner_path
    )
    if (
        amendment.get("schema_version")
        != "scaffoldseal-d123-external-dependency-resume-v1"
        or amendment.get("accepted") is not True
        or amendment.get("plan_sha256") != EXPECTED_PLAN_SHA256
        or amendment.get("plan_file_sha256") != EXPECTED_PLAN_FILE_SHA256
        or amendment.get("runner_file_sha256") != EXPECTED_RUNNER_SHA256
        or amendment.get("governance_file_sha256") != EXPECTED_GOVERNANCE_SHA256
        or amendment.get("wrapper_file_sha256")
        != runner.stream_sha256(wrapper_path)
    ):
        raise RuntimeError("D123 external-dependency resume lacks exact acceptance")
    frozen_external_identities(plan)
    return plan, acceptance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    plan, acceptance = validate_resume_boundary(root)
    governance.next_dependency_ready_stage = next_stage_with_frozen_external_receipts
    ResumeSafeD123LockedAdapter.runtime_root = root
    import d123_locked_adapter

    d123_locked_adapter.D123LockedAdapter = ResumeSafeD123LockedAdapter
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "plan_sha256": plan["plan_sha256"],
                    "external_receipts": 5,
                },
                sort_keys=True,
            )
        )
    if args.execute:
        runner.D123Executor(
            plan, acceptance, root, root / runner.DEFAULT_PLAN
        ).run()


if __name__ == "__main__":
    main()
