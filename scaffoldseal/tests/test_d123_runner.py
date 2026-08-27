import json
from pathlib import Path
import shutil
import sys
import tempfile

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import d123_runner as runner


def test_candidate_validates_but_execution_is_source_locked():
    root = Path(__file__).resolve().parents[1]
    path = root / "d123_plan_candidate.json"
    plan = json.loads(path.read_text())
    runner.validate_plan(plan, root)
    try:
        runner.assert_execution_authorized(
            plan,
            {
                "accepted": True,
                "execution_authorized": True,
                "gpu_training_allowed": True,
            },
            path,
            Path(runner.__file__),
        )
    except RuntimeError as error:
        assert "source locks" in str(error)
    else:
        raise AssertionError("Candidate D123 runner executed while source-locked")


def test_cpu_stage_methods_are_present_but_unreachable_while_locked():
    assert callable(runner.D123Executor._run_label_gate_stage)
    assert callable(runner.D123Executor._run_metric_stage)
    assert callable(runner.D123Executor._run_selection_stage)
    assert callable(runner.D123Executor._run_descriptor_pretraining_stage)
    assert callable(runner.D123Executor._run_inner_stage)
    assert callable(runner.D123Executor._run_outer_stage)
    assert callable(runner.D123Executor._dispatch_stage)
    assert callable(runner.D123Executor.run)
    assert runner.PREFIT_REVIEW_SOURCE_LOCK is False
    assert runner.REAL_EXECUTION_SOURCE_LOCK is False


def test_frozen_selection_rule_is_ceil_median_of_exactly_four():
    select = runner.D123Executor._ceil_median_four
    assert select([1, 2, 3, 4]) == 3
    assert select([200, 200, 201, 900]) == 201
    for invalid in ([1, 2, 3], [0, 1, 2, 3], [1, 2, 3, 2001]):
        try:
            select(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid D123 stopping epochs were selected")


def test_revalidation_rejects_locked_byte_mutation_before_claim(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temp:
        mirror = Path(temp) / "project"
        shutil.copytree(root, mirror)
        plan_path = mirror / "d123_plan_candidate.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        acceptance = json.loads(
            (mirror / runner.DEFAULT_ACCEPTANCE).read_text(encoding="utf-8")
        )
        executor = object.__new__(runner.D123Executor)
        executor.plan = plan
        executor.acceptance = acceptance
        executor.project_root = mirror
        executor.plan_path = plan_path
        executor.runner_path = mirror / "src/d123_runner.py"
        monkeypatch.setattr(runner, "PREFIT_REVIEW_SOURCE_LOCK", True)
        monkeypatch.setattr(runner, "REAL_EXECUTION_SOURCE_LOCK", True)
        locked_path = mirror / next(iter(plan["locked_files"]))
        locked_path.write_bytes(locked_path.read_bytes() + b"\n")
        try:
            executor._revalidate_before_claim()
        except RuntimeError as error:
            assert "locked file validation failed" in str(error)
        else:
            raise AssertionError("D123 claim proceeded after locked bytes drifted")
