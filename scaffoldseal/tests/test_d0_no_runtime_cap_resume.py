from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import d0_no_runtime_cap_resume as amendment


class _LedgerOnlyExecutor:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def _accepted_gpu_seconds(self) -> float:
        return self.seconds


def test_finite_accounting_has_no_elapsed_time_ceiling() -> None:
    executor = _LedgerOnlyExecutor(10_000 * 3600.0)
    amendment._finite_accounting_only(executor)


@pytest.mark.parametrize("seconds", [math.nan, math.inf, -1.0])
def test_invalid_accounting_remains_blocked(seconds: float) -> None:
    with pytest.raises(RuntimeError, match="accounting is invalid"):
        amendment._finite_accounting_only(_LedgerOnlyExecutor(seconds))


def test_unaccepted_amendment_is_fail_closed(tmp_path: Path) -> None:
    (tmp_path / amendment.AMENDMENT_NAME).write_text(
        json.dumps(
            {
                "schema_version": amendment.SCHEMA_VERSION,
                "accepted": False,
                "user_authorized": True,
                "scope": "remove_cumulative_runtime_stop_only",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lacks independent acceptance"):
        amendment._load_authorized_amendment(tmp_path)
