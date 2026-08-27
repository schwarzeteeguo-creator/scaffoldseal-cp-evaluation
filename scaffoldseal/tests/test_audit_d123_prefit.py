from pathlib import Path

from audit_d123_prefit import audit


def test_d123_prefit_candidate_audit_passes_without_execution():
    root = Path(__file__).resolve().parents[1]
    report = audit(root)
    assert report["status"] == "PASS_ZERO_TRAINING_CANDIDATE"
    assert report["stages"] == 551
    assert report["scientific_fits"] == 491
    assert report["execution_namespaces_absent"] is True
    assert report["gpu_training_invoked"] is False
    assert len(report["mutation_rejections"]) == 4
