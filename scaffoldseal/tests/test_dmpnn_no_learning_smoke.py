import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "baseline_candidates" / "BenchmarkCycPeptMP"
PINNED_PYTHON = BASELINE / ".venv" / "Scripts" / "python.exe"
SCRIPT = ROOT / "src" / "dmpnn_no_learning_smoke.py"
EVIDENCE = ROOT / "audit" / "dmpnn_no_learning_smoke.json"


class PinnedDMPNNNoLearningSmokeTests(unittest.TestCase):
    def test_pinned_smoke_reproduces_evidence_without_touching_baseline(self):
        before = subprocess.run(
            ["git", "-C", str(BASELINE), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dmpnn_no_learning_smoke.json"
            completed = subprocess.run(
                [
                    str(PINNED_PYTHON),
                    str(SCRIPT),
                    "--baseline-root",
                    str(BASELINE),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            observed = json.loads(output.read_text(encoding="utf-8"))
            committed = json.loads(EVIDENCE.read_text(encoding="utf-8"))
            self.assertEqual(observed, committed, completed.stdout + completed.stderr)
            self.assertEqual(observed["status"], "PASS")
            self.assertFalse(observed["real_model_fit_or_weight_update_performed"])
            self.assertTrue(all(observed["assertions"].values()))
        after = subprocess.run(
            ["git", "-C", str(BASELINE), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
