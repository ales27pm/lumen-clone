from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from audit_package_inspector import inspect_audit_file  # noqa: E402


class AuditPackageInspectorReportTests(unittest.TestCase):
    def test_live_e2e_evidence_layer_is_counted_not_unrecognized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live-e2e.json"
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": "1.0.0",
                        "generatedAt": "2026-06-14T00:00:00Z",
                        "exportPolicy": {
                            "format": "live-e2e-test-report-json",
                            "sourceLayer": "e2eTestReport",
                            "ownsLiveE2EScenarios": True,
                        },
                        "payload": {
                            "passed": 1,
                            "failed": 1,
                            "results": [
                                {"title": "passes", "passed": True},
                                {"title": "fails", "passed": False},
                            ],
                            "trainingSignals": ["capture failed prompts"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            inspection = inspect_audit_file(path)

        self.assertEqual("live-e2e-test-report-json", inspection.source_format)
        self.assertEqual("e2eTestReport", inspection.source_layer)
        self.assertEqual(2, inspection.e2e_scenario_count)
        self.assertEqual(1, inspection.e2e_failed_count)
        self.assertEqual(1, inspection.e2e_training_signal_count)
        self.assertNotIn("unrecognized JSON audit shape", inspection.warnings)

    def test_persistent_runtime_diagnostics_export_is_counted_not_unrecognized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "persistent-runtime-diagnostics.json"
            path.write_text(
                json.dumps(
                    {
                        "exportedAt": "2026-06-14T00:00:00Z",
                        "ndjson": "{}\n{}\n",
                        "state": {
                            "records": [
                                {"status": "passed"},
                                {
                                    "status": "cancelled",
                                    "metrics": {
                                        "didCancel": True,
                                        "cancellationReason": "persistent-diagnostics-agent-cancel",
                                    },
                                },
                                {"status": "failed"},
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            inspection = inspect_audit_file(path)

        self.assertEqual("persistent_runtime_diagnostics_export", inspection.source_format)
        self.assertEqual("persistentRuntimeDiagnostics", inspection.source_layer)
        self.assertEqual(3, inspection.diagnostics_record_count)
        self.assertEqual(1, inspection.diagnostics_failed_count)
        self.assertNotIn("unrecognized JSON audit shape", inspection.warnings)


if __name__ == "__main__":
    unittest.main()
