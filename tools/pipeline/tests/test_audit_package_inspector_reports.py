from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from audit_package_inspector import inspect_audit_file  # noqa: E402
from audit_to_adapter_contract import (  # noqa: E402
    REDACTED_IN_APP_DATASET_EXPORT_FORMAT,
    REDACTED_IN_APP_DATASET_EXPORT_KIND,
    REDACTED_IN_APP_DATASET_FILE_PREFIX,
    REDACTED_IN_APP_DATASET_PACKAGE_SCHEMA_VERSION,
    REDACTED_IN_APP_DATASET_PRIVACY_POLICY,
    REDACTED_IN_APP_DATASET_PROMPT_POLICY,
    REDACTED_IN_APP_DATASET_SOURCE_ACTIONS,
)


def redacted_v1_package() -> dict:
    return {
        "schemaVersion": REDACTED_IN_APP_DATASET_PACKAGE_SCHEMA_VERSION,
        "generatedAt": "2026-08-10T03:20:38Z",
        "exportKind": REDACTED_IN_APP_DATASET_EXPORT_KIND,
        "app": {"bundleIdentifier": "com.27pm.lumenclone", "buildNumber": "42"},
        "testFlight": {
            "sourceAction": REDACTED_IN_APP_DATASET_SOURCE_ACTIONS[1],
            "filePrefix": REDACTED_IN_APP_DATASET_FILE_PREFIX,
            "liveE2EReportIncluded": True,
        },
        "manifestSource": "interactive-model-tool-validation-live-e2e",
        "usedRuntimeFallback": False,
        "scenarioResults": [],
        "recentTraces": [{"event": "modelTurn", "slot": "executor", "modelFamily": "qwen3"}],
        "liveE2EReport": {
            "schemaVersion": "1.0.0",
            "exportPolicy": {
                "format": "live-e2e-test-report-json",
                "sourceLayer": "e2eTestReport",
                "ownsLiveE2EScenarios": True,
            },
            "payload": {"passed": 1, "failed": 0, "results": [{"passed": True}]},
            "traceSidecarField": "recentTraces",
        },
        "traceSelectedToolAllowedCount": 0,
        "traceParseErrorCount": 0,
        "exportQualityFailures": [],
        "improveLoop": {
            "acceptedTraining": [],
            "quarantinedSamples": [],
            "regressionTests": [],
        },
        "exportPolicy": {
            "format": REDACTED_IN_APP_DATASET_EXPORT_FORMAT,
            "privacy": REDACTED_IN_APP_DATASET_PRIVACY_POLICY,
            "promptPolicy": REDACTED_IN_APP_DATASET_PROMPT_POLICY,
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
            "includesDeterministicStaticScenarios": False,
        },
    }


class AuditPackageInspectorReportTests(unittest.TestCase):
    def inspect(self, payload: dict):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lumen-testflight-agent-grounding-redacted-v1-test.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return inspect_audit_file(path)

    def test_current_redacted_v1_package_is_inspected(self) -> None:
        inspection = self.inspect(redacted_v1_package())

        self.assertTrue(inspection.is_in_app_package)
        self.assertEqual(REDACTED_IN_APP_DATASET_EXPORT_FORMAT, inspection.source_format)
        self.assertEqual(1, inspection.trace_count)
        self.assertEqual(1, inspection.e2e_scenario_count)
        self.assertEqual([], inspection.errors)

    def test_redacted_v1_lookalikes_fail_closed(self) -> None:
        def mutate(path: tuple[str, ...], value: object):
            def apply(package: dict) -> None:
                target = package
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
            return apply

        cases = {
            "future schema": mutate(("schemaVersion",), "2.1.0"),
            "wrong format": mutate(("exportPolicy", "format"), "generic-json"),
            "privacy claim drift": mutate(("exportPolicy", "privacy"), "redacted"),
            "parent owns E2E": mutate(("exportPolicy", "ownsLiveE2EScenarios"), True),
            "unversioned prefix": mutate(("testFlight", "filePrefix"), "lumen-testflight-agent-grounding"),
            "wrong interactive source": mutate(("manifestSource",), "generic-runtime-audit"),
            "quality failures present": mutate(("exportQualityFailures",), [{"type": "trace_gap"}]),
            "training payload included": mutate(("improveLoop", "acceptedTraining"), [{"prompt": "raw"}]),
            "embedded E2E does not own results": mutate(("liveE2EReport", "exportPolicy", "ownsLiveE2EScenarios"), False),
        }
        for label, apply in cases.items():
            with self.subTest(label=label):
                package = copy.deepcopy(redacted_v1_package())
                apply(package)
                inspection = self.inspect(package)
                self.assertFalse(inspection.is_in_app_package)
                self.assertTrue(inspection.errors)

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
                                {
                                    "status": "skipped",
                                    "remediationProposals": [
                                        {
                                            "id": "manual-scenario-foreground",
                                            "title": "Run the diagnostic from the foreground control",
                                            "action": "Open Runtime Diagnostics and start the matching manual probe from the foreground UI.",
                                            "severity": "info",
                                        }
                                    ],
                                },
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            inspection = inspect_audit_file(path)

        self.assertEqual("persistent_runtime_diagnostics_export", inspection.source_format)
        self.assertEqual("persistentRuntimeDiagnostics", inspection.source_layer)
        self.assertEqual(4, inspection.diagnostics_record_count)
        self.assertEqual(2, inspection.diagnostics_failed_count)
        self.assertEqual(1, inspection.diagnostics_remediation_proposal_count)
        self.assertNotIn("unrecognized JSON audit shape", inspection.warnings)


if __name__ == "__main__":
    unittest.main()
