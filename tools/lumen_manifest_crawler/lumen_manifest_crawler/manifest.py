from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

DETERMINISTIC_GENERATED_AT = "1970-01-01T00:00:00+00:00"


class SourceFileHash(BaseModel):
    path: str
    sha256: str


class SourceIntegrity(BaseModel):
    commit: str | None = None
    files: list[SourceFileHash] = Field(default_factory=list)


class AppManifestInfo(BaseModel):
    name: str = "Lumen"
    bundleIdentifier: str | None = None
    buildVersion: str | None = None
    generatedAt: str = DETERMINISTIC_GENERATED_AT


class ModelSlotManifest(BaseModel):
    id: str
    role: str
    modelFamily: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    source: str | None = None


class FleetManifest(BaseModel):
    contractVersion: str = "unknown"
    slots: list[ModelSlotManifest] = Field(default_factory=list)


class FleetTopologySlotManifest(BaseModel):
    role: str
    purpose: str
    inputSignature: str
    outputSignature: str
    calls: list[str] = Field(default_factory=list)
    calledBy: list[str] = Field(default_factory=list)
    memoryScopes: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)

    def snake_dump(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "purpose": self.purpose,
            "input_signature": self.inputSignature,
            "output_signature": self.outputSignature,
            "calls": self.calls,
            "called_by": self.calledBy,
            "memory_scopes": self.memoryScopes,
            "responsibilities": self.responsibilities,
        }


class FleetTopologyManifest(BaseModel):
    slots: dict[str, FleetTopologySlotManifest] = Field(default_factory=dict)
    externalHandoffTools: list[str] = Field(default_factory=list)

    def snake_dump(self) -> dict[str, Any]:
        return {
            "slots": {slot_id: slot.snake_dump() for slot_id, slot in sorted(self.slots.items())},
            "external_handoff_tools": self.externalHandoffTools,
        }


class ToolArgumentManifest(BaseModel):
    name: str
    type: str
    required: bool = True
    description: str | None = None
    source: str | None = None


class ToolManifest(BaseModel):
    id: str
    displayName: str | None = None
    description: str | None = None
    requiresApproval: bool = False
    permissionKey: str | None = None
    arguments: list[ToolArgumentManifest] = Field(default_factory=list)
    source: str | None = None
    inferred: bool = False
    inferredSource: str | None = None


class IntentManifest(BaseModel):
    id: str
    allowedToolIDs: list[str] = Field(default_factory=list)
    source: str | None = None


class RoutingMatrixEntry(BaseModel):
    intent: str
    allowedTools: list[str] = Field(default_factory=list)
    forbiddenTools: list[str] = Field(default_factory=list)


class FreshnessClassManifest(BaseModel):
    id: str
    ttlSeconds: int | None = None
    durable: bool = False
    source: str | None = None


class MemoryManifest(BaseModel):
    scopes: list[str] = Field(default_factory=list)
    freshnessClasses: list[FreshnessClassManifest] = Field(default_factory=list)


class SentinelManifest(BaseModel):
    forbiddenInUserOutput: list[str] = Field(default_factory=list)


class AgentProtocolManifest(BaseModel):
    cortexOutput: dict[str, Any] = Field(default_factory=lambda: {
        "requiredFields": ["intent", "selectedToolID", "requiresApproval", "nextModel", "reasoningSummary"]
    })
    executorOutput: dict[str, Any] = Field(default_factory=lambda: {
        "format": "strict_json",
        "noMarkdown": True,
        "noExplanation": True,
    })


class ValidationFailure(BaseModel):
    code: str
    message: str
    path: str | None = None


class ValidationWarning(BaseModel):
    code: str
    message: str
    path: str | None = None


class ValidationReport(BaseModel):
    passed: bool
    failures: list[ValidationFailure] = Field(default_factory=list)
    warnings: list[ValidationWarning] = Field(default_factory=list)


class AgentBehaviorManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemaVersion: str = "1.0.0"
    app: AppManifestInfo = Field(default_factory=AppManifestInfo)
    sourceIntegrity: SourceIntegrity = Field(default_factory=SourceIntegrity)
    fleet: FleetManifest = Field(default_factory=FleetManifest)
    fleetTopology: FleetTopologyManifest = Field(default_factory=FleetTopologyManifest)
    tools: list[ToolManifest] = Field(default_factory=list)
    intents: list[IntentManifest] = Field(default_factory=list)
    routingMatrix: list[RoutingMatrixEntry] = Field(default_factory=list)
    memory: MemoryManifest = Field(default_factory=MemoryManifest)
    sentinels: SentinelManifest = Field(default_factory=SentinelManifest)
    agentProtocols: AgentProtocolManifest = Field(default_factory=AgentProtocolManifest)

    def output_dict(self) -> dict[str, Any]:
        payload = self.model_dump()
        payload["fleet_topology"] = self.fleetTopology.snake_dump()
        return payload

    def write_json(self, path: Path, *, pretty: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        indent = 2 if pretty else None
        path.write_text(json.dumps(self.output_dict(), ensure_ascii=False, indent=indent, sort_keys=False) + "\n", encoding="utf-8")
