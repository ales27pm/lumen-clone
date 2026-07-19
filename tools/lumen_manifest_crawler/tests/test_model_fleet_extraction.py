from pathlib import Path

import pytest

from lumen_manifest_crawler.crawler import _build_fleet_topology
from lumen_manifest_crawler.manifest import AgentBehaviorManifest
from lumen_manifest_crawler.swift_extractors.base import SwiftFile
from lumen_manifest_crawler.swift_extractors.model_fleet import ModelFleetExtractor


REPO_ROOT = Path(__file__).resolve().parents[3]
EMBEDDING_RESPONSIBILITIES = [
    "semantic memory embedding",
    "embedding vector generation",
]


def _extract_model_fleet(text: str) -> AgentBehaviorManifest:
    manifest = AgentBehaviorManifest()
    ModelFleetExtractor().extract(
        SwiftFile(
            Path("ModelFleet.swift"),
            "ios/Lumen/Services/ModelFleet.swift",
            text,
        ),
        manifest,
    )
    return manifest


def test_embedding_responsibilities_are_bound_to_exact_swift_contract() -> None:
    manifest = _extract_model_fleet(
        """
        static let embedding = LumenModelSlotContract(
            slot: .embedding,
            systemContract: "Embedding model slot for semantic memory.",
            outputContract: .embeddingVector
        )
        """
    )

    assert len(manifest.fleet.slots) == 1
    embedding = manifest.fleet.slots[0]
    assert embedding.id == "embedding"
    assert embedding.role == "embedding"
    assert embedding.responsibilities == EMBEDDING_RESPONSIBILITIES


@pytest.mark.parametrize(
    ("system_contract", "output_contract"),
    [
        ("Embedding model slot for semantic retrieval.", "embeddingVector"),
        ("Embedding model slot for semantic memory.", "finalText"),
    ],
)
def test_embedding_responsibility_fallback_fails_closed_on_contract_drift(
    system_contract: str,
    output_contract: str,
) -> None:
    manifest = _extract_model_fleet(
        f"""
        static let embedding = LumenModelSlotContract(
            slot: .embedding,
            systemContract: "{system_contract}",
            outputContract: .{output_contract}
        )
        """
    )

    assert manifest.fleet.slots[0].responsibilities == []


def test_current_swift_embedding_contract_materializes_specific_topology() -> None:
    source = REPO_ROOT / "ios" / "Lumen" / "Services" / "ModelFleet.swift"
    manifest = _extract_model_fleet(source.read_text(encoding="utf-8"))
    embedding = next(slot for slot in manifest.fleet.slots if slot.id == "embedding")

    assert embedding.responsibilities == EMBEDDING_RESPONSIBILITIES

    topology = _build_fleet_topology(manifest).slots["embedding"]
    assert topology.purpose == (
        "Generate semantic vector representations for memory indexing and retrieval."
    )
    assert topology.inputSignature == (
        "Text or content selected for semantic memory indexing or retrieval."
    )
    assert topology.outputSignature == (
        "Embedding vector only; no user-facing text, tool call, or hidden reasoning."
    )
    assert topology.responsibilities == sorted(EMBEDDING_RESPONSIBILITIES)
