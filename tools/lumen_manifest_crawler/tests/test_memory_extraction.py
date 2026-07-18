from pathlib import Path

import pytest

from lumen_manifest_crawler.crawler import _apply_freshness_defaults
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, FreshnessClassManifest
from lumen_manifest_crawler.swift_extractors.base import SwiftFile
from lumen_manifest_crawler.swift_extractors.memory import MemoryExtractor


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_runtime_ttl_policies_are_bound_to_memory_store_source() -> None:
    root = _repo_root()
    model_path = root / "ios" / "Lumen" / "Models" / "MemoryItem.swift"
    store_path = root / "ios" / "Lumen" / "Services" / "MemoryStore.swift"
    manifest = AgentBehaviorManifest()
    extractor = MemoryExtractor()

    for path in (model_path, store_path):
        extractor.extract(
            SwiftFile(
                path=path,
                relpath=path.relative_to(root).as_posix(),
                text=path.read_text(encoding="utf-8"),
            ),
            manifest,
        )

    freshness = {item.id: item for item in manifest.memory.freshnessClasses}
    assert freshness["volatile"].ttlSeconds == 2700
    assert freshness["volatile"].durable is False
    assert freshness["volatile"].source == "ios/Lumen/Services/MemoryStore.swift"
    assert freshness["shortLived"].ttlSeconds == 21600
    assert freshness["shortLived"].durable is False
    assert freshness["shortLived"].source == "ios/Lumen/Services/MemoryStore.swift"


def test_runtime_aligned_fallbacks_do_not_restore_stale_ttls() -> None:
    manifest = AgentBehaviorManifest()
    manifest.memory.freshnessClasses = [
        FreshnessClassManifest(id="volatile"),
        FreshnessClassManifest(id="shortLived"),
    ]

    _apply_freshness_defaults(manifest)

    freshness = {item.id: item for item in manifest.memory.freshnessClasses}
    assert freshness["volatile"].ttlSeconds == 2700
    assert freshness["shortLived"].ttlSeconds == 21600


def test_conflicting_runtime_ttls_fail_closed() -> None:
    text = """
    return TTLPolicy(freshness: .volatile, ttl: 45 * 60)
    return TTLPolicy(freshness: .volatile, ttl: 60 * 60)
    """

    with pytest.raises(ValueError, match="conflicting runtime TTLs"):
        MemoryExtractor._runtime_ttl_policies(text)


def test_commented_runtime_ttl_does_not_override_live_source() -> None:
    text = """
    // return TTLPolicy(freshness: .volatile, ttl: 60 * 60)
    return TTLPolicy(freshness: .volatile, ttl: 45 * 60)
    """

    assert MemoryExtractor._runtime_ttl_policies(text) == {
        "volatile": 2700
    }
