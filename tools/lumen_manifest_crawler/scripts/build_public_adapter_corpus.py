#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lumen_manifest_crawler.dataset.public_adapter_corpus import (  # noqa: E402
    SOURCE_MANIFEST_PATH,
    PublicCorpusError,
    build_public_adapter_corpus,
)


def _artifact_override(value: str) -> tuple[str, Path]:
    source_id, separator, raw_path = value.partition("=")
    if not separator or not source_id or not raw_path:
        raise argparse.ArgumentTypeError("artifact override must be SOURCE_ID=PATH")
    return source_id, Path(raw_path).expanduser().resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Lumen's deterministic, license-pinned public adapter corpus snapshot."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/public_adapter_corpus"),
        help="Snapshot directory for records.jsonl, manifest.json, and attribution.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(".cache/lumen-public-adapter-corpus"),
        help="Raw artifact cache. Raw source files are never copied into the snapshot.",
    )
    parser.add_argument(
        "--lumen-manifest",
        type=Path,
        default=Path("generated/agent_manifest/AgentBehaviorManifest.json"),
        help="Current AgentBehaviorManifest used to validate every Executor tool envelope.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=SOURCE_MANIFEST_PATH,
        help="Pinned public source manifest.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid downloads and require every hash-verified artifact in the cache or an override.",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        type=_artifact_override,
        metavar="SOURCE_ID=PATH",
        help="Use a local raw artifact for a source. The pinned SHA-256 is still required.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = dict(args.artifact)
    try:
        result = build_public_adapter_corpus(
            args.output.resolve(),
            cache_dir=args.cache.resolve(),
            lumen_manifest_path=args.lumen_manifest.resolve(),
            source_manifest_path=args.source_manifest.resolve(),
            offline=args.offline,
            artifact_paths=overrides,
        )
    except (OSError, PublicCorpusError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "countsByAgent": result.counts_by_agent,
                "manifest": str(result.manifest_path),
                "recordCount": result.record_count,
                "records": str(result.records_path),
                "recordsSHA256": result.records_sha256,
                "thirdPartyAttribution": str(result.attribution_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
