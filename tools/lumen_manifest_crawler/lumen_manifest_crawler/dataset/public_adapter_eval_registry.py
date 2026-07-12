"""Pinned, evaluation-only public corpus declarations and hash-only fingerprints.

The registry deliberately contains no adapter training target. Raw BFCL rows
are read only while building a fingerprint bundle; the emitted artifact keeps
only normalized row hashes and bounded token-shingle sketches. This supports
exact and near-leakage checks without retaining benchmark prompts or schemas.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping


REGISTRY_PATH = Path(__file__).with_name("public_adapter_eval_sources.json")
FINGERPRINT_BUNDLE_PATH = Path(__file__).with_name("public_adapter_eval_fingerprints.json")
REGISTRY_SCHEMA_PREFIX = "lumen.public-adapter-evaluation-sources"
FINGERPRINT_SCHEMA = "lumen.public-adapter-evaluation-fingerprints/1.1.0"
SUPPORTED_REGISTRY_MAJOR = 1
ROW_NORMALIZATION = "nfkc-casefold-whitespace-canonical-json-v1"
TOKENIZATION = "unicode-word-or-punctuation-v1"
TOKEN_SHINGLE_WIDTH = 5
TOKEN_SHINGLE_SKETCH_SIZE = 64
TOKEN_SHINGLE_DIGEST_HEX_LENGTH = 32
_SCHEMA_RE = re.compile(
    rf"^{re.escape(REGISTRY_SCHEMA_PREFIX)}/(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data with a stable canonical encoding."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_public_adapter_eval_registry(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the forward-compatible, same-major source registry."""
    registry_path = path or REGISTRY_PATH
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("public adapter evaluation registry must be a JSON object")
    validate_public_adapter_eval_registry(payload)
    return payload


def validate_public_adapter_eval_registry(registry: Mapping[str, Any]) -> None:
    """Reject any registry that could make public evaluation data trainable."""
    schema = registry.get("schema")
    match = _SCHEMA_RE.fullmatch(schema) if isinstance(schema, str) else None
    if match is None or int(match.group("major")) != SUPPORTED_REGISTRY_MAJOR:
        raise ValueError(f"unsupported public adapter evaluation registry schema: {schema!r}")
    if registry.get("purpose") != "evaluation_only":
        raise ValueError("public adapter evaluation registry must be evaluation_only")
    if registry.get("hashOnly") is not True:
        raise ValueError("public adapter evaluation registry must emit hashes only")
    if registry.get("trainingEligible") is not False or registry.get("trainingTargets") != []:
        raise ValueError("public adapter evaluation registry cannot declare training targets")
    if registry.get("license") != "Apache-2.0":
        raise ValueError("BFCL registry must retain its Apache-2.0 license")

    revision = registry.get("revision")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("public adapter evaluation revision must be a pinned commit SHA")
    source_url = registry.get("sourceURL")
    if not isinstance(source_url, str) or revision not in source_url:
        raise ValueError("public adapter evaluation sourceURL must include the pinned revision")

    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("public adapter evaluation registry must declare artifacts")
    seen_ids: set[str] = set()
    seen_categories: set[str] = set()
    seen_namespaces: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("public adapter evaluation artifact must be an object")
        artifact_id = _required_string(artifact, "id")
        category = _required_string(artifact, "category")
        namespace = _required_string(artifact, "contaminationNamespace")
        if artifact_id in seen_ids or category in seen_categories or namespace in seen_namespaces:
            raise ValueError("public adapter evaluation artifact identifiers must be unique")
        seen_ids.add(artifact_id)
        seen_categories.add(category)
        seen_namespaces.add(namespace)

        relative_path = Path(_required_string(artifact, "path"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe public evaluation artifact path: {relative_path}")
        artifact_url = _required_string(artifact, "artifactURL")
        if revision not in artifact_url or relative_path.as_posix() not in artifact_url:
            raise ValueError(f"artifactURL is not pinned to the declared artifact: {artifact_id}")
        artifact_sha = artifact.get("artifactSHA256")
        if not isinstance(artifact_sha, str) or _SHA256_RE.fullmatch(artifact_sha) is None:
            raise ValueError(f"invalid artifact SHA-256: {artifact_id}")
        if not isinstance(artifact.get("artifactBytes"), int) or artifact["artifactBytes"] <= 0:
            raise ValueError(f"invalid artifact byte count: {artifact_id}")
        if artifact.get("format") != "jsonl":
            raise ValueError(f"unsupported public evaluation artifact format: {artifact_id}")
        if artifact.get("trainingEligible") is not False or artifact.get("trainingTargets") != []:
            raise ValueError(f"public evaluation artifact cannot declare training targets: {artifact_id}")
        _required_string_list(artifact, "evaluationAdapters")
        _required_string_list(artifact, "metrics")


def build_public_adapter_eval_fingerprint_bundle(
    registry: Mapping[str, Any] | None = None,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Load or build deterministic per-row, hash-only leakage evidence.

    Normal callers receive the committed bundle derived from the pinned BFCL
    artifacts. Passing ``artifact_root`` rebuilds the same evidence from local
    bytes after verifying their size and SHA-256 against the registry.
    """
    use_default_registry = registry is None
    source = dict(registry or load_public_adapter_eval_registry())
    validate_public_adapter_eval_registry(source)
    if artifact_root is None:
        if use_default_registry:
            return _load_default_public_adapter_eval_fingerprint_bundle()
        return load_public_adapter_eval_fingerprint_bundle(registry=source)

    artifact_paths = _verify_declared_artifacts(source, artifact_root)

    source_identity = {
        "datasetID": source["datasetID"],
        "revision": source["revision"],
        "license": source["license"],
        "sourceURL": source["sourceURL"],
    }
    artifacts: list[dict[str, Any]] = []
    total_rows = 0
    for declaration in sorted(source["artifacts"], key=lambda item: item["id"]):
        locator = {
            "datasetID": source["datasetID"],
            "revision": source["revision"],
            "path": declaration["path"],
            "artifactSHA256": declaration["artifactSHA256"],
        }
        artifact_fingerprint = {
            "id": declaration["id"],
            "category": declaration["category"],
            "artifactSHA256": declaration["artifactSHA256"],
            "artifactBytes": declaration["artifactBytes"],
            "sourceLocatorSHA256": canonical_sha256(locator),
            "evaluationAdapters": sorted(declaration["evaluationAdapters"]),
            "metrics": sorted(declaration["metrics"]),
            "contaminationNamespace": declaration["contaminationNamespace"],
            "trainingEligible": False,
            "trainingTargets": [],
        }
        rows = _fingerprint_jsonl_rows(artifact_paths[declaration["id"]])
        total_rows += len(rows)
        artifact_fingerprint.update(
            {
                "rowCount": len(rows),
                "rowFingerprintAggregateSHA256": canonical_sha256(rows),
                "rows": rows,
            }
        )
        artifact_fingerprint["declarationSHA256"] = canonical_sha256(artifact_fingerprint)
        artifacts.append(artifact_fingerprint)

    payload = {
        "schema": FINGERPRINT_SCHEMA,
        "purpose": "evaluation_only_contamination_and_provenance",
        "hashOnly": True,
        "rawEvaluationTextIncluded": False,
        "trainingEligible": False,
        "trainingTargets": [],
        "registrySHA256": canonical_sha256(source),
        "source": source_identity,
        "sourceFingerprintSHA256": canonical_sha256(source_identity),
        "rowFingerprintContract": {
            "normalization": ROW_NORMALIZATION,
            "rowDigest": "sha256",
            "tokenization": TOKENIZATION,
            "tokenShingleWidth": TOKEN_SHINGLE_WIDTH,
            "tokenShingleSketch": f"bottom-{TOKEN_SHINGLE_SKETCH_SIZE}",
            "tokenShingleDigest": f"sha256-{TOKEN_SHINGLE_DIGEST_HEX_LENGTH * 4}",
        },
        "rowCount": total_rows,
        "artifacts": artifacts,
    }
    payload["bundleSHA256"] = canonical_sha256(payload)
    validate_public_adapter_eval_fingerprint_bundle(payload, registry=source)
    return payload


def load_public_adapter_eval_fingerprint_bundle(
    path: Path | None = None,
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate the committed hash-only BFCL row bundle."""
    bundle_path = path or FINGERPRINT_BUNDLE_PATH
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"unable to read public adapter evaluation fingerprints: {bundle_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("public adapter evaluation fingerprint bundle must be a JSON object")
    validate_public_adapter_eval_fingerprint_bundle(
        payload,
        registry=registry or load_public_adapter_eval_registry(),
    )
    return payload


@lru_cache(maxsize=1)
def _load_default_public_adapter_eval_fingerprint_bundle() -> dict[str, Any]:
    return load_public_adapter_eval_fingerprint_bundle()


def validate_public_adapter_eval_fingerprint_bundle(
    bundle: Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | None = None,
) -> None:
    """Validate provenance, completeness, and the absence of raw evaluation text."""
    source = dict(registry or load_public_adapter_eval_registry())
    validate_public_adapter_eval_registry(source)
    expected_bundle_keys = {
        "schema",
        "purpose",
        "hashOnly",
        "rawEvaluationTextIncluded",
        "trainingEligible",
        "trainingTargets",
        "registrySHA256",
        "source",
        "sourceFingerprintSHA256",
        "rowFingerprintContract",
        "rowCount",
        "artifacts",
        "bundleSHA256",
    }
    if set(bundle) != expected_bundle_keys:
        raise ValueError("public evaluation fingerprint bundle contains undeclared fields")
    if bundle.get("schema") != FINGERPRINT_SCHEMA:
        raise ValueError(f"unsupported public evaluation fingerprint schema: {bundle.get('schema')!r}")
    if bundle.get("purpose") != "evaluation_only_contamination_and_provenance":
        raise ValueError("public evaluation fingerprints must remain evaluation-only")
    if (
        bundle.get("hashOnly") is not True
        or bundle.get("rawEvaluationTextIncluded") is not False
        or bundle.get("trainingEligible") is not False
        or bundle.get("trainingTargets") != []
    ):
        raise ValueError("public evaluation fingerprint bundle violates its hash-only contract")
    if bundle.get("registrySHA256") != canonical_sha256(source):
        raise ValueError("public evaluation fingerprint registry hash mismatch")
    expected_source = {
        "datasetID": source["datasetID"],
        "revision": source["revision"],
        "license": source["license"],
        "sourceURL": source["sourceURL"],
    }
    if bundle.get("source") != expected_source:
        raise ValueError("public evaluation source identity mismatch")
    if bundle.get("sourceFingerprintSHA256") != canonical_sha256(expected_source):
        raise ValueError("public evaluation source fingerprint mismatch")

    contract = bundle.get("rowFingerprintContract")
    expected_contract = {
        "normalization": ROW_NORMALIZATION,
        "rowDigest": "sha256",
        "tokenization": TOKENIZATION,
        "tokenShingleWidth": TOKEN_SHINGLE_WIDTH,
        "tokenShingleSketch": f"bottom-{TOKEN_SHINGLE_SKETCH_SIZE}",
        "tokenShingleDigest": f"sha256-{TOKEN_SHINGLE_DIGEST_HEX_LENGTH * 4}",
    }
    if contract != expected_contract:
        raise ValueError("public evaluation row fingerprint contract mismatch")

    declarations = {item["id"]: item for item in source["artifacts"]}
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(declarations):
        raise ValueError("public evaluation fingerprint artifact coverage mismatch")
    seen: set[str] = set()
    total_rows = 0
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("public evaluation fingerprint artifact must be an object")
        expected_artifact_keys = {
            "id",
            "category",
            "artifactSHA256",
            "artifactBytes",
            "sourceLocatorSHA256",
            "evaluationAdapters",
            "metrics",
            "contaminationNamespace",
            "trainingEligible",
            "trainingTargets",
            "rowCount",
            "rowFingerprintAggregateSHA256",
            "rows",
            "declarationSHA256",
        }
        if set(artifact) != expected_artifact_keys:
            raise ValueError("public evaluation fingerprint artifact contains undeclared fields")
        artifact_id = artifact.get("id")
        if not isinstance(artifact_id, str) or artifact_id in seen or artifact_id not in declarations:
            raise ValueError("public evaluation fingerprint artifact ID mismatch")
        seen.add(artifact_id)
        declaration = declarations[artifact_id]
        expected_locator = {
            "datasetID": source["datasetID"],
            "revision": source["revision"],
            "path": declaration["path"],
            "artifactSHA256": declaration["artifactSHA256"],
        }
        if (
            artifact.get("category") != declaration["category"]
            or artifact.get("artifactSHA256") != declaration["artifactSHA256"]
            or artifact.get("artifactBytes") != declaration["artifactBytes"]
            or artifact.get("sourceLocatorSHA256") != canonical_sha256(expected_locator)
            or artifact.get("evaluationAdapters") != sorted(declaration["evaluationAdapters"])
            or artifact.get("metrics") != sorted(declaration["metrics"])
            or artifact.get("contaminationNamespace") != declaration["contaminationNamespace"]
        ):
            raise ValueError(f"public evaluation artifact hash mismatch: {artifact_id}")
        if artifact.get("trainingEligible") is not False or artifact.get("trainingTargets") != []:
            raise ValueError(f"public evaluation row fingerprints cannot be trainable: {artifact_id}")

        rows = artifact.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"public evaluation artifact has no row fingerprints: {artifact_id}")
        if artifact.get("rowCount") != len(rows):
            raise ValueError(f"public evaluation row count mismatch: {artifact_id}")
        if artifact.get("rowFingerprintAggregateSHA256") != canonical_sha256(rows):
            raise ValueError(f"public evaluation row aggregate mismatch: {artifact_id}")
        for ordinal, row in enumerate(rows):
            _validate_row_fingerprint(row, expected_ordinal=ordinal, artifact_id=artifact_id)
        total_rows += len(rows)

        without_declaration_hash = dict(artifact)
        declared_hash = without_declaration_hash.pop("declarationSHA256", None)
        if declared_hash != canonical_sha256(without_declaration_hash):
            raise ValueError(f"public evaluation artifact declaration hash mismatch: {artifact_id}")

    if bundle.get("rowCount") != total_rows:
        raise ValueError("public evaluation total row count mismatch")
    without_bundle_hash = dict(bundle)
    declared_bundle_hash = without_bundle_hash.pop("bundleSHA256", None)
    if declared_bundle_hash != canonical_sha256(without_bundle_hash):
        raise ValueError("public evaluation bundle hash mismatch")


def public_adapter_eval_source_descriptors(
    registry: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return acquisition metadata for an isolated adapter-evaluation consumer."""
    source = dict(registry or load_public_adapter_eval_registry())
    validate_public_adapter_eval_registry(source)
    return [
        {
            "datasetID": source["datasetID"],
            "revision": source["revision"],
            "license": source["license"],
            "id": artifact["id"],
            "category": artifact["category"],
            "artifactURL": artifact["artifactURL"],
            "artifactSHA256": artifact["artifactSHA256"],
            "artifactBytes": artifact["artifactBytes"],
            "format": artifact["format"],
            "evaluationAdapters": sorted(artifact["evaluationAdapters"]),
            "metrics": sorted(artifact["metrics"]),
            "contaminationNamespace": artifact["contaminationNamespace"],
            "trainingEligible": False,
            "trainingTargets": [],
        }
        for artifact in sorted(source["artifacts"], key=lambda item: item["id"])
    ]


def public_evaluation_text_shingle_hashes(values: Iterable[str]) -> set[str]:
    """Hash all normalized five-token shingles without retaining source text."""

    tokens: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("public evaluation contamination inputs must be strings")
        normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
        tokens.extend(re.findall(r"[^\W_]+|[^\s\w]", normalized, flags=re.UNICODE))
    if not tokens:
        return set()
    width = min(TOKEN_SHINGLE_WIDTH, len(tokens))
    return {
        hashlib.sha256("\u241f".join(tokens[index : index + width]).encode("utf-8")).hexdigest()[
            :TOKEN_SHINGLE_DIGEST_HEX_LENGTH
        ]
        for index in range(len(tokens) - width + 1)
    }


def _verify_declared_artifacts(
    registry: Mapping[str, Any], artifact_root: Path
) -> dict[str, Path]:
    root = artifact_root.resolve()
    verified: dict[str, Path] = {}
    for artifact in registry["artifacts"]:
        path = (root / artifact["path"]).resolve()
        if root not in path.parents:
            raise ValueError(f"artifact escapes verification root: {artifact['id']}")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"unable to read public evaluation artifact: {artifact['id']}") from exc
        if len(payload) != artifact["artifactBytes"]:
            raise ValueError(f"public evaluation artifact byte count mismatch: {artifact['id']}")
        if hashlib.sha256(payload).hexdigest() != artifact["artifactSHA256"]:
            raise ValueError(f"public evaluation artifact hash mismatch: {artifact['id']}")
        verified[artifact["id"]] = path
    return verified


def _fingerprint_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid public evaluation JSONL at {path.name}:{line_number}") from exc
        if not isinstance(value, dict) or not value:
            raise ValueError(f"public evaluation row must be a non-empty object: {path.name}:{line_number}")
        normalized = _normalize_json_value(value)
        tokens = list(_iter_normalized_tokens(normalized))
        shingle_hashes = _bottom_k_shingle_hashes(tokens)
        rows.append(
            {
                "rowOrdinal": len(rows),
                "normalizedRowSHA256": canonical_sha256(normalized),
                "tokenCount": len(tokens),
                "tokenShingleCount": max(1, len(tokens) - TOKEN_SHINGLE_WIDTH + 1) if tokens else 0,
                "tokenShingleSketch": shingle_hashes,
            }
        )
    if not rows:
        raise ValueError(f"public evaluation artifact contains no rows: {path.name}")
    return rows


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in sorted(value.items())}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError(f"unsupported public evaluation JSON value: {type(value).__name__}")


def _iter_normalized_tokens(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield from re.findall(r"[^\W_]+|[^\s\w]", value, flags=re.UNICODE)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_normalized_tokens(item)
    elif isinstance(value, dict):
        for key, item in sorted(value.items()):
            yield from re.findall(r"[^\W_]+|[^\s\w]", key.casefold(), flags=re.UNICODE)
            yield from _iter_normalized_tokens(item)
    elif value is not None:
        yield json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bottom_k_shingle_hashes(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    width = min(TOKEN_SHINGLE_WIDTH, len(tokens))
    digests = {
        hashlib.sha256("\u241f".join(tokens[index : index + width]).encode("utf-8")).hexdigest()[
            :TOKEN_SHINGLE_DIGEST_HEX_LENGTH
        ]
        for index in range(len(tokens) - width + 1)
    }
    return sorted(digests)[:TOKEN_SHINGLE_SKETCH_SIZE]


def _validate_row_fingerprint(
    row: Any,
    *,
    expected_ordinal: int,
    artifact_id: str,
) -> None:
    expected_keys = {
        "rowOrdinal",
        "normalizedRowSHA256",
        "tokenCount",
        "tokenShingleCount",
        "tokenShingleSketch",
    }
    if not isinstance(row, Mapping) or set(row) != expected_keys:
        raise ValueError(f"public evaluation row contains non-hash fields: {artifact_id}")
    if type(row.get("rowOrdinal")) is not int or row.get("rowOrdinal") != expected_ordinal:
        raise ValueError(f"public evaluation row ordinal mismatch: {artifact_id}")
    row_hash = row.get("normalizedRowSHA256")
    if not isinstance(row_hash, str) or _SHA256_RE.fullmatch(row_hash) is None:
        raise ValueError(f"invalid public evaluation row hash: {artifact_id}")
    token_count = row.get("tokenCount")
    shingle_count = row.get("tokenShingleCount")
    if type(token_count) is not int or token_count <= 0:
        raise ValueError(f"invalid public evaluation token count: {artifact_id}")
    if type(shingle_count) is not int or shingle_count <= 0:
        raise ValueError(f"invalid public evaluation shingle count: {artifact_id}")
    sketch = row.get("tokenShingleSketch")
    digest_re = re.compile(rf"^[0-9a-f]{{{TOKEN_SHINGLE_DIGEST_HEX_LENGTH}}}$")
    if (
        not isinstance(sketch, list)
        or not sketch
        or len(sketch) > TOKEN_SHINGLE_SKETCH_SIZE
        or sketch != sorted(set(sketch))
        or any(not isinstance(item, str) or digest_re.fullmatch(item) is None for item in sketch)
    ):
        raise ValueError(f"invalid public evaluation shingle sketch: {artifact_id}")


def _required_string(value: Mapping[str, Any], key: str) -> str:
    candidate = value.get(key)
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError(f"public adapter evaluation field must be a non-empty string: {key}")
    return candidate


def _required_string_list(value: Mapping[str, Any], key: str) -> list[str]:
    candidate = value.get(key)
    if (
        not isinstance(candidate, list)
        or not candidate
        or any(not isinstance(item, str) or not item.strip() for item in candidate)
        or len(candidate) != len(set(candidate))
    ):
        raise ValueError(f"public adapter evaluation field must contain unique strings: {key}")
    return candidate
