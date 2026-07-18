#!/usr/bin/env python3
"""Build and attest Lumen's exact Qwen3 shared-base GGUF.

The shared base is a release artifact, not a convenient third-party download.
This utility therefore converts the exact controlled Hugging Face snapshot with
the exact controlled llama.cpp revision, quantizes it deterministically, reads
the finished GGUF with the pinned llama.cpp reader, and writes a self-hashed
attestation.  Verification fails closed when any source shard, index binding,
chat template, converter checkout, or GGUF semantic field drifts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence


EXPECTED_FILE_NAME = "lumen-qwen3-fast-shared-q4_k_m.gguf"
EXPECTED_ATTESTATION_FILE_NAME = (
    "lumen-qwen3-fast-shared-q4_k_m.attestation.json"
)
DEFAULT_OUTPUT = Path("models/base_qwen3_fast") / EXPECTED_FILE_NAME
DEFAULT_ATTESTATION_OUTPUT = (
    Path("models/base_qwen3_fast") / EXPECTED_ATTESTATION_FILE_NAME
)
DEFAULT_TARGET_REPO = "ales27pm/lumen-qwen3-bootstrap-gguf"

SOURCE_MODEL_ID = "Qwen/Qwen3-1.7B"
SOURCE_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SOURCE_MODEL_INDEX_DIGEST = (
    "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d"
)
SOURCE_MODEL_TOKENIZER_DIGEST = (
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
)
SOURCE_MODEL_ARTIFACT_DIGEST = (
    "f0fcc7921091130524a2c1ab3d063a02dcc7327e6970279e3742c86de1737218"
)
SOURCE_MODEL_INDEX_SHARD_BINDING_SHA256 = (
    "c7bc19b0bb18c6b3ac476c4d0c97b9eb3a430cdaf190376d6bb3c0b3369630af"
)
PINNED_QWEN3_CHAT_TEMPLATE_SHA256 = (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
)

LLAMA_CPP_REPOSITORY = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_REVISION = "34558825a27f4d74dcfd7a91bfde4464baa2a30a"
LLAMA_CPP_TREE_SHA1 = "c1a1276ee3e64576d3f91150f2410b4e9291ee45"
CONVERTER_RELATIVE_PATH = Path("convert_hf_to_gguf.py")
GGUF_READER_RELATIVE_PATH = Path("gguf-py/gguf/scripts/gguf_dump.py")
CONVERTER_REQUIREMENTS_RELATIVE_PATH = Path(
    "requirements/requirements-convert_hf_to_gguf.txt"
)
CONVERTER_GIT_BLOB_SHA1 = "3b23d5ebc0d303f4b5280d18207b642f7b4b4bab"
GGUF_READER_GIT_BLOB_SHA1 = "8177dff386c7ee1e130ed0a013c2d297ee866439"
CONVERTER_SHA256 = (
    "c819f18fb22927b49fabc3b35d1c9e21ee638b3817eccd1bd4efbcc7116eeb4d"
)
GGUF_READER_SHA256 = (
    "d8b8fc28e96d15d8a4d6f05cdff4a747f5a06f31172efee9dfc971998ed0203f"
)
CONVERTER_REQUIREMENTS_SHA256 = (
    "d6ee53814f8069932540c3c06f03121914098b3485c7a48bb7baa4e6358943d8"
)

ATTESTATION_SCHEMA_VERSION = "lumen.shared-base-gguf/1.0.0"
SOURCE_SNAPSHOT_SCHEMA_VERSION = "lumen.base-model-source-snapshot/1.0.0"
SOURCE_SHARD_SCHEMA_VERSION = "lumen.base-model-weight-shards/1.0.0"
INDEX_SHARD_BINDING_SCHEMA_VERSION = (
    "lumen.base-model-index-shard-binding/1.0.0"
)
CONVERTER_CLOSURE_SCHEMA_VERSION = "lumen.shared-base-converter/1.0.0"
BUILD_RECIPE_SCHEMA_VERSION = "lumen.shared-base-build-recipe/1.0.0"

EXPECTED_GGUF_ARCHITECTURE = "qwen3"
EXPECTED_GGUF_GENERAL_TYPE = "model"
EXPECTED_GGUF_FILE_TYPE = 15  # llama.cpp LlamaFileType.MOSTLY_Q4_K_M
EXPECTED_GGUF_QUANTIZATION_VERSION = 2
EXPECTED_GGUF_TENSOR_COUNT = 311
EXPECTED_QUANTIZATION = "Q4_K_M"
MIN_BYTES = 1_000_000_000
GGUF_FIXED_HEADER_SIZE = 24
GGUF_SUPPORTED_VERSIONS = frozenset({2, 3})
GGUF_READER_TIMEOUT_SECONDS = 120
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")

SOURCE_MODEL_WEIGHT_SHARDS: tuple[dict[str, Any], ...] = (
    {
        "filename": "model-00001-of-00002.safetensors",
        "size": 3_441_185_608,
        "sha256": (
            "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5"
        ),
    },
    {
        "filename": "model-00002-of-00002.safetensors",
        "size": 622_329_984,
        "sha256": (
            "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"
        ),
    },
)

# This is the complete isolated directory supplied to convert_hf_to_gguf.py.
# Exact hashes make a pinned Hub commit necessary but not sufficient: corrupted
# cache entries and server-side drift are rejected before converter code runs.
SOURCE_FILE_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "filename": "config.json",
        "size": 726,
        "sha256": (
            "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
        ),
    },
    {
        "filename": "generation_config.json",
        "size": 239,
        "sha256": (
            "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"
        ),
    },
    {
        "filename": "merges.txt",
        "size": 1_671_853,
        "sha256": (
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"
        ),
    },
    *SOURCE_MODEL_WEIGHT_SHARDS,
    {
        "filename": "model.safetensors.index.json",
        "size": 25_605,
        "sha256": SOURCE_MODEL_INDEX_DIGEST,
    },
    {
        "filename": "tokenizer.json",
        "size": 11_422_654,
        "sha256": SOURCE_MODEL_TOKENIZER_DIGEST,
    },
    {
        "filename": "tokenizer_config.json",
        "size": 9_732,
        "sha256": (
            "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
        ),
    },
    {
        "filename": "vocab.json",
        "size": 2_776_833,
        "sha256": (
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
        ),
    },
)
SOURCE_SNAPSHOT_SHA256 = (
    "6987f015a22615cf72fee744e0b09823a43bd749cbd7060535f8a9b656619f6c"
)

CMAKE_CONFIGURATION: tuple[str, ...] = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_NATIVE=OFF",
    "-DGGML_OPENMP=OFF",
    "-DGGML_LLAMAFILE=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_SERVER=OFF",
)
CONVERSION_ARGUMENTS: tuple[str, ...] = (
    "convert_hf_to_gguf.py",
    "--outfile",
    "<unquantized-f16.gguf>",
    "--outtype",
    "f16",
    "<verified-source-snapshot>",
)
QUANTIZATION_ARGUMENTS: tuple[str, ...] = (
    "llama-quantize",
    "<unquantized-f16.gguf>",
    EXPECTED_FILE_NAME,
    EXPECTED_QUANTIZATION,
    "1",
)

_GGUF_READER_FD_BOOTSTRAP = """
import os
import sys

reader_fd = int(sys.argv[1])
reader_path = sys.argv[2]
artifact_path = sys.argv[3]
chunks = []
offset = 0
while True:
    chunk = os.pread(reader_fd, 1 << 20, offset)
    if not chunk:
        break
    chunks.append(chunk)
    offset += len(chunk)
source = b"".join(chunks)
sys.argv = [reader_path, artifact_path, "--json"]
sys.path[0] = os.path.dirname(os.path.abspath(reader_path))
namespace = {
    "__name__": "__main__",
    "__file__": reader_path,
    "__package__": None,
    "__cached__": None,
}
exec(compile(source, reader_path, "exec"), namespace, namespace)
"""


@dataclass(frozen=True)
class VerifiedLlamaCppCheckout:
    path: Path
    converter_script: Path
    reader_script: Path
    requirements_file: Path
    revision: str
    tree_sha1: str
    converter_git_blob_sha1: str
    reader_git_blob_sha1: str
    converter_sha256: str
    reader_sha256: str
    requirements_sha256: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key is not allowed: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_nonfinite(value)
    return parsed


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_finite_float,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"Unable to read strict JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _file_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_regular_readonly(path: Path, *, label: str) -> tuple[BinaryIO, os.stat_result]:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise RuntimeError(f"{label} verification requires O_NOFOLLOW support")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable: {path}") from exc
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError(f"{label} must be a regular file: {path}")
        return os.fdopen(descriptor, "rb", closefd=True), observed
    except BaseException:
        os.close(descriptor)
        raise


def _require_stable_descriptor(
    handle: BinaryIO,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    if _file_signature(os.fstat(handle.fileno())) != _file_signature(expected):
        raise RuntimeError(f"{label} changed while it was being verified")


def _require_path_matches_descriptor(
    path: Path,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    try:
        observed = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"{label} changed while it was being verified") from exc
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_dev != expected.st_dev
        or observed.st_ino != expected.st_ino
    ):
        raise RuntimeError(f"{label} changed while it was being verified")


def _hash_descriptor(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(handle.fileno(), 1 << 20, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def file_sha256(path: Path, *, label: str = "File") -> str:
    handle, observed = _open_regular_readonly(path, label=label)
    try:
        digest = _hash_descriptor(handle)
        _require_stable_descriptor(handle, observed, label=label)
        _require_path_matches_descriptor(path, observed, label=label)
        return digest
    finally:
        handle.close()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_regular_file(path: Path, *, label: str) -> None:
    handle, observed = _open_regular_readonly(path, label=label)
    try:
        os.fsync(handle.fileno())
        _require_stable_descriptor(handle, observed, label=label)
        _require_path_matches_descriptor(path, observed, label=label)
    finally:
        handle.close()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _read_descriptor_bytes(handle: BinaryIO) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while True:
        chunk = os.pread(handle.fileno(), 1 << 20, offset)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        offset += len(chunk)


def _source_file_contracts() -> list[dict[str, Any]]:
    return [dict(item) for item in SOURCE_FILE_CONTRACTS]


def _source_snapshot_payload() -> dict[str, Any]:
    return {
        "schemaVersion": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "files": _source_file_contracts(),
    }


def _shard_contract() -> dict[str, Any]:
    return {
        "schemaVersion": SOURCE_SHARD_SCHEMA_VERSION,
        "shards": sorted(
            (dict(item) for item in SOURCE_MODEL_WEIGHT_SHARDS),
            key=lambda item: item["filename"],
        ),
    }


def _index_shard_binding() -> dict[str, Any]:
    return {
        "schemaVersion": INDEX_SHARD_BINDING_SCHEMA_VERSION,
        "indexDigest": SOURCE_MODEL_INDEX_DIGEST,
        "referencedShardNames": sorted(
            item["filename"] for item in SOURCE_MODEL_WEIGHT_SHARDS
        ),
        "shardContractDigest": SOURCE_MODEL_ARTIFACT_DIGEST,
    }


def _expected_source_attestation() -> dict[str, Any]:
    return {
        "modelID": SOURCE_MODEL_ID,
        "revision": SOURCE_MODEL_REVISION,
        "artifactDigest": SOURCE_MODEL_ARTIFACT_DIGEST,
        "indexDigest": SOURCE_MODEL_INDEX_DIGEST,
        "indexReferencedShardNames": sorted(
            item["filename"] for item in SOURCE_MODEL_WEIGHT_SHARDS
        ),
        "indexShardBindingSHA256": SOURCE_MODEL_INDEX_SHARD_BINDING_SHA256,
        "tokenizerDigest": SOURCE_MODEL_TOKENIZER_DIGEST,
        "chatTemplateSHA256": PINNED_QWEN3_CHAT_TEMPLATE_SHA256,
        "snapshot": _source_snapshot_payload(),
        "snapshotSHA256": SOURCE_SNAPSHOT_SHA256,
    }


def verify_source_snapshot(source_dir: Path) -> dict[str, Any]:
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise RuntimeError(f"Source snapshot must be a regular directory: {source_dir}")
    expected_names = {item["filename"] for item in SOURCE_FILE_CONTRACTS}
    observed_names = {item.name for item in source_dir.iterdir()}
    if observed_names != expected_names:
        raise RuntimeError("Pinned source snapshot has missing or unexpected entries")

    for contract in SOURCE_FILE_CONTRACTS:
        path = source_dir / contract["filename"]
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Pinned source artifact must be a regular file: {path}")
        if path.stat().st_size != contract["size"]:
            raise RuntimeError(f"Pinned source artifact size drifted: {path.name}")
        if file_sha256(path, label="Pinned source artifact") != contract["sha256"]:
            raise RuntimeError(f"Pinned source artifact digest drifted: {path.name}")

    if canonical_sha256(_shard_contract()) != SOURCE_MODEL_ARTIFACT_DIGEST:
        raise RuntimeError("Pinned source shard contract digest drifted")
    if canonical_sha256(_source_snapshot_payload()) != SOURCE_SNAPSHOT_SHA256:
        raise RuntimeError("Pinned source snapshot contract digest drifted")

    index = read_json_object(source_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or any(not isinstance(value, str) for value in weight_map.values())
    ):
        raise RuntimeError("Pinned source index has an invalid weight_map")
    referenced = sorted(set(weight_map.values()))
    expected_referenced = sorted(
        item["filename"] for item in SOURCE_MODEL_WEIGHT_SHARDS
    )
    if referenced != expected_referenced:
        raise RuntimeError("Pinned source index does not close over exact weight shards")
    if canonical_sha256(_index_shard_binding()) != SOURCE_MODEL_INDEX_SHARD_BINDING_SHA256:
        raise RuntimeError("Pinned source index-to-shard binding digest drifted")

    tokenizer_config = read_json_object(source_dir / "tokenizer_config.json")
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise RuntimeError("Pinned source tokenizer lacks a chat template")
    if hashlib.sha256(chat_template.encode("utf-8")).hexdigest() != (
        PINNED_QWEN3_CHAT_TEMPLATE_SHA256
    ):
        raise RuntimeError("Pinned source chat template digest drifted")
    return _expected_source_attestation()


def _copy_verified_source(source: Path, destination: Path, contract: Mapping[str, Any]) -> None:
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Pinned source artifact is unavailable: {source}") from exc
    source_handle, source_stat = _open_regular_readonly(
        resolved_source,
        label="Downloaded pinned source artifact",
    )
    temporary: Path | None = None
    digest = hashlib.sha256()
    try:
        if source_stat.st_size != contract["size"]:
            raise RuntimeError(
                f"Downloaded pinned source artifact size drifted: {contract['filename']}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_handle:
            temporary = Path(output_handle.name)
            offset = 0
            while True:
                chunk = os.pread(source_handle.fileno(), 1 << 20, offset)
                if not chunk:
                    break
                digest.update(chunk)
                output_handle.write(chunk)
                offset += len(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        _require_stable_descriptor(
            source_handle,
            source_stat,
            label="Downloaded pinned source artifact",
        )
        _require_path_matches_descriptor(
            resolved_source,
            source_stat,
            label="Downloaded pinned source artifact",
        )
        if digest.hexdigest() != contract["sha256"]:
            raise RuntimeError(
                f"Downloaded pinned source artifact digest drifted: {contract['filename']}"
            )
        os.replace(temporary, destination)
        temporary = None
    finally:
        source_handle.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def stage_source_snapshot(
    destination: Path,
    *,
    source_snapshot: Path | None = None,
) -> dict[str, Any]:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"Source staging directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if source_snapshot is not None:
        source_root = source_snapshot.resolve()
        source_paths = {
            contract["filename"]: source_root / contract["filename"]
            for contract in SOURCE_FILE_CONTRACTS
        }
    else:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError(
                "Missing huggingface_hub; install it to fetch the pinned source snapshot"
            ) from exc
        source_paths = {
            contract["filename"]: Path(
                hf_hub_download(
                    repo_id=SOURCE_MODEL_ID,
                    filename=contract["filename"],
                    revision=SOURCE_MODEL_REVISION,
                    repo_type="model",
                )
            )
            for contract in SOURCE_FILE_CONTRACTS
        }

    for contract in SOURCE_FILE_CONTRACTS:
        _copy_verified_source(
            source_paths[contract["filename"]],
            destination / contract["filename"],
            contract,
        )
    return verify_source_snapshot(destination)


def _git_environment() -> dict[str, str]:
    value = dict(os.environ)
    value.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    return value


def _git_output(checkout: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "--no-optional-locks",
                "-C",
                str(checkout),
                *arguments,
            ],
            text=True,
            capture_output=True,
            check=True,
            env=_git_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or ""
        raise RuntimeError(f"Unable to verify pinned llama.cpp checkout: {detail.strip()}") from exc
    return completed.stdout.strip()


def verify_llama_cpp_checkout(checkout: Path) -> VerifiedLlamaCppCheckout:
    if checkout.is_symlink() or not checkout.is_dir():
        raise RuntimeError(f"llama.cpp checkout must be a regular directory: {checkout}")
    revision = _git_output(checkout, "rev-parse", "HEAD")
    if revision != LLAMA_CPP_REVISION:
        raise RuntimeError("llama.cpp revision drifted from the pinned commit")
    tree_sha1 = _git_output(checkout, "rev-parse", "HEAD^{tree}")
    if tree_sha1 != LLAMA_CPP_TREE_SHA1:
        raise RuntimeError("llama.cpp source tree drifted from the pinned commit")
    if _git_output(checkout, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("llama.cpp checkout must be completely clean")
    remote = _git_output(checkout, "remote", "get-url", "origin")
    if remote.rstrip("/") != LLAMA_CPP_REPOSITORY:
        raise RuntimeError("llama.cpp origin is not the controlled repository")

    converter = checkout / CONVERTER_RELATIVE_PATH
    reader = checkout / GGUF_READER_RELATIVE_PATH
    requirements = checkout / CONVERTER_REQUIREMENTS_RELATIVE_PATH
    expected_files = (
        (converter, CONVERTER_RELATIVE_PATH, CONVERTER_GIT_BLOB_SHA1, CONVERTER_SHA256),
        (reader, GGUF_READER_RELATIVE_PATH, GGUF_READER_GIT_BLOB_SHA1, GGUF_READER_SHA256),
        (
            requirements,
            CONVERTER_REQUIREMENTS_RELATIVE_PATH,
            None,
            CONVERTER_REQUIREMENTS_SHA256,
        ),
    )
    for path, relative, expected_blob, expected_sha256 in expected_files:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Pinned llama.cpp input is unavailable: {relative}")
        _git_output(checkout, "ls-files", "--error-unmatch", relative.as_posix())
        blob = _git_output(checkout, "rev-parse", f"HEAD:{relative.as_posix()}")
        if GIT_SHA1_PATTERN.fullmatch(blob) is None:
            raise RuntimeError(f"Pinned llama.cpp input has invalid Git identity: {relative}")
        if expected_blob is not None and blob != expected_blob:
            raise RuntimeError(f"Pinned llama.cpp input Git identity drifted: {relative}")
        if file_sha256(path, label="Pinned llama.cpp input") != expected_sha256:
            raise RuntimeError(f"Pinned llama.cpp input content drifted: {relative}")

    return VerifiedLlamaCppCheckout(
        path=checkout,
        converter_script=converter,
        reader_script=reader,
        requirements_file=requirements,
        revision=revision,
        tree_sha1=tree_sha1,
        converter_git_blob_sha1=CONVERTER_GIT_BLOB_SHA1,
        reader_git_blob_sha1=GGUF_READER_GIT_BLOB_SHA1,
        converter_sha256=CONVERTER_SHA256,
        reader_sha256=GGUF_READER_SHA256,
        requirements_sha256=CONVERTER_REQUIREMENTS_SHA256,
    )


def clone_pinned_llama_cpp(destination: Path) -> VerifiedLlamaCppCheckout:
    if destination.exists():
        raise RuntimeError(f"Refusing to clone over an existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    commands = (
        ["git", "init", str(destination)],
        ["git", "-C", str(destination), "remote", "add", "origin", LLAMA_CPP_REPOSITORY],
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--no-tags",
            "--depth=1",
            "origin",
            LLAMA_CPP_REVISION,
        ],
        [
            "git",
            "-C",
            str(destination),
            "checkout",
            "--detach",
            LLAMA_CPP_REVISION,
        ],
    )
    for command in commands:
        controlled = [command[0], "-c", "core.hooksPath=/dev/null", *command[1:]]
        subprocess.run(controlled, check=True, env=_git_environment())
    return verify_llama_cpp_checkout(destination)


def _scalar_metadata(
    metadata: Mapping[str, Any],
    key: str,
    *,
    expected_type: str,
) -> Any:
    field = metadata.get(key)
    if (
        not isinstance(field, Mapping)
        or set(field) != {"index", "type", "offset", "value"}
        or type(field.get("index")) is not int
        or field["index"] < 0
        or field.get("type") != expected_type
        or type(field.get("offset")) is not int
        or field["offset"] < 0
    ):
        raise RuntimeError(f"Pinned GGUF reader returned invalid metadata for {key}")
    return field["value"]


def _verify_gguf_semantics(
    result: Mapping[str, Any],
    *,
    path: Path,
    tensor_count: int,
    metadata_kv_count: int,
) -> dict[str, Any]:
    metadata = result.get("metadata")
    tensors = result.get("tensors")
    if (
        set(result) != {"filename", "endian", "metadata", "tensors"}
        or Path(str(result.get("filename") or "")).resolve() != path.resolve()
        or result.get("endian") not in {"LITTLE", "BIG"}
        or not isinstance(metadata, Mapping)
        or not isinstance(tensors, Mapping)
        or len(metadata) != metadata_kv_count + 3
        or len(tensors) != tensor_count
        or not all(isinstance(value, Mapping) for value in metadata.values())
        or not all(isinstance(value, Mapping) for value in tensors.values())
    ):
        raise RuntimeError("Pinned GGUF reader evidence does not match the fixed header")

    fixed_values = {
        "GGUF.version": _scalar_metadata(metadata, "GGUF.version", expected_type="UINT32"),
        "GGUF.tensor_count": _scalar_metadata(
            metadata, "GGUF.tensor_count", expected_type="UINT64"
        ),
        "GGUF.kv_count": _scalar_metadata(metadata, "GGUF.kv_count", expected_type="UINT64"),
    }
    if (
        fixed_values["GGUF.tensor_count"] != tensor_count
        or fixed_values["GGUF.kv_count"] != metadata_kv_count
    ):
        raise RuntimeError("Pinned GGUF reader fixed metadata count drifted")

    required_strings = {
        "general.architecture": EXPECTED_GGUF_ARCHITECTURE,
        "general.type": EXPECTED_GGUF_GENERAL_TYPE,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
    }
    for key, expected in required_strings.items():
        observed = _scalar_metadata(metadata, key, expected_type="STRING")
        if not isinstance(observed, str) or observed != expected:
            raise RuntimeError(f"Shared-base GGUF semantic metadata drifted: {key}")

    required_uint32 = {
        "general.file_type": EXPECTED_GGUF_FILE_TYPE,
        "general.quantization_version": EXPECTED_GGUF_QUANTIZATION_VERSION,
        "qwen3.block_count": 28,
        "qwen3.context_length": 40_960,
        "qwen3.embedding_length": 2_048,
        "qwen3.feed_forward_length": 6_144,
        "qwen3.attention.head_count": 16,
        "qwen3.attention.head_count_kv": 8,
        "tokenizer.ggml.bos_token_id": 151_643,
        "tokenizer.ggml.eos_token_id": 151_645,
        "tokenizer.ggml.padding_token_id": 151_643,
    }
    for key, expected in required_uint32.items():
        if _scalar_metadata(metadata, key, expected_type="UINT32") != expected:
            raise RuntimeError(f"Shared-base GGUF semantic metadata drifted: {key}")
    if _scalar_metadata(metadata, "tokenizer.ggml.add_bos_token", expected_type="BOOL") is not False:
        raise RuntimeError("Shared-base GGUF tokenizer BOS behavior drifted")

    chat_template = _scalar_metadata(
        metadata,
        "tokenizer.chat_template",
        expected_type="STRING",
    )
    if not isinstance(chat_template, str) or not chat_template:
        raise RuntimeError("Shared-base GGUF lacks the controlled chat template")
    chat_template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if chat_template_sha256 != PINNED_QWEN3_CHAT_TEMPLATE_SHA256:
        raise RuntimeError("Shared-base GGUF chat template drifted")
    if "adapter.type" in metadata:
        raise RuntimeError("Shared-base GGUF must not contain adapter metadata")

    base_count = metadata.get("general.base_model.count")
    if base_count is not None:
        if _scalar_metadata(metadata, "general.base_model.count", expected_type="UINT32") != 1:
            raise RuntimeError("Shared-base GGUF must bind at most one exact source base")
        repo_url = _scalar_metadata(
            metadata,
            "general.base_model.0.repo_url",
            expected_type="STRING",
        )
        if repo_url != f"https://huggingface.co/{SOURCE_MODEL_ID}":
            raise RuntimeError("Shared-base GGUF embedded base-model identity drifted")

    if tensor_count != EXPECTED_GGUF_TENSOR_COUNT:
        raise RuntimeError("Shared-base GGUF tensor closure drifted")
    return {
        "ggufVersion": fixed_values["GGUF.version"],
        "tensorCount": tensor_count,
        "metadataKVCount": metadata_kv_count,
        "architecture": EXPECTED_GGUF_ARCHITECTURE,
        "generalType": EXPECTED_GGUF_GENERAL_TYPE,
        "fileType": EXPECTED_GGUF_FILE_TYPE,
        "quantization": EXPECTED_QUANTIZATION,
        "quantizationVersion": EXPECTED_GGUF_QUANTIZATION_VERSION,
        "chatTemplateSHA256": chat_template_sha256,
    }


def _run_pinned_reader(
    path: Path,
    *,
    artifact_handle: BinaryIO,
    checkout: VerifiedLlamaCppCheckout,
    tensor_count: int,
    metadata_kv_count: int,
) -> dict[str, Any]:
    reader_handle, reader_stat = _open_regular_readonly(
        checkout.reader_script,
        label="Pinned llama.cpp GGUF reader",
    )
    try:
        reader_source = _read_descriptor_bytes(reader_handle)
        if (
            hashlib.sha256(reader_source).hexdigest() != checkout.reader_sha256
            or _git_blob_sha1(reader_source) != checkout.reader_git_blob_sha1
        ):
            raise RuntimeError("Pinned llama.cpp GGUF reader content drifted")
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _GGUF_READER_FD_BOOTSTRAP,
                str(reader_handle.fileno()),
                str(checkout.reader_script),
                f"/proc/self/fd/{artifact_handle.fileno()}",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=GGUF_READER_TIMEOUT_SECONDS,
            pass_fds=(reader_handle.fileno(), artifact_handle.fileno()),
        )
        _require_stable_descriptor(
            reader_handle,
            reader_stat,
            label="Pinned llama.cpp GGUF reader",
        )
        _require_path_matches_descriptor(
            checkout.reader_script,
            reader_stat,
            label="Pinned llama.cpp GGUF reader",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Pinned llama.cpp GGUF reader could not inspect artifact") from exc
    finally:
        reader_handle.close()
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise RuntimeError(f"Pinned llama.cpp GGUF reader rejected artifact{suffix}")
    try:
        result = json.loads(
            completed.stdout,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_finite_float,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Pinned llama.cpp GGUF reader returned invalid JSON") from exc
    if not isinstance(result, Mapping):
        raise RuntimeError("Pinned llama.cpp GGUF reader returned invalid evidence")
    return _verify_gguf_semantics(
        result,
        path=path,
        tensor_count=tensor_count,
        metadata_kv_count=metadata_kv_count,
    )


def validate_gguf(
    path: Path,
    *,
    checkout: VerifiedLlamaCppCheckout,
    min_bytes: int = MIN_BYTES,
) -> dict[str, Any]:
    handle, observed = _open_regular_readonly(path, label="Shared-base GGUF")
    try:
        if observed.st_size < min_bytes or observed.st_size <= GGUF_FIXED_HEADER_SIZE:
            raise RuntimeError(
                f"Shared-base GGUF is too small: {observed.st_size} bytes"
            )
        header = os.pread(handle.fileno(), GGUF_FIXED_HEADER_SIZE, 0)
        if len(header) != GGUF_FIXED_HEADER_SIZE or header[:4] != b"GGUF":
            raise RuntimeError("Shared-base artifact has invalid GGUF magic/header")
        version = int.from_bytes(header[4:8], byteorder="little", signed=False)
        tensor_count = int.from_bytes(header[8:16], byteorder="little", signed=False)
        metadata_kv_count = int.from_bytes(
            header[16:24], byteorder="little", signed=False
        )
        if version not in GGUF_SUPPORTED_VERSIONS:
            raise RuntimeError(f"Shared-base GGUF version is unsupported: {version}")
        if tensor_count <= 0 or metadata_kv_count <= 0:
            raise RuntimeError("Shared-base GGUF fixed-header counts are invalid")
        digest = _hash_descriptor(handle)
        semantics = _run_pinned_reader(
            path,
            artifact_handle=handle,
            checkout=checkout,
            tensor_count=tensor_count,
            metadata_kv_count=metadata_kv_count,
        )
        if semantics["ggufVersion"] != version:
            raise RuntimeError("Shared-base GGUF reader version drifted from fixed header")
        _require_stable_descriptor(handle, observed, label="Shared-base GGUF")
        _require_path_matches_descriptor(path, observed, label="Shared-base GGUF")
    finally:
        handle.close()
    return {
        "fileName": EXPECTED_FILE_NAME,
        "sha256": digest,
        "sizeBytes": observed.st_size,
        **semantics,
    }


def _controlled_process_environment() -> dict[str, str]:
    value = dict(os.environ)
    value.update(
        {
            "HF_HUB_OFFLINE": "1",
            "LC_ALL": "C",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "TZ": "UTC",
        }
    )
    return value


def build_quantizer(
    checkout: VerifiedLlamaCppCheckout,
    build_dir: Path,
    *,
    jobs: int,
) -> Path:
    if jobs <= 0:
        raise RuntimeError("Build jobs must be positive")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(checkout.path),
            "-B",
            str(build_dir),
            *CMAKE_CONFIGURATION,
        ],
        check=True,
        env=_controlled_process_environment(),
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "llama-quantize",
            "--parallel",
            str(jobs),
        ],
        check=True,
        env=_controlled_process_environment(),
    )
    candidates = (
        build_dir / "bin" / "llama-quantize",
        build_dir / "bin" / "Release" / "llama-quantize",
    )
    binaries = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(binaries) != 1:
        raise RuntimeError("Pinned llama.cpp build did not produce one exact quantizer")
    return binaries[0]


def _command_version(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            text=True,
            capture_output=True,
            check=True,
            env=_controlled_process_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to record build tool version: {command[0]}") from exc
    output = (completed.stdout or completed.stderr).strip().splitlines()
    if not output:
        raise RuntimeError(f"Build tool did not report a version: {command[0]}")
    return output[0]


def _converter_closure(
    checkout: VerifiedLlamaCppCheckout,
    *,
    quantizer_sha256: str,
) -> dict[str, Any]:
    value = {
        "schemaVersion": CONVERTER_CLOSURE_SCHEMA_VERSION,
        "repository": LLAMA_CPP_REPOSITORY,
        "revision": checkout.revision,
        "treeSHA1": checkout.tree_sha1,
        "converterGitBlobSHA1": checkout.converter_git_blob_sha1,
        "converterSHA256": checkout.converter_sha256,
        "ggufReaderGitBlobSHA1": checkout.reader_git_blob_sha1,
        "ggufReaderSHA256": checkout.reader_sha256,
        "requirementsSHA256": checkout.requirements_sha256,
        "quantizerSHA256": quantizer_sha256,
    }
    return {**value, "closureSHA256": canonical_sha256(value)}


def _build_recipe() -> dict[str, Any]:
    return {
        "schemaVersion": BUILD_RECIPE_SCHEMA_VERSION,
        "cmakeConfiguration": list(CMAKE_CONFIGURATION),
        "conversionArguments": list(CONVERSION_ARGUMENTS),
        "quantizationArguments": list(QUANTIZATION_ARGUMENTS),
        "offlineConversion": True,
        "quantizationThreads": 1,
    }


def make_attestation(
    *,
    artifact: Mapping[str, Any],
    source: Mapping[str, Any],
    checkout: VerifiedLlamaCppCheckout,
    quantizer_sha256: str,
    target_repo: str,
    build_environment: Mapping[str, str],
) -> dict[str, Any]:
    if SHA256_PATTERN.fullmatch(quantizer_sha256) is None:
        raise RuntimeError("Quantizer digest is invalid")
    value = {
        "schemaVersion": ATTESTATION_SCHEMA_VERSION,
        "distribution": {
            "repositoryID": target_repo,
            "fileName": EXPECTED_FILE_NAME,
            "attestationFileName": EXPECTED_ATTESTATION_FILE_NAME,
        },
        "artifact": dict(artifact),
        "sourceBaseModel": dict(source),
        "converter": _converter_closure(
            checkout,
            quantizer_sha256=quantizer_sha256,
        ),
        "buildRecipe": _build_recipe(),
        "buildEnvironment": dict(build_environment),
    }
    return {**value, "attestationSHA256": canonical_sha256(value)}


def _expected_artifact_fields(evidence: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "fileName",
        "sha256",
        "sizeBytes",
        "ggufVersion",
        "tensorCount",
        "metadataKVCount",
        "architecture",
        "generalType",
        "fileType",
        "quantization",
        "quantizationVersion",
        "chatTemplateSHA256",
    )
    return {field: evidence[field] for field in fields}


def verify_attestation(
    attestation: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any],
    checkout: VerifiedLlamaCppCheckout,
    expected_target_repo: str,
) -> dict[str, Any]:
    expected_top_level = {
        "schemaVersion",
        "distribution",
        "artifact",
        "sourceBaseModel",
        "converter",
        "buildRecipe",
        "buildEnvironment",
        "attestationSHA256",
    }
    if set(attestation) != expected_top_level:
        raise RuntimeError("Shared-base attestation has an invalid field set")
    unsigned = dict(attestation)
    declared_sha256 = unsigned.pop("attestationSHA256", None)
    if (
        attestation.get("schemaVersion") != ATTESTATION_SCHEMA_VERSION
        or declared_sha256 != canonical_sha256(unsigned)
    ):
        raise RuntimeError("Shared-base attestation failed its self-hash")
    if attestation.get("distribution") != {
        "repositoryID": expected_target_repo,
        "fileName": EXPECTED_FILE_NAME,
        "attestationFileName": EXPECTED_ATTESTATION_FILE_NAME,
    }:
        raise RuntimeError("Shared-base distribution identity drifted")
    if attestation.get("artifact") != _expected_artifact_fields(artifact):
        raise RuntimeError("Shared-base attestation does not bind the GGUF bytes and semantics")
    if attestation.get("sourceBaseModel") != _expected_source_attestation():
        raise RuntimeError("Shared-base attestation source-model closure drifted")
    if attestation.get("buildRecipe") != _build_recipe():
        raise RuntimeError("Shared-base attestation build recipe drifted")

    converter = attestation.get("converter")
    if not isinstance(converter, Mapping):
        raise RuntimeError("Shared-base attestation converter closure is invalid")
    quantizer_sha256 = converter.get("quantizerSHA256")
    if not isinstance(quantizer_sha256, str) or SHA256_PATTERN.fullmatch(quantizer_sha256) is None:
        raise RuntimeError("Shared-base attestation quantizer digest is invalid")
    if dict(converter) != _converter_closure(
        checkout,
        quantizer_sha256=quantizer_sha256,
    ):
        raise RuntimeError("Shared-base attestation converter closure drifted")
    build_environment = attestation.get("buildEnvironment")
    if (
        not isinstance(build_environment, Mapping)
        or set(build_environment) != {
            "cmakeVersion",
            "machine",
            "platform",
            "pythonImplementation",
            "pythonVersion",
        }
        or not all(isinstance(value, str) and value for value in build_environment.values())
    ):
        raise RuntimeError("Shared-base attestation build environment is invalid")
    return dict(attestation)


def write_attestation(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _build_environment() -> dict[str, str]:
    return {
        "cmakeVersion": _command_version(["cmake", "--version"]),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "pythonImplementation": platform.python_implementation(),
        "pythonVersion": platform.python_version(),
    }


def build_shared_base(
    *,
    output: Path,
    attestation_path: Path,
    work_dir: Path,
    source_snapshot: Path | None,
    llama_cpp_dir: Path | None,
    target_repo: str,
    jobs: int,
    replace: bool,
) -> dict[str, Any]:
    if output.name != EXPECTED_FILE_NAME:
        raise RuntimeError(f"Shared-base output must be named {EXPECTED_FILE_NAME}")
    if attestation_path.name != EXPECTED_ATTESTATION_FILE_NAME:
        raise RuntimeError(
            f"Shared-base attestation must be named {EXPECTED_ATTESTATION_FILE_NAME}"
        )
    if not replace and (output.exists() or attestation_path.exists()):
        raise RuntimeError("Shared-base output already exists; pass --replace explicitly")
    if work_dir.exists() and any(work_dir.iterdir()):
        raise RuntimeError(f"Build work directory must be empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    staged_source = work_dir / "source-snapshot"
    source_contract = stage_source_snapshot(
        staged_source,
        source_snapshot=source_snapshot,
    )
    checkout = (
        verify_llama_cpp_checkout(llama_cpp_dir.resolve())
        if llama_cpp_dir is not None
        else clone_pinned_llama_cpp(work_dir / "llama.cpp")
    )
    quantizer = build_quantizer(checkout, work_dir / "llama-build", jobs=jobs)
    quantizer_sha256 = file_sha256(quantizer, label="Pinned llama.cpp quantizer")

    unquantized = work_dir / "qwen3-1.7b-f16.gguf"
    environment = _controlled_process_environment()
    subprocess.run(
        [
            sys.executable,
            str(checkout.converter_script),
            "--outfile",
            str(unquantized),
            "--outtype",
            "f16",
            str(staged_source),
        ],
        check=True,
        cwd=checkout.path,
        env=environment,
    )
    if unquantized.is_symlink() or not unquantized.is_file():
        raise RuntimeError("Pinned converter did not produce a regular unquantized GGUF")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".building",
    )
    os.close(descriptor)
    temporary_output = Path(temporary_name)
    temporary_output.unlink()
    try:
        subprocess.run(
            [
                str(quantizer),
                str(unquantized),
                str(temporary_output),
                EXPECTED_QUANTIZATION,
                "1",
            ],
            check=True,
            env=environment,
        )
        if temporary_output.is_symlink() or not temporary_output.is_file():
            raise RuntimeError("Pinned quantizer did not produce a regular GGUF")
        # Preserve any previously attested artifact until the replacement has
        # passed the pinned structural and semantic reader.  This intentionally
        # performs a second verification after the atomic rename so both the
        # candidate bytes and the final distribution path are bound.
        checkout = verify_llama_cpp_checkout(checkout.path)
        source_contract = verify_source_snapshot(staged_source)
        validate_gguf(temporary_output, checkout=checkout)
        _fsync_regular_file(
            temporary_output,
            label="Verified shared-base GGUF candidate",
        )
        os.replace(temporary_output, output)
        _fsync_directory(output.parent)
    finally:
        temporary_output.unlink(missing_ok=True)

    artifact = validate_gguf(output, checkout=checkout)
    attestation = make_attestation(
        artifact=artifact,
        source=source_contract,
        checkout=checkout,
        quantizer_sha256=quantizer_sha256,
        target_repo=target_repo,
        build_environment=_build_environment(),
    )
    verify_attestation(
        attestation,
        artifact=artifact,
        checkout=checkout,
        expected_target_repo=target_repo,
    )
    write_attestation(attestation_path, attestation)
    return attestation


def verify_existing(
    *,
    output: Path,
    attestation_path: Path,
    llama_cpp_dir: Path,
    target_repo: str,
) -> dict[str, Any]:
    if output.name != EXPECTED_FILE_NAME or attestation_path.name != (
        EXPECTED_ATTESTATION_FILE_NAME
    ):
        raise RuntimeError("Shared-base distribution filenames drifted")
    checkout = verify_llama_cpp_checkout(llama_cpp_dir.resolve())
    artifact = validate_gguf(output, checkout=checkout)
    return verify_attestation(
        read_json_object(attestation_path),
        artifact=artifact,
        checkout=checkout,
        expected_target_repo=target_repo,
    )


def hf_cli(root: Path) -> str:
    venv_hf = root / ".venv" / "bin" / "hf"
    return str(venv_hf) if venv_hf.is_file() else "hf"


def upload(
    output: Path,
    attestation_path: Path,
    *,
    target_repo: str,
    private: bool,
) -> None:
    hf = hf_cli(repo_root())
    create = [hf, "repos", "create", target_repo, "--type", "model", "--exist-ok"]
    if private:
        create.append("--private")
    subprocess.run(create, check=True)
    for path, remote_name in (
        (output, EXPECTED_FILE_NAME),
        (attestation_path, EXPECTED_ATTESTATION_FILE_NAME),
    ):
        subprocess.run(
            [
                hf,
                "upload",
                target_repo,
                str(path),
                remote_name,
                "--repo-type",
                "model",
                "--commit-message",
                "Upload attested Lumen Qwen3 shared base",
            ],
            check=True,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify Lumen's attested Qwen3 shared-base GGUF."
    )
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--attestation-output",
        type=Path,
        default=DEFAULT_ATTESTATION_OUTPUT,
    )
    parser.add_argument("--method", choices=["build", "verify"], default="build")
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        help="Optional local copy of the exact pinned Hugging Face snapshot.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        help="Exact clean pinned checkout; build clones the pinned commit when omitted.",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--target-repo", default=DEFAULT_TARGET_REPO)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = resolve(Path.cwd(), args.root).resolve()
    output = resolve(root, args.output).resolve()
    attestation_path = resolve(root, args.attestation_output).resolve()
    source_snapshot = (
        resolve(root, args.source_snapshot).resolve()
        if args.source_snapshot is not None
        else None
    )
    llama_cpp_dir = (
        resolve(root, args.llama_cpp_dir).resolve()
        if args.llama_cpp_dir is not None
        else None
    )

    temporary_work: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.method == "build":
            if args.work_dir is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary_work = tempfile.TemporaryDirectory(
                    prefix=".qwen3-shared-base-build-",
                    dir=output.parent,
                )
                work_dir = Path(temporary_work.name)
            else:
                work_dir = resolve(root, args.work_dir).resolve()
            build_shared_base(
                output=output,
                attestation_path=attestation_path,
                work_dir=work_dir,
                source_snapshot=source_snapshot,
                llama_cpp_dir=llama_cpp_dir,
                target_repo=args.target_repo,
                jobs=args.jobs,
                replace=args.replace,
            )
        else:
            if llama_cpp_dir is None:
                raise RuntimeError("--method verify requires --llama-cpp-dir")
            verify_existing(
                output=output,
                attestation_path=attestation_path,
                llama_cpp_dir=llama_cpp_dir,
                target_repo=args.target_repo,
            )
        if args.upload:
            if llama_cpp_dir is None and args.method == "verify":
                raise RuntimeError("Verified upload requires the pinned llama.cpp checkout")
            upload(
                output,
                attestation_path,
                target_repo=args.target_repo,
                private=args.private,
            )
    finally:
        if temporary_work is not None:
            temporary_work.cleanup()
    print(f"Verified shared base: {output}")
    print(f"Verified attestation: {attestation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
