from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import tools.prepare_qwen3_shared_base as shared_base


def _field(index: int, value_type: str, value: Any) -> dict[str, Any]:
    return {
        "index": index,
        "type": value_type,
        "offset": index * 16,
        "value": value,
    }


def _semantic_metadata(
    *,
    architecture: str = shared_base.EXPECTED_GGUF_ARCHITECTURE,
    general_type: str = shared_base.EXPECTED_GGUF_GENERAL_TYPE,
    file_type: int = shared_base.EXPECTED_GGUF_FILE_TYPE,
    chat_template: str = "controlled-chat-template",
) -> dict[str, dict[str, Any]]:
    values: list[tuple[str, str, Any]] = [
        ("GGUF.version", "UINT32", 3),
        ("GGUF.tensor_count", "UINT64", shared_base.EXPECTED_GGUF_TENSOR_COUNT),
        ("GGUF.kv_count", "UINT64", 17),
        ("general.architecture", "STRING", architecture),
        ("general.type", "STRING", general_type),
        ("general.file_type", "UINT32", file_type),
        (
            "general.quantization_version",
            "UINT32",
            shared_base.EXPECTED_GGUF_QUANTIZATION_VERSION,
        ),
        ("qwen3.block_count", "UINT32", 28),
        ("qwen3.context_length", "UINT32", 40_960),
        ("qwen3.embedding_length", "UINT32", 2_048),
        ("qwen3.feed_forward_length", "UINT32", 6_144),
        ("qwen3.attention.head_count", "UINT32", 16),
        ("qwen3.attention.head_count_kv", "UINT32", 8),
        ("tokenizer.ggml.bos_token_id", "UINT32", 151_643),
        ("tokenizer.ggml.eos_token_id", "UINT32", 151_645),
        ("tokenizer.ggml.padding_token_id", "UINT32", 151_643),
        ("tokenizer.ggml.add_bos_token", "BOOL", False),
        ("tokenizer.ggml.model", "STRING", "gpt2"),
        ("tokenizer.ggml.pre", "STRING", "qwen2"),
        ("tokenizer.chat_template", "STRING", chat_template),
    ]
    return {
        key: _field(index, value_type, value)
        for index, (key, value_type, value) in enumerate(values)
    }


def _fake_checkout(
    tmp_path: Path,
    artifact: Path,
    *,
    metadata: dict[str, dict[str, Any]],
) -> shared_base.VerifiedLlamaCppCheckout:
    tmp_path.mkdir(parents=True, exist_ok=True)
    reader = tmp_path / "gguf_dump.py"
    result = {
        "filename": str(artifact.resolve()),
        "endian": "LITTLE",
        "metadata": metadata,
        "tensors": {
            f"tensor.{index}": {"name": f"tensor.{index}"}
            for index in range(shared_base.EXPECTED_GGUF_TENSOR_COUNT)
        },
    }
    reader.write_text(
        "import json\n"
        f"print(json.dumps({result!r}, sort_keys=True))\n",
        encoding="utf-8",
    )
    source = reader.read_bytes()
    converter = tmp_path / "convert_hf_to_gguf.py"
    requirements = tmp_path / "requirements.txt"
    converter.write_text("# controlled converter\n", encoding="utf-8")
    requirements.write_text("# controlled requirements\n", encoding="utf-8")
    return shared_base.VerifiedLlamaCppCheckout(
        path=tmp_path,
        converter_script=converter,
        reader_script=reader,
        requirements_file=requirements,
        revision=shared_base.LLAMA_CPP_REVISION,
        tree_sha1=shared_base.LLAMA_CPP_TREE_SHA1,
        converter_git_blob_sha1="1" * 40,
        reader_git_blob_sha1=shared_base._git_blob_sha1(source),
        converter_sha256=hashlib.sha256(converter.read_bytes()).hexdigest(),
        reader_sha256=hashlib.sha256(source).hexdigest(),
        requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest(),
    )


def _fake_gguf(tmp_path: Path) -> Path:
    path = tmp_path / shared_base.EXPECTED_FILE_NAME
    path.write_bytes(
        b"".join(
            (
                b"GGUF",
                (3).to_bytes(4, byteorder="little", signed=False),
                shared_base.EXPECTED_GGUF_TENSOR_COUNT.to_bytes(
                    8, byteorder="little", signed=False
                ),
                (17).to_bytes(8, byteorder="little", signed=False),
                b"controlled-payload",
            )
        )
    )
    return path


def _patch_small_source_contract(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    *,
    index_shards: tuple[str, ...] = ("shard-1.safetensors", "shard-2.safetensors"),
) -> None:
    template = "unit-test-chat-template"
    payloads = {
        "config.json": json.dumps({"model_type": "qwen3"}).encode(),
        "generation_config.json": b"{}",
        "merges.txt": b"a b\n",
        "shard-1.safetensors": b"shard-one",
        "shard-2.safetensors": b"shard-two",
        "model.safetensors.index.json": json.dumps(
            {
                "metadata": {"total_size": 2},
                "weight_map": {
                    f"tensor.{index}": shard
                    for index, shard in enumerate(index_shards)
                },
            },
            sort_keys=True,
        ).encode(),
        "tokenizer.json": b'{"version":"1.0"}',
        "tokenizer_config.json": json.dumps({"chat_template": template}).encode(),
        "vocab.json": b"{}",
    }
    source.mkdir()
    for filename, payload in payloads.items():
        (source / filename).write_bytes(payload)

    shards = tuple(
        {
            "filename": filename,
            "size": len(payloads[filename]),
            "sha256": hashlib.sha256(payloads[filename]).hexdigest(),
        }
        for filename in ("shard-1.safetensors", "shard-2.safetensors")
    )
    contracts = tuple(
        {
            "filename": filename,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for filename, payload in payloads.items()
    )
    monkeypatch.setattr(shared_base, "SOURCE_MODEL_WEIGHT_SHARDS", shards)
    monkeypatch.setattr(shared_base, "SOURCE_FILE_CONTRACTS", contracts)
    monkeypatch.setattr(
        shared_base,
        "SOURCE_MODEL_INDEX_DIGEST",
        hashlib.sha256(payloads["model.safetensors.index.json"]).hexdigest(),
    )
    monkeypatch.setattr(
        shared_base,
        "SOURCE_MODEL_TOKENIZER_DIGEST",
        hashlib.sha256(payloads["tokenizer.json"]).hexdigest(),
    )
    monkeypatch.setattr(
        shared_base,
        "PINNED_QWEN3_CHAT_TEMPLATE_SHA256",
        hashlib.sha256(template.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        shared_base,
        "SOURCE_MODEL_ARTIFACT_DIGEST",
        shared_base.canonical_sha256(shared_base._shard_contract()),
    )
    monkeypatch.setattr(
        shared_base,
        "SOURCE_MODEL_INDEX_SHARD_BINDING_SHA256",
        shared_base.canonical_sha256(shared_base._index_shard_binding()),
    )
    monkeypatch.setattr(
        shared_base,
        "SOURCE_SNAPSHOT_SHA256",
        shared_base.canonical_sha256(shared_base._source_snapshot_payload()),
    )


def test_default_path_builds_only_exact_pinned_source() -> None:
    args = shared_base.parse_args([])

    assert args.method == "build"
    assert shared_base.SOURCE_MODEL_ID == "Qwen/Qwen3-1.7B"
    assert shared_base.SOURCE_MODEL_REVISION == (
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    assert shared_base.LLAMA_CPP_REVISION == (
        "34558825a27f4d74dcfd7a91bfde4464baa2a30a"
    )
    assert shared_base.SOURCE_MODEL_ARTIFACT_DIGEST == (
        "f0fcc7921091130524a2c1ab3d063a02dcc7327e6970279e3742c86de1737218"
    )
    with pytest.raises(SystemExit):
        shared_base.parse_args(["--method", "download"])
    with pytest.raises(SystemExit):
        shared_base.parse_args(["--method", "unsloth"])


def test_source_snapshot_verifies_exact_index_shard_and_template_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _patch_small_source_contract(monkeypatch, source)

    verified = shared_base.verify_source_snapshot(source)

    assert verified["modelID"] == shared_base.SOURCE_MODEL_ID
    assert verified["revision"] == shared_base.SOURCE_MODEL_REVISION
    assert verified["artifactDigest"] == shared_base.SOURCE_MODEL_ARTIFACT_DIGEST
    assert verified["snapshotSHA256"] == shared_base.SOURCE_SNAPSHOT_SHA256


def test_source_snapshot_rejects_index_outside_exact_shard_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _patch_small_source_contract(
        monkeypatch,
        source,
        index_shards=("shard-1.safetensors", "rogue.safetensors"),
    )

    with pytest.raises(RuntimeError, match="does not close over exact weight shards"):
        shared_base.verify_source_snapshot(source)


def test_valid_gguf_requires_exact_model_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = "controlled-chat-template"
    monkeypatch.setattr(
        shared_base,
        "PINNED_QWEN3_CHAT_TEMPLATE_SHA256",
        hashlib.sha256(template.encode()).hexdigest(),
    )
    artifact = _fake_gguf(tmp_path)
    checkout = _fake_checkout(
        tmp_path / "reader",
        artifact,
        metadata=_semantic_metadata(chat_template=template),
    )

    evidence = shared_base.validate_gguf(artifact, checkout=checkout, min_bytes=25)

    assert evidence["architecture"] == "qwen3"
    assert evidence["generalType"] == "model"
    assert evidence["fileType"] == shared_base.EXPECTED_GGUF_FILE_TYPE
    assert evidence["tensorCount"] == shared_base.EXPECTED_GGUF_TENSOR_COUNT
    assert evidence["chatTemplateSHA256"] == (
        shared_base.PINNED_QWEN3_CHAT_TEMPLATE_SHA256
    )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (_semantic_metadata(architecture="llama"), "general.architecture"),
        (_semantic_metadata(general_type="adapter"), "general.type"),
        (_semantic_metadata(file_type=2), "general.file_type"),
        (_semantic_metadata(chat_template="drifted"), "chat template drifted"),
    ],
)
def test_gguf_rejects_semantic_drift_even_with_valid_magic_and_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, dict[str, Any]],
    message: str,
) -> None:
    monkeypatch.setattr(
        shared_base,
        "PINNED_QWEN3_CHAT_TEMPLATE_SHA256",
        hashlib.sha256(b"controlled-chat-template").hexdigest(),
    )
    artifact = _fake_gguf(tmp_path)
    reader_root = tmp_path / "reader"
    reader_root.mkdir()
    checkout = _fake_checkout(reader_root, artifact, metadata=metadata)

    with pytest.raises(RuntimeError, match=message):
        shared_base.validate_gguf(artifact, checkout=checkout, min_bytes=25)


def test_attestation_binds_artifact_source_converter_and_self_hash(
    tmp_path: Path,
) -> None:
    checkout_root = tmp_path / "checkout"
    checkout_root.mkdir()
    artifact_path = _fake_gguf(tmp_path)
    checkout = _fake_checkout(
        checkout_root,
        artifact_path,
        metadata=_semantic_metadata(),
    )
    artifact = {
        "fileName": shared_base.EXPECTED_FILE_NAME,
        "sha256": "a" * 64,
        "sizeBytes": 1_234,
        "ggufVersion": 3,
        "tensorCount": shared_base.EXPECTED_GGUF_TENSOR_COUNT,
        "metadataKVCount": 17,
        "architecture": "qwen3",
        "generalType": "model",
        "fileType": shared_base.EXPECTED_GGUF_FILE_TYPE,
        "quantization": "Q4_K_M",
        "quantizationVersion": 2,
        "chatTemplateSHA256": shared_base.PINNED_QWEN3_CHAT_TEMPLATE_SHA256,
    }
    environment = {
        "cmakeVersion": "cmake version 4.0",
        "machine": "x86_64",
        "platform": "Linux-test",
        "pythonImplementation": "CPython",
        "pythonVersion": "3.12.0",
    }

    attestation = shared_base.make_attestation(
        artifact=artifact,
        source=shared_base._expected_source_attestation(),
        checkout=checkout,
        quantizer_sha256="b" * 64,
        target_repo=shared_base.DEFAULT_TARGET_REPO,
        build_environment=environment,
    )

    assert attestation["sourceBaseModel"]["artifactDigest"] == (
        shared_base.SOURCE_MODEL_ARTIFACT_DIGEST
    )
    assert attestation["converter"]["revision"] == shared_base.LLAMA_CPP_REVISION
    assert attestation["attestationSHA256"] == shared_base.canonical_sha256(
        {key: value for key, value in attestation.items() if key != "attestationSHA256"}
    )
    assert shared_base.verify_attestation(
        attestation,
        artifact=artifact,
        checkout=checkout,
        expected_target_repo=shared_base.DEFAULT_TARGET_REPO,
    ) == attestation

    rehashed_drift = copy.deepcopy(attestation)
    rehashed_drift["artifact"]["sha256"] = "c" * 64
    rehashed_drift["attestationSHA256"] = shared_base.canonical_sha256(
        {
            key: value
            for key, value in rehashed_drift.items()
            if key != "attestationSHA256"
        }
    )
    with pytest.raises(RuntimeError, match="does not bind the GGUF"):
        shared_base.verify_attestation(
            rehashed_drift,
            artifact=artifact,
            checkout=checkout,
            expected_target_repo=shared_base.DEFAULT_TARGET_REPO,
        )


def test_checkout_verifier_rejects_untracked_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "llama.cpp"
    reader = checkout / shared_base.GGUF_READER_RELATIVE_PATH
    converter = checkout / shared_base.CONVERTER_RELATIVE_PATH
    requirements = checkout / shared_base.CONVERTER_REQUIREMENTS_RELATIVE_PATH
    reader.parent.mkdir(parents=True)
    requirements.parent.mkdir(parents=True)
    reader.write_text("# reader\n", encoding="utf-8")
    converter.write_text("# converter\n", encoding="utf-8")
    requirements.write_text("# requirements\n", encoding="utf-8")
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Lumen Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "lumen@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "remote",
            "add",
            "origin",
            shared_base.LLAMA_CPP_REPOSITORY,
        ],
        check=True,
    )
    monkeypatch.setattr(
        shared_base,
        "LLAMA_CPP_REVISION",
        subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    monkeypatch.setattr(
        shared_base,
        "LLAMA_CPP_TREE_SHA1",
        subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"], text=True
        ).strip(),
    )
    monkeypatch.setattr(
        shared_base,
        "CONVERTER_GIT_BLOB_SHA1",
        subprocess.check_output(
            [
                "git",
                "-C",
                str(checkout),
                "rev-parse",
                f"HEAD:{shared_base.CONVERTER_RELATIVE_PATH.as_posix()}",
            ],
            text=True,
        ).strip(),
    )
    monkeypatch.setattr(
        shared_base,
        "GGUF_READER_GIT_BLOB_SHA1",
        subprocess.check_output(
            [
                "git",
                "-C",
                str(checkout),
                "rev-parse",
                f"HEAD:{shared_base.GGUF_READER_RELATIVE_PATH.as_posix()}",
            ],
            text=True,
        ).strip(),
    )
    monkeypatch.setattr(
        shared_base,
        "CONVERTER_SHA256",
        hashlib.sha256(converter.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        shared_base,
        "GGUF_READER_SHA256",
        hashlib.sha256(reader.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        shared_base,
        "CONVERTER_REQUIREMENTS_SHA256",
        hashlib.sha256(requirements.read_bytes()).hexdigest(),
    )

    shared_base.verify_llama_cpp_checkout(checkout)
    (checkout / "untracked_startup_hook.py").write_text("raise SystemExit\n")

    with pytest.raises(RuntimeError, match="completely clean"):
        shared_base.verify_llama_cpp_checkout(checkout)


def test_clone_fetches_and_checks_out_only_the_full_pinned_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    sentinel = object()
    monkeypatch.setattr(shared_base.subprocess, "run", fake_run)
    monkeypatch.setattr(shared_base, "verify_llama_cpp_checkout", lambda _: sentinel)

    result = shared_base.clone_pinned_llama_cpp(tmp_path / "llama.cpp")

    assert result is sentinel
    assert any(
        "fetch" in command and shared_base.LLAMA_CPP_REVISION in command
        for command in calls
    )
    assert any(
        "checkout" in command and shared_base.LLAMA_CPP_REVISION in command
        for command in calls
    )
    assert all("HEAD" not in command for command in calls)
    assert all("core.hooksPath=/dev/null" in command for command in calls)


def test_build_recipe_is_offline_single_threaded_and_q4_k_m() -> None:
    recipe = shared_base._build_recipe()

    assert recipe["offlineConversion"] is True
    assert recipe["quantizationThreads"] == 1
    assert "Q4_K_M" in recipe["quantizationArguments"]
    assert "<verified-source-snapshot>" in recipe["conversionArguments"]
    assert "-DGGML_NATIVE=OFF" in recipe["cmakeConfiguration"]
