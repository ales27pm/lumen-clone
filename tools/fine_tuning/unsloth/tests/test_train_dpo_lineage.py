from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import types
from pathlib import Path

import pytest

from tools.fine_tuning.unsloth import train_dpo, train_sft
from tools.fine_tuning.unsloth.adapter_artifact import write_adapter_artifact_manifest


QWEN_MODEL_ID = "Qwen/Qwen3-1.7B"
QWEN_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SFT_CODE_SHA256 = "e" * 64
DEPENDENCY_LOCK_SHA256 = "f" * 64
REQUIREMENTS_SHA256 = "0" * 64
RUNTIME_SOURCE_REVISION = "a" * 40


def _safetensors_bytes(data: bytes = b"\x00\x00\x00\x00") -> bytes:
    header = json.dumps(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, len(data)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    return len(header).to_bytes(8, "little") + header + data


def _write_sft_adapter(path: Path) -> dict:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(_safetensors_bytes())
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return write_adapter_artifact_manifest(path, training_phase="sft")


def _write_finalized_sft_manifest(
    path: Path,
    artifact: dict,
    *,
    agent: str = "executor",
    variant: str = "internal_plus_public_optimized",
    source_variant_sha256: str = "c" * 64,
    seed: int = 42,
) -> None:
    payload = {
        "agent": agent,
        "variant": variant,
        "seed": seed,
        "sourceVariantManifestSHA256": source_variant_sha256,
        "trainingEnvironment": {"effectiveSeed": seed},
        "trainingCodeSHA256": SFT_CODE_SHA256,
        "trainingDependencyLockSHA256": DEPENDENCY_LOCK_SHA256,
        "requirementsSHA256": REQUIREMENTS_SHA256,
        "runtimeSourceRevision": RUNTIME_SOURCE_REVISION,
        "artifact": {
            "status": "trained",
            "trainingPhase": "sft",
            "adapterSHA256": artifact["adapterSHA256"],
            "adapterManifestSHA256": artifact["adapterSHA256"],
            "effectiveSeed": seed,
        },
    }
    payload["variantManifestSHA256"] = train_dpo._canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verified_sft_parent_rejects_identity_digest_and_file_drift(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "sft" / "executor"
    adapter.parent.mkdir()
    artifact = _write_sft_adapter(adapter)
    finalized = tmp_path / "sft-finalized.json"
    cfg = {
        "agent": "executor",
        "variant": "internal_plus_public_optimized",
        "variantManifestSHA256": "c" * 64,
        "seed": 42,
        "trainingCodeSHA256ByPhase": {"sft": SFT_CODE_SHA256},
        "trainingDependencyLockSHA256": DEPENDENCY_LOCK_SHA256,
        "requirementsSHA256": REQUIREMENTS_SHA256,
    }

    _write_finalized_sft_manifest(finalized, artifact, agent="cortex")
    with pytest.raises(RuntimeError, match="finalized SFT artifact"):
        train_dpo._verified_sft_parent(
            cfg, adapter_dir=adapter, finalized_manifest_path=finalized
        )

    _write_finalized_sft_manifest(finalized, artifact, variant="internal_only")
    with pytest.raises(RuntimeError, match="finalized SFT artifact"):
        train_dpo._verified_sft_parent(
            cfg, adapter_dir=adapter, finalized_manifest_path=finalized
        )

    _write_finalized_sft_manifest(finalized, artifact, source_variant_sha256="d" * 64)
    with pytest.raises(RuntimeError, match="finalized SFT artifact"):
        train_dpo._verified_sft_parent(
            cfg, adapter_dir=adapter, finalized_manifest_path=finalized
        )

    _write_finalized_sft_manifest(finalized, artifact)
    payload = json.loads(finalized.read_text(encoding="utf-8"))
    payload["artifact"]["adapterSHA256"] = "f" * 64
    payload["variantManifestSHA256"] = train_dpo._canonical_sha256(
        {key: value for key, value in payload.items() if key != "variantManifestSHA256"}
    )
    finalized.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected finalized lineage"):
        train_dpo._verified_sft_parent(
            cfg, adapter_dir=adapter, finalized_manifest_path=finalized
        )

    _write_finalized_sft_manifest(finalized, artifact)
    (adapter / "adapter_model.safetensors").write_bytes(
        _safetensors_bytes(b"\x01\x00\x00\x00")
    )
    with pytest.raises(ValueError, match="do not match"):
        train_dpo._verified_sft_parent(
            cfg, adapter_dir=adapter, finalized_manifest_path=finalized
        )


def test_dpo_output_path_must_be_role_scoped_and_separate(tmp_path: Path) -> None:
    sft = tmp_path / "lora" / "executor"
    cfg = {
        "agent": "executor",
        "adapter_output_dir": str(sft),
        "output_dir": str(tmp_path / "training" / "executor"),
        "dpo_output_dir": str(tmp_path / "lora_dpo" / "executor"),
    }
    work, output = train_dpo.validate_dpo_artifact_paths(cfg, sft_adapter_dir=sft)
    assert work.name == "dpo"
    assert output == Path(cfg["dpo_output_dir"])

    cfg["dpo_output_dir"] = str(tmp_path / "result")
    with pytest.raises(ValueError, match="agent role"):
        train_dpo.validate_dpo_artifact_paths(cfg, sft_adapter_dir=sft)


def test_controlled_seed_rejects_cli_and_environment_drift() -> None:
    assert train_sft._resolve_controlled_seed(
        {"seed": 42}, cli_seed=42, environ={"LUMEN_TRAIN_SEED": "42"}
    ) == (42, "cli_verified")

    with pytest.raises(ValueError, match="CLI seed override"):
        train_sft._resolve_controlled_seed({"seed": 42}, cli_seed=7, environ={})
    with pytest.raises(ValueError, match="LUMEN_TRAIN_SEED override"):
        train_sft._resolve_controlled_seed(
            {"seed": 42}, environ={"LUMEN_TRAIN_SEED": "7"}
        )


def _valid_preference_row() -> dict:
    return {
        "prompt": [
            {"role": "system", "content": "Ground the answer in trusted observations."},
            {"role": "user", "content": "What did the tool report?"},
        ],
        "chosen": {"role": "assistant", "content": "The tool reported success."},
        "rejected": {"role": "assistant", "content": "I guessed that it worked."},
        "metadata": {"source": "test"},
    }


def test_preference_rows_remain_conversational_for_trl_chat_templates() -> None:
    source = _valid_preference_row()
    normalized = train_dpo.row_to_preference(source)

    assert normalized == {
        "prompt": source["prompt"],
        "chosen": [source["chosen"]],
        "rejected": [source["rejected"]],
    }
    assert isinstance(normalized["prompt"], list)
    assert normalized["prompt"][-1]["role"] == "user"
    assert normalized["chosen"][0]["role"] == "assistant"
    assert normalized["rejected"][0]["role"] == "assistant"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.pop("prompt"), "non-empty message list"),
        (lambda row: row.__setitem__("prompt", "question"), "non-empty message list"),
        (lambda row: row.__setitem__("prompt", []), "non-empty message list"),
        (
            lambda row: row.__setitem__("prompt", [{"role": "tool", "content": "result"}]),
            "unsupported role",
        ),
        (
            lambda row: row.__setitem__("prompt", [{"role": "user", "content": "  "}]),
            "non-empty text",
        ),
        (
            lambda row: row.__setitem__(
                "prompt", [{"role": "user", "content": "question", "name": "ignored"}]
            ),
            "only role and content",
        ),
        (
            lambda row: row.__setitem__(
                "prompt",
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "partial answer"},
                ],
            ),
            "must end with a user message",
        ),
        (
            lambda row: row.__setitem__(
                "prompt",
                [
                    {"role": "user", "content": "one"},
                    {"role": "user", "content": "two"},
                ],
            ),
            "must alternate",
        ),
        (lambda row: row.pop("chosen"), "include chosen and rejected"),
        (lambda row: row.pop("rejected"), "include chosen and rejected"),
        (lambda row: row.__setitem__("chosen", "answer"), "exactly one assistant message"),
        (
            lambda row: row.__setitem__(
                "chosen",
                [
                    {"role": "assistant", "content": "one"},
                    {"role": "assistant", "content": "two"},
                ],
            ),
            "exactly one assistant message",
        ),
        (
            lambda row: row.__setitem__("chosen", {"role": "user", "content": "answer"}),
            "unsupported role",
        ),
        (
            lambda row: row.__setitem__("rejected", {"role": "assistant", "content": "\t"}),
            "non-empty text",
        ),
        (
            lambda row: row.__setitem__(
                "rejected", {"role": "assistant", "content": "The  tool\nreported success."}
            ),
            "must differ",
        ),
    ],
)
def test_preference_rows_fail_closed_instead_of_synthesizing_defaults(
    mutate: object,
    message: str,
) -> None:
    row = _valid_preference_row()
    mutate(row)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        train_dpo.row_to_preference(row)


@pytest.mark.e2e
def test_pinned_qwen_trl_chat_template_preserves_assistant_boundaries() -> None:
    transformers = pytest.importorskip(
        "transformers",
        reason="Pinned Qwen chat-template integration requires transformers==4.57.6",
    )
    trl = pytest.importorskip(
        "trl",
        reason="Pinned Qwen chat-template integration requires trl==0.24.0",
    )
    if trl.__version__ != "0.24.0":
        pytest.skip(f"Pinned chat-template integration requires trl==0.24.0, found {trl.__version__}")
    if transformers.__version__ != "4.57.6":
        pytest.skip(
            "Pinned chat-template integration requires transformers==4.57.6, "
            f"found {transformers.__version__}"
        )

    allow_network = os.environ.get("LUMEN_ENABLE_NETWORK_TESTS") == "1"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            QWEN_MODEL_ID,
            revision=QWEN_REVISION,
            local_files_only=not allow_network,
        )
    except (OSError, ValueError):
        if allow_network:
            raise
        pytest.skip(
            "Pinned Qwen tokenizer is not cached; set LUMEN_ENABLE_NETWORK_TESTS=1 "
            "to enable this network-backed integration test"
        )

    from trl.data_utils import maybe_apply_chat_template

    normalized = train_dpo.row_to_preference(_valid_preference_row())
    prepared = maybe_apply_chat_template(normalized, tokenizer)
    without_generation_boundary = tokenizer.apply_chat_template(
        normalized["prompt"],
        tokenize=False,
        add_generation_prompt=False,
    )
    generation_boundary = prepared["prompt"][len(without_generation_boundary) :]

    assert prepared["prompt"].startswith(without_generation_boundary)
    assert generation_boundary
    assert "assistant" in generation_boundary
    assert prepared["chosen"].rstrip().endswith(tokenizer.eos_token)
    assert prepared["rejected"].rstrip().endswith(tokenizer.eos_token)
    assert "The tool reported success." in prepared["chosen"]
    assert "I guessed that it worked." in prepared["rejected"]

    old_flattened = {
        "prompt": without_generation_boundary,
        "chosen": normalized["chosen"][0]["content"],
        "rejected": normalized["rejected"][0]["content"],
    }
    assert maybe_apply_chat_template(old_flattened, tokenizer) == old_flattened
    assert prepared != old_flattened


def test_pinned_trl_024_dpo_and_orpo_constructor_contracts() -> None:
    class ConfigProbe:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class DPOTrainerProbe:
        def __init__(
            self,
            *,
            model: object,
            ref_model: object,
            args: ConfigProbe,
            train_dataset: object,
            eval_dataset: object,
            processing_class: object,
        ) -> None:
            self.inputs = (model, ref_model, args, train_dataset, eval_dataset, processing_class)

    class ORPOTrainerProbe:
        def __init__(
            self,
            *,
            model: object,
            args: ConfigProbe,
            train_dataset: object,
            eval_dataset: object,
            processing_class: object,
        ) -> None:
            self.inputs = (model, args, train_dataset, eval_dataset, processing_class)

    cfg = {
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "num_train_epochs": 1,
        "warmup_steps": 0,
        "max_seq_length": 512,
    }
    common = {
        "seed": 42,
        "model": object(),
        "tokenizer": object(),
        "train_dataset": object(),
        "val_dataset": None,
        "output_dir": Path("executor-dpo-training"),
        "dpo_config_class": ConfigProbe,
        "dpo_trainer_class": DPOTrainerProbe,
        "orpo_config_class": ConfigProbe,
        "orpo_trainer_class": ORPOTrainerProbe,
    }

    dpo_trainer, dpo_args = train_dpo._build_preference_trainer(
        cfg, preference_trainer="dpo", **common
    )
    assert isinstance(dpo_trainer, DPOTrainerProbe)
    assert dpo_args.kwargs["model_adapter_name"] == train_dpo.POLICY_ADAPTER_NAME
    assert dpo_args.kwargs["ref_adapter_name"] == train_dpo.REFERENCE_ADAPTER_NAME
    assert dpo_args.kwargs["seed"] == dpo_args.kwargs["data_seed"] == 42

    orpo_trainer, orpo_args = train_dpo._build_preference_trainer(
        cfg, preference_trainer="orpo", **common
    )
    assert isinstance(orpo_trainer, ORPOTrainerProbe)
    assert "model_adapter_name" not in orpo_args.kwargs
    assert orpo_args.kwargs["seed"] == orpo_args.kwargs["data_seed"] == 42


@pytest.mark.e2e
def test_actual_pinned_trl_024_config_and_trainer_constructor_apis() -> None:
    trl = pytest.importorskip(
        "trl",
        reason="Pinned constructor integration requires trl==0.24.0",
    )
    if trl.__version__ != "0.24.0":
        pytest.skip(
            f"Pinned constructor integration requires trl==0.24.0, found {trl.__version__}"
        )

    class DPOConstructorBinding:
        def __init__(self, **kwargs: object) -> None:
            inspect.signature(trl.DPOTrainer.__init__).bind(object(), **kwargs)

    class ORPOConstructorBinding:
        def __init__(self, **kwargs: object) -> None:
            inspect.signature(trl.ORPOTrainer.__init__).bind(object(), **kwargs)

    cfg = {
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "num_train_epochs": 1,
        "warmup_steps": 0,
        "max_seq_length": 512,
        "fp16": False,
    }
    common = {
        "seed": 42,
        "model": object(),
        "tokenizer": object(),
        "train_dataset": object(),
        "val_dataset": None,
        "output_dir": Path("executor-preference-training"),
        "dpo_config_class": trl.DPOConfig,
        "dpo_trainer_class": DPOConstructorBinding,
        "orpo_config_class": trl.ORPOConfig,
        "orpo_trainer_class": ORPOConstructorBinding,
    }

    _, dpo_args = train_dpo._build_preference_trainer(
        cfg,
        preference_trainer="dpo",
        **common,
    )
    _, orpo_args = train_dpo._build_preference_trainer(
        cfg,
        preference_trainer="orpo",
        **common,
    )

    assert isinstance(dpo_args, trl.DPOConfig)
    assert isinstance(orpo_args, trl.ORPOConfig)
    assert dpo_args.model_adapter_name == train_dpo.POLICY_ADAPTER_NAME
    assert dpo_args.ref_adapter_name == train_dpo.REFERENCE_ADAPTER_NAME
    assert dpo_args.seed == dpo_args.data_seed == 42
    assert orpo_args.seed == orpo_args.data_seed == 42


def test_dpo_loads_frozen_sft_reference_and_saves_only_policy(tmp_path: Path) -> None:
    events: list[tuple[str, object]] = []

    class Model:
        def load_adapter(self, path: str, *, adapter_name: str, is_trainable: bool) -> None:
            events.append(("load", (path, adapter_name, is_trainable)))

        def set_adapter(self, adapter_name: str) -> None:
            events.append(("set", adapter_name))

        def save_pretrained(self, path: str, **kwargs: object) -> None:
            events.append(("save", (path, kwargs)))

    class PeftModelProbe:
        @staticmethod
        def from_pretrained(
            base_model: object,
            path: str,
            *,
            adapter_name: str,
            is_trainable: bool,
        ) -> Model:
            events.append(
                ("from", (base_model, path, adapter_name, is_trainable))
            )
            return Model()

    adapter_dir = tmp_path / "executor-sft-adapter"
    model = train_dpo._load_sft_policy(
        object(),
        peft_model_class=PeftModelProbe,
        sft_adapter_dir=adapter_dir,
        preference_trainer="dpo",
    )
    train_dpo._save_policy_adapter(model, tmp_path / "executor-dpo-adapter")

    assert events[0][1][2:] == (train_dpo.POLICY_ADAPTER_NAME, True)
    assert events[1] == (
        "load",
        (str(adapter_dir), train_dpo.REFERENCE_ADAPTER_NAME, False),
    )
    assert events[-1][1][1] == {
        "safe_serialization": True,
        "selected_adapters": [train_dpo.POLICY_ADAPTER_NAME],
    }


def test_verify_base_model_lineage_checks_pinned_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    shard_name = "weights.safetensors"
    artifacts = {
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"layer.weight": shard_name}}, sort_keys=True
        ).encode(),
        "tokenizer.json": b"tokenizer",
        shard_name: b"weights",
    }
    downloaded: list[tuple[str, str, str]] = []
    for filename, content in artifacts.items():
        (tmp_path / filename).write_bytes(content)

    def hf_hub_download(*, repo_id: str, filename: str, revision: str) -> str:
        downloaded.append((repo_id, filename, revision))
        return str(tmp_path / filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download),
    )
    shard_contract = {
        "schemaVersion": "lumen.base-model-weight-shards/1.0.0",
        "shards": [
            {
                "filename": shard_name,
                "size": len(artifacts[shard_name]),
                "sha256": hashlib.sha256(artifacts[shard_name]).hexdigest(),
            }
        ],
    }
    index_digest = hashlib.sha256(artifacts["model.safetensors.index.json"]).hexdigest()
    artifact_digest = train_dpo._canonical_sha256(shard_contract)
    train_dpo._verify_base_model_lineage(
        {
            "base_model_name": "example/model",
            "baseModelRevision": revision,
            "baseModelIndexDigest": index_digest,
            "baseModelIndexReferencedShardNames": [shard_name],
            "baseModelIndexShardBindingSHA256": train_dpo._canonical_sha256(
                {
                    "schemaVersion": "lumen.base-model-index-shard-binding/1.0.0",
                    "indexDigest": index_digest,
                    "referencedShardNames": [shard_name],
                    "shardContractDigest": artifact_digest,
                }
            ),
            "baseModelWeightShards": shard_contract["shards"],
            "baseModelTokenizerDigest": hashlib.sha256(artifacts["tokenizer.json"]).hexdigest(),
            "baseModelArtifactDigest": artifact_digest,
        }
    )

    assert downloaded == [
        ("example/model", "model.safetensors.index.json", revision),
        ("example/model", "tokenizer.json", revision),
        ("example/model", shard_name, revision),
    ]


def test_verify_base_model_lineage_rejects_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "model.safetensors.index.json"
    artifact.write_bytes(b"unexpected")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda **_: str(artifact)),
    )

    with pytest.raises(RuntimeError, match="Pinned base-model artifact digest mismatch"):
        train_dpo._verify_base_model_lineage(
            {
                "base_model_name": "example/model",
                "baseModelRevision": "a" * 40,
                "baseModelIndexDigest": "0" * 64,
                "baseModelArtifactDigest": train_dpo._canonical_sha256(
                    {
                        "schemaVersion": "lumen.base-model-weight-shards/1.0.0",
                        "shards": [
                            {
                                "filename": "weights.safetensors",
                                "size": 1,
                                "sha256": "2" * 64,
                            }
                        ],
                    }
                ),
                "baseModelWeightShards": [
                    {
                        "filename": "weights.safetensors",
                        "size": 1,
                        "sha256": "2" * 64,
                    }
                ],
                "baseModelTokenizerDigest": "1" * 64,
            }
        )


def test_sft_lineage_rejects_modified_weight_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = tmp_path / "model.safetensors.index.json"
    tokenizer = tmp_path / "tokenizer.json"
    shard = tmp_path / "weights.safetensors"
    index.write_text(
        json.dumps({"weight_map": {"layer.weight": shard.name}}),
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"tokenizer")
    shard.write_bytes(b"modified")
    files = {path.name: path for path in (index, tokenizer, shard)}
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            hf_hub_download=lambda **kwargs: str(files[kwargs["filename"]])
        ),
    )
    declared_shards = [
        {"filename": shard.name, "size": 8, "sha256": hashlib.sha256(b"expected").hexdigest()}
    ]
    index_digest = hashlib.sha256(index.read_bytes()).hexdigest()
    artifact_digest = train_sft._canonical_sha256(
        {
            "schemaVersion": "lumen.base-model-weight-shards/1.0.0",
            "shards": declared_shards,
        }
    )
    with pytest.raises(RuntimeError, match="weight shard digest mismatch"):
        train_sft._verify_base_model_lineage(
            {
                "base_model_name": "example/model",
                "baseModelRevision": "a" * 40,
                "baseModelIndexDigest": index_digest,
                "baseModelIndexReferencedShardNames": [shard.name],
                "baseModelIndexShardBindingSHA256": train_sft._canonical_sha256(
                    {
                        "schemaVersion": "lumen.base-model-index-shard-binding/1.0.0",
                        "indexDigest": index_digest,
                        "referencedShardNames": [shard.name],
                        "shardContractDigest": artifact_digest,
                    }
                ),
                "baseModelArtifactDigest": artifact_digest,
                "baseModelWeightShards": declared_shards,
                "baseModelTokenizerDigest": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            }
        )
