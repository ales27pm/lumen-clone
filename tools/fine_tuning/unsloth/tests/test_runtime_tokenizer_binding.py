from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools.fine_tuning.unsloth import evaluate_adapter, export_gguf, train_dpo, train_sft


_MODEL_CONFIG_PAYLOAD = b'{"max_position_embeddings":40960}'


class _Backend:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def to_str(self) -> str:
        return json.dumps(
            {"model": {"type": "WordLevel", "marker": self.marker}},
            sort_keys=True,
        )


class _RuntimeTokenizer:
    is_fast = True
    chat_template = "{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}"
    bos_token_id = None
    eos_token_id = 2
    pad_token_id = 2
    unk_token_id = 0
    mask_token_id = None
    all_special_ids = [0, 2]
    all_special_tokens = ["<unk>", "<eos>"]
    padding_side = "left"
    truncation_side = "right"

    def __init__(
        self,
        source: Path,
        *,
        marker: str = "pinned",
        drift_on: str | None = None,
        model_max_length: int = 40_960,
    ) -> None:
        self.name_or_path = str(source)
        self.backend_tokenizer = _Backend(marker)
        self.model_max_length = model_max_length
        self._drift_on = drift_on

    def get_vocab(self) -> dict[str, int]:
        return {"<unk>": 0, "<eos>": 2, "Lumen": 10}

    def get_added_vocab(self) -> dict[str, int]:
        return {"<eos>": 2}

    def __call__(
        self,
        value: str,
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        marker = 1000 if self._drift_on and self._drift_on in value else 0
        ids = [marker + sum(word.encode("utf-8")) for word in value.split()]
        if add_special_tokens:
            ids = [2, *ids]
        return {"input_ids": ids}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        enable_thinking: bool,
        tokenize: bool,
        add_generation_prompt: bool,
        **_: Any,
    ) -> str | list[int]:
        assert enable_thinking is False
        rendered = "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "|assistant:"
        marker = 1000 if self._drift_on and self._drift_on in rendered else 0
        return (
            [marker + byte for byte in rendered.encode("utf-8")]
            if tokenize
            else rendered
        )


def _snapshot_record(snapshot: Path, marker: str = "a") -> dict[str, Any]:
    config_path = snapshot / "config.json"
    if not config_path.exists():
        config_path.write_bytes(_MODEL_CONFIG_PAYLOAD)
    payload = {
        "schemaVersion": (
            "lumen.private-base-model-conversion-snapshot-verification/1.1.0"
        ),
        "baseModelID": "example/model",
        "baseModelRevision": "b" * 40,
        "baseModelTokenizerDigest": "c" * 64,
        "baseModelTokenizerFiles": [],
        "baseModelTokenizerClosureSHA256": "d" * 64,
        "snapshotPath": str(snapshot),
        "verificationMethod": "private_regular_file_closure",
        "snapshotDirectorySignature": {"marker": marker},
        "fileStabilitySignatures": [],
    }
    return {
        **payload,
        "snapshotVerificationSHA256": train_sft._canonical_sha256(payload),
    }


def _config() -> dict[str, Any]:
    return {
        "baseModelID": "example/model",
        "base_model_name": "example/model",
        "baseModelRevision": "b" * 40,
        "baseModelTokenizerClosureSHA256": "d" * 64,
        "baseModelTokenizerFiles": [
            {
                "path": "config.json",
                "sizeBytes": len(_MODEL_CONFIG_PAYLOAD),
                "sha256": hashlib.sha256(_MODEL_CONFIG_PAYLOAD).hexdigest(),
                "huggingFaceBlobID": "a" * 40,
            }
        ],
        "baseModelRuntimeSnapshotPath": "/placeholder/runtime-snapshot",
        "baseModelRuntimeSnapshotVerification": {},
        "max_seq_length": 4_096,
        "load_in_4bit": True,
        "bf16": False,
        "fp16": True,
    }


def test_runtime_binding_accepts_only_equivalent_snapshot_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    verification = _snapshot_record(snapshot)
    monkeypatch.setattr(
        train_sft,
        "_verified_private_runtime_model_snapshot",
        lambda _cfg: (snapshot, verification),
    )

    binding = train_sft._verify_runtime_tokenizer_binding(
        _config(),
        expected_tokenizer=_RuntimeTokenizer(snapshot),
        runtime_tokenizer=_RuntimeTokenizer(snapshot),
        snapshot_path=snapshot,
        snapshot_verification=verification,
    )

    assert binding["schemaVersion"] == train_sft.RUNTIME_TOKENIZER_BINDING_SCHEMA
    assert binding["runtimeSnapshotVerificationSHA256"] == verification[
        "snapshotVerificationSHA256"
    ]
    assert binding["allowedRuntimeTransformations"]["modelMaxLength"] == 40_960


def test_runtime_binding_rejects_requested_length_when_model_context_is_larger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    verification = _snapshot_record(snapshot)
    monkeypatch.setattr(
        train_sft,
        "_verified_private_runtime_model_snapshot",
        lambda _cfg: (snapshot, verification),
    )

    with pytest.raises(RuntimeError, match="unapproved runtime transformation"):
        train_sft._verify_runtime_tokenizer_binding(
            _config(),
            expected_tokenizer=_RuntimeTokenizer(snapshot),
            runtime_tokenizer=_RuntimeTokenizer(
                snapshot,
                model_max_length=4_096,
            ),
            snapshot_path=snapshot,
            snapshot_verification=verification,
        )


def test_runtime_binding_rejects_unseen_prompt_token_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    verification = _snapshot_record(snapshot)
    monkeypatch.setattr(
        train_sft,
        "_verified_private_runtime_model_snapshot",
        lambda _cfg: (snapshot, verification),
    )

    with pytest.raises(RuntimeError, match="behavior drifted"):
        train_sft._verify_runtime_tokenizer_binding(
            _config(),
            expected_tokenizer=_RuntimeTokenizer(snapshot),
            runtime_tokenizer=_RuntimeTokenizer(
                snapshot,
                drift_on="report.json",
            ),
            snapshot_path=snapshot,
            snapshot_verification=verification,
        )


def test_runtime_binding_rejects_shared_cache_tokenizer_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    cache = tmp_path / "writable-hf-cache"
    cache.mkdir()
    verification = _snapshot_record(snapshot)
    monkeypatch.setattr(
        train_sft,
        "_verified_private_runtime_model_snapshot",
        lambda _cfg: (snapshot, verification),
    )

    with pytest.raises(RuntimeError, match="ignored the private snapshot"):
        train_sft._verify_runtime_tokenizer_binding(
            _config(),
            expected_tokenizer=_RuntimeTokenizer(snapshot),
            runtime_tokenizer=_RuntimeTokenizer(cache),
            snapshot_path=snapshot,
            snapshot_verification=verification,
        )


def test_runtime_binding_rejects_snapshot_replacement_after_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    before = _snapshot_record(snapshot, "before")
    after = _snapshot_record(snapshot, "after")
    monkeypatch.setattr(
        train_sft,
        "_verified_private_runtime_model_snapshot",
        lambda _cfg: (snapshot, after),
    )

    with pytest.raises(RuntimeError, match="changed during Unsloth loading"):
        train_sft._verify_runtime_tokenizer_binding(
            _config(),
            expected_tokenizer=_RuntimeTokenizer(snapshot),
            runtime_tokenizer=_RuntimeTokenizer(snapshot),
            snapshot_path=snapshot,
            snapshot_verification=before,
        )


class _SerializableConfig:
    def __init__(self, payload: dict[str, Any], *, name_or_path: Path | None = None) -> None:
        self._payload = payload
        self._name_or_path = str(name_or_path) if name_or_path is not None else ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class QuantState:
    __module__ = "bitsandbytes.functional"
    quant_type = "nf4"
    nested = True
    blocksize = 64
    dtype = "torch.float16"
    state2 = SimpleNamespace(blocksize=256)


class Params4bit:
    __module__ = "bitsandbytes.nn.modules"

    def __init__(self) -> None:
        self.device = SimpleNamespace(type="cuda")
        self.dtype = "torch.uint8"
        self.quant_type = "nf4"
        self.compress_statistics = True
        self.bnb_quantized = True
        self.requires_grad = False
        self.quant_state = QuantState()


class Linear4bit:
    __module__ = "bitsandbytes.nn.modules"

    def __init__(self) -> None:
        self.weight = Params4bit()
        self.compute_dtype = "torch.float16"


class BitsAndBytesConfig(_SerializableConfig):
    __module__ = "transformers.utils.quantization_config"


class Bnb4BitHfQuantizer:
    __module__ = "transformers.quantizers.quantizer_bnb_4bit"

    def __init__(self, quantization_config: BitsAndBytesConfig) -> None:
        self.quantization_config = quantization_config


class _FakeTensor:
    def __init__(
        self,
        values: Any,
        *,
        dtype: str,
        device: Any,
        shape: tuple[int, ...] | None = None,
        finite: bool = True,
    ) -> None:
        self.values = values
        self.dtype = dtype
        self.device = device
        self.shape = shape or (len(values), len(values[0]))
        self.requires_grad = False
        self.finite = finite


class _FakeReduction:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> _FakeReduction:
        return self

    def item(self) -> bool:
        return self.value


class _InferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None


@pytest.fixture(autouse=True)
def _controlled_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(
        long="torch.int64",
        float16="torch.float16",
        bfloat16="torch.bfloat16",
        tensor=lambda values, *, dtype, device: _FakeTensor(
            values,
            dtype=dtype,
            device=device,
        ),
        inference_mode=lambda: _InferenceMode(),
        isfinite=lambda tensor: _FakeReduction(tensor.finite),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _quantization_payload(*, active: bool = False) -> dict[str, Any]:
    payload = {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "quant_method": "bitsandbytes",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
        "llm_int8_enable_fp32_cpu_offload": False,
        "llm_int8_has_fp16_weight": False,
        "llm_int8_threshold": 6.0,
        "llm_int8_skip_modules": None,
    }
    if active:
        payload["llm_int8_skip_modules"] = [
            "lm_head",
            "multi_modal_projector",
            "merger",
            "modality_projection",
            "router",
            "mlp.gate",
            "block_sparse_moe.gate",
            "mamba",
            "audio_tower",
            "vision_tower",
            "vision_embedder",
            "embed_vision",
            "embed_audio",
            "score",
            "classifier",
            "qa_outputs",
        ]
    return payload


def _quantized_runtime_model(
    *,
    snapshot: Path,
    config_payload: dict[str, Any],
    generation_payload: dict[str, Any],
    max_seq_length: int,
    forward_dtype: str = "torch.float16",
    forward_device_type: str = "cuda",
    forward_finite: bool = True,
) -> Any:
    config_payload = {
        "attention_bias": False,
        "tie_word_embeddings": True,
        "vocab_size": 151_936,
        **config_payload,
        "num_hidden_layers": 28,
    }
    config = _SerializableConfig(config_payload, name_or_path=snapshot)
    config.quantization_config = _quantization_payload()
    quantization_config = BitsAndBytesConfig(_quantization_payload(active=True))
    modules = []
    for layer in range(28):
        modules.extend(
            (
                (f"model.layers.{layer}.self_attn.{projection}", Linear4bit())
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
        )
        modules.extend(
            (
                (f"model.layers.{layer}.mlp.{projection}", Linear4bit())
                for projection in ("gate_proj", "up_proj", "down_proj")
            )
        )
    parameter_by_name = {
        f"{module_name}.weight": module.weight
        for module_name, module in modules
    }
    def cuda_parameter() -> SimpleNamespace:
        return SimpleNamespace(device=SimpleNamespace(type="cuda"))

    parameter_by_name.update(
        {
            "model.embed_tokens.weight": cuda_parameter(),
            "model.norm.weight": cuda_parameter(),
        }
    )
    parameter_by_name.update(
        {
            f"model.layers.{layer}.{name}": cuda_parameter()
            for layer in range(28)
            for name in (
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            )
        }
    )
    class QuantizedRuntimeModel:
        def __init__(self) -> None:
            self.config = config
            self.generation_config = _SerializableConfig(generation_payload)
            self.max_seq_length = max_seq_length
            self.is_loaded_in_4bit = True
            self.is_quantized = True
            self.quantization_method = "bitsandbytes"
            self.hf_quantizer = Bnb4BitHfQuantizer(quantization_config)
            self.training = False

        def named_modules(self):
            return iter(modules)

        def named_parameters(self):
            return iter(parameter_by_name.items())

        def eval(self) -> QuantizedRuntimeModel:
            self.training = False
            return self

        def train(self) -> QuantizedRuntimeModel:
            self.training = True
            return self

        def __call__(
            self,
            *,
            input_ids: _FakeTensor,
            attention_mask: _FakeTensor,
            use_cache: bool,
            return_dict: bool,
        ) -> Any:
            assert input_ids.values == [(1, 2, 3, 4)]
            assert attention_mask.values == [(1, 1, 1, 1)]
            assert input_ids.device.type == "cuda"
            assert use_cache is False
            assert return_dict is True
            return SimpleNamespace(
                logits=_FakeTensor(
                    [],
                    dtype=forward_dtype,
                    device=SimpleNamespace(type=forward_device_type),
                    shape=(1, 4, config_payload["vocab_size"]),
                    finite=forward_finite,
                )
            )

    return QuantizedRuntimeModel()


def test_runtime_model_binding_uses_private_config_and_bound_generation_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    verification = _snapshot_record(snapshot)
    cfg = {
        **_config(),
        "baseModelIndexDigest": "1" * 64,
        "baseModelIndexShardBindingSHA256": "2" * 64,
        "baseModelArtifactDigest": "3" * 64,
        "baseModelGenerationConfigFile": {
            "path": "generation_config.json",
            "sizeBytes": 1,
            "sha256": "4" * 64,
            "huggingFaceBlobID": "5" * 40,
        },
    }
    monkeypatch.setattr(
        train_sft,
        "_verified_private_runtime_model_snapshot",
        lambda _cfg: (snapshot, verification),
    )
    expected_generation = {
        "eos_token_id": 2,
        "do_sample": False,
        "max_length": 20,
    }

    class GenerationConfig:
        @classmethod
        def from_pretrained(cls, path: str, *, local_files_only: bool):
            assert Path(path) == snapshot
            assert local_files_only is True
            return _SerializableConfig(expected_generation)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(GenerationConfig=GenerationConfig),
    )
    model = _quantized_runtime_model(
        snapshot=snapshot,
        config_payload={"model_type": "qwen3", "max_position_embeddings": 40_960},
        generation_payload={**expected_generation, "max_length": 40_960},
        max_seq_length=4_096,
    )

    binding = train_sft._verify_runtime_model_binding(
        cfg,
        runtime_model=model,
        snapshot_path=snapshot,
        snapshot_verification=verification,
    )

    assert binding["generationConfigSource"] == (
        "verified_private_generation_config_file"
    )
    assert binding["localFilesOnly"] is True
    assert binding["modelConfigVerificationStatus"] == (
        "attested_runtime_observation_not_independently_reconstructed"
    )
    assert binding["allowedGenerationConfigTransformations"] == {
        "maxLength": {
            "source": "verified_runtime_model.config.max_position_embeddings",
            "sourceValue": 40_960,
            "originalValue": 20,
            "runtimeValue": 40_960,
        }
    }
    materialization = binding["runtimeLoadMaterialization"]
    assert materialization["requestedMaxSequenceLength"] == 4_096
    assert materialization["runtimeMaxSequenceLength"] == 4_096
    assert materialization["requestedComputeDType"] == "float16"
    assert materialization["expectedTargetModuleCount"] == 196
    assert materialization["materializedTargetModuleCount"] == 196
    assert materialization["runtimeIsLoadedIn4Bit"] is True
    assert materialization["runtimeIsQuantized"] is True
    assert materialization["parameterPlacement"]["totalParameterCount"] == 310
    assert materialization["parameterPlacement"]["deviceTypeCounts"] == {
        "cuda": 310
    }
    assert materialization["forwardKernelProbe"] == {
        **{
            "schemaVersion": "lumen.runtime-forward-kernel-probe/1.0.0",
            "status": "passed",
            "fixedInputSHA256": train_sft._canonical_sha256(
                {
                    "inputIDs": [[1, 2, 3, 4]],
                    "attentionMask": [[1, 1, 1, 1]],
                    "useCache": False,
                }
            ),
            "batchSize": 1,
            "tokenCount": 4,
            "logitsShape": [1, 4, 151_936],
            "logitsDType": "float16",
            "logitsDeviceType": "cuda",
            "allFinite": True,
            "requiresGrad": False,
            "useCache": False,
        },
        "runtimeForwardKernelProbeSHA256": materialization[
            "forwardKernelProbe"
        ]["runtimeForwardKernelProbeSHA256"],
    }


def test_runtime_model_binding_rejects_partial_or_wrong_dtype_4bit_materialization(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model = _quantized_runtime_model(
        snapshot=snapshot,
        config_payload={"model_type": "qwen3", "max_position_embeddings": 40_960},
        generation_payload={"max_length": 40_960},
        max_seq_length=4_096,
    )
    modules = list(model.named_modules())
    model.named_modules = lambda: iter(modules[:-1])
    with pytest.raises(RuntimeError, match="exact fully quantized"):
        train_sft._runtime_4bit_materialization_evidence(model, _config())

    model = _quantized_runtime_model(
        snapshot=snapshot,
        config_payload={"model_type": "qwen3", "max_position_embeddings": 40_960},
        generation_payload={"max_length": 40_960},
        max_seq_length=4_096,
    )
    first_module = next(model.named_modules())[1]
    first_module.compute_dtype = "torch.bfloat16"
    with pytest.raises(RuntimeError, match="materialized CUDA NF4"):
        train_sft._runtime_4bit_materialization_evidence(model, _config())


def test_runtime_model_binding_rejects_unbound_model_max_sequence_length(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model = _quantized_runtime_model(
        snapshot=snapshot,
        config_payload={"model_type": "qwen3", "max_position_embeddings": 40_960},
        generation_payload={"max_length": 40_960},
        max_seq_length=8_192,
    )
    with pytest.raises(RuntimeError, match="max_seq_length drifted"):
        train_sft._runtime_4bit_materialization_evidence(model, _config())


def test_runtime_model_binding_rejects_cpu_parameter_offload(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model = _quantized_runtime_model(
        snapshot=snapshot,
        config_payload={"model_type": "qwen3", "max_position_embeddings": 40_960},
        generation_payload={"max_length": 40_960},
        max_seq_length=4_096,
    )
    parameters = list(model.named_parameters())
    parameters[-1][1].device.type = "cpu"
    model.named_parameters = lambda: iter(parameters)
    with pytest.raises(RuntimeError, match="non-CUDA parameter"):
        train_sft._runtime_4bit_materialization_evidence(model, _config())


@pytest.mark.parametrize(
    ("forward_dtype", "forward_device_type", "forward_finite"),
    (
        ("torch.bfloat16", "cuda", True),
        ("torch.float16", "cpu", True),
        ("torch.float16", "cuda", False),
    ),
)
def test_runtime_model_binding_rejects_invalid_cuda_forward_probe(
    tmp_path: Path,
    forward_dtype: str,
    forward_device_type: str,
    forward_finite: bool,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model = _quantized_runtime_model(
        snapshot=snapshot,
        config_payload={"model_type": "qwen3", "max_position_embeddings": 40_960},
        generation_payload={"max_length": 40_960},
        max_seq_length=4_096,
        forward_dtype=forward_dtype,
        forward_device_type=forward_device_type,
        forward_finite=forward_finite,
    )
    with pytest.raises(RuntimeError, match="forward probe logits"):
        train_sft._runtime_4bit_materialization_evidence(model, _config())


def test_all_real_unsloth_loaders_pass_the_explicit_controlled_dtype() -> None:
    for module in (train_sft, train_dpo, evaluate_adapter, export_gguf):
        tree = ast.parse(inspect.getsource(module))
        loader_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and (
                (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {"FastLanguageModel", "fast_language_model"}
                )
            )
        ]
        assert len(loader_calls) == 1
        dtype = next(
            (keyword.value for keyword in loader_calls[0].keywords if keyword.arg == "dtype"),
            None,
        )
        assert isinstance(dtype, ast.Call)
        assert isinstance(dtype.func, ast.Name)
        assert dtype.func.id == "_controlled_torch_dtype"


def test_runtime_model_binding_rejects_generation_config_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    verification = _snapshot_record(snapshot)
    monkeypatch.setattr(
        train_sft,
        "_verified_private_runtime_model_snapshot",
        lambda _cfg: (snapshot, verification),
    )

    class GenerationConfig:
        @classmethod
        def from_pretrained(cls, _path: str, *, local_files_only: bool):
            assert local_files_only is True
            return _SerializableConfig({"eos_token_id": 2})

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(GenerationConfig=GenerationConfig),
    )
    model = _quantized_runtime_model(
        snapshot=snapshot,
        config_payload={"model_type": "qwen3", "max_position_embeddings": 40_960},
        generation_payload={"eos_token_id": 99, "max_length": 40_960},
        max_seq_length=4_096,
    )

    with pytest.raises(RuntimeError, match="unapproved transformation"):
        train_sft._verify_runtime_model_binding(
            _config(),
            runtime_model=model,
            snapshot_path=snapshot,
            snapshot_verification=verification,
        )


def test_runtime_model_binding_tracks_requested_context_above_base_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    verification = _snapshot_record(snapshot)
    monkeypatch.setattr(
        train_sft,
        "_verified_private_runtime_model_snapshot",
        lambda _cfg: (snapshot, verification),
    )
    cfg = {**_config(), "max_seq_length": 65_536}
    pristine_generation = {"eos_token_id": 2, "max_length": 20}

    class GenerationConfig:
        @classmethod
        def from_pretrained(cls, _path: str, *, local_files_only: bool):
            assert local_files_only is True
            return _SerializableConfig(pristine_generation)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(GenerationConfig=GenerationConfig),
    )
    model = _quantized_runtime_model(
        snapshot=snapshot,
        config_payload={"model_type": "qwen3", "max_position_embeddings": 65_536},
        generation_payload={**pristine_generation, "max_length": 65_536},
        max_seq_length=65_536,
    )

    binding = train_sft._verify_runtime_model_binding(
        cfg,
        runtime_model=model,
        snapshot_path=snapshot,
        snapshot_verification=verification,
    )

    assert binding["allowedGenerationConfigTransformations"]["maxLength"][
        "sourceValue"
    ] == 65_536


def test_runtime_model_binding_rejects_pristine_unpatched_max_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    verification = _snapshot_record(snapshot)
    monkeypatch.setattr(
        train_sft,
        "_verified_private_runtime_model_snapshot",
        lambda _cfg: (snapshot, verification),
    )

    class GenerationConfig:
        @classmethod
        def from_pretrained(cls, _path: str, *, local_files_only: bool):
            assert local_files_only is True
            return _SerializableConfig({"eos_token_id": 2, "max_length": 20})

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(GenerationConfig=GenerationConfig),
    )
    model = _quantized_runtime_model(
        snapshot=snapshot,
        config_payload={"model_type": "qwen3", "max_position_embeddings": 40_960},
        generation_payload={"eos_token_id": 2, "max_length": 20},
        max_seq_length=4_096,
    )

    with pytest.raises(RuntimeError, match="unapproved transformation"):
        train_sft._verify_runtime_model_binding(
            _config(),
            runtime_model=model,
            snapshot_path=snapshot,
            snapshot_verification=verification,
        )


def test_peft_identity_normalization_removes_private_path_and_pins_revision() -> None:
    private_path = "/run/private/base-model"

    class RuntimeBaseConfig:
        def __init__(self) -> None:
            self._name_or_path = private_path
            self.name_or_path = private_path

        def to_dict(self) -> dict[str, str]:
            return {"_name_or_path": self._name_or_path}

    runtime_base_config = RuntimeBaseConfig()
    runtime_base_model = SimpleNamespace(
        config=runtime_base_config,
        name_or_path=private_path,
    )
    peft_configs = {
        "default": SimpleNamespace(
            base_model_name_or_path=private_path,
            revision=None,
        ),
        "reference": SimpleNamespace(
            base_model_name_or_path=private_path,
            revision="main",
        ),
    }
    cfg = _config()
    model = SimpleNamespace(
        peft_config=peft_configs,
        base_model=runtime_base_model,
        config=runtime_base_config,
        get_base_model=lambda: runtime_base_model,
    )

    evidence = train_sft._normalize_peft_base_model_identity(
        model,
        cfg,
    )

    assert evidence["baseModelID"] == cfg["baseModelID"]
    assert evidence["baseModelRevision"] == cfg["baseModelRevision"]
    assert evidence["privateRuntimePathPersisted"] is False
    assert all(
        item.base_model_name_or_path == cfg["baseModelID"]
        and item.revision == cfg["baseModelRevision"]
        for item in peft_configs.values()
    )
    assert model.config.to_dict()["_name_or_path"] == cfg["baseModelID"]
    assert model.base_model.name_or_path == cfg["baseModelID"]

    # These are the two independent PEFT 0.19.1 sources used for Trainer
    # checkpoint README generation. Neither may retain the private snapshot.
    checkpoint_model_card = (
        f"base_model: {model.config.to_dict()['_name_or_path']}\n"
        f"tag: base_model:adapter:{model.base_model.name_or_path}\n"
    )
    assert private_path not in checkpoint_model_card
    assert checkpoint_model_card.count(cfg["baseModelID"]) == 2

    with pytest.raises(RuntimeError, match="canonical and immutable"):
        train_sft._normalize_peft_base_model_identity(
            model,
            {
                **cfg,
                "baseModelID": private_path,
                "base_model_name": private_path,
            },
        )


def _publication_config(snapshot: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    payloads = {
        "tokenizer.json": b'{"version":"1.0"}',
        "tokenizer_config.json": b'{"chat_template":"pinned"}',
    }
    files = [
        {
            "path": filename,
            "sizeBytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "huggingFaceBlobID": "a" * 40,
        }
        for filename, payload in payloads.items()
    ]
    cfg = {
        **_config(),
        "baseModelTokenizerFiles": files,
        "baseModelTokenizerSnapshotPath": str(snapshot),
    }
    return cfg, payloads


def test_adapter_publication_uses_exact_base_bytes_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    cfg, payloads = _publication_config(snapshot)
    for filename, payload in payloads.items():
        (snapshot / filename).write_bytes(payload)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    verification = _snapshot_record(snapshot)
    monkeypatch.setattr(
        train_sft,
        "_verified_private_runtime_model_snapshot",
        lambda _cfg: (snapshot, verification),
    )

    binding = train_sft._publish_exact_base_tokenizer_subset(
        cfg,
        adapter_output_dir=adapter,
        snapshot_path=snapshot,
        snapshot_verification=verification,
    )

    assert {path.name for path in adapter.iterdir()} == set(payloads)
    assert all((adapter / name).read_bytes() == payload for name, payload in payloads.items())
    assert binding["transformation"] == "exact_byte_subset_no_derived_tokenizer"


def test_adapter_publication_rejects_derived_chat_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    cfg, payloads = _publication_config(snapshot)
    for filename, payload in payloads.items():
        (snapshot / filename).write_bytes(payload)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "chat_template.jinja").write_text("derived", encoding="utf-8")
    verification = _snapshot_record(snapshot)
    monkeypatch.setattr(
        train_sft,
        "_verified_private_runtime_model_snapshot",
        lambda _cfg: (snapshot, verification),
    )

    with pytest.raises(RuntimeError, match="unapproved derived tokenizer"):
        train_sft._publish_exact_base_tokenizer_subset(
            cfg,
            adapter_output_dir=adapter,
            snapshot_path=snapshot,
            snapshot_verification=verification,
        )


@pytest.mark.parametrize(
    "module",
    (train_sft, train_dpo, evaluate_adapter, export_gguf),
)
def test_all_runtime_model_loads_bind_explicit_snapshot_tokenizer(module: Any) -> None:
    source = inspect.getsource(module)
    loader_call = (
        "fast_language_model.from_pretrained("
        if module is train_sft
        else "FastLanguageModel.from_pretrained("
    )
    calls = source.count(loader_call)
    assert calls >= 1
    snapshot_name = (
        "runtime_snapshot_path"
        if module is train_sft
        else "runtime_tokenizer_snapshot_path"
    )
    assert source.count(f"tokenizer_name=str({snapshot_name})") == calls
    assert source.count(f"model_name=str({snapshot_name})") == calls
    assert source.count("local_files_only=True") >= calls


def test_trainers_never_save_mutated_runtime_tokenizer() -> None:
    assert "tokenizer.save_pretrained" not in inspect.getsource(train_sft.main)
    assert "tokenizer.save_pretrained" not in inspect.getsource(train_dpo.main)


def test_runtime_binding_smoke_is_self_hashed_and_stops_before_peft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    snapshot_verification = _snapshot_record(snapshot)
    model_binding = {
        "schemaVersion": train_sft.RUNTIME_MODEL_BINDING_SCHEMA,
        "runtimeModelBindingSHA256": "1" * 64,
    }
    tokenizer_binding = {
        "schemaVersion": train_sft.RUNTIME_TOKENIZER_BINDING_SCHEMA,
        "runtimeTokenizerBindingSHA256": "2" * 64,
    }
    monkeypatch.setattr(
        train_sft,
        "_load_verified_unsloth_runtime",
        lambda _cfg, *, fast_language_model: (
            object(),
            object(),
            snapshot,
            snapshot_verification,
            model_binding,
            tokenizer_binding,
        ),
    )
    expected_evidence = {
        "baseModelRuntimeSnapshotPath": str(snapshot),
        "runtimeModelBinding": model_binding,
        "runtimeTokenizerBinding": tokenizer_binding,
    }
    monkeypatch.setattr(
        train_sft,
        "_runtime_tokenizer_evidence",
        lambda *_args, **_kwargs: expected_evidence,
    )

    smoke = train_sft._run_runtime_binding_smoke(
        _config(),
        fast_language_model=object(),
    )

    unsigned = dict(smoke)
    declared = unsigned.pop("runtimeBindingSmokeSHA256")
    assert smoke["schemaVersion"] == "lumen.runtime-binding-smoke/1.0.0"
    assert smoke["runtimeModelBinding"] == model_binding
    assert declared == train_sft._canonical_sha256(unsigned)
    smoke_source = inspect.getsource(train_sft._run_runtime_binding_smoke)
    assert "get_peft_model" not in smoke_source
    assert "Trainer(" not in smoke_source
