from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    from .export_gguf import (
        _verified_release_bake_lineage,
        load_config as load_export_config,
    )
    from .train_sft import _require_unsloth_before_transformers, _seed_everything
except ImportError:
    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    from export_gguf import (  # type: ignore
        _verified_release_bake_lineage,
        load_config as load_export_config,
    )
    from train_sft import (  # type: ignore
        _require_unsloth_before_transformers,
        _seed_everything,
    )


EVALUATION_RUN_SCHEMA_VERSION = "lumen.adapter-evaluation-run/1.0.0"
CANDIDATE_OUTPUT_SCHEMA_VERSION = "lumen.adapter-eval-candidate/1.0.0"
SUPPORTED_AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")
JSON_OUTPUT_AGENTS = frozenset({"cortex", "executor"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOOL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.]*$")
_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a finalized per-agent LoRA adapter over its frozen evaluation "
            "suite and produce promotion-compatible candidate outputs and scores."
        )
    )
    parser.add_argument("--config", required=True, help="Finalized adapter config JSON.")
    parser.add_argument(
        "--adapter-dir",
        help="Override config adapter_output_dir (the finalized manifest must still bind it).",
    )
    parser.add_argument(
        "--finalized-variant-manifest",
        help=(
            "Override the finalized manifest path. Defaults to config "
            "finalized_variant_manifest or <output_dir>/finalized_variant_manifest.json."
        ),
    )
    parser.add_argument(
        "--eval-jsonl",
        help="Override the frozen eval JSONL. Defaults to <dataset_dir>/eval.jsonl.",
    )
    parser.add_argument(
        "--behavior-manifest",
        default="generated/agent_manifest/AgentBehaviorManifest.json",
        help="Behavior manifest supplying the exact tool and fleet contracts used by scoring.",
    )
    parser.add_argument(
        "--output-dir",
        help="Evaluation output directory. Defaults to <finalized-manifest-dir>/evaluation.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Deterministic prefix length for a smoke run. Omit to run the full frozen suite.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Bounded generation budget per example (default: 1024; maximum: 4096).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing evaluator output files.",
    )
    return parser.parse_args(argv)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _load_evaluation_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    crawler_root = repo_root / "tools" / "lumen_manifest_crawler"
    if crawler_root.is_dir() and str(crawler_root) not in sys.path:
        sys.path.insert(0, str(crawler_root))
    from lumen_manifest_crawler.dataset import adapter_evaluation

    return adapter_evaluation


def _validate_prompt_messages(value: Any, *, path: Path, line_number: int) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}:{line_number} messages must be a non-empty list")
    expected_role = "user"
    for index, message in enumerate(value):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(
                f"{path}:{line_number} messages[{index}] must contain only role and content"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in _MESSAGE_ROLES or not isinstance(content, str) or not content.strip():
            raise ValueError(f"{path}:{line_number} messages[{index}] is invalid")
        if index == 0 and role == "system":
            continue
        if role != expected_role:
            raise ValueError(
                f"{path}:{line_number} prompt roles must alternate after an optional leading system message"
            )
        expected_role = "assistant" if expected_role == "user" else "user"
    if value[-1].get("role") != "user":
        raise ValueError(f"{path}:{line_number} prompt must end with a user message")


def load_evaluation_records(
    path: Path,
    *,
    agent: str,
    evaluation_module: ModuleType,
) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen evaluation JSONL not found: {path}")
    records: list[dict[str, Any]] = []
    seen_eval_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            raw_record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(raw_record, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        _validate_prompt_messages(
            raw_record.get("messages"),
            path=path,
            line_number=line_number,
        )
        record = evaluation_module.upgrade_evaluation_record(raw_record)
        eval_id = record.get("evalID")
        record_agent = str((record.get("metadata") or {}).get("agent") or "").strip().lower()
        if not isinstance(eval_id, str) or not eval_id:
            raise ValueError(f"{path}:{line_number} has no stable evalID")
        if eval_id in seen_eval_ids:
            raise ValueError(f"{path}:{line_number} duplicates evalID {eval_id}")
        if record_agent != agent:
            raise ValueError(
                f"{path}:{line_number} belongs to agent {record_agent or '<missing>'}, expected {agent}"
            )
        metrics = record.get("metrics")
        if not isinstance(metrics, list) or not metrics or any(
            not isinstance(metric, dict) for metric in metrics
        ):
            raise ValueError(f"{path}:{line_number} has an invalid metric contract")
        seen_eval_ids.add(eval_id)
        records.append(record)
    if not records:
        raise ValueError(f"Frozen evaluation JSONL is empty: {path}")
    return records, evaluation_module.canonical_sha256(records)


def load_behavior_contract(path: Path) -> tuple[dict[str, Any], set[str], str]:
    manifest = _load_json_object(path, label="Behavior manifest")
    raw_tools = manifest.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ValueError("Behavior manifest tools must be a non-empty list")
    tool_contracts: dict[str, Any] = {}
    for index, tool in enumerate(raw_tools):
        if not isinstance(tool, dict):
            raise ValueError(f"Behavior manifest tools[{index}] must be an object")
        tool_id = tool.get("id")
        arguments = tool.get("arguments")
        if (
            not isinstance(tool_id, str)
            or _TOOL_ID_PATTERN.fullmatch(tool_id) is None
            or tool_id in tool_contracts
            or not isinstance(arguments, list)
        ):
            raise ValueError(f"Behavior manifest tools[{index}] has an invalid or duplicate ID")
        argument_names: set[str] = set()
        for argument_index, argument in enumerate(arguments):
            if not isinstance(argument, dict):
                raise ValueError(
                    f"Behavior manifest {tool_id} arguments[{argument_index}] must be an object"
                )
            name = argument.get("name")
            declared_type = argument.get("type")
            required = argument.get("required")
            allowed_values = argument.get("allowedValues")
            if (
                not isinstance(name, str)
                or not name
                or name in argument_names
                or not isinstance(declared_type, str)
                or not declared_type
                or type(required) is not bool
                or (allowed_values is not None and not isinstance(allowed_values, list))
            ):
                raise ValueError(
                    f"Behavior manifest {tool_id} arguments[{argument_index}] is invalid"
                )
            argument_names.add(name)
        tool_contracts[tool_id] = tool

    fleet = manifest.get("fleet")
    raw_slots = fleet.get("slots") if isinstance(fleet, dict) else None
    if not isinstance(raw_slots, list) or not raw_slots:
        raise ValueError("Behavior manifest fleet.slots must be a non-empty list")
    allowed_slots: set[str] = set()
    for index, slot in enumerate(raw_slots):
        slot_id = slot.get("id") if isinstance(slot, dict) else None
        if (
            not isinstance(slot_id, str)
            or not slot_id
            or slot_id in allowed_slots
        ):
            raise ValueError(f"Behavior manifest fleet.slots[{index}] is invalid")
        allowed_slots.add(slot_id)
    return tool_contracts, allowed_slots, _canonical_sha256(manifest)


def validate_scoring_contracts(
    records: Sequence[Mapping[str, Any]],
    *,
    tool_contracts: Mapping[str, Any],
    allowed_slots: set[str],
) -> None:
    referenced_tools: set[str] = set()
    referenced_slots: set[str] = set()
    for record in records:
        for metric in record["metrics"]:
            expected_tool = metric.get("expectedToolID")
            if isinstance(expected_tool, str):
                referenced_tools.add(expected_tool)
            raw_allowed_tools = metric.get("allowedToolIDs")
            if isinstance(raw_allowed_tools, list):
                referenced_tools.update(
                    value for value in raw_allowed_tools if isinstance(value, str)
                )
            candidates: list[Mapping[str, Any]] = [metric]
            if isinstance(metric.get("contract"), Mapping):
                candidates.append(metric["contract"])
            for candidate in candidates:
                for key in (
                    "expectedSlot",
                    "expectedAggregationOwnerSlotID",
                ):
                    value = candidate.get(key)
                    if isinstance(value, str):
                        referenced_slots.add(value)
                for key in (
                    "allowedSlots",
                    "knownSlotIDs",
                    "expectedDelegatedSlotIDs",
                ):
                    value = candidate.get(key)
                    if isinstance(value, list):
                        referenced_slots.update(
                            item for item in value if isinstance(item, str)
                        )
    missing_tools = sorted(referenced_tools - set(tool_contracts))
    missing_slots = sorted(referenced_slots - allowed_slots)
    if missing_tools:
        raise ValueError(
            "Behavior manifest is missing evaluation tool contracts: "
            + ", ".join(missing_tools)
        )
    if missing_slots:
        raise ValueError(
            "Behavior manifest is missing evaluation fleet slots: "
            + ", ".join(missing_slots)
        )


def load_finalized_manifest(
    path: Path,
    *,
    cfg: Mapping[str, Any],
    evaluation_sha256: str,
    evaluation_module: ModuleType,
) -> dict[str, Any]:
    finalized = _load_json_object(path, label="Finalized variant manifest")
    expected_sha = finalized.get("variantManifestSHA256")
    unsigned = dict(finalized)
    unsigned.pop("variantManifestSHA256", None)
    if (
        not isinstance(expected_sha, str)
        or _SHA256_PATTERN.fullmatch(expected_sha) is None
        or evaluation_module.canonical_sha256(unsigned) != expected_sha
    ):
        raise ValueError("Finalized variant manifest integrity check failed")
    agent = str(cfg.get("agent") or "").strip().lower()
    variant = cfg.get("variant")
    source_sha = cfg.get("variantManifestSHA256")
    artifact = finalized.get("artifact")
    if (
        finalized.get("agent") != agent
        or finalized.get("variant") != variant
        or finalized.get("sourceVariantManifestSHA256") != source_sha
        or finalized.get("frozenEvaluationSHA256") != evaluation_sha256
        or not isinstance(artifact, dict)
        or artifact.get("status") != "trained"
        or _SHA256_PATTERN.fullmatch(str(artifact.get("adapterSHA256") or "")) is None
    ):
        raise ValueError(
            "Finalized variant manifest is not bound to the selected config, adapter, and frozen evaluation"
        )
    validator = getattr(evaluation_module, "_valid_variant_manifest", None)
    if validator is None or not validator(
        finalized,
        agent=agent,
        expected_variant=variant,
        require_trained_artifact=True,
    ):
        raise ValueError("Finalized variant manifest failed the controlled lineage contract")
    return finalized


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError) as exc:
        raise RuntimeError("Unable to resolve the adapter model device") from exc


def _move_model_inputs(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    return value


def generate_completion(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_seq_length: int,
    max_new_tokens: int,
    torch_module: ModuleType | Any | None = None,
) -> tuple[str, int, int]:
    try:
        encoded = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Tokenizer could not render the frozen evaluation prompt") from exc
    if isinstance(encoded, Mapping):
        model_inputs = dict(encoded)
    else:
        model_inputs = {"input_ids": encoded}
    input_ids = model_inputs.get("input_ids")
    shape = getattr(input_ids, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[0]) != 1:
        raise RuntimeError("Tokenizer must return one rank-two input_ids tensor")
    input_token_count = int(shape[-1])
    remaining_tokens = max_seq_length - input_token_count
    generation_budget = min(max_new_tokens, remaining_tokens)
    if input_token_count <= 0 or generation_budget <= 0:
        raise RuntimeError(
            "Frozen evaluation prompt consumes the configured maximum sequence length"
        )
    device = _model_device(model)
    moved_inputs = {
        key: _move_model_inputs(value, device)
        for key, value in model_inputs.items()
    }
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": generation_budget,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
    }
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None or eos_token_id is not None:
        generation_kwargs["pad_token_id"] = (
            pad_token_id if pad_token_id is not None else eos_token_id
        )
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    with torch_module.inference_mode():
        generated = model.generate(**moved_inputs, **generation_kwargs)
    sequences = getattr(generated, "sequences", generated)
    output_shape = getattr(sequences, "shape", None)
    if output_shape is None or len(output_shape) != 2 or int(output_shape[0]) != 1:
        raise RuntimeError("Model generation must return one rank-two token sequence")
    generated_token_count = max(0, int(output_shape[-1]) - input_token_count)
    generated_ids = sequences[0][input_token_count:]
    completion = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(completion, str):
        raise RuntimeError("Tokenizer decode did not return text")
    return completion.strip(), input_token_count, generated_token_count


def normalize_candidate_output(
    agent: str,
    completion: str,
    *,
    evaluation_module: ModuleType,
) -> tuple[Any, str, str | None]:
    if agent not in JSON_OUTPUT_AGENTS:
        return completion, "text" if completion else "empty_text", (
            None if completion else "empty_candidate_output"
        )
    parsed, json_error = evaluation_module._parse_candidate_json(completion)
    if json_error is not None:
        return completion, "invalid_json", json_error
    if not isinstance(parsed, dict):
        return completion, "invalid_json", "json_output_must_be_an_object"
    return parsed, "json_object", None


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    agent: str,
    model: Any,
    tokenizer: Any,
    max_seq_length: int,
    max_new_tokens: int,
    evaluation_module: ModuleType,
    torch_module: ModuleType | Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    outputs: dict[str, Any] = {}
    output_rows: list[dict[str, Any]] = []
    format_failure_count = 0
    for index, record in enumerate(records, start=1):
        completion, input_tokens, generated_tokens = generate_completion(
            model,
            tokenizer,
            record["messages"],
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            torch_module=torch_module,
        )
        output, output_kind, format_error = normalize_candidate_output(
            agent,
            completion,
            evaluation_module=evaluation_module,
        )
        if format_error is not None:
            format_failure_count += 1
        eval_id = str(record["evalID"])
        outputs[eval_id] = output
        row = {
            "schemaVersion": CANDIDATE_OUTPUT_SCHEMA_VERSION,
            "evalID": eval_id,
            "agent": agent,
            "output": output,
            "outputKind": output_kind,
            "formatError": format_error,
            "inputTokenCount": input_tokens,
            "generatedTokenCount": generated_tokens,
        }
        row["candidateRecordSHA256"] = _canonical_sha256(row)
        output_rows.append(row)
        print(
            f"[{agent}] evaluated {index}/{len(records)} {eval_id} ({output_kind})",
            flush=True,
        )
    return outputs, output_rows, format_failure_count


def load_inference_model(
    cfg: Mapping[str, Any],
    *,
    adapter_dir: Path,
) -> tuple[Any, Any]:
    if cfg.get("load_in_4bit") is not True:
        raise ValueError("Evaluation requires the controlled load_in_4bit=true config")
    _require_unsloth_before_transformers()
    try:
        from unsloth import FastLanguageModel  # type: ignore
        from peft import PeftModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation requires the pinned Unsloth, PEFT, and PyTorch environment"
        ) from exc
    # Seed only after Unsloth has patched Transformers, but before model loading.
    _seed_everything(int(cfg["seed"]))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model_name"],
        revision=cfg["baseModelRevision"],
        max_seq_length=int(cfg["max_seq_length"]),
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(
        model,
        str(adapter_dir),
        is_trainable=False,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    return model, tokenizer


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def load_candidate_outputs(path: Path, *, agent: str) -> dict[str, Any]:
    """Load evaluator output without trusting duplicate or mutated JSONL rows."""

    if not path.is_file():
        raise FileNotFoundError(f"Candidate output JSONL not found: {path}")
    outputs: dict[str, Any] = {}
    expected_keys = {
        "schemaVersion",
        "evalID",
        "agent",
        "output",
        "outputKind",
        "formatError",
        "inputTokenCount",
        "generatedTokenCount",
        "candidateRecordSHA256",
    }
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ValueError(f"{path}:{line_number} has an invalid candidate record")
        expected_sha256 = row.get("candidateRecordSHA256")
        unsigned = dict(row)
        unsigned.pop("candidateRecordSHA256", None)
        eval_id = row.get("evalID")
        if (
            row.get("schemaVersion") != CANDIDATE_OUTPUT_SCHEMA_VERSION
            or row.get("agent") != agent
            or not isinstance(eval_id, str)
            or not eval_id
            or eval_id in outputs
            or not isinstance(expected_sha256, str)
            or _SHA256_PATTERN.fullmatch(expected_sha256) is None
            or _canonical_sha256(unsigned) != expected_sha256
            or row.get("outputKind")
            not in {"json_object", "invalid_json", "text", "empty_text"}
            or type(row.get("inputTokenCount")) is not int
            or row["inputTokenCount"] <= 0
            or type(row.get("generatedTokenCount")) is not int
            or row["generatedTokenCount"] < 0
        ):
            raise ValueError(f"{path}:{line_number} failed candidate lineage validation")
        outputs[eval_id] = row["output"]
    if not outputs:
        raise ValueError(f"Candidate output JSONL is empty: {path}")
    return outputs


def _check_output_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Evaluator output paths must be distinct")
    for path in resolved:
        if path.exists() and (path.is_dir() or path.is_symlink()):
            raise ValueError(f"Evaluator output path is not a regular file: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Evaluator output already exists (pass --overwrite to replace it): {path}"
            )


def run(args: argparse.Namespace) -> int:
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if args.max_new_tokens <= 0 or args.max_new_tokens > 4096:
        raise ValueError("--max-new-tokens must be between 1 and 4096")

    config_path = Path(args.config).resolve()
    cfg = load_export_config(config_path)
    agent = str(cfg["agent"]).strip().lower()
    if agent not in SUPPORTED_AGENTS:
        raise ValueError(f"Unsupported evaluation agent: {agent}")
    adapter_dir = Path(args.adapter_dir or cfg["adapter_output_dir"]).resolve()
    finalized_path = Path(
        args.finalized_variant_manifest
        or cfg.get("finalized_variant_manifest")
        or (Path(str(cfg["output_dir"])) / "finalized_variant_manifest.json")
    ).resolve()
    eval_path = Path(
        args.eval_jsonl or (Path(str(cfg["dataset_dir"])) / "eval.jsonl")
    ).resolve()
    behavior_manifest_path = Path(args.behavior_manifest).resolve()
    output_dir = Path(args.output_dir or (finalized_path.parent / "evaluation")).resolve()
    candidate_path = output_dir / "candidate_outputs.jsonl"
    report_path = output_dir / "evaluation_report.json"
    run_manifest_path = output_dir / "evaluation_run_manifest.json"
    _check_output_paths(
        (candidate_path, report_path, run_manifest_path),
        overwrite=bool(args.overwrite),
    )

    evaluation_module = _load_evaluation_module()
    all_records, evaluation_sha256 = load_evaluation_records(
        eval_path,
        agent=agent,
        evaluation_module=evaluation_module,
    )
    tool_contracts, allowed_slots, behavior_manifest_sha256 = load_behavior_contract(
        behavior_manifest_path
    )
    expected_behavior_file_sha256 = cfg.get("behaviorManifestFileSHA256")
    if (
        expected_behavior_file_sha256 is not None
        and _file_sha256(behavior_manifest_path) != expected_behavior_file_sha256
    ):
        raise ValueError("Behavior manifest file drifted from the finalized evaluation config")
    validate_scoring_contracts(
        all_records,
        tool_contracts=tool_contracts,
        allowed_slots=allowed_slots,
    )
    finalized = load_finalized_manifest(
        finalized_path,
        cfg=cfg,
        evaluation_sha256=evaluation_sha256,
        evaluation_module=evaluation_module,
    )

    cfg["adapter_output_dir"] = str(adapter_dir)
    cfg["finalized_variant_manifest"] = str(finalized_path)
    lineage = _verified_release_bake_lineage(cfg)
    artifact_sha256 = lineage["adapterSHA256"]
    if artifact_sha256 != finalized["artifact"]["adapterSHA256"]:
        raise ValueError("Verified adapter artifact digest does not match finalized lineage")

    selected_records = (
        all_records[: args.max_examples]
        if args.max_examples is not None
        else all_records
    )
    complete_evaluation = len(selected_records) == len(all_records)
    model, tokenizer = load_inference_model(cfg, adapter_dir=adapter_dir)
    outputs, output_rows, format_failure_count = evaluate_records(
        selected_records,
        agent=agent,
        model=model,
        tokenizer=tokenizer,
        max_seq_length=int(cfg["max_seq_length"]),
        max_new_tokens=int(args.max_new_tokens),
        evaluation_module=evaluation_module,
    )
    controlled_lineage_builder = getattr(
        evaluation_module,
        "_variant_controlled_lineage",
        None,
    )
    if controlled_lineage_builder is None:
        raise RuntimeError("Evaluation module lacks controlled-lineage scoring support")
    report = evaluation_module.score_evaluation_suite(
        all_records,
        outputs,
        tool_contracts=tool_contracts,
        allowed_slots=allowed_slots,
        agent=agent,
        variant=cfg["variant"],
        controlled_lineage=controlled_lineage_builder(finalized),
        variant_manifest=finalized,
        artifact_sha256=artifact_sha256,
    )
    if report.get("promotionEvidenceBound") is not True:
        raise RuntimeError("Evaluation report could not bind to finalized adapter lineage")

    candidate_bytes = _jsonl_bytes(output_rows)
    report_bytes = _json_bytes(report)
    candidate_file_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    report_file_sha256 = hashlib.sha256(report_bytes).hexdigest()
    quality_gate_passed = (
        complete_evaluation
        and format_failure_count == 0
        and report.get("evidenceComplete") is True
        and report.get("criticalFailureCount") == 0
        and report.get("passedCaseCount") == report.get("caseCount")
    )
    status = (
        "format_failed"
        if format_failure_count
        else "quality_gate_passed"
        if quality_gate_passed
        else "quality_gate_failed"
        if complete_evaluation
        else "smoke_complete"
    )
    run_manifest: dict[str, Any] = {
        "schemaVersion": EVALUATION_RUN_SCHEMA_VERSION,
        "status": status,
        "evaluatorCodePath": str(Path(__file__).resolve()),
        "evaluatorCodeSHA256": _file_sha256(Path(__file__).resolve()),
        "agent": agent,
        "variant": cfg["variant"],
        "configPath": str(config_path),
        "configSHA256": _file_sha256(config_path),
        "adapterDirectory": str(adapter_dir),
        "adapterSHA256": artifact_sha256,
        "finalizedVariantManifestPath": str(finalized_path),
        "finalizedVariantManifestSHA256": finalized["variantManifestSHA256"],
        "evaluationJSONLPath": str(eval_path),
        "evaluationSHA256": evaluation_sha256,
        "behaviorManifestPath": str(behavior_manifest_path),
        "behaviorManifestSHA256": behavior_manifest_sha256,
        "candidateOutputsPath": str(candidate_path),
        "candidateOutputsFileSHA256": candidate_file_sha256,
        "candidateOutputsSHA256": report["candidateOutputsSHA256"],
        "evaluationReportPath": str(report_path),
        "evaluationReportFileSHA256": report_file_sha256,
        "evaluationReportSHA256": report["reportSHA256"],
        "fullCaseCount": len(all_records),
        "generatedCaseCount": len(selected_records),
        "completeEvaluation": complete_evaluation,
        "formatFailureCount": format_failure_count,
        "criticalFailureCount": report["criticalFailureCount"],
        "qualityGatePassed": quality_gate_passed,
        "generation": {
            "doSample": False,
            "numBeams": 1,
            "thinkingEnabled": False,
            "maxNewTokens": int(args.max_new_tokens),
            "maxSequenceLength": int(cfg["max_seq_length"]),
            "seed": int(cfg["seed"]),
        },
    }
    run_manifest["runManifestSHA256"] = _canonical_sha256(run_manifest)

    _atomic_write_bytes(candidate_path, candidate_bytes)
    _atomic_write_bytes(report_path, report_bytes)
    _atomic_write_bytes(run_manifest_path, _json_bytes(run_manifest))
    print(f"Wrote candidate outputs: {candidate_path}")
    print(f"Wrote scored report: {report_path}")
    print(f"Wrote evaluation run manifest: {run_manifest_path}")
    print(
        f"Evaluation status={status} weightedScore={report['weightedScore']} "
        f"criticalFailures={report['criticalFailureCount']} "
        f"formatFailures={format_failure_count}"
    )
    if format_failure_count:
        return 2
    if complete_evaluation and not quality_gate_passed:
        return 3
    return 0


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
