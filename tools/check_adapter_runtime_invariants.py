#!/usr/bin/env python3
"""Static drift guard for Lumen's Qwen3 adapter runtime.

This script intentionally avoids importing app modules. It scans source files for
architecture invariants that must remain true after the Qwen3 shared-base + LoRA
adapter migration.

It is not a replacement for Xcode/device validation. It is a fast guard against
regressing to the slow five-full-GGUF runtime shape.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_FAMILY_SELECTION = ROOT / "ios/Lumen/Services/ModelFamilySelection.swift"
MODEL_FLEET = ROOT / "ios/Lumen/Services/ModelFleet.swift"
LLAMA_SERVICE = ROOT / "ios/Lumen/Services/LlamaService.swift"
SLOT_COORDINATOR = ROOT / "ios/Lumen/Services/SlotModelRuntimeCoordinator.swift"
MODELS_VIEW = ROOT / "ios/Lumen/Views/ModelsView.swift"
EXPORT_GGUF = ROOT / "tools/fine_tuning/unsloth/export_gguf.py"
DOC = ROOT / "docs/ADAPTER_RUNTIME_IMPROVE_LOOP.md"
TERMINAL_LOOP = ROOT / "tools/lumen_terminal_improve_loop.py"
TRAIN_SFT = ROOT / "tools/fine_tuning/unsloth/train_sft.py"
QWEN3_CONFIG_DIR = ROOT / "tools/fine_tuning/unsloth/configs_qwen3_bootstrap"

EXPECTED_ADAPTERS = {
    "lumen-cortex-lora.gguf",
    "lumen-executor-lora.gguf",
    "lumen-mouth-lora.gguf",
    "lumen-mimicry-lora.gguf",
    "lumen-rem-lora.gguf",
    "lumen-fleet-lora.gguf",
}

RELEASE_BAKE_DEFAULTS = {
    "lumen-cortex-release-bake-q4_k_m.gguf",
    "lumen-executor-release-bake-q4_k_m.gguf",
    "lumen-mouth-release-bake-q4_k_m.gguf",
    "lumen-mimicry-release-bake-q4_k_m.gguf",
    "lumen-rem-release-bake-q4_k_m.gguf",
}


def read(path: Path) -> str:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def fail(message: str) -> None:
    raise AssertionError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def section_after_marker(text: str, marker: str) -> str:
    index = text.find(marker)
    require(index >= 0, f"missing marker: {marker}")
    return text[index:]


def check_catalog() -> None:
    text = read(MODEL_FAMILY_SELECTION)
    qwen3 = section_after_marker(text, "static var qwen3BootstrapModels")

    require(
        qwen3.count("lumen-qwen3-fast-shared-q4_k_m.gguf") == 1,
        "Qwen3 default catalog must contain exactly one shared chat base filename.",
    )
    for adapter in sorted(EXPECTED_ADAPTERS):
        require(adapter in qwen3, f"Qwen3 default catalog missing adapter: {adapter}")
    for release_bake in sorted(RELEASE_BAKE_DEFAULTS):
        require(
            release_bake not in qwen3,
            f"Qwen3 default catalog must not include release-bake artifact: {release_bake}",
        )
    require(
        "lumen-fleet-lora.gguf" in qwen3 and "roleID" in qwen3,
        "Fleet adapter must be represented as a role adapter, not by abusing embedding slot metadata.",
    )


def check_fleet_resolver() -> None:
    text = read(MODEL_FLEET)
    require(
        "fallbackFamily: LumenModelFamily? = selectedFamily == .qwen3 ? nil : selectedFamily" in text,
        "Qwen3 fallback path must not label non-Qwen3 fallback assignments as Qwen3.",
    )
    require(
        "if text.contains(slotToken) { return (model, 2) }" in text,
        "Adapter ranking must prefer exact role adapter matches.",
    )
    require(
        "if slot == .cortex, text.contains(\"fleet\") { return (model, 1) }" in text,
        "Fleet adapter fallback must rank below exact cortex adapter matches.",
    )


def check_runtime() -> None:
    text = read(LLAMA_SERVICE)
    require("private actor AdapterChatRuntime" in text, "AdapterChatRuntime must remain actor-isolated.")
    require(
        "clearAdapters()" in text and "context.apply(loraAdapter: adapter, scale: scale)" in text,
        "Adapter activation must clear before applying LoRA.",
    )
    require("let isLast = index == lastIndex" in text, "Prompt decode must mark the final prompt token for logits.")
    require("outputTokenCount: nil" in text, "outputTokenCount must stay nil unless real token counts are threaded through.")
    require(
        "sanitized.split(whereSeparator: { $0.isWhitespace }).count" not in text,
        "Do not report whitespace word count as outputTokenCount.",
    )
    require(
        "adapterApplied" in text and "adapterSlot" in text,
        "Runtime trace metadata must include adapterApplied and adapterSlot.",
    )


def check_slot_coordinator() -> None:
    text = read(SLOT_COORDINATOR)
    adapter_section_match = re.search(r"private func ensureAdapterRuntimeReady[\s\S]+?private func ensureLegacyRuntimeReady", text)
    require(adapter_section_match is not None, "Missing ensureAdapterRuntimeReady/ensureLegacyRuntimeReady split.")
    adapter_section = adapter_section_match.group(0)
    require("unloadAllChat" not in adapter_section, "Qwen3 adapter slot switch must not call unloadAllChat().")
    require("unloadRoleAdapter(slot: slot)" in adapter_section, "Failed role adapter activation must unload the failed adapter handle.")


def check_models_view() -> None:
    text = read(MODELS_VIEW)
    require(
        "isAdapter: sm.modelRole == .roleAdapter" in text,
        "Downloaded model rows must know when a row is a role adapter.",
    )
    require(
        "catalog.role == .roleAdapter" in text,
        "Featured model cards must treat role adapters as non-activatable adapter artifacts.",
    )
    require(
        "stored.modelRole != .roleAdapter" in text,
        "Stored role adapters must not be activatable as chat/embedding models.",
    )


def check_export_policy() -> None:
    text = read(EXPORT_GGUF)
    require("--release-bake" in text, "export_gguf.py must require explicit --release-bake for merged GGUF export.")
    require("Skipped GGUF release bake by default" in text, "export_gguf.py must skip release-bake by default.")
    require(
        "merge_adapters_by_default" in text and "release_bake_enabled_by_default" in text,
        "export_gguf.py must validate adapter-first config flags.",
    )


def check_docs() -> None:
    text = read(DOC)
    require("Non-negotiable runtime invariant" in text, "Adapter runtime doctrine doc missing invariant section.")
    require("Default Qwen3 runtime must not" in text, "Adapter runtime doctrine doc must explicitly forbid five full role GGUFs.")
    require("Improve-loop drift checks" in text, "Adapter runtime doctrine doc must include improve-loop drift checks.")


def check_terminal_loop() -> None:
    text = read(TERMINAL_LOOP)
    require(
        "hf\", \"repos\", \"create\"" in text or '"repos", "create"' in text,
        "Terminal improve-loop must call 'hf repos create' (current Hugging Face CLI), not the legacy 'hf repo create'.",
    )
    require(
        "hf repo create" not in text,
        "Terminal improve-loop must not reference the legacy 'hf repo create' subcommand.",
    )
    require(
        "--base-model-id" in text and "--base" in text,
        "Terminal improve-loop convert stage must thread an explicit --base / --base-model-id to convert_lora_to_gguf.py.",
    )
    require(
        "_resolve_base_for_convert" in text,
        "Terminal improve-loop must validate base model resolution before invoking the LoRA→GGUF converter.",
    )
    require(
        "pipeline_state.json" in text and "--resume" in text and "--state-file" in text,
        "Terminal improve-loop must support resumable pipeline_state.json with --resume / --state-file.",
    )
    require(
        "configs_qwen3_bootstrap" in text,
        "Terminal improve-loop must default to the Qwen3 bootstrap config dir.",
    )
    require(
        "validate_qwen3_configs" in text,
        "Terminal improve-loop must include a strict Qwen3 config validator (no Qwen2.x base in the bootstrap dir).",
    )


def check_qwen3_configs_alignment() -> None:
    require(QWEN3_CONFIG_DIR.exists(), f"missing Qwen3 bootstrap config dir: {QWEN3_CONFIG_DIR.relative_to(ROOT)}")
    forbidden = ("qwen2", "qwen-2")
    for cfg_path in sorted(QWEN3_CONFIG_DIR.glob("*.json")):
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(f"invalid JSON in {cfg_path.relative_to(ROOT)}: {exc}")
        base = str(cfg.get("base_model_name", "")).lower()
        require(
            "qwen3" in base,
            f"{cfg_path.relative_to(ROOT)}: base_model_name must reference Qwen3 (got '{cfg.get('base_model_name')}').",
        )
        for token in forbidden:
            require(
                token not in base,
                f"{cfg_path.relative_to(ROOT)}: base_model_name still references a pre-Qwen3 family ('{cfg.get('base_model_name')}').",
            )
        require(
            cfg.get("merge_adapters_by_default", False) is False,
            f"{cfg_path.relative_to(ROOT)}: merge_adapters_by_default must remain false (adapter-first).",
        )
        require(
            cfg.get("release_bake_enabled_by_default", False) is False,
            f"{cfg_path.relative_to(ROOT)}: release_bake_enabled_by_default must remain false (adapter-first).",
        )


def check_train_sft_reproducibility() -> None:
    text = read(TRAIN_SFT)
    require("--seed" in text, "train_sft.py must accept --seed for reproducibility.")
    require(
        "--resume-from-checkpoint" in text,
        "train_sft.py must accept --resume-from-checkpoint for resumable training.",
    )
    require(
        "--assistant-only-loss" in text or "assistant_only_loss" in text,
        "train_sft.py must support assistant-only loss for instruction-tuning.",
    )
    require(
        "train_manifest.json" in text,
        "train_sft.py must write a train_manifest.json with reproducibility metadata.",
    )


def main() -> int:
    checks = [
        check_catalog,
        check_fleet_resolver,
        check_runtime,
        check_slot_coordinator,
        check_models_view,
        check_export_policy,
        check_docs,
        check_terminal_loop,
        check_qwen3_configs_alignment,
        check_train_sft_reproducibility,
    ]
    failures: list[str] = []
    for check in checks:
        try:
            check()
            print(f"PASS {check.__name__}")
        except Exception as exc:  # noqa: BLE001 - command-line checker should report every failure
            failures.append(f"FAIL {check.__name__}: {exc}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("All Qwen3 adapter runtime invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
