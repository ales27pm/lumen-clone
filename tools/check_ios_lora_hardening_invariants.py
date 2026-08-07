#!/usr/bin/env python3
"""Focused hardening checks for Lumen's iOS LoRA runtime.

This is a static source scanner. It does not replace Xcode builds or real-device
smoke testing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_SERVICE = ROOT / "ios/Lumen/Services/LlamaService.swift"
XCODE_PROJECT = ROOT / "ios/Lumen.xcodeproj/project.pbxproj"
HARDENING_DOC = ROOT / "docs/HARDENING_IOS_LORA_ADAPTER_RUNTIME.md"

REQUIRED_DOC_TOKENS = (
    "Poison and Antidote",
    "single-adapter",
    "Jetsam",
    "exact version `1.2.0`",
    "adapterApplied",
    "lastAdapterFailureReason",
)

REQUIRED_SYMBOL_TOKENS = (
    "llama_adapter_lora_init(model, path)",
    "llama_adapter_lora_free(adapter)",
    "llama_set_adapter_lora(ctx, adapter, scale)",
    "llama_rm_adapter_lora(ctx, adapter)",
    "llama_clear_adapter_lora(ctx)",
    "LlamaLoraAdapter(model:path:)",
    "LlamaContext.apply(loraAdapter:scale:)",
    "LlamaContext.removeAllLoraAdapters()",
)


def read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing required file: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def extract_function(text: str, signature_fragment: str) -> str:
    start = text.find(signature_fragment)
    require(start >= 0, f"missing function fragment: {signature_fragment}")
    brace = text.find("{", start)
    require(brace >= 0, f"missing function body for: {signature_fragment}")
    depth = 0
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise AssertionError(f"unterminated function body for: {signature_fragment}")


def check_swift_llama_pin() -> None:
    project = read(XCODE_PROJECT)
    match = re.search(
        r'XCRemoteSwiftPackageReference "swift-llama-cpp"[\s\S]+?repositoryURL = "https://github\.com/pgorzelany/swift-llama-cpp\.git";[\s\S]+?requirement = \{[\s\S]+?kind = exactVersion;[\s\S]+?version = 1\.2\.0;',
        project,
    )
    require(match is not None, "swift-llama-cpp must be pinned to exactVersion 1.2.0")
    require("kind = branch" not in project, "swift-llama-cpp must not use branch-based resolution")


def check_single_adapter_activation() -> None:
    text = read(LLAMA_SERVICE)
    runtime_activation = extract_function(
        text,
        "func activateRoleAdapter(slot: LumenModelSlot, scale: Float, operationGeneration: UInt64) throws",
    )
    claim_index = runtime_activation.find("claimAdapterActivation(generation: operationGeneration)")
    clear_index = runtime_activation.find("clearAdaptersUnconditionally()")
    apply_index = runtime_activation.find("context.apply(loraAdapter: adapter, scale: scale)")
    require(
        claim_index >= 0 and clear_index >= 0 and apply_index >= 0 and claim_index < clear_index < apply_index,
        "runtime must claim newest activation ownership and clear adapters before applying selected LoRA",
    )

    app_activation = extract_function(text, "func activateRoleAdapter(slot: LumenModelSlot) async throws")
    require(
        "let activationGeneration = beginAdapterActivation()" in app_activation
        and "let activated = try await runtime.activateRoleAdapter(" in app_activation
        and "operationGeneration: activationGeneration" in app_activation,
        "app activation must register and pass newest-operation ownership to runtime activation",
    )
    require(app_activation.find("try await runtime.activateRoleAdapter") < app_activation.find("activeAdapterSlot = slot"), "activeAdapterSlot must be set only after successful apply")
    require("await runtime.clearAdapters(operationGeneration: activationGeneration)" in app_activation, "failure path must clear adapters under the same activation ownership")
    require("activationGeneration == adapterActivationGeneration" in app_activation, "app activation must reject stale post-await publication")
    require("activeAdapterSlot = nil" in app_activation, "failure path must reset activeAdapterSlot")
    require("lastAdapterFailureReason = error.localizedDescription" in app_activation, "failure path must preserve lastAdapterFailureReason")


def check_hardening_doc() -> None:
    doc = read(HARDENING_DOC)
    for token in REQUIRED_DOC_TOKENS + REQUIRED_SYMBOL_TOKENS:
        require(token in doc, f"hardening doc missing token: {token}")


def main() -> int:
    checks = (check_swift_llama_pin, check_single_adapter_activation, check_hardening_doc)
    failures: list[str] = []
    for check in checks:
        try:
            check()
            print(f"PASS {check.__name__}")
        except Exception as exc:
            failures.append(f"FAIL {check.__name__}: {exc}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("All iOS LoRA hardening invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
