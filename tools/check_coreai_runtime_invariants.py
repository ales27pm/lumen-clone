#!/usr/bin/env python3
"""Static Core AI wiring/identity gate; does not establish Apple/device execution."""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def check(root: pathlib.Path) -> list[str]:
    failures: list[str] = []

    def read(relative: str) -> str:
        path = root / relative
        if not path.is_file():
            failures.append(f"missing required source: {relative}")
            return ""
        return path.read_text(encoding="utf-8")

    project = read("ios/Lumen.xcodeproj/project.pbxproj")
    targets = re.findall(r"IPHONEOS_DEPLOYMENT_TARGET\s*=\s*([^;]+);", project)
    if not targets or any(value.strip() != "27.2" for value in targets):
        failures.append("Core AI app and test deployment targets must be iOS 27.2")
    for repository, revision, product in (
        ("apple/coreai-models", "9db747612f972001279f02adb04d3ef982310d0e", "CoreAILM"),
        ("huggingface/swift-transformers", "eed7264ac5e4ec5dfa6165c6e5c5577364344fe4", "Tokenizers"),
    ):
        package = re.search(
            r"(?P<id>[A-F0-9]{24})[^\n]*=\s*\{\s*isa = XCRemoteSwiftPackageReference;"
            r"\s*repositoryURL = \"https://github.com/" + re.escape(repository)
            + r"\";\s*requirement = \{kind = revision; revision = " + revision + r"; \};",
            project,
        )
        if not package:
            failures.append(f"missing immutable Core AI dependency pin: {repository}")
            continue
        dependency = re.search(
            r"(?P<id>[A-F0-9]{24})[^\n]*=\s*\{\s*isa = XCSwiftPackageProductDependency;"
            r"\s*package = " + package["id"] + r"[^;]*;\s*productName = " + product + r";",
            project,
        )
        if not dependency or not re.search(r"productRef = " + dependency["id"], project):
            failures.append(f"Core AI product must be linked: {product}")

    service = read("ios/Lumen/Services/LLM/CoreAI/CoreAIGemmaService.swift")
    for required in (
        "!targetEnvironment(simulator)", "CoreAIRunner(bundle:",
        "AutoTokenizer.from(modelFolder:", "CoreAIGemmaModelStore.configuredURL()",
        "withTaskCancellationHandler", "suspendForCoreAI()", "engine.reset(to: 0)",
        "DiskWriteBudget.shared.beginGeneration()", "ResourceBudgetGate.allowsForegroundModelLoad",
    ):
        if required not in service:
            failures.append(f"missing Core AI lifecycle boundary: {required}")
    code = "\n".join(line for line in service.splitlines() if not line.lstrip().startswith("//"))
    if re.search(r"URLSession|URLRequest|from\s*\(\s*pretrained\s*:|\.loadTokenizer\s*\(", code):
        failures.append("Core AI tokenizer/model loading must remain entirely local")
    executor = read("ios/Lumen/Assistant/StructuredAgentKernelExecutor.swift")
    if 'if generation.preventsToolExecution {' not in executor:
        failures.append("Core AI runtime errors must terminate before semantic action repairs")
    chat = read("ios/Lumen/Views/ChatView.swift")
    if re.search(r"AppLlamaService\.shared\.stream\s*\(", chat):
        failures.append("Plain chat must use the production kernel rather than a direct llama stream")
    contracts = read("ios/Lumen/Assistant/AgentKernelContracts.swift")
    if "let allowsToolExecution: Bool" not in contracts:
        failures.append("The kernel must distinguish tools disabled from an unrestricted tool list")
    classifier = read("ios/Lumen/Services/IntentClassification/IntentClassifierService.swift")
    preference = classifier.find("if CoreAIGemmaModelStore.isEnabled {")
    learned_call = classifier.find("await BundledIntentClassifier.shared.classify")
    if preference < 0 or learned_call < preference:
        failures.append("Gemma mode must bypass the auxiliary learned intent classifier")
    return failures


if __name__ == "__main__":
    errors = check(ROOT)
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        raise SystemExit(1)
    print("Core AI runtime static invariants passed (not compile or device evidence).")
