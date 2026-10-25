#!/usr/bin/env bash
set -euo pipefail

# Apple executor only. No Core AI SDK, model assets, native tool IO, signing,
# network, or third-party Swift packages. App source remains unchanged.
if [[ "$(uname -s)" != "Darwin" ]]; then
  printf 'Self-prediction Swift tests require macOS and Swift 6.2+.\n' >&2
  exit 2
fi
command -v swift >/dev/null 2>&1 || { printf 'swift is unavailable.\n' >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { printf 'python3 is unavailable.\n' >&2; exit 2; }
swift --version | python3 -c 'import re,sys; s=sys.stdin.read(); m=re.search(r"Swift version (\d+)\.(\d+)",s); sys.exit(0 if m and tuple(map(int,m.groups())) >= (6,2) else 2)' || {
  printf 'Swift 6.2+ is required for the app source isolation syntax.\n' >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lumen-self-prediction.XXXXXX")"
trap 'rm -rf "$PACKAGE_DIR"' EXIT
mkdir -p "$PACKAGE_DIR/Sources/Lumen" "$PACKAGE_DIR/Tests/LumenTests" "$PACKAGE_DIR/fixtures"
export TMPDIR="$(cd "$PACKAGE_DIR/fixtures" && pwd -P)/"

for source in \
  ios/Lumen/Services/SelfModel/SelfPredictionLedger.swift \
  ios/Lumen/Services/SelfModel/SelfPredictionFeedback.swift \
  ios/Lumen/Assistant/HostSelfModelContext.swift \
  ios/Lumen/Assistant/SelfModelContextProvider.swift \
  ios/Lumen/Assistant/PromptGroundingSection.swift \
  ios/Lumen/Services/PromptBudget.swift \
  ios/Lumen/Services/PromptContextBuilder.swift \
  ios/Lumen/Models/MemoryContextItem.swift \
  ios/Lumen/Services/AgentJSONKeyIntegrity.swift \
  ios/Lumen/Models/AgentJSONValue.swift \
  ios/Lumen/Models/ToolDefinition.swift \
  ios/Lumen/Permissions/PermissionDomain.swift \
  ios/Lumen/Permissions/PermissionState.swift \
  ios/Lumen/Tools/ToolID.swift \
  ios/Lumen/Tools/ToolCategory.swift \
  ios/Lumen/Tools/ToolInvocation.swift \
  ios/Lumen/Tools/ToolResult.swift \
  ios/Lumen/Tools/SecureToolDefinition.swift \
  ios/Lumen/Tools/LocalTool.swift \
  ios/Lumen/Tools/ToolExecutionContext.swift \
  ios/Lumen/Tools/ToolApprovalPolicy.swift \
  ios/Lumen/Tools/SafeToolOutputLimiter.swift \
  ios/Lumen/Tools/ToolSchemaBridge.swift \
  ios/Lumen/Tools/ToolRegistry.swift \
  ios/Lumen/Views/SelfPredictionSettingsSection.swift \
  ios/LumenTests/Support/SelfPredictionHarnessSupport.swift; do
  cp "$ROOT_DIR/$source" "$PACKAGE_DIR/Sources/Lumen/"
done
for test in SelfPredictionLedgerTests.swift SelfPredictionFeedbackTests.swift \
  SelfPredictionPerturbationTests.swift SelfPredictionFeedbackContextTests.swift \
  ExecutorDirectAnswerPromptTests.swift \
  AgentTurnPredictionTests.swift \
  AgentPredictionKeyIntegrityTests.swift ToolPredictionIntegrationTests.swift; do
  cp "$ROOT_DIR/ios/LumenTests/$test" "$PACKAGE_DIR/Tests/LumenTests/"
done

# Extract exact self-contained declarations from larger app files. Unique
# anchors fail closed on source drift; implementation bodies are never doubled.
python3 - "$ROOT_DIR" "$PACKAGE_DIR/Sources/Lumen" <<'PYTHON'
from pathlib import Path
import json
import re
import sys

root, target = Path(sys.argv[1]), Path(sys.argv[2])

def source(path):
    return (root / path).read_text(encoding="utf-8")

def between(text, start, end):
    assert text.count(start) == 1, f"Ambiguous source start: {start}"
    assert text.count(end) == 1, f"Ambiguous source end: {end}"
    lo, hi = text.index(start), text.index(end)
    assert lo < hi, "Reversed source anchors"
    return text[lo:hi]

def write(name, body):
    (target / name).write_text(
        "import Foundation\nimport CoreFoundation\n"
        "// Exact declarations extracted by run_self_prediction_tests.sh.\n" + body + "\n",
        encoding="utf-8",
    )

parser = source("ios/Lumen/Services/AgentService.swift")
write("AgentTurnParser.swift", between(parser, "// MARK: - Structured turn model", "// MARK: - Streaming JSON string extractor"))

executor = source("ios/Lumen/Assistant/StructuredAgentKernelExecutor.swift")
schemas = re.findall(r'^    nonisolated static let (structuredAgent\w*Schema) = #"(.*?)"#$', executor, re.M)
assert len(schemas) == 3, "Expected exactly three runtime schemas"
for _, schema in schemas:
    json.loads(schema)
schema_declarations = "\n".join(f'    nonisolated static let {name} = #"{schema}"#' for name, schema in schemas)
runtime_prompts = between(executor, "    nonisolated static let executorRuntimeSystemPromptContract =", "    private let kernel:")
provenance = between(executor, "    /// Compare the typed original action, before argument string coercion or dedupe.", "    static func buildSystemPrompt(")
write("StructuredPredictionContract.swift", "@MainActor struct StructuredAgentKernelExecutor {\n" + schema_declarations + "\n" + runtime_prompts + "\n" + provenance + "}\n")

routes = source("ios/Lumen/Services/ToolExecutor.swift")
approval = between(routes, "nonisolated enum ToolExecutionApproval", "nonisolated enum ToolRouteGuard")
normalization = between(routes, "    static func canonicalToolID(", "    static func canExecuteTool(")
write("ToolRouteGuard.swift", approval + "nonisolated enum ToolRouteGuard {\n" + normalization + "}\n")

permissions = source("ios/Lumen/Services/PermissionsCenter.swift")
write("PermissionKind.swift", between(permissions, "nonisolated enum PermissionKind:", "@MainActor\n@Observable\nfinal class PermissionsCenter"))
trigger = source("ios/Lumen/Models/Trigger.swift")
write("TriggerScheduleContract.swift", between(trigger, "nonisolated enum TriggerScheduleContract", "@Model\nfinal class Trigger"))
manifest = source("ios/Lumen/Services/AgentGrounding/AgentBehaviorManifest.swift")
argument_start = "public struct RuntimeToolArgument:"
assert manifest.count(argument_start) == 1
# RuntimeToolArgument is the last declaration in this file.
argument = manifest[manifest.index(argument_start):]
assert argument.rstrip().endswith("}"), "Incomplete runtime argument declaration"
write("RuntimeToolArgument.swift", argument)

# Compile the real host/context/assembler logic without the application kernel,
# SwiftData snapshot builder, or attachment IO. Shape/decoder declarations and
# the revision validator below are exact source; unused adapters fail closed in
# SelfPredictionHarnessSupport rather than manufacturing successful IO/trace.
snapshot = source("ios/Lumen/Services/SelfModel/SelfModelSnapshot.swift")
shape = between(snapshot, "nonisolated struct SelfModelSnapshot:", "nonisolated enum SelfModelSnapshotBuilder")
revision = between(snapshot, "    static func validatedSourceRevision(", "    private static func inferredSlot(")
decoder_anchor = "extension SelfModelSnapshot.AppInfo {"
assert snapshot.count(decoder_anchor) == 1
decoders = snapshot[snapshot.index(decoder_anchor):]
write("SelfModelSnapshotShape.swift", shape + "\nnonisolated enum SelfModelSnapshotBuilder {\n" + revision + "}\n" + decoders)
budget = source("ios/Lumen/Assistant/ContextBudgetAllocator.swift")
write("ContextBudgetShapes.swift", between(budget, "struct ContextBudgetTokenSections:", "enum ContextBudgetAllocator"))
chat = source("ios/Lumen/Models/ChatMessage.swift")
write("MessageRole.swift", between(chat, "enum MessageRole:", "enum ToolStatus:"))
attachments = source("ios/Lumen/Models/ChatAttachment.swift")
write("ChatAttachmentShape.swift", between(attachments, "nonisolated struct ChatAttachment:", "nonisolated enum AttachmentResolver"))
print("Exact source declarations and three JSON schemas extracted; Swift execution follows.")
PYTHON

cat > "$PACKAGE_DIR/Package.swift" <<'SWIFT'
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LumenSelfPredictionRegression",
    platforms: [.macOS(.v15)],
    products: [.library(name: "Lumen", targets: ["Lumen"])],
    targets: [
        .target(name: "Lumen"),
        .testTarget(name: "LumenTests", dependencies: ["Lumen"])
    ],
    swiftLanguageModes: [.v5]
)
SWIFT

swift test --package-path "$PACKAGE_DIR" -c release \
  -Xswiftc -DSELF_PREDICTION_PORTABLE_TESTS \
  -Xswiftc -enable-upcoming-feature -Xswiftc MemberImportVisibility \
  --filter '(SelfPredictionLedger|SelfPredictionFeedback|SelfPredictionPerturbation|SelfPredictionFeedbackContext|ExecutorDirectAnswerPrompt|AgentTurnPrediction|AgentPredictionKeyIntegrity|ToolPredictionIntegration)Tests'
