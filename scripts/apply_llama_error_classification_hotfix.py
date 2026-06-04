from pathlib import Path

path = Path("ios/Lumen/Services/LlamaService.swift")
text = path.read_text()

text = text.replace(
    "private enum LlamaErrorCode: String {",
    "private enum LlamaErrorCode: String, Sendable {",
    1,
)
text = text.replace(
    "    private func classifyError(_ error: Error) -> LlamaErrorCode {",
    "    private nonisolated func classifyError(_ error: Error) -> LlamaErrorCode {",
    1,
)

path.write_text(text)
