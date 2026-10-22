#!/usr/bin/env bash
set -euo pipefail

# Execute the real Foundation/Darwin implementation without any model assets,
# app signing, Core AI framework or third-party Swift package dependencies.
if [[ "$(uname -s)" != "Darwin" ]]; then
  printf 'Core AI package IO tests require macOS and a Swift 6.2+ toolchain.\n' >&2
  exit 2
fi
command -v swift >/dev/null 2>&1 || { printf 'swift is unavailable.\n' >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lumen-coreai-package-io.XXXXXX")"
trap 'rm -rf "$PACKAGE_DIR"' EXIT
mkdir -p "$PACKAGE_DIR/Sources/Lumen" "$PACKAGE_DIR/Tests/LumenTests"
cp "$ROOT_DIR/ios/Lumen/Services/LLM/CoreAILocalPackageIO.swift" "$PACKAGE_DIR/Sources/Lumen/"
cp "$ROOT_DIR/ios/LumenTests/CoreAILocalPackageIOTests.swift" "$PACKAGE_DIR/Tests/LumenTests/"
cat > "$PACKAGE_DIR/Package.swift" <<'SWIFT'
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CoreAILocalPackageIORegression",
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
  -Xswiftc -DCOREAI_PACKAGE_IO_MEMORY_TESTS \
  --filter CoreAILocalPackageIOTests
