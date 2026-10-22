from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]


def test_both_model_settings_surfaces_use_owned_cancelled_operations():
    for name in ("CoreAIGemmaSettingsSection.swift", "CoreAITextEmbeddingSettingsSection.swift"):
        source = (ROOT / "ios/Lumen/Views" / name).read_text()
        assert "ModelSettingsOperationOwner()" in source
        assert ".onDisappear { operationOwner.deactivate() }" in source
        assert ".onChange(of: scenePhase)" in source
        assert "operationOwner.cancel()" in source
        assert "operationOwner.start(" in source
        assert "guard operationOwner.isActive else { return }" in source
        assert not re.search(r"\bTask\s*\{", source)


def test_memory_and_background_cancellation_precede_coalescing_or_suspension():
    memory = (ROOT / "ios/Lumen/System/MemoryPressureMonitor.swift").read_text()
    warning = re.sub(r"//[^\n]*", "", memory.split("func handleWarning(", 1)[1])
    cancellation = warning.index("AppCancellationBus.shared.cancel(.modelMaintenance)")
    assert cancellation < warning.index("if let lastWarningAt")
    assert cancellation < warning.index("await")
    assert cancellation < warning.index("async let")
    scene = (ROOT / "ios/Lumen/System/SceneTransitionCoordinator.swift").read_text()
    for function in ("handleScenePhaseChange", "handleDidEnterBackground"):
        body = scene.split(f"func {function}(", 1)[1].split("\n    }", 1)[0]
        assert body.index("AppCancellationBus.shared.cancel(.modelMaintenance)") < body.index("guard ")
