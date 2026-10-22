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
        assert ".onChange(of: operationOwner.configurationRevision)" in source
        assert "guard operationOwner.isActive else { return }" in source
        assert not re.search(r"\bTask\s*\{", source)


def test_memory_and_background_cancellation_precede_coalescing_or_suspension():
    memory = (ROOT / "ios/Lumen/System/MemoryPressureMonitor.swift").read_text()
    warning = re.sub(r"//[^\n]*", "", memory.split("func handleWarning(", 1)[1])
    cancellation = warning.index("AppCancellationBus.shared.cancel(.modelMaintenance)")
    assert cancellation < warning.index("if let lastWarningAt")
    assert cancellation < warning.index("await")
    assert cancellation < warning.index("async let")
    load_cancellation = warning.index("AppCancellationBus.shared.cancel(.modelLoad)")
    assert load_cancellation < warning.index("if let lastWarningAt")
    assert load_cancellation < warning.index("await")
    loader = (ROOT / "ios/Lumen/Services/ModelLoader.swift").read_text()
    cancel_loads = loader.split("static func cancelActiveLoads()", 1)[1].split("\n    }", 1)[0]
    assert cancel_loads.index("AppCancellationBus.shared.cancel(.modelLoad)") < cancel_loads.index("Task {")
    scene = (ROOT / "ios/Lumen/System/SceneTransitionCoordinator.swift").read_text()
    for function in ("handleScenePhaseChange", "handleDidEnterBackground"):
        body = scene.split(f"func {function}(", 1)[1].split("\n    }", 1)[0]
        assert body.index("AppCancellationBus.shared.cancel(.modelMaintenance)") < body.index("guard ")


def test_embedding_metadata_discards_a_superseded_snapshot_after_await():
    source = (ROOT / "ios/Lumen/Views/CoreAITextEmbeddingSettingsSection.swift").read_text()
    refresh = source.split("private func refreshMetadata()", 1)[1].split("\n    }", 1)[0]
    assert refresh.index("let requestID = metadataRefreshID") < refresh.index("await ")
    assert refresh.index("await ") < refresh.index("requestID == metadataRefreshID")
    assert refresh.index("requestID == metadataRefreshID") < refresh.index("isConfigured = configured")
    assert refresh.index("let configurationRevision = operationOwner.configurationRevision") < refresh.index("await ")
    assert refresh.index("await ") < refresh.index("configurationRevision == operationOwner.configurationRevision")


def test_model_actions_preserve_the_source_rebuild_cursor_until_identity_is_checked():
    source = (ROOT / "ios/Lumen/Views/CoreAITextEmbeddingSettingsSection.swift").read_text()
    perform = source.split("private func perform(", 1)[1].split("private func advanceRebuild(", 1)[0]
    assert "rebuild = nil" not in perform
    assert "await rebuild.advance(matching: metadata)" in source


def test_parent_context_metadata_uses_a_versioned_view_task_for_success_and_failure():
    source = (ROOT / "ios/Lumen/Views/SettingsView.swift").read_text()
    assert "@MainActor\nstruct SettingsView: View" in source
    assert "CoreAIGemmaSettingsSection()" in source
    assert ".task(id: contextRequest) { await refreshGemmaContext(matching: contextRequest) }" in source
    request = source.split("private var gemmaContextRequest:", 1)[1].split("\n    }", 1)[0]
    assert "modelConfigurationState.revision" in request
    assert "usesGemma: usesGemma" in request
    refresh = source.split("private func refreshGemmaContext(", 1)[1].split("private var currentVoiceName:", 1)[0]
    assert refresh.count("!Task.isCancelled, request == gemmaContextRequest") == 3
    success = refresh.split("let length = try await", 1)[1].split("} catch", 1)[0]
    failure = refresh.split("} catch", 1)[1]
    assert success.index("request == gemmaContextRequest") < success.index("gemmaContextLength = length")
    assert failure.index("request == gemmaContextRequest") < failure.index("gemmaContextIssue =")


def test_models_view_refreshes_from_shared_revision_without_a_child_callback_task():
    source = (ROOT / "ios/Lumen/Views/ModelsView.swift").read_text()
    assert "@MainActor\nstruct ModelsView: View" in source
    assert ".task(id: modelConfigurationState.revision) { await refreshLoaded() }" in source
    assert "CoreAITextEmbeddingSettingsSection()" in source
    assert "CoreAITextEmbeddingSettingsSection(onConfigurationChanged:" not in source
    refresh = source.split("private func refreshLoaded()", 1)[1].split("private func load(", 1)[0]
    assert refresh.index("let configurationRevision = modelConfigurationState.revision") < refresh.index("await ")
    assert refresh.index("await runtimeController.refreshLoadedPaths()") < refresh.index("configurationRevision == modelConfigurationState.revision")
    for guard in (
        "gemmaSelected == usesGemma",
        "coreAIEmbeddingsSelected == usesCoreAIEmbeddings",
        "chatModelID == appState.activeChatModelID",
        "embeddingModelID == appState.activeEmbeddingModelID",
    ):
        assert refresh.index(guard) < refresh.index("isGemmaLoaded = gemmaLoaded")
    assert refresh.index("configurationRevision == modelConfigurationState.revision") < refresh.index("isCoreAIEmbeddingLoaded = coreAIEmbeddingLoaded")


def test_chat_home_revalidates_provisioning_after_shared_revision_and_rejects_stale_results():
    source = (ROOT / "ios/Lumen/Views/ChatHomeView.swift").read_text()
    assert "@MainActor\nstruct ChatHomeView: View" in source
    assert "let validationKey = provisioningValidationKey" in source
    assert ".task(id: validationKey)" in source
    assert "await refreshProvisioningPresentation(matching: validationKey)" in source
    key = source.split("private var provisioningValidationKey:", 1)[1].split("private func refreshProvisioningPresentation(", 1)[0]
    assert "modelConfigurationState.revision.uuidString" in key
    refresh = source.split("private func refreshProvisioningPresentation(", 1)[1].split("\nstruct EmptyChatPlaceholder:", 1)[0]
    assert refresh.count("!Task.isCancelled, validationKey == provisioningValidationKey") == 2
    validation = refresh.split("let valid = await", 1)[1]
    for guard in (
        "validationKey == provisioningValidationKey, !usesGemma",
        "chatModelID == appState.activeChatModelID",
        "embeddingModelID == appState.activeEmbeddingModelID",
    ):
        assert validation.index(guard) < validation.index("ModelProvisioningReceipt.invalidate()")
        assert validation.index(guard) < validation.index("showingOnboarding = !valid")
