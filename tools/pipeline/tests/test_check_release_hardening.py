import tools.check_release_hardening as release_hardening


def _states_for_directive(directive: str) -> list[bool]:
    return release_hardening.debug_stack_for_lines(
        [
            directive,
            "UnavailableGGUFNativeBridge()",
            "#else",
            "UnavailableGGUFNativeBridge()",
            "#endif",
        ]
    )


def test_parenthesized_debug_condition_is_debug_only():
    assert _states_for_directive("#if (DEBUG)") == [True, True, False, False, False]


def test_parenthesized_negated_debug_condition_is_release_branch():
    assert _states_for_directive("#if !(DEBUG)") == [False, False, True, True, False]


def test_spaced_parenthesized_negated_debug_condition_is_release_branch():
    assert _states_for_directive("#if ! ( DEBUG )") == [False, False, True, True, False]


def test_defined_debug_condition_is_debug_only():
    assert _states_for_directive("#if defined(DEBUG)") == [True, True, False, False, False]


def test_negated_defined_debug_condition_is_release_branch():
    assert _states_for_directive("#if !defined(DEBUG)") == [False, False, True, True, False]


def test_spaced_negated_defined_debug_condition_is_release_branch():
    assert _states_for_directive("#if ! defined ( DEBUG )") == [False, False, True, True, False]


def test_nested_negated_defined_debug_condition_is_release_branch():
    assert _states_for_directive("#if !(defined(DEBUG))") == [False, False, True, True, False]


def test_lossy_rag_memory_product_paths_are_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen"
    source_root.mkdir(parents=True)
    (source_root / "ProductMemory.swift").write_text(
        """
        func unsafe(context: ModelContext) async {
            _ = await MemoryStore.recall(query: "x", context: context)
            _ = MemoryStore.exportJSON(context: context)
            _ = RAGStore.counts(context: context)
            _ = RAGStore.chunks(for: .note, context: context)
            _ = await RAGStore.indexImportedFiles(context: context)
            _ = await RAGStore.indexPhotos(monthsBack: 6, context: context)
            _ = await RAGStore.indexNote(title: "n", body: "b", context: context)
            _ = await RAGEngine().retrieve(query: "x", limit: 1, context: context)
            _ = await MemoryEngine().search(query: "x", limit: 1, context: context)
            let chunks = (try? context.fetch(FetchDescriptor<RAGChunk>())) ?? []
            _ = chunks
            return "[]"
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [source_root])

    violations = release_hardening.scan_source()

    assert any("lossy memory recall wrapper" in item for item in violations)
    assert any("lossy memory export wrapper" in item for item in violations)
    assert any("lossy RAG counts wrapper" in item for item in violations)
    assert any("lossy RAG chunks wrapper" in item for item in violations)
    assert any("lossy RAG imported-file index wrapper" in item for item in violations)
    assert any("lossy RAG photo index wrapper" in item for item in violations)
    assert any("lossy RAG note index wrapper" in item for item in violations)
    assert any("lossy RAG retrieve wrapper" in item for item in violations)
    assert any("lossy memory engine search wrapper" in item for item in violations)
    assert any("lossy SwiftData fetch empty fallback" in item for item in violations)
    assert any("lossy empty JSON export fallback" in item for item in violations)


def test_rag_memory_diagnostic_product_paths_are_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen"
    source_root.mkdir(parents=True)
    (source_root / "ProductMemory.swift").write_text(
        """
        func safe(context: ModelContext) async {
            _ = await MemoryStore.recallWithDiagnostics(query: "x", context: context)
            _ = MemoryStore.exportJSONWithDiagnostics(context: context)
            _ = RAGStore.countsWithDiagnostics(context: context)
            _ = RAGStore.chunksWithDiagnostics(for: .note, context: context)
            _ = await RAGStore.indexImportedFilesWithDiagnostics(context: context)
            _ = await RAGStore.indexPhotosWithDiagnostics(monthsBack: 6, context: context)
            _ = await RAGStore.indexNoteWithDiagnostics(title: "n", body: "b", context: context)
            _ = await RAGEngine().retrieveWithDiagnostics(query: "x", limit: 1, context: context)
            _ = await MemoryEngine().searchWithDiagnostics(query: "x", limit: 1, context: context)
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [source_root])

    assert release_hardening.scan_source() == []


def test_imported_file_empty_fallback_product_paths_are_rejected(tmp_path, monkeypatch):
    services_root = tmp_path / "ios" / "Lumen" / "Services"
    tools_root = services_root / "Tools"
    tools_root.mkdir(parents=True)
    services_root.mkdir(parents=True, exist_ok=True)
    (services_root / "RAGStore.swift").write_text(
        """
        func unsafe() {
            let files = FileStore.importedFiles()
            _ = files
        }
        """,
        encoding="utf-8",
    )
    (tools_root / "FilesTools.swift").write_text(
        """
        func unsafe(fm: FileManager, dir: URL) {
            let files = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
            _ = files
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy imported files wrapper" in item for item in violations)
    assert any("lossy imported files directory empty fallback" in item for item in violations)


def test_imported_file_diagnostic_product_paths_are_allowed(tmp_path, monkeypatch):
    services_root = tmp_path / "ios" / "Lumen" / "Services"
    tools_root = services_root / "Tools"
    tools_root.mkdir(parents=True)
    services_root.mkdir(parents=True, exist_ok=True)
    (services_root / "RAGStore.swift").write_text(
        """
        func safe() {
            let imports = FileStore.importedFilesWithDiagnostics()
            _ = imports.diagnostic
        }
        """,
        encoding="utf-8",
    )
    (tools_root / "FilesTools.swift").write_text(
        """
        func safe(fm: FileManager) {
            let imports = FileStore.importedFilesWithDiagnostics(fileManager: fm)
            _ = imports.files
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_imported_file_write_wrapper_product_paths_are_rejected(tmp_path, monkeypatch):
    views_root = tmp_path / "ios" / "Lumen" / "Views"
    views_root.mkdir(parents=True)
    (views_root / "ChatView.swift").write_text(
        """
        func unsafe(url: URL) {
            guard let imported = FileStore.importFile(from: url) else { return }
            _ = imported
        }
        """,
        encoding="utf-8",
    )
    (views_root / "SourcesView.swift").write_text(
        """
        func unsafe(url: URL) {
            if let imported = FileStore.importFile(from: url) {
                _ = imported
            }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert sum("lossy imported-file write wrapper" in item for item in violations) == 2


def test_imported_file_write_diagnostics_product_paths_are_allowed(tmp_path, monkeypatch):
    views_root = tmp_path / "ios" / "Lumen" / "Views"
    views_root.mkdir(parents=True)
    (views_root / "ChatView.swift").write_text(
        """
        func safe(url: URL) {
            let imported = FileStore.importFileWithDiagnostics(from: url)
            _ = imported.diagnostic
        }
        """,
        encoding="utf-8",
    )
    (views_root / "SourcesView.swift").write_text(
        """
        func safe(url: URL) {
            let imported = FileStore.importFileWithDiagnostics(from: url)
            _ = imported.url
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_attachment_size_zero_fallback_is_rejected(tmp_path, monkeypatch):
    models_root = tmp_path / "ios" / "Lumen" / "Models"
    models_root.mkdir(parents=True)
    (models_root / "ChatAttachment.swift").write_text(
        """
        func unsafe(url: URL) {
            let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
            let size = (attrs?[.size] as? NSNumber)?.intValue ?? 0
            _ = size
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy attachment size zero fallback" in item for item in violations)


def test_attachment_metadata_guard_is_allowed(tmp_path, monkeypatch):
    models_root = tmp_path / "ios" / "Lumen" / "Models"
    models_root.mkdir(parents=True)
    (models_root / "ChatAttachment.swift").write_text(
        """
        func safe(url: URL) -> Int? {
            guard
                let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
                let size = (attrs[.size] as? NSNumber)?.intValue
            else {
                return nil
            }
            return size
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_attachment_extraction_empty_fallbacks_are_rejected(tmp_path, monkeypatch):
    models_root = tmp_path / "ios" / "Lumen" / "Models"
    services_root = tmp_path / "ios" / "Lumen" / "Services"
    models_root.mkdir(parents=True)
    services_root.mkdir(parents=True)
    (models_root / "ChatAttachment.swift").write_text(
        """
        func unsafe(url: URL, data: Data) -> String {
            guard let pdf = PDFDocument(url: url) else { return "" }
            guard let bytes = try? Data(contentsOf: url) else { return "" }
            let attr = try? NSAttributedString(data: data, options: [:], documentAttributes: nil)
            _ = (pdf, bytes, attr)
            return rawExtractText(attachment)
        }
        """,
        encoding="utf-8",
    )
    (services_root / "PromptBudget.swift").write_text(
        """
        func unsafe(attachment: ChatAttachment) {
            let raw = AttachmentResolver.rawExtractText(attachment)
            _ = raw
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy attachment PDF empty fallback" in item for item in violations)
    assert any("lossy attachment data read fallback" in item for item in violations)
    assert any("lossy attachment attributed decode fallback" in item for item in violations)
    assert any("lossy raw attachment extraction wrapper" in item for item in violations)


def test_attachment_extraction_diagnostic_path_is_allowed(tmp_path, monkeypatch):
    models_root = tmp_path / "ios" / "Lumen" / "Models"
    services_root = tmp_path / "ios" / "Lumen" / "Services"
    models_root.mkdir(parents=True)
    services_root.mkdir(parents=True)
    (models_root / "ChatAttachment.swift").write_text(
        """
        func safe(attachment: ChatAttachment) {
            let result = AttachmentResolver.extractTextWithDiagnostics(attachment)
            _ = result.diagnostic
        }
        """,
        encoding="utf-8",
    )
    (services_root / "PromptBudget.swift").write_text(
        """
        func safe(attachment: ChatAttachment) {
            let extraction = AttachmentResolver.extractTextWithDiagnostics(attachment)
            _ = extraction.mode
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_developer_trace_raw_persistence_and_attachment_context_are_rejected(tmp_path, monkeypatch):
    service_root = tmp_path / "ios" / "Lumen" / "Services" / "LLM"
    views_root = tmp_path / "ios" / "Lumen" / "Views"
    service_root.mkdir(parents=True)
    views_root.mkdir(parents=True)
    (service_root / "DeveloperTrace.swift").write_text(
        """
        func unsafe(trace: DeveloperTrace, encoder: JSONEncoder) throws {
            _ = try encoder.encode(trace)
        }
        """,
        encoding="utf-8",
    )
    (views_root / "ChatView.swift").write_text(
        """
        func unsafe(item: MessageItem, attachment: ChatAttachment) {
            _ = TraceContextItem(
                title: attachment.name,
                content: item.content,
                source: attachment.path
            )
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("raw developer trace encoding" in item for item in violations)
    assert any("raw attachment name in trace context" in item for item in violations)
    assert any("raw attachment path in trace context" in item for item in violations)
    assert any("raw history content in trace context" in item for item in violations)


def test_developer_trace_redacted_persistence_and_attachment_context_are_allowed(tmp_path, monkeypatch):
    service_root = tmp_path / "ios" / "Lumen" / "Services" / "LLM"
    views_root = tmp_path / "ios" / "Lumen" / "Views"
    service_root.mkdir(parents=True)
    views_root.mkdir(parents=True)
    (service_root / "DeveloperTrace.swift").write_text(
        """
        func safe(trace: DeveloperTrace, encoder: JSONEncoder) throws {
            _ = try encoder.encode(trace.redactedForPersistence())
        }
        """,
        encoding="utf-8",
    )
    (views_root / "ChatView.swift").write_text(
        """
        func safe(item: MessageItem, attachment: ChatAttachment) {
            _ = TraceContextItem(
                title: "Attachment",
                content: "history_chars=12;sha256=abcd",
                source: "attachment_path_sha256=abcd"
            )
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_file_tool_generic_read_failures_are_rejected(tmp_path, monkeypatch):
    tools_root = tmp_path / "ios" / "Lumen" / "Services" / "Tools"
    tools_root.mkdir(parents=True)
    (tools_root / "FilesTools.swift").write_text(
        """
        func unsafe(url: URL) {
            guard let data = try? Data(contentsOf: url) else { return "Couldn't read secret.txt." }
            guard let pdf = PDFDocument(url: url) else { return "Couldn't open PDF." }
            _ = data
            _ = pdf
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy file tool data read fallback" in item for item in violations)
    assert any("generic file tool read failure" in item for item in violations)
    assert any("generic file tool open failure" in item for item in violations)


def test_file_tool_diagnostic_read_path_is_allowed(tmp_path, monkeypatch):
    tools_root = tmp_path / "ios" / "Lumen" / "Services" / "Tools"
    tools_root.mkdir(parents=True)
    (tools_root / "FilesTools.swift").write_text(
        """
        func safe(url: URL) {
            let result = readMatchedFileWithDiagnostics(url: url)
            _ = diagnosticText(result.diagnostic)
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_slot_model_integrity_boolean_filter_is_rejected(tmp_path, monkeypatch):
    services_root = tmp_path / "ios" / "Lumen" / "Services"
    services_root.mkdir(parents=True)
    (services_root / "SlotModelRuntimeCoordinator.swift").write_text(
        """
        func unsafe(candidates: [StoredModel]) {
            let pool = candidates.filter { ModelFileIntegrity.validateInstalledFile($0) }
            _ = pool
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy installed-model integrity filter" in item for item in violations)


def test_slot_model_integrity_diagnostic_filter_is_allowed(tmp_path, monkeypatch):
    services_root = tmp_path / "ios" / "Lumen" / "Services"
    services_root.mkdir(parents=True)
    (services_root / "SlotModelRuntimeCoordinator.swift").write_text(
        """
        func safe(candidates: [StoredModel]) {
            var pool: [StoredModel] = []
            for candidate in candidates {
                switch ModelFileIntegrity.validateInstalledFileWithDiagnostics(candidate) {
                case .success:
                    pool.append(candidate)
                case .failure(let failure):
                    logger.error("diagnostic=\\(failure.diagnosticCode, privacy: .public)")
                }
            }
            _ = pool
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_memory_remember_try_fallback_is_rejected(tmp_path, monkeypatch):
    views_root = tmp_path / "ios" / "Lumen" / "Views"
    views_root.mkdir(parents=True)
    (views_root / "ChatView.swift").write_text(
        """
        func unsafe(context: ModelContext) async {
            try? await MemoryStore.remember("remember this", kind: .fact, source: "chat", context: context)
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy memory remember try fallback" in item for item in violations)


def test_memory_remember_diagnostic_path_is_allowed(tmp_path, monkeypatch):
    views_root = tmp_path / "ios" / "Lumen" / "Views"
    views_root.mkdir(parents=True)
    (views_root / "ChatView.swift").write_text(
        """
        func safe(context: ModelContext) async {
            let result = await MemoryStore.rememberWithDiagnostics("remember this", kind: .fact, source: "chat", context: context)
            _ = result.diagnostic
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_memory_tool_raw_save_content_and_error_are_rejected(tmp_path, monkeypatch):
    tools_root = tmp_path / "ios" / "Lumen" / "Services" / "Tools"
    tools_root.mkdir(parents=True)
    (tools_root / "MemoryTools.swift").write_text(
        """
        func save(content: String, context: ModelContext) async -> String {
            let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
            do {
                try await MemoryStore.remember(trimmed, kind: .fact, source: "agent", context: context)
                return "Saved: \\(trimmed)"
            } catch {
                return "Failed to save memory: \\(error.localizedDescription)"
            }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("throwing memory tool save path" in item for item in violations)
    assert any("raw memory save content echo" in item for item in violations)
    assert any("raw memory save localized error" in item for item in violations)


def test_memory_tool_diagnostic_save_message_is_allowed(tmp_path, monkeypatch):
    tools_root = tmp_path / "ios" / "Lumen" / "Services" / "Tools"
    tools_root.mkdir(parents=True)
    (tools_root / "MemoryTools.swift").write_text(
        """
        func save(content: String, context: ModelContext) async -> String {
            let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
            let result = await MemoryStore.rememberWithDiagnostics(trimmed, kind: .fact, source: "agent", context: context)
            return saveMessage(from: result)
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_calendar_tool_raw_localized_error_is_rejected(tmp_path, monkeypatch):
    tools_root = tmp_path / "ios" / "Lumen" / "Services" / "Tools"
    tools_root.mkdir(parents=True)
    (tools_root / "CalendarTools.swift").write_text(
        """
        enum CalendarTools {
            static func createReminder(title: String) async -> String {
                do {
                    return "Added reminder"
                } catch {
                    return "Couldn't add reminder: \\(error.localizedDescription)"
                }
            }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("raw calendar tool localized error" in item for item in violations)


def test_calendar_tool_sanitized_reminder_failure_is_allowed(tmp_path, monkeypatch):
    tools_root = tmp_path / "ios" / "Lumen" / "Services" / "Tools"
    tools_root.mkdir(parents=True)
    (tools_root / "CalendarTools.swift").write_text(
        """
        enum CalendarTools {
            static func createReminder(title: String) async -> String {
                do {
                    return "Added reminder"
                } catch {
                    return reminderFailureMessage(action: "add")
                }
            }

            static func reminderFailureMessage(action: String) -> String {
                "I couldn't \\(action) reminders right now. Try again later."
            }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_native_tool_raw_localized_errors_are_rejected(tmp_path, monkeypatch):
    tools_root = tmp_path / "ios" / "Lumen" / "Services" / "Tools"
    tools_root.mkdir(parents=True)
    for filename, message in {
        "AlarmTools.swift": "Alarm scheduling failed",
        "HealthTools.swift": "Couldn't request Health access",
        "ContactsTools.swift": "Couldn't search contacts",
    }.items():
        (tools_root / filename).write_text(
            f"""
            enum Tool {{
                static func run() -> String {{
                    do {{
                        return "ok"
                    }} catch {{
                        return "{message}: \\(error.localizedDescription)"
                    }}
                }}
            }}
            """,
            encoding="utf-8",
        )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("raw alarm tool localized error" in item for item in violations)
    assert any("raw health tool localized error" in item for item in violations)
    assert any("raw contacts tool localized error" in item for item in violations)


def test_native_tool_sanitized_failures_are_allowed(tmp_path, monkeypatch):
    tools_root = tmp_path / "ios" / "Lumen" / "Services" / "Tools"
    tools_root.mkdir(parents=True)
    (tools_root / "AlarmTools.swift").write_text(
        """
        enum AlarmTools {
            static func run() -> String {
                do {
                    return "ok"
                } catch {
                    return schedulingFailureMessage()
                }
            }

            static func schedulingFailureMessage() -> String {
                "Alarm scheduling failed. Try again later."
            }
        }
        """,
        encoding="utf-8",
    )
    (tools_root / "HealthTools.swift").write_text(
        """
        enum HealthTools {
            static func run() -> String {
                do {
                    return "ok"
                } catch {
                    return authorizationFailureMessage()
                }
            }

            static func authorizationFailureMessage() -> String {
                "Couldn't request Health access right now. Try again later."
            }
        }
        """,
        encoding="utf-8",
    )
    (tools_root / "ContactsTools.swift").write_text(
        """
        enum ContactsTools {
            static func run() -> String {
                do {
                    return "ok"
                } catch {
                    return searchFailureMessage()
                }
            }

            static func searchFailureMessage() -> String {
                "Couldn't search contacts right now. Try again later."
            }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_rag_file_extraction_try_fallbacks_are_rejected(tmp_path, monkeypatch):
    services_root = tmp_path / "ios" / "Lumen" / "Services"
    services_root.mkdir(parents=True)
    (services_root / "RAGStore.swift").write_text(
        """
        func unsafe(url: URL, data: Data) {
            let fileData = try? Data(contentsOf: url)
            let attr = try? NSAttributedString(data: data, options: [:], documentAttributes: nil)
            _ = fileData
            _ = attr
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy RAG file read fallback" in item for item in violations)
    assert any("lossy RAG attributed decode fallback" in item for item in violations)


def test_rag_file_extraction_diagnostic_path_is_allowed(tmp_path, monkeypatch):
    services_root = tmp_path / "ios" / "Lumen" / "Services"
    services_root.mkdir(parents=True)
    (services_root / "RAGStore.swift").write_text(
        """
        func safe(url: URL) {
            let result = extractFileTextWithDiagnostics(url: url)
            _ = result.diagnostic
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_memory_capture_pending_count_fallbacks_are_rejected(tmp_path, monkeypatch):
    memory_root = tmp_path / "ios" / "Lumen" / "Memory"
    intent_root = tmp_path / "ios" / "Lumen" / "AppIntents"
    memory_root.mkdir(parents=True)
    intent_root.mkdir(parents=True)
    (memory_root / "MemoryCaptureQueue.swift").write_text(
        """
        func unsafe() {
            let remaining = (try? pendingCount(fileURL: fileURL)) ?? 0
            _ = remaining
        }
        """,
        encoding="utf-8",
    )
    (intent_root / "LumenAddMemoryIntent.swift").write_text(
        """
        func unsafe() {
            let pending = (try? MemoryCaptureQueue.pendingCount()) ?? 1
            _ = pending
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert sum("lossy memory capture pending count fallback" in item for item in violations) == 2


def test_memory_capture_pending_count_diagnostic_paths_are_allowed(tmp_path, monkeypatch):
    memory_root = tmp_path / "ios" / "Lumen" / "Memory"
    intent_root = tmp_path / "ios" / "Lumen" / "AppIntents"
    memory_root.mkdir(parents=True)
    intent_root.mkdir(parents=True)
    (memory_root / "MemoryCaptureQueue.swift").write_text(
        """
        func safe() {
            let pending = pendingCountWithDiagnostics(fileURL: fileURL)
            _ = pending.diagnostic
        }
        """,
        encoding="utf-8",
    )
    (intent_root / "LumenAddMemoryIntent.swift").write_text(
        """
        func safe() {
            let pending = MemoryCaptureQueue.pendingCountWithDiagnostics()
            _ = pending.count
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_headless_stored_model_empty_fleet_fallback_is_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Assistant"
    source_root.mkdir(parents=True)
    (source_root / "HeadlessAgentKernelRunner.swift").write_text(
        """
        func unsafe(context: ModelContext) async {
            let stored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
            _ = stored
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy headless stored-model fetch empty fallback" in item for item in violations)


def test_headless_stored_model_throwing_fetch_is_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Assistant"
    source_root.mkdir(parents=True)
    (source_root / "HeadlessAgentKernelRunner.swift").write_text(
        """
        func safe(context: ModelContext) throws {
            let stored = try context.fetch(FetchDescriptor<StoredModel>())
            _ = stored
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_trigger_persist_nil_fallback_is_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Services"
    source_root.mkdir(parents=True)
    (source_root / "TriggerScheduler.swift").write_text(
        """
        func runTrigger(context: ModelContext) -> String? {
            do { try persist(context, operation: "runTrigger", scope: "Trigger") } catch { return nil }
            return "ok"
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy trigger persist nil fallback" in item for item in violations)


def test_trigger_persist_failure_message_is_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Services"
    source_root.mkdir(parents=True)
    (source_root / "TriggerScheduler.swift").write_text(
        """
        func runTrigger(context: ModelContext) -> String? {
            do {
                try persist(context, operation: "runTrigger", scope: "Trigger")
            } catch {
                return Self.triggerPersistenceFailureMessage(error: error)
            }
            return "ok"
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_trigger_scheduler_fetch_silent_return_is_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Services"
    source_root.mkdir(parents=True)
    (source_root / "TriggerScheduler.swift").write_text(
        """
        func refreshNextFireTimes(context: ModelContext) {
            guard let all = try? context.fetch(FetchDescriptor<Trigger>()) else { return }
            _ = all
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy trigger scheduler fetch silent return" in item for item in violations)


def test_trigger_scheduler_fetch_failure_message_is_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Services"
    source_root.mkdir(parents=True)
    (source_root / "TriggerScheduler.swift").write_text(
        """
        func refreshNextFireTimes(context: ModelContext) -> String? {
            let all: [Trigger]
            do {
                all = try context.fetch(FetchDescriptor<Trigger>())
            } catch {
                return Self.triggerFetchFailureMessage(error: error)
            }
            _ = all
            return nil
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_trigger_tool_ignored_save_and_empty_fetch_are_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Services" / "Tools"
    source_root.mkdir(parents=True)
    (source_root / "TriggerTools.swift").write_text(
        """
        func unsafe(ctx: ModelContext) {
            try? ctx.save()
            let all = (try? ctx.fetch(FetchDescriptor<Trigger>())) ?? []
            _ = all
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy trigger tool ignored save" in item for item in violations)
    assert any("lossy trigger tool fetch empty fallback" in item for item in violations)


def test_trigger_tool_explicit_fetch_and_save_failures_are_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Services" / "Tools"
    source_root.mkdir(parents=True)
    (source_root / "TriggerTools.swift").write_text(
        """
        func safe(ctx: ModelContext) -> String {
            let all: [Trigger]
            do {
                all = try ctx.fetch(FetchDescriptor<Trigger>())
            } catch {
                return triggerFetchFailureMessage(error: error)
            }
            do {
                try ctx.save()
            } catch {
                return triggerSaveFailureMessage(operation: "create", error: error)
            }
            return "\\(all.count)"
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_run_trigger_intent_no_result_fallback_is_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "AppIntents"
    source_root.mkdir(parents=True)
    (source_root / "LumenRunTriggerIntent.swift").write_text(
        """
        func perform() async -> String {
            let result = await TriggerScheduler.shared.runTrigger(trigger, context: ctx, settings: settings, notify: false) ?? "No result."
            return result
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("generic trigger intent no-result fallback" in item for item in violations)


def test_run_trigger_intent_degraded_empty_result_is_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "AppIntents"
    source_root.mkdir(parents=True)
    (source_root / "LumenRunTriggerIntent.swift").write_text(
        """
        func perform() async -> String {
            let result = await TriggerScheduler.shared.runTrigger(trigger, context: ctx, settings: settings, notify: false)
            return renderedTriggerResult(result)
        }

        static func renderedTriggerResult(_ result: String?) -> String {
            let trimmed = result?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !trimmed.isEmpty else {
                return LumenIntentResultRenderer.degraded("trigger returned empty result")
            }
            return String(trimmed.prefix(500))
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_model_bootstrap_stored_model_empty_fetch_fallback_is_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Services"
    source_root.mkdir(parents=True)
    (source_root / "ModelLaunchBootstrap.swift").write_text(
        """
        func unsafe(context: ModelContext) {
            let stored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
            _ = stored
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy model bootstrap stored-model fetch empty fallback" in item for item in violations)


def test_model_bootstrap_diagnostic_fetch_path_is_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Services"
    source_root.mkdir(parents=True)
    (source_root / "ModelLaunchBootstrap.swift").write_text(
        """
        func safe(context: ModelContext) {
            guard let stored = fetchStoredModels(context: context, operation: "repairFleet") else {
                return
            }
            _ = stored
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_rem_cycle_stored_model_empty_fetch_fallback_is_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Services"
    source_root.mkdir(parents=True)
    (source_root / "RemCycleService.swift").write_text(
        """
        func unsafe(context: ModelContext) {
            let stored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
            _ = stored
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy rem cycle stored-model fetch empty fallback" in item for item in violations)


def test_rem_cycle_diagnostic_fetch_path_is_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Services"
    source_root.mkdir(parents=True)
    (source_root / "RemCycleService.swift").write_text(
        """
        func safe(context: ModelContext) {
            let catalog = storedModelCatalogSnapshot {
                try context.fetch(FetchDescriptor<StoredModel>())
            }
            _ = catalog.diagnostic
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_settings_live_e2e_stored_model_empty_fetch_fallback_is_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Views"
    source_root.mkdir(parents=True)
    (source_root / "SettingsView.swift").write_text(
        """
        func unsafe(modelContext: ModelContext) {
            let stored = (try? modelContext.fetch(FetchDescriptor<StoredModel>())) ?? []
            _ = stored
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy settings e2e stored-model fetch empty fallback" in item for item in violations)


def test_settings_live_e2e_diagnostic_model_snapshot_is_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Views"
    source_root.mkdir(parents=True)
    (source_root / "SettingsView.swift").write_text(
        """
        func safe(appState: AppState, modelContext: ModelContext) {
            let modelLoadSnapshotResult = ModelLoader.modelLoadSnapshot(appState: appState, context: modelContext)
            guard let modelLoadSnapshot = modelLoadSnapshotResult.snapshot else {
                _ = modelLoadSnapshotResult.diagnostic
                return
            }
            _ = modelLoadSnapshot
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_settings_model_directory_empty_fallback_is_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Views"
    source_root.mkdir(parents=True)
    (source_root / "SettingsView.swift").write_text(
        """
        func unsafe() {
            let modelsDirectory = try? ModelStorage.modelsDirectoryURLOrThrow()
            let files = modelsDirectory.flatMap {
                try? FileManager.default.contentsOfDirectory(at: $0, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            } ?? []
            _ = files
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy settings model directory fallback" in item for item in violations)
    assert any("lossy settings model files directory empty fallback" in item for item in violations)


def test_settings_raw_model_path_diagnostic_is_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Views"
    source_root.mkdir(parents=True)
    (source_root / "SettingsView.swift").write_text(
        """
        func unsafe() -> String {
            let modelFilesResult = ModelStorage.modelFilesWithDiagnostics()
            return "Models path: \\(modelFilesResult.directory?.path ?? \"unavailable\")"
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("raw settings models path" in item for item in violations)


def test_settings_model_directory_diagnostic_listing_is_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Views"
    source_root.mkdir(parents=True)
    (source_root / "SettingsView.swift").write_text(
        """
        func safe() {
            let modelFilesResult = ModelStorage.modelFilesWithDiagnostics()
            let modelsPathSummary = modelFilesResult.directory.map(pathSummary) ?? "unavailable"
            _ = modelFilesResult.files.count
            _ = modelFilesResult.diagnostic
            _ = modelsPathSummary
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_settings_imported_files_empty_fallback_is_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Views"
    source_root.mkdir(parents=True)
    (source_root / "SettingsView.swift").write_text(
        """
        func unsafe() {
            let imported = FileStore.importedFiles()
            let importsDirectory = try? FileStore.importsDirectoryOrThrow(fileManager: .default)
            _ = (imported, importsDirectory)
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy settings imported files wrapper" in item for item in violations)
    assert any("lossy settings imports directory fallback" in item for item in violations)


def test_settings_imported_files_diagnostic_listing_is_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Views"
    source_root.mkdir(parents=True)
    (source_root / "SettingsView.swift").write_text(
        """
        func safe() {
            let importedFilesResult = FileStore.importedFilesWithDiagnostics()
            _ = importedFilesResult.files.count
            _ = importedFilesResult.directory
            _ = importedFilesResult.diagnostic
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_release_unfinished_markers_are_rejected_outside_debug(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen"
    source_root.mkdir(parents=True)
    (source_root / "UnfinishedProduct.swift").write_text(
        """
        func unfinished() {
            // TODO: wire this for Release
            let status = "stubbed backend"
            let message = "not implemented"
            _ = (status, message)
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [source_root])

    violations = release_hardening.scan_source()

    assert any("production TODO marker" in item for item in violations)
    assert any("production stub marker" in item for item in violations)
    assert any("production not-implemented marker" in item for item in violations)


def test_gguf_release_precondition_crashes_are_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen"
    source_root.mkdir(parents=True)
    (source_root / "GGUFEngine.swift").write_text(
        """
        func unsafe(includeUnavailableGGUF: Bool) {
            preconditionFailure("GGUFEngine requires a compiled native bridge in Release builds.")
            precondition(!includeUnavailableGGUF, "Unavailable GGUF native bridge registration is DEBUG-only.")
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [source_root])

    violations = release_hardening.scan_source()

    assert sum("gguf release precondition crash" in item for item in violations) == 2
    assert sum("production precondition crash" in item for item in violations) == 2


def test_debug_only_unfinished_markers_are_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen"
    source_root.mkdir(parents=True)
    (source_root / "DebugDiagnostics.swift").write_text(
        """
        func diagnostics() {
            #if DEBUG
            // TODO: debug harness expansion
            let status = "stubbed debug probe"
            let message = "not implemented in debug probe"
            _ = (status, message)
            #endif
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [source_root])

    assert release_hardening.scan_source() == []


def test_unsafe_public_diagnostics_are_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen"
    source_root.mkdir(parents=True)
    (source_root / "UnsafeDiagnostics.swift").write_text(
        """
        import OSLog

        func unsafe(
            logger: Logger,
            query: String,
            userPrompt: String,
            rawOutput: String,
            title: String,
            arguments: [String: String],
            error: Error
        ) {
            logger.info("query=\\(query)")
            logger.error("prompt=\\(userPrompt, privacy: .public)")
            logger.error("raw_output=\\(rawOutput, privacy: .public)")
            logger.error("title=\\(title, privacy: .public)")
            logger.error("error=\\(String(describing: error), privacy: .public)")
            _ = PersistentDiagnosticEvent(code: "unsafe", message: userPrompt)
            _ = PersistentDiagnosticEvent(code: "unsafe", message: "tool", values: ["arguments": arguments.description])
            _ = try? encoder.encode(trace)
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [source_root])

    violations = release_hardening.scan_source()

    assert any("raw sensitive logger interpolation" in item for item in violations)
    assert any("public raw sensitive diagnostic interpolation" in item for item in violations)
    assert any("public raw error diagnostic" in item for item in violations)
    assert any("raw persistent diagnostic field" in item for item in violations)
    assert any("developer trace raw encoding" in item for item in violations)


def test_private_and_sanitized_diagnostics_are_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen"
    source_root.mkdir(parents=True)
    (source_root / "SafeDiagnostics.swift").write_text(
        """
        import OSLog

        func safe(logger: Logger, query: String, title: String, error: Error) {
            logger.info("query_hash=\\(RuntimeFallbackLogger.promptHash(query), privacy: .public)")
            logger.error("source_hash=\\(RuntimeFallbackLogger.promptHash(title), privacy: .public)")
            logger.error("query=\\(query, privacy: .private)")
            logger.error("title=\\(title, privacy: .private)")
            logger.error("error_code=\\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            _ = PersistentDiagnosticEvent(code: "safe", message: "sanitized", values: ["promptChars": String(query.count)])
            #if DEBUG
            _ = try? encoder.encode(trace)
            #endif
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [source_root])

    assert release_hardening.scan_source() == []


def test_removed_legacy_tool_command_and_background_bridge_names_are_rejected(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen"
    source_root.mkdir(parents=True)
    (source_root / "ProductToolExecution.swift").write_text(
        """
        func unsafe(registry: SecureToolRegistry) async {
            _ = await registry.executeLegacyTool("device.status", arguments: .init(), approval: .autonomous)
            _ = BackgroundToolBridgePolicy.self
            let status = BackgroundToolBridgeAssessment.Status.bridgeMappingUnavailable
            let message = "Legacy agent bridge is excluded from Release builds."
            // Temporary compatibility shim
            _ = status
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [source_root])

    violations = release_hardening.scan_source()

    assert any("removed legacy tool command API" in item for item in violations)
    assert any("background compatibility bridge surface" in item for item in violations)
    assert any("release legacy bridge exclusion wording" in item for item in violations)


def test_legacy_compatibility_bridge_implementation_must_be_debug_only(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Assistant"
    source_root.mkdir(parents=True)
    (source_root / "LegacyAgentCompatibilityBridge.swift").write_text(
        """
        enum LegacyAgentCompatibilityBridge {
            static func runLegacyAgentService() {}
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("legacy compatibility bridge implementation must be inside #if DEBUG" in item for item in violations)


def test_legacy_compatibility_bridge_file_must_be_fully_debug_only(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Assistant"
    source_root.mkdir(parents=True)
    (source_root / "LegacyAgentCompatibilityBridge.swift").write_text(
        """
        #if DEBUG
        enum LegacyAgentCompatibilityBridge {
            static func runLegacyAgentService() {}
        }
        #endif

        private extension AgentEvent {
            var agentKernelEvent: AgentKernelEvent { .final("") }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("legacy compatibility bridge file must be fully inside #if DEBUG" in item for item in violations)


def test_debug_only_legacy_compatibility_bridge_implementation_is_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Assistant"
    source_root.mkdir(parents=True)
    (source_root / "LegacyAgentCompatibilityBridge.swift").write_text(
        """
        #if DEBUG
        enum LegacyAgentCompatibilityBridge {
            static func runLegacyAgentService() {}
        }
        #endif
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_run_legacy_agent_bridge_api_must_be_debug_only(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Assistant"
    source_root.mkdir(parents=True)
    (source_root / "AssistantKernel+Streaming.swift").write_text(
        """
        extension AssistantKernel {
            func runLegacyAgentBridge(_ request: AgentRequest, options: LegacyAgentRunOptions) -> AsyncStream<AgentKernelEvent> {
                AsyncStream { continuation in
                    continuation.yield(.error("DEBUG-only"))
                    continuation.finish()
                }
            }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("legacy bridge API surface must be inside #if DEBUG" in item for item in violations)


def test_debug_only_run_legacy_agent_bridge_api_is_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Assistant"
    source_root.mkdir(parents=True)
    (source_root / "AssistantKernel+Streaming.swift").write_text(
        """
        extension AssistantKernel {
            #if DEBUG
            func runLegacyAgentBridge(_ request: AgentRequest, options: LegacyAgentRunOptions) -> AsyncStream<AgentKernelEvent> {
                LegacyAgentCompatibilityBridge.runLegacyAgentService(request, options: options)
            }
            #endif
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_shipped_release_docs_reject_unproven_status_words(tmp_path, monkeypatch):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    app_intents = docs_root / "APP_INTENTS.md"
    background = docs_root / "BACKGROUND_PROCESSING.md"
    tool_security = docs_root / "TOOL_SECURITY_MODEL.md"
    app_intents.write_text("This planned AppIntent surface is partial.\n", encoding="utf-8")
    background.write_text("The compatibility bridge runs triggers through BackgroundToolBridgePolicy.\n", encoding="utf-8")
    tool_security.write_text("Legacy compatibility bridge remains shipped.\n", encoding="utf-8")
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "DOC_ROOTS", [app_intents, background, tool_security])

    violations = release_hardening.scan_docs()

    assert any("shipped planned wording" in item for item in violations)
    assert any("shipped partial wording" in item for item in violations)
    assert any("shipped compatibility bridge wording" in item for item in violations)
    assert any("stale background bridge policy wording" in item for item in violations)


def test_developer_console_storage_diagnostics_reject_lossy_counts_and_raw_paths(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Developer"
    source_root.mkdir(parents=True)
    (source_root / "DeveloperFramework.swift").write_text(
        """
        func logsText() -> String {
            let modelsDirectory = ModelStorage.modelsDirectoryURL()
            let imported = FileStore.importedFiles()
            let modelFiles = (try? FileManager.default.contentsOfDirectory(at: modelsDirectory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
            return "Imported \\(imported.count) model files \\(modelFiles.count) Models path: \\(modelsDirectory.path)"
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("lossy developer imported-files wrapper" in item for item in violations)
    assert any("lossy developer model-files fallback" in item for item in violations)
    assert any("raw developer models path" in item for item in violations)


def test_developer_console_storage_diagnostic_apis_and_path_hash_are_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Developer"
    source_root.mkdir(parents=True)
    (source_root / "DeveloperFramework.swift").write_text(
        """
        func logsText() -> String {
            let imported = FileStore.importedFilesWithDiagnostics()
            let modelFiles = ModelStorage.modelFilesWithDiagnostics()
            let modelsPathSummary = modelFiles.directory.map(Self.pathSummary) ?? "unavailable"
            return "Imported \\(imported.files.count) mode \\(imported.mode) model files \\(modelFiles.files.count) path \\(modelsPathSummary)"
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_release_model_catalog_rejects_fallback_surface_wording(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Services" / "LLM" / "Models"
    source_root.mkdir(parents=True)
    (source_root / "BuiltInModelCatalog.swift").write_text(
        """
        enum BuiltInModelCatalog {
            static let all = [
                ModelCatalogEntry(
                    id: "builtin.tiny-intent",
                    displayName: "Mock Fallback",
                    backend: .tinyIntent,
                    recommendedUse: .tinyIntent,
                    source: .bundled,
                    contextLength: 512,
                    minimumRecommendedTier: .constrained,
                    tags: ["fallback"],
                    notes: "Staged unavailable routing."
                )
            ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("release model catalog fallback wording" in item for item in violations)
    assert any("release model catalog mock wording" in item for item in violations)
    assert any("release model catalog staged wording" in item for item in violations)
    assert any("release model catalog unavailable wording" in item for item in violations)


def test_debug_only_model_catalog_diagnostic_descriptor_wording_is_allowed(tmp_path, monkeypatch):
    source_root = tmp_path / "ios" / "Lumen" / "Services" / "LLM" / "Models"
    source_root.mkdir(parents=True)
    (source_root / "BuiltInModelCatalog.swift").write_text(
        """
        enum BuiltInModelCatalog {
            static let all = {
                var entries = [
                    ModelCatalogEntry(
                        id: "qwen",
                        displayName: "Qwen",
                        backend: .gguf,
                        recommendedUse: .standardChat,
                        source: .unknown,
                        contextLength: 512,
                        minimumRecommendedTier: .constrained,
                        tags: ["gguf"],
                        notes: "Bundled model descriptor."
                    )
                ]
                #if DEBUG
                entries.append(ModelCatalogEntry(
                    id: "debug.embedding",
                    displayName: "Debug Unavailable Fallback",
                    backend: .gguf,
                    recommendedUse: .testing,
                    source: .unknown,
                    contextLength: 512,
                    minimumRecommendedTier: .constrained,
                    tags: ["fallback"],
                    notes: "DEBUG-only unavailable diagnostic descriptor."
                ))
                #endif
                return entries
            }()
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    assert release_hardening.scan_source() == []


def test_release_family_and_fleet_catalogs_reject_fallback_surface_wording(tmp_path, monkeypatch):
    services_root = tmp_path / "ios" / "Lumen" / "Services"
    services_root.mkdir(parents=True)
    (services_root / "ModelFamilySelection.swift").write_text(
        """
        enum LumenModelFamily {
            var description: String { "Embedding fallback family" }
        }
        """,
        encoding="utf-8",
    )
    (services_root / "ModelFleetCatalog.swift").write_text(
        """
        enum LumenModelFleetCatalog {
            static let all = [
                CatalogModel(
                    id: "fleet-fallback",
                    name: "Fleet Fallback",
                    repoId: "owner/repo",
                    fileName: "model.gguf",
                    parameters: "1B",
                    quantization: "Q4",
                    sizeBytes: 1,
                    role: .chat,
                    description: "Use when adapter loading is unavailable.",
                    tags: ["fallback"]
                )
            ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("ModelFamilySelection.swift" in item and "release model catalog fallback wording" in item for item in violations)
    assert any("ModelFleetCatalog.swift" in item and "release model catalog fallback wording" in item for item in violations)
    assert any("ModelFleetCatalog.swift" in item and "release model catalog unavailable wording" in item for item in violations)


def test_release_compiled_services_must_use_effective_deterministic_compatibility_gate(tmp_path, monkeypatch):
    services_root = tmp_path / "ios" / "Lumen" / "Services"
    services_root.mkdir(parents=True)
    (services_root / "SlotAgentService.swift").write_text(
        """
        func unsafe(options: LegacyAgentRunOptions) {
            if options.allowDeterministicCompatibility {
                runDeterministicCompatibility()
            }
        }
        """,
        encoding="utf-8",
    )
    (services_root / "AgentService.swift").write_text(
        """
        func safe(options: LegacyAgentRunOptions) {
            if options.allowsDeterministicCompatibilityExecution {
                runDebugCompatibility()
            }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(release_hardening, "ROOT", tmp_path)
    monkeypatch.setattr(release_hardening, "SOURCE_ROOTS", [tmp_path / "ios" / "Lumen"])

    violations = release_hardening.scan_source()

    assert any("raw deterministic compatibility flag in Release-compiled execution path" in item for item in violations)
