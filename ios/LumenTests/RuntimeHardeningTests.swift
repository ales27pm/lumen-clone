import XCTest
@testable import Lumen

final class RuntimeHardeningTests: XCTestCase {
    func testModelStorageDocumentsDirectoryThrowsWhenUnavailable() {
        XCTAssertThrowsError(try ModelStorage.documentsDirectoryURL(candidateDirectories: [])) { error in
            XCTAssertEqual(error as? ModelStorage.StorageError, .documentDirectoryUnavailable)
        }
    }

    func testModelStorageFallsBackToApplicationSupportWhenDocumentsUnavailable() throws {
        let appSupport = URL(fileURLWithPath: "/app-support", isDirectory: true)
        let resolved = try ModelStorage.persistentBaseDirectoryURL(documentDirectories: [], applicationSupportDirectories: [appSupport])
        XCTAssertEqual(resolved, appSupport)
    }

    func testModelStoragePersistentDirectoryThrowsWhenAllUnavailable() {
        XCTAssertThrowsError(try ModelStorage.persistentBaseDirectoryURL(documentDirectories: [], applicationSupportDirectories: [])) { error in
            XCTAssertEqual(error as? ModelStorage.StorageError, .persistentDirectoryUnavailable)
        }
    }

    func testModelStorageModelsDirectoryThrowsWhenPersistentDirectoryUnavailable() {
        XCTAssertThrowsError(
            try ModelStorage.modelsDirectoryURLOrThrow(documentDirectories: [], applicationSupportDirectories: [])
        ) { error in
            XCTAssertEqual(error as? ModelStorage.StorageError, .persistentDirectoryUnavailable)
        }
    }

    func testModelStorageModelFilesDirectoryFailureIsDiagnosticAndSanitized() {
        let result = ModelStorage.modelFilesWithDiagnosticsForTests(
            modelsDirectory: {
                throw ModelStorage.StorageError.directoryCreationFailed("/private/raw/models/path")
            },
            contents: { _ in [] }
        )

        XCTAssertNil(result.directory)
        XCTAssertTrue(result.files.isEmpty)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("models_directory_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains("/private/raw/models/path") == true)
        XCTAssertNotEqual(result.diagnostic, "empty_models_directory")
    }

    func testModelStorageModelFilesListFailureIsDiagnosticAndSanitized() {
        let rawPath = "/private/raw/models/path"
        let directory = URL(fileURLWithPath: rawPath, isDirectory: true)
        let result = ModelStorage.modelFilesWithDiagnosticsForTests(
            modelsDirectory: { directory },
            contents: { _ in
                throw ModelStorage.StorageError.contentsUnavailable(rawPath)
            }
        )

        XCTAssertEqual(result.directory, directory)
        XCTAssertTrue(result.files.isEmpty)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("models_list_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(rawPath) == true)
        XCTAssertNotEqual(result.diagnostic, "empty_models_directory")
    }

    func testModelStorageModelFilesEmptyDirectoryIsDistinctFromFailure() {
        let directory = URL(fileURLWithPath: "/tmp/models", isDirectory: true)
        let result = ModelStorage.modelFilesWithDiagnosticsForTests(
            modelsDirectory: { directory },
            contents: { _ in [] }
        )

        XCTAssertEqual(result.directory, directory)
        XCTAssertTrue(result.files.isEmpty)
        XCTAssertEqual(result.mode, "loaded")
        XCTAssertEqual(result.diagnostic, "empty_models_directory")
    }

    func testFileStoreDocumentsDirectoryThrowsWhenUnavailable() {
        XCTAssertThrowsError(try FileStore.documentsDirectoryURL(candidateDirectories: [])) { error in
            XCTAssertEqual(error as? FileStore.FileStoreError, .documentDirectoryUnavailable)
        }
    }

    func testFileStoreFallsBackToApplicationSupportWhenDocumentsUnavailable() throws {
        let appSupport = URL(fileURLWithPath: "/app-support", isDirectory: true)
        let resolved = try FileStore.persistentBaseDirectoryURL(documentDirectories: [], applicationSupportDirectories: [appSupport])
        XCTAssertEqual(resolved, appSupport)
    }

    func testFileStorePersistentDirectoryThrowsWhenAllUnavailable() {
        XCTAssertThrowsError(try FileStore.persistentBaseDirectoryURL(documentDirectories: [], applicationSupportDirectories: [])) { error in
            XCTAssertEqual(error as? FileStore.FileStoreError, .persistentDirectoryUnavailable)
        }
    }

    func testFileStoreImportsDirectoryCompatibilityAccessorDoesNotCrash() {
        XCTAssertFalse(FileStore.importsDirectory.path.isEmpty)
    }

    func testFileStoreImportedFilesDirectoryFailureIsDiagnosticAndSanitized() {
        let result = FileStore.importedFilesWithDiagnosticsForTests(
            importsDirectory: {
                throw FileStore.FileStoreError.directoryCreationFailed("/private/raw/imports/path")
            },
            contents: { _ in [] }
        )

        XCTAssertNil(result.directory)
        XCTAssertTrue(result.files.isEmpty)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("imports_directory_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains("/private/raw/imports/path") == true)
        XCTAssertNotEqual(result.diagnostic, "empty_imports")
    }

    func testFileStoreImportedFilesListFailureIsDiagnosticAndSanitized() {
        let result = FileStore.importedFilesWithDiagnosticsForTests(
            importsDirectory: {
                URL(fileURLWithPath: "/tmp/imports", isDirectory: true)
            },
            contents: { _ in
                throw FileStore.FileStoreError.contentsUnavailable("/private/raw/imports/path")
            }
        )

        XCTAssertEqual(result.directory, URL(fileURLWithPath: "/tmp/imports", isDirectory: true))
        XCTAssertTrue(result.files.isEmpty)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("imports_list_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains("/private/raw/imports/path") == true)
        XCTAssertNotEqual(result.diagnostic, "empty_imports")
    }

    func testFileStoreImportedFilesEmptyDirectoryIsDistinctFromFailure() {
        let result = FileStore.importedFilesWithDiagnosticsForTests(
            importsDirectory: {
                URL(fileURLWithPath: "/tmp/imports", isDirectory: true)
            },
            contents: { _ in [] }
        )

        XCTAssertEqual(result.directory, URL(fileURLWithPath: "/tmp/imports", isDirectory: true))
        XCTAssertTrue(result.files.isEmpty)
        XCTAssertEqual(result.mode, "loaded")
        XCTAssertEqual(result.diagnostic, "empty_imports")
    }

    func testFileStoreImportedFilesHidesOnlyInternalStagingArtifacts() {
        let directory = URL(fileURLWithPath: "/tmp/imports", isDirectory: true)
        let visible = directory.appendingPathComponent("notes.txt")
        let ordinaryHidden = directory.appendingPathComponent(".notes")
        let malformedStage = directory.appendingPathComponent(".lumen-import-not-a-uuid.staged")
        let legacyStage = directory.appendingPathComponent(".lumen-import-\(UUID().uuidString).staged")
        let processStage = directory.appendingPathComponent(
            ".lumen-import-\(UUID().uuidString)-\(UUID().uuidString).staged"
        )
        let result = FileStore.importedFilesWithDiagnosticsForTests(
            importsDirectory: { directory },
            contents: { _ in [visible, ordinaryHidden, malformedStage, legacyStage, processStage] }
        )

        XCTAssertEqual(result.files, [visible, ordinaryHidden, malformedStage])
        XCTAssertEqual(result.mode, "loaded")
        XCTAssertNil(result.diagnostic)
    }

    func testFileStorePurgesInterruptedLegacyStageWithoutRemovingOtherHiddenFiles() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("lumen-stage-cleanup-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let legacyStage = root.appendingPathComponent(".lumen-import-\(UUID().uuidString).staged")
        let ordinaryHidden = root.appendingPathComponent(".notes")
        let malformedStage = root.appendingPathComponent(".lumen-import-not-a-uuid.staged")
        try Data("interrupted private bytes".utf8).write(to: legacyStage)
        try Data("keep".utf8).write(to: ordinaryHidden)
        try Data("keep".utf8).write(to: malformedStage)

        try FileStore.purgeStaleImportStages(in: root)

        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyStage.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: ordinaryHidden.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: malformedStage.path))
    }

    func testFileStoreDoesNotPurgeAnActiveProcessStageDuringConcurrentDirectoryAccess() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("lumen-active-stage-\(UUID().uuidString)", isDirectory: true)
        let source = root.appendingPathComponent("source.txt")
        let imports = root.appendingPathComponent("imports", isDirectory: true)
        try FileManager.default.createDirectory(at: imports, withIntermediateDirectories: true)
        try Data("replacement".utf8).write(to: source)
        defer { try? FileManager.default.removeItem(at: root) }

        let result = FileStore.importFileWithDiagnosticsForTests(
            source: source,
            importsDirectory: { imports },
            destinationExists: { _ in false },
            stageItem: { source, staged in
                try FileManager.default.copyItem(at: source, to: staged)
                try FileStore.purgeStaleImportStages(in: imports)
                XCTAssertTrue(FileManager.default.fileExists(atPath: staged.path))
            },
            commitStagedItem: { staged, destination, _ in
                try FileManager.default.moveItem(at: staged, to: destination)
            },
            cleanupStagedItem: { try? FileManager.default.removeItem(at: $0) }
        )

        XCTAssertEqual(result.mode, "imported")
        XCTAssertEqual(try String(contentsOf: imports.appendingPathComponent("source.txt")), "replacement")
    }

    func testFileStoreImportFileDirectoryFailureIsDiagnosticAndSanitized() {
        let rawPath = "/private/raw/imports/path"
        let result = FileStore.importFileWithDiagnosticsForTests(
            source: URL(fileURLWithPath: "/tmp/source/secret.txt"),
            importsDirectory: {
                throw FileStore.FileStoreError.directoryCreationFailed(rawPath)
            },
            destinationExists: { _ in false },
            stageItem: { _, _ in },
            commitStagedItem: { _, _, _ in },
            cleanupStagedItem: { _ in }
        )

        XCTAssertNil(result.url)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("imports_directory_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(rawPath) == true)
    }

    func testFileStoreImportFileCommitFailureIsDiagnosticAndSanitized() {
        let rawPath = "/private/raw/imports/secret.txt"
        var staged = false
        var cleaned = false
        let result = FileStore.importFileWithDiagnosticsForTests(
            source: URL(fileURLWithPath: "/tmp/source/secret.txt"),
            importsDirectory: {
                URL(fileURLWithPath: "/tmp/imports", isDirectory: true)
            },
            destinationExists: { _ in true },
            stageItem: { _, _ in staged = true },
            commitStagedItem: { _, _, destinationExists in
                XCTAssertTrue(destinationExists)
                throw NSError(domain: rawPath, code: 19)
            },
            cleanupStagedItem: { _ in cleaned = true }
        )

        XCTAssertNil(result.url)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("import_commit_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(rawPath) == true)
        XCTAssertTrue(staged)
        XCTAssertTrue(cleaned)
    }

    func testFileStoreImportFileCopyFailureIsDiagnosticAndSanitized() {
        let rawPath = "/private/raw/source/secret.txt"
        let result = FileStore.importFileWithDiagnosticsForTests(
            source: URL(fileURLWithPath: "/tmp/source/secret.txt"),
            importsDirectory: {
                URL(fileURLWithPath: "/tmp/imports", isDirectory: true)
            },
            destinationExists: { _ in false },
            stageItem: { _, _ in
                throw NSError(domain: rawPath, code: 23)
            },
            commitStagedItem: { _, _, _ in XCTFail("commit must not run after copy failure") },
            cleanupStagedItem: { _ in }
        )

        XCTAssertNil(result.url)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("import_copy_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(rawPath) == true)
    }

    func testFileStoreImportFileSuccessReturnsDestination() {
        let result = FileStore.importFileWithDiagnosticsForTests(
            source: URL(fileURLWithPath: "/tmp/source/secret.txt"),
            importsDirectory: {
                URL(fileURLWithPath: "/tmp/imports", isDirectory: true)
            },
            destinationExists: { _ in false },
            stageItem: { source, staged in
                XCTAssertEqual(source, URL(fileURLWithPath: "/tmp/source/secret.txt"))
                XCTAssertEqual(staged.deletingLastPathComponent(), URL(fileURLWithPath: "/tmp/imports", isDirectory: true))
            },
            commitStagedItem: { staged, dest, destinationExists in
                XCTAssertEqual(staged.deletingLastPathComponent(), URL(fileURLWithPath: "/tmp/imports", isDirectory: true))
                XCTAssertEqual(dest, URL(fileURLWithPath: "/tmp/imports/secret.txt"))
                XCTAssertFalse(destinationExists)
            },
            cleanupStagedItem: { _ in XCTFail("cleanup must not run after success") }
        )

        XCTAssertEqual(result.url, URL(fileURLWithPath: "/tmp/imports/secret.txt"))
        XCTAssertEqual(result.mode, "imported")
        XCTAssertNil(result.diagnostic)
    }

    func testFileStoreFailedReplacementPreservesExistingImportedFile() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("lumen-file-import-\(UUID().uuidString)", isDirectory: true)
        let imports = root.appendingPathComponent("imports", isDirectory: true)
        let missingSourceDirectory = root.appendingPathComponent("missing", isDirectory: true)
        let missingSource = missingSourceDirectory.appendingPathComponent("document.txt")
        let existing = imports.appendingPathComponent("document.txt")
        try FileManager.default.createDirectory(at: imports, withIntermediateDirectories: true)
        try Data("existing contents".utf8).write(to: existing, options: [.atomic])
        defer { try? FileManager.default.removeItem(at: root) }

        let result = FileStore.importFileWithDiagnosticsForTests(
            source: missingSource,
            importsDirectory: imports
        )

        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("import_copy_failed:") == true)
        XCTAssertEqual(try String(contentsOf: existing, encoding: .utf8), "existing contents")
    }

    func testFileStoreSuccessfulReplacementCommitsNewBytes() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("lumen-file-import-\(UUID().uuidString)", isDirectory: true)
        let sourceDirectory = root.appendingPathComponent("source", isDirectory: true)
        let imports = root.appendingPathComponent("imports", isDirectory: true)
        let source = sourceDirectory.appendingPathComponent("document.txt")
        let existing = imports.appendingPathComponent("document.txt")
        try FileManager.default.createDirectory(at: sourceDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: imports, withIntermediateDirectories: true)
        try Data("replacement contents".utf8).write(to: source, options: [.atomic])
        try Data("existing contents".utf8).write(to: existing, options: [.atomic])
        defer { try? FileManager.default.removeItem(at: root) }

        let result = FileStore.importFileWithDiagnosticsForTests(
            source: source,
            importsDirectory: imports
        )

        XCTAssertEqual(result.mode, "imported")
        XCTAssertEqual(result.url, existing)
        XCTAssertEqual(try String(contentsOf: existing, encoding: .utf8), "replacement contents")
    }

    func testAttachmentResolverDoesNotInventZeroSizeWhenMetadataUnavailable() {
        let missing = URL(fileURLWithPath: "/tmp/lumen-missing-\(UUID().uuidString).txt")
        XCTAssertNil(AttachmentResolver.make(from: missing))
    }

    func testAttachmentResolverReadsImportedAttachmentSize() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("lumen-attachment-\(UUID().uuidString).txt")
        let data = Data("hello".utf8)
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let attachment = AttachmentResolver.make(from: url)

        XCTAssertEqual(attachment?.byteSize, data.count)
        XCTAssertEqual(attachment?.kind.rawValue, ChatAttachment.Kind.text.rawValue)
    }

    func testAttachmentTextReadFailureIsDiagnosticAndSanitized() {
        let rawPath = "/private/raw/imports/secret.txt"
        let attachment = ChatAttachment(name: "secret.txt", kind: .text, path: rawPath, byteSize: 128)
        let result = AttachmentResolver.extractTextWithDiagnosticsForTests(
            attachment,
            readData: { _ in throw NSError(domain: rawPath, code: 33) },
            pdfText: { _ in "unused" },
            attributedText: { _ in "unused" }
        )

        XCTAssertNil(result.text)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("attachment_read_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(rawPath) == true)
    }

    func testAttachmentPDFOpenFailureIsDiagnosticAndSanitized() {
        let rawPath = "/private/raw/imports/secret.pdf"
        let attachment = ChatAttachment(name: "secret.pdf", kind: .pdf, path: rawPath, byteSize: 128)
        let result = AttachmentResolver.extractTextWithDiagnosticsForTests(
            attachment,
            readData: { _ in Data() },
            pdfText: { _ in throw NSError(domain: rawPath, code: 37) },
            attributedText: { _ in "unused" }
        )

        XCTAssertNil(result.text)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("attachment_pdf_open_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(rawPath) == true)
    }

    func testAttachmentAttributedDecodeFailureIsDiagnosticAndSanitized() {
        let rawPath = "/private/raw/imports/secret.rtf"
        let attachment = ChatAttachment(name: "secret.rtf", kind: .text, path: rawPath, byteSize: 3)
        let result = AttachmentResolver.extractTextWithDiagnosticsForTests(
            attachment,
            readData: { _ in Data([0xff, 0xfe, 0xfd]) },
            pdfText: { _ in "unused" },
            attributedText: { _ in throw NSError(domain: rawPath, code: 39) }
        )

        XCTAssertNil(result.text)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("attachment_attributed_decode_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(rawPath) == true)
    }

    func testAttachmentEmptyTextIsDistinctFromExtractionFailure() {
        let attachment = ChatAttachment(name: "empty.txt", kind: .text, path: "/tmp/empty.txt", byteSize: 0)
        let result = AttachmentResolver.extractTextWithDiagnosticsForTests(
            attachment,
            readData: { _ in Data() },
            pdfText: { _ in "unused" },
            attributedText: { _ in "unused" }
        )

        XCTAssertEqual(result.text, "")
        XCTAssertEqual(result.mode, "loaded")
        XCTAssertEqual(result.diagnostic, "empty_attachment_text")
    }

    func testPromptAssemblerSurfacesAttachmentExtractionFailureDiagnostic() {
        let rawPath = "/private/raw/imports/missing.txt"
        let attachment = ChatAttachment(name: "missing.txt", kind: .text, path: rawPath, byteSize: 12)
        let assembly = PromptAssembler.assemble(
            systemPrompt: "sys",
            history: [],
            userMessage: "summarize",
            memories: [],
            attachments: [attachment],
            budget: PromptBudget(totalChars: 4_000, attachmentsShare: 1_000, memoriesShare: 0, historyShare: 0)
        )

        XCTAssertEqual(assembly.attachmentStates.first?.mode, "failed")
        XCTAssertTrue(assembly.attachmentStates.first?.diagnostic?.hasPrefix("attachment_read_failed:") == true)
        XCTAssertTrue(assembly.systemPrompt.contains("Diagnostic: attachment_read_failed:"))
        XCTAssertFalse(assembly.systemPrompt.contains(rawPath))
    }

    func testFilesToolReadFailureIsDiagnosticAndSanitized() {
        let rawPath = "/private/raw/imports/secret.txt"
        let result = FilesTools.readMatchedFileWithDiagnosticsForTests(
            url: URL(fileURLWithPath: rawPath),
            readData: { _ in
                throw NSError(domain: rawPath, code: 31)
            },
            pdfText: { _ in "unused" }
        )

        XCTAssertNil(result.text)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("file_read_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(rawPath) == true)
    }

    func testFilesToolTextDecodeFailureIsDistinctFromReadFailure() {
        let rawPath = "/private/raw/imports/secret.txt"
        let result = FilesTools.readMatchedFileWithDiagnosticsForTests(
            url: URL(fileURLWithPath: rawPath),
            readData: { _ in Data([0xff, 0xfe, 0xfd]) },
            pdfText: { _ in "unused" }
        )

        XCTAssertNil(result.text)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertEqual(result.diagnostic, "text_decode_failed")
    }

    func testFilesToolPDFOpenFailureIsDiagnosticAndSanitized() {
        let rawPath = "/private/raw/imports/secret.pdf"
        let result = FilesTools.readMatchedFileWithDiagnosticsForTests(
            url: URL(fileURLWithPath: rawPath),
            readData: { _ in Data() },
            pdfText: { _ in
                throw NSError(domain: rawPath, code: 41)
            }
        )

        XCTAssertNil(result.text)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("pdf_open_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(rawPath) == true)
    }

    func testFilesToolEmptyTextIsDistinctFromReadFailure() {
        let result = FilesTools.readMatchedFileWithDiagnosticsForTests(
            url: URL(fileURLWithPath: "/tmp/empty.txt"),
            readData: { _ in Data("   \n".utf8) },
            pdfText: { _ in "unused" }
        )

        XCTAssertNil(result.text)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertEqual(result.diagnostic, "empty_text")
    }

    func testLocationReferenceExtractorReturnsFailureForInvalidPattern() {
        let result = LocationReferenceExtractor.makeCoordinateRegex(pattern: "[")
        guard case let .failure(error) = result else {
            return XCTFail("Expected regex compilation to fail")
        }
        XCTAssertEqual(error, .invalidPattern("["))
    }
}
