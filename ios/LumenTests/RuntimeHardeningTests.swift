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

    func testFileStoreImportFileDirectoryFailureIsDiagnosticAndSanitized() {
        let rawPath = "/private/raw/imports/path"
        let result = FileStore.importFileWithDiagnosticsForTests(
            source: URL(fileURLWithPath: "/tmp/source/secret.txt"),
            importsDirectory: {
                throw FileStore.FileStoreError.directoryCreationFailed(rawPath)
            },
            destinationExists: { _ in false },
            removeExisting: { _ in },
            copyItem: { _, _ in }
        )

        XCTAssertNil(result.url)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("imports_directory_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(rawPath) == true)
    }

    func testFileStoreImportFileRemoveExistingFailureIsDiagnosticAndSanitized() {
        let rawPath = "/private/raw/imports/secret.txt"
        let result = FileStore.importFileWithDiagnosticsForTests(
            source: URL(fileURLWithPath: "/tmp/source/secret.txt"),
            importsDirectory: {
                URL(fileURLWithPath: "/tmp/imports", isDirectory: true)
            },
            destinationExists: { _ in true },
            removeExisting: { _ in
                throw NSError(domain: rawPath, code: 19)
            },
            copyItem: { _, _ in XCTFail("copy must not run after remove failure") }
        )

        XCTAssertNil(result.url)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("import_remove_existing_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(rawPath) == true)
    }

    func testFileStoreImportFileCopyFailureIsDiagnosticAndSanitized() {
        let rawPath = "/private/raw/source/secret.txt"
        let result = FileStore.importFileWithDiagnosticsForTests(
            source: URL(fileURLWithPath: "/tmp/source/secret.txt"),
            importsDirectory: {
                URL(fileURLWithPath: "/tmp/imports", isDirectory: true)
            },
            destinationExists: { _ in false },
            removeExisting: { _ in XCTFail("remove must not run when destination does not exist") },
            copyItem: { _, _ in
                throw NSError(domain: rawPath, code: 23)
            }
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
            removeExisting: { _ in XCTFail("remove must not run when destination does not exist") },
            copyItem: { source, dest in
                XCTAssertEqual(source, URL(fileURLWithPath: "/tmp/source/secret.txt"))
                XCTAssertEqual(dest, URL(fileURLWithPath: "/tmp/imports/secret.txt"))
            }
        )

        XCTAssertEqual(result.url, URL(fileURLWithPath: "/tmp/imports/secret.txt"))
        XCTAssertEqual(result.mode, "imported")
        XCTAssertNil(result.diagnostic)
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
