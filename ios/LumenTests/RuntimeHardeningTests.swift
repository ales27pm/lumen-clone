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

    func testLocationReferenceExtractorReturnsFailureForInvalidPattern() {
        let result = LocationReferenceExtractor.makeCoordinateRegex(pattern: "[")
        guard case let .failure(error) = result else {
            return XCTFail("Expected regex compilation to fail")
        }
        XCTAssertEqual(error, .invalidPattern("["))
    }
}
