import CryptoKit
import Darwin
import Foundation
import Testing
@testable import Lumen

@Suite(.serialized)
struct CoreAILocalPackageIOTests {
    @Test func copyAndHashPreserveMultipleChunksAndPartialTail() throws {
        let fixture = try Fixture(prefixBytes: 2 * 1_024 * 1_024, tail: "coreai-tail\n")
        defer { fixture.remove() }
        let io = CoreAILocalPackageIO()
        let directory = try io.openDirectory(fixture.root)
        defer { close(directory) }
        #expect(try io.fileSize(relative: "payload.bin", descriptor: directory) == 2_097_164)
        let copied = try io.digest(relative: "payload.bin", descriptor: directory, copyTo: fixture.output,
                                   expectedBytes: 2_097_164, maximumBytes: 2_097_164)
        #expect(copied.bytes == 2_097_164)
        #expect(copied.sha256 == "c0af7d44539783d97cd9e48cab93e1547736e2cb6851b87e948f6a8d2ca830a2")
        #expect(try io.digest(relative: "copied.bin", descriptor: directory).sha256 == copied.sha256)
        let handle = try FileHandle(forReadingFrom: fixture.output)
        defer { try? handle.close() }
        try handle.seek(toOffset: 2 * 1_024 * 1_024)
        #expect(try handle.read(upToCount: 13) == Data("coreai-tail\n".utf8))
    }

    @Test func emptyFileAndExactChunkEOFHaveKnownHashes() throws {
        for (size, expected) in [
            (0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            (1_024 * 1_024, "30e14955ebf1352266dc2ff8067e68104607e750abb9d3b36582b8af909fcb58")
        ] {
            let fixture = try Fixture(prefixBytes: size, tail: "")
            defer { fixture.remove() }
            let io = CoreAILocalPackageIO()
            let directory = try io.openDirectory(fixture.root)
            defer { close(directory) }
            let result = try io.digest(relative: "payload.bin", descriptor: directory, copyTo: fixture.output)
            #expect(result.bytes == Int64(size))
            #expect(result.sha256 == expected)
            #expect(try io.digest(relative: "copied.bin", descriptor: directory).sha256 == expected)
        }
    }

    @Test func symbolicLinkCannotBecomeAHashOrCopySource() throws {
        let fixture = try Fixture(prefixBytes: 0, tail: "synthetic")
        defer { fixture.remove() }
        try FileManager.default.createSymbolicLink(at: fixture.root.appendingPathComponent("linked.bin"), withDestinationURL: fixture.source)
        let io = CoreAILocalPackageIO()
        let directory = try io.openDirectory(fixture.root)
        defer { close(directory) }
        #expect(throws: CoreAILocalPackageIO.Failure.localDirectory) {
            try io.digest(relative: "linked.bin", descriptor: directory, copyTo: fixture.output)
        }
        #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
    }

    @Test func cancelledDetachedWorkerCannotReturnASuccessfulDigest() async throws {
        let fixture = try Fixture(prefixBytes: 2 * 1_024 * 1_024, tail: "coreai-tail\n")
        defer { fixture.remove() }
        let result = await Task.detached {
            let io = CoreAILocalPackageIO()
            let directory = try io.openDirectory(fixture.root)
            defer { close(directory) }
            withUnsafeCurrentTask { $0?.cancel() }
            return try io.digest(relative: "payload.bin", descriptor: directory, copyTo: fixture.output)
        }.result
        switch result {
        case .success: Issue.record("A cancelled package copy returned success")
        case .failure(let error): #expect(error is CancellationError)
        }
        #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
    }

    @Test func expectedLengthIsRejectedBeforeCreatingTheDestination() throws {
        let fixture = try Fixture(prefixBytes: 0, tail: "synthetic")
        defer { fixture.remove() }
        let io = CoreAILocalPackageIO()
        let directory = try io.openDirectory(fixture.root)
        defer { close(directory) }
        #expect(throws: CoreAILocalPackageIO.Failure.integrity) {
            try io.digest(relative: "payload.bin", descriptor: directory, copyTo: fixture.output, expectedBytes: 10)
        }
        #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
    }

    @Test func remainingByteBudgetIsRejectedBeforeTouchingTheDestination() throws {
        let fixture = try Fixture(prefixBytes: 0, tail: "synthetic")
        defer { fixture.remove() }
        let sentinel = Data("keep-existing-output".utf8)
        try sentinel.write(to: fixture.output)
        let io = CoreAILocalPackageIO()
        let directory = try io.openDirectory(fixture.root)
        defer { close(directory) }
        for budget in [Int64(8), -1] {
            #expect(throws: CoreAILocalPackageIO.Failure.invalidPackage) {
                try io.digest(relative: "payload.bin", descriptor: directory, copyTo: fixture.output, maximumBytes: budget)
            }
            #expect(try Data(contentsOf: fixture.output) == sentinel)
        }
    }

    @Test func inventoryReturnsSortedRelativeFilesAndIgnoresEmptyDirectories() throws {
        let fixture = try InventoryFixture()
        defer { fixture.remove() }
        try FileManager.default.createDirectory(at: fixture.root.appendingPathComponent("nested/empty"), withIntermediateDirectories: true)
        for path in ["z.bin", "nested/β.bin", "a.bin"] {
            try Data("synthetic".utf8).write(to: fixture.root.appendingPathComponent(path))
        }
        #expect(try CoreAILocalPackageIO().inventory(fixture.root) == ["a.bin", "nested/β.bin", "z.bin"])
    }

    @Test func inventoryAdmitsItsExactEntryLimitThenRejectsOneMore() throws {
        let fixture = try InventoryFixture(fileCount: 4_095)
        defer { fixture.remove() }
        let io = CoreAILocalPackageIO()
        // The root directory itself is one of the 4,096 admitted entries.
        #expect(try io.inventory(fixture.root).count == 4_095)
        try fixture.createEmptyFiles(4_095..<4_096)
        #expect(throws: CoreAILocalPackageIO.Failure.invalidPackage) { try io.inventory(fixture.root) }
    }

    @Test func inventoryRejectsSymbolicLinks() throws {
        let fixture = try InventoryFixture(fileCount: 1)
        defer { fixture.remove() }
        try FileManager.default.createSymbolicLink(at: fixture.root.appendingPathComponent("linked"), withDestinationURL: fixture.root)
        #expect(throws: CoreAILocalPackageIO.Failure.localDirectory) { try CoreAILocalPackageIO().inventory(fixture.root) }
    }

    @Test func cancelledInventoryCannotReturnAResult() async throws {
        let fixture = try InventoryFixture(fileCount: 1)
        defer { fixture.remove() }
        let result = await Task.detached {
            withUnsafeCurrentTask { $0?.cancel() }
            return try CoreAILocalPackageIO().inventory(fixture.root)
        }.result
        switch result {
        case .success: Issue.record("A cancelled package inventory returned success")
        case .failure(let error): #expect(error is CancellationError)
        }
    }

    #if COREAI_PACKAGE_IO_MEMORY_TESTS
    // Enabled only by the standalone Darwin runner. Process-wide memory must
    // not be sampled while unrelated full-app test suites run concurrently.
    @Test func detachedCopyBoundsTemporaryMemoryBeforeTheOuterPoolDrains() async throws {
        try await Self.checkMemoryBound(copy: true)
    }

    @Test func detachedHashOnlyBoundsTemporaryMemoryBeforeTheOuterPoolDrains() async throws {
        try await Self.checkMemoryBound(copy: false)
    }

    @Test func wideInventoryIsRejectedWithoutRetainingTheWholeDirectory() async throws {
        // Match the independently reproduced 32k short-name directory. The
        // eager listing exceeded the 64 MiB memory budget before rejection.
        let fixture = try InventoryFixture(fileCount: 32_768)
        defer { fixture.remove() }
        let measurement = try await Task.detached {
            try autoreleasepool {
                let before = try Self.physicalFootprint()
                let rejected: Bool
                do {
                    _ = try CoreAILocalPackageIO().inventory(fixture.root)
                    rejected = false
                } catch CoreAILocalPackageIO.Failure.invalidPackage {
                    rejected = true
                }
                // This measures retained memory at rejection, before the outer
                // pool drains; it is not a peak-memory measurement.
                let after = try Self.physicalFootprint()
                return (rejected, after > before ? after - before : 0)
            }
        }.value
        #expect(measurement.0)
        print("CoreAI package IO memory: mode=inventory entries=32768 growth=\(measurement.1) limit=67108864")
        #expect(measurement.1 < 64 * 1_024 * 1_024)
    }

    private static func checkMemoryBound(copy: Bool) async throws {
        let fixture = try Fixture(prefixBytes: 128 * 1_024 * 1_024, tail: "coreai-memory-tail\n")
        defer { fixture.remove() }
        let measurement = try await Task.detached {
            try autoreleasepool {
                let io = CoreAILocalPackageIO()
                let directory = try io.openDirectory(fixture.root)
                defer { close(directory) }
                let before = try physicalFootprint()
                let identity = try io.digest(relative: "payload.bin", descriptor: directory, copyTo: copy ? fixture.output : nil)
                // Measure while the OUTER pool is still alive. The old helper
                // retains its temporary Foundation buffers until this drains.
                let after = try physicalFootprint()
                return Measurement(identity: identity, growth: after > before ? after - before : 0)
            }
        }.value
        #expect(measurement.identity.bytes == 134_217_747)
        // Independently computed SHA-256 of 128 MiB zero bytes followed by the
        // 19 UTF-8 bytes "coreai-memory-tail\n"; no whole-file Data is created.
        #expect(measurement.identity.sha256 == "54c1334f6ca546334e61614aacbaf95759e0f1e568bc15eca03f9b73d2fdbccf")
        print("CoreAI package IO memory: mode=\(copy ? "copy" : "hash") bytes=\(measurement.identity.bytes) growth=\(measurement.growth) limit=67108864")
        #expect(measurement.growth < 64 * 1_024 * 1_024)
    }

    private struct Measurement: Sendable {
        let identity: CoreAILocalPackageIO.FileIdentity
        let growth: UInt64
    }

    private enum MeasurementFailure: Error {
        case taskInfo(kern_return_t)
    }

    private static func physicalFootprint() throws -> UInt64 {
        var information = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let capacity = Int(count)
        let result = withUnsafeMutablePointer(to: &information) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: capacity) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { throw MeasurementFailure.taskInfo(result) }
        return information.phys_footprint
    }
    #endif

    private struct InventoryFixture: Sendable {
        let root: URL

        init(fileCount: Int = 0) throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("CoreAIInventory-\(UUID().uuidString)", isDirectory: true).resolvingSymlinksInPath()
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            do { try createEmptyFiles(0..<fileCount) }
            catch { remove(); throw error }
        }

        func createEmptyFiles(_ indices: Range<Int>) throws {
            let directory = Darwin.open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard directory >= 0 else { throw FixtureFailure.fileCreation }
            defer { close(directory) }
            for index in indices {
                let file = openat(directory, "\(index).bin", O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, S_IRUSR | S_IWUSR)
                guard file >= 0 else { throw FixtureFailure.fileCreation }
                close(file)
            }
        }

        func remove() { try? FileManager.default.removeItem(at: root) }
    }

    /// Sparse synthetic source; fixture setup allocates only a short tail.
    private struct Fixture: Sendable {
        let root: URL
        var source: URL { root.appendingPathComponent("payload.bin") }
        var output: URL { root.appendingPathComponent("copied.bin") }

        init(prefixBytes: Int, tail: String) throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("CoreAIPackageIO-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let file = Darwin.open(source.path, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, S_IRUSR | S_IWUSR)
            guard file >= 0 else { throw FixtureFailure.fileCreation }
            defer { close(file) }
            do {
                guard ftruncate(file, off_t(prefixBytes)) == 0 else { throw FixtureFailure.fileCreation }
                let data = Data(tail.utf8)
                let written = data.withUnsafeBytes { bytes in
                    pwrite(file, bytes.baseAddress, bytes.count, off_t(prefixBytes))
                }
                guard written == data.count else { throw FixtureFailure.fileCreation }
            } catch {
                try? FileManager.default.removeItem(at: root)
                throw error
            }
        }

        func remove() { try? FileManager.default.removeItem(at: root) }
    }

    private enum FixtureFailure: Error { case fileCreation }
}
