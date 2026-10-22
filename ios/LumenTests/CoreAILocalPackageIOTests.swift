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
        let copied = try io.digest(relative: "payload.bin", descriptor: directory, copyTo: fixture.output)
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
