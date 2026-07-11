import Foundation
import SwiftData
import Accelerate
import OSLog

nonisolated struct VectorIndexLoadResult: Sendable, Equatable {
    let loadedCount: Int
    let mode: String
    let diagnostic: String?
}

/// In-memory, pre-normalized Float32 vector index with Accelerate-accelerated search.
///
/// Design:
/// - Embeddings are stored on disk as `[Double]` on the SwiftData models (unchanged for
///   backwards compatibility). At first search we load them into a contiguous Float32
///   matrix laid out row-major so a single `vDSP_mmul` computes every cosine score
///   in one BLAS call.
/// - Embeddings from `AppLlamaService` are already L2-normalized, so `dot == cosine`.
///   We defensively re-normalize on load so stale data can't poison the index.
/// - Bucket metadata (e.g. `RAGSourceType.rawValue`) is kept parallel to the matrix
///   so source-aware filtering happens before ranking with no extra allocation.
/// - The index is incrementally appended on new inserts and invalidated on wipes to
///   avoid re-reading SwiftData on every query.
@MainActor
final class RAGVectorIndex {
    static let shared = RAGVectorIndex()
    private static let logger = Logger(subsystem: "ai.lumen.app", category: "vector-index")

    private var ids: [PersistentIdentifier] = []
    private var buckets: [String] = []
    private var matrix: [Float] = []
    private var dim: Int = 0
    private var loaded: Bool = false
    private var loadedFormatVersion: Int?
    private var loadedModelIdentifier: String?

    private init() {}

    var count: Int { ids.count }
    var dimension: Int { dim }

    func invalidate() {
        ids.removeAll(keepingCapacity: false)
        buckets.removeAll(keepingCapacity: false)
        matrix.removeAll(keepingCapacity: false)
        dim = 0
        loaded = false
        loadedFormatVersion = nil
        loadedModelIdentifier = nil
    }

    @discardableResult
    func ensureLoaded(
        context: ModelContext,
        formatVersion: Int = SemanticEmbeddingText.formatVersion,
        modelIdentifier: String = "assistant-kernel-embedding",
        dimension: Int? = nil
    ) -> VectorIndexLoadResult {
        ensureLoaded(formatVersion: formatVersion, modelIdentifier: modelIdentifier, dimension: dimension, fetch: { try context.fetch(FetchDescriptor<RAGChunk>()) })
    }

    @discardableResult
    func ensureLoadedForTests(fetch: () throws -> [RAGChunk]) -> VectorIndexLoadResult {
        ensureLoaded(formatVersion: SemanticEmbeddingText.formatVersion, modelIdentifier: "assistant-kernel-embedding", dimension: nil, fetch: fetch)
    }

    private func ensureLoaded(
        formatVersion: Int,
        modelIdentifier: String,
        dimension requiredDimension: Int?,
        fetch: () throws -> [RAGChunk]
    ) -> VectorIndexLoadResult {
        if loaded, loadedFormatVersion != formatVersion || loadedModelIdentifier != modelIdentifier || (requiredDimension != nil && dim != requiredDimension) {
            invalidate()
        }
        guard !loaded else {
            return VectorIndexLoadResult(loadedCount: ids.count, mode: "already_loaded", diagnostic: nil)
        }
        let fetched: [RAGChunk]
        do {
            fetched = try fetch()
        } catch {
            let diagnostic = "rag_vector_index_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            Self.logger.error("vector_index_load_failed kind=rag diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return VectorIndexLoadResult(loadedCount: 0, mode: "failed", diagnostic: diagnostic)
        }
        loaded = true
        loadedFormatVersion = formatVersion
        loadedModelIdentifier = modelIdentifier
        ids.reserveCapacity(fetched.count)
        buckets.reserveCapacity(fetched.count)
        var d = 0
        for c in fetched where !c.embedding.isEmpty
            && c.embeddingFormatVersion == formatVersion
            && c.embeddingModelIdentifier == modelIdentifier
            && c.embeddingDimension == c.embedding.count
            && (requiredDimension == nil || c.embeddingDimension == requiredDimension) {
            if d == 0 { d = c.embedding.count }
            guard c.embedding.count == d else { continue }
            appendRow(id: c.persistentModelID, bucket: c.sourceType, vector: c.embedding, expectedDim: d)
        }
        dim = d
        return VectorIndexLoadResult(loadedCount: ids.count, mode: "loaded", diagnostic: nil)
    }

    func append(id: PersistentIdentifier, bucket: String, vector: [Double]) {
        guard loaded, !vector.isEmpty else { return }
        if dim == 0 { dim = vector.count }
        guard vector.count == dim else { return }
        appendRow(id: id, bucket: bucket, vector: vector, expectedDim: dim)
    }

    func removeBucket(_ bucket: String) {
        guard loaded, !ids.isEmpty else { return }
        var keep = [Bool](repeating: true, count: ids.count)
        var removed = 0
        for i in 0..<ids.count where buckets[i] == bucket {
            keep[i] = false
            removed += 1
        }
        if removed == 0 { return }
        if removed == ids.count {
            ids.removeAll(keepingCapacity: true)
            buckets.removeAll(keepingCapacity: true)
            matrix.removeAll(keepingCapacity: true)
            return
        }
        compact(keep: keep)
    }

    func removeAll() {
        ids.removeAll(keepingCapacity: true)
        buckets.removeAll(keepingCapacity: true)
        matrix.removeAll(keepingCapacity: true)
    }

    func search(
        query: [Double],
        topK: Int,
        allowedBuckets: Set<String>? = nil,
        minScore: Float = 0.0
    ) -> [(id: PersistentIdentifier, score: Float)] {
        guard loaded, !ids.isEmpty, dim > 0, query.count == dim, topK > 0 else { return [] }
        let q = VectorMath.toFloat32Normalized(query)
        let rows = ids.count
        var scores = [Float](repeating: 0, count: rows)
        matrix.withUnsafeBufferPointer { m in
            q.withUnsafeBufferPointer { qp in
                scores.withUnsafeMutableBufferPointer { sp in
                    vDSP_mmul(
                        m.baseAddress!, 1,
                        qp.baseAddress!, 1,
                        sp.baseAddress!, 1,
                        vDSP_Length(rows),
                        1,
                        vDSP_Length(dim)
                    )
                }
            }
        }
        return VectorMath.topK(
            scores: scores,
            k: topK,
            minScore: minScore,
            includeIndex: { idx in
                guard let allowed = allowedBuckets else { return true }
                return allowed.contains(buckets[idx])
            }
        ).map { (ids[$0.0], $0.1) }
    }

    private func appendRow(id: PersistentIdentifier, bucket: String, vector: [Double], expectedDim: Int) {
        ids.append(id)
        buckets.append(bucket)
        matrix.reserveCapacity(matrix.count + expectedDim)
        let normalized = VectorMath.toFloat32Normalized(vector)
        matrix.append(contentsOf: normalized)
    }

    private func compact(keep: [Bool]) {
        var newIds: [PersistentIdentifier] = []
        var newBuckets: [String] = []
        var newMatrix: [Float] = []
        newIds.reserveCapacity(ids.count)
        newBuckets.reserveCapacity(ids.count)
        newMatrix.reserveCapacity(matrix.count)
        for i in 0..<ids.count where keep[i] {
            newIds.append(ids[i])
            newBuckets.append(buckets[i])
            let start = i * dim
            newMatrix.append(contentsOf: matrix[start..<(start + dim)])
        }
        ids = newIds
        buckets = newBuckets
        matrix = newMatrix
    }
}

@MainActor
final class MemoryVectorIndex {
    static let shared = MemoryVectorIndex()
    private static let logger = Logger(subsystem: "ai.lumen.app", category: "vector-index")

    private var ids: [PersistentIdentifier] = []
    private var pinned: [Bool] = []
    private var matrix: [Float] = []
    private var dim: Int = 0
    private var loaded: Bool = false

    private init() {}

    func invalidate() {
        ids.removeAll(keepingCapacity: false)
        pinned.removeAll(keepingCapacity: false)
        matrix.removeAll(keepingCapacity: false)
        dim = 0
        loaded = false
    }

    @discardableResult
    func ensureLoaded(context: ModelContext) -> VectorIndexLoadResult {
        ensureLoaded(fetch: { try context.fetch(FetchDescriptor<MemoryItem>()) })
    }

    @discardableResult
    func ensureLoadedForTests(fetch: () throws -> [MemoryItem]) -> VectorIndexLoadResult {
        ensureLoaded(fetch: fetch)
    }

    private func ensureLoaded(fetch: () throws -> [MemoryItem]) -> VectorIndexLoadResult {
        guard !loaded else {
            return VectorIndexLoadResult(loadedCount: ids.count, mode: "already_loaded", diagnostic: nil)
        }
        let fetched: [MemoryItem]
        do {
            fetched = try fetch()
        } catch {
            let diagnostic = "memory_vector_index_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            Self.logger.error("vector_index_load_failed kind=memory diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return VectorIndexLoadResult(loadedCount: 0, mode: "failed", diagnostic: diagnostic)
        }
        loaded = true
        var d = 0
        for m in fetched where !m.embedding.isEmpty {
            if d == 0 { d = m.embedding.count }
            guard m.embedding.count == d else { continue }
            append(id: m.persistentModelID, isPinned: m.isPinned, vector: m.embedding, expectedDim: d)
        }
        dim = d
        return VectorIndexLoadResult(loadedCount: ids.count, mode: "loaded", diagnostic: nil)
    }

    func append(id: PersistentIdentifier, isPinned: Bool, vector: [Double]) {
        guard loaded, !vector.isEmpty else { return }
        if dim == 0 { dim = vector.count }
        guard vector.count == dim else { return }
        append(id: id, isPinned: isPinned, vector: vector, expectedDim: dim)
    }

    func removeAll(keepPinned: Bool) {
        guard loaded else { return }
        if !keepPinned {
            ids.removeAll(keepingCapacity: true)
            pinned.removeAll(keepingCapacity: true)
            matrix.removeAll(keepingCapacity: true)
            return
        }
        var keep = [Bool](repeating: true, count: ids.count)
        var removed = 0
        for i in 0..<ids.count where !pinned[i] {
            keep[i] = false
            removed += 1
        }
        if removed == 0 { return }
        var newIds: [PersistentIdentifier] = []
        var newPinned: [Bool] = []
        var newMatrix: [Float] = []
        for i in 0..<ids.count where keep[i] {
            newIds.append(ids[i])
            newPinned.append(pinned[i])
            let start = i * dim
            newMatrix.append(contentsOf: matrix[start..<(start + dim)])
        }
        ids = newIds
        pinned = newPinned
        matrix = newMatrix
    }

    func search(
        query: [Double],
        topK: Int,
        pinBonus: Float = 0.15
    ) -> [(id: PersistentIdentifier, score: Float)] {
        guard loaded, !ids.isEmpty, dim > 0, query.count == dim, topK > 0 else { return [] }
        let q = VectorMath.toFloat32Normalized(query)
        let rows = ids.count
        var scores = [Float](repeating: 0, count: rows)
        matrix.withUnsafeBufferPointer { m in
            q.withUnsafeBufferPointer { qp in
                scores.withUnsafeMutableBufferPointer { sp in
                    vDSP_mmul(
                        m.baseAddress!, 1,
                        qp.baseAddress!, 1,
                        sp.baseAddress!, 1,
                        vDSP_Length(rows),
                        1,
                        vDSP_Length(dim)
                    )
                }
            }
        }
        if pinBonus != 0 {
            for i in 0..<rows where pinned[i] { scores[i] += pinBonus }
        }
        return VectorMath.topK(scores: scores, k: topK, minScore: -Float.infinity) { _ in true }
            .map { (ids[$0.0], $0.1) }
    }

    private func append(id: PersistentIdentifier, isPinned: Bool, vector: [Double], expectedDim: Int) {
        ids.append(id)
        pinned.append(isPinned)
        matrix.reserveCapacity(matrix.count + expectedDim)
        matrix.append(contentsOf: VectorMath.toFloat32Normalized(vector))
    }
}

enum VectorMath {
    static func toFloat32Normalized(_ v: [Double]) -> [Float] {
        guard !v.isEmpty else { return [] }
        var out = [Float](repeating: 0, count: v.count)
        for i in 0..<v.count { out[i] = Float(v[i]) }
        var ss: Float = 0
        out.withUnsafeBufferPointer { bp in
            vDSP_svesq(bp.baseAddress!, 1, &ss, vDSP_Length(out.count))
        }
        let norm = sqrtf(ss)
        if norm > 1e-8 {
            var inv: Float = 1.0 / norm
            let count = vDSP_Length(out.count)
            out.withUnsafeMutableBufferPointer { bp in
                let base = bp.baseAddress!
                vDSP_vsmul(base, 1, &inv, base, 1, count)
            }
        }
        return out
    }

    /// Unordered bounded selection using a small max-heap style scan. O(n log k).
    static func topK(
        scores: [Float],
        k: Int,
        minScore: Float,
        includeIndex: (Int) -> Bool
    ) -> [(Int, Float)] {
        guard k > 0, !scores.isEmpty else { return [] }
        var heap: [(Int, Float)] = []
        heap.reserveCapacity(k + 1)
        for i in 0..<scores.count {
            let s = scores[i]
            if s < minScore { continue }
            if !includeIndex(i) { continue }
            if heap.count < k {
                heap.append((i, s))
                if heap.count == k { heap.sort { $0.1 < $1.1 } }
            } else if s > heap[0].1 {
                heap[0] = (i, s)
                var j = 0
                while true {
                    let l = 2 * j + 1, r = 2 * j + 2
                    var smallest = j
                    if l < heap.count && heap[l].1 < heap[smallest].1 { smallest = l }
                    if r < heap.count && heap[r].1 < heap[smallest].1 { smallest = r }
                    if smallest == j { break }
                    heap.swapAt(j, smallest)
                    j = smallest
                }
            }
        }
        return heap.sorted { $0.1 > $1.1 }
    }
}
