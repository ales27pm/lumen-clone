import Foundation
import SwiftData
import Accelerate
import OSLog

nonisolated struct VectorIndexLoadResult: Sendable, Equatable {
    let loadedCount: Int
    let mode: String
    let diagnostic: String?
}

nonisolated struct RAGEmbeddingIndexMetadata: Sendable, Equatable {
    let formatVersion: Int
    let modelIdentifier: String
    let dimension: Int
}

/// In-memory, pre-normalized Float32 vector index with Accelerate-accelerated search.
///
/// Design:
/// - Embeddings are stored on disk as `[Double]` on the SwiftData models. At first
///   search we load them into a contiguous Float32
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
        formatVersion: Int,
        modelIdentifier: String,
        dimension: Int
    ) -> VectorIndexLoadResult {
        let stagingPrefix = RAGChunk.replacementStagingSourceTypePrefix
        var descriptor = FetchDescriptor<RAGChunk>(
            predicate: #Predicate<RAGChunk> { !$0.sourceType.starts(with: stagingPrefix) }
        )
        descriptor.includePendingChanges = false
        let readContext = ModelContext(context.container)
        readContext.autosaveEnabled = false
        return ensureLoaded(
            formatVersion: formatVersion,
            modelIdentifier: modelIdentifier,
            dimension: dimension,
            fetch: { try readContext.fetch(descriptor) }
        )
    }

    @discardableResult
    func ensureLoaded(context: ModelContext, metadata: RAGEmbeddingIndexMetadata) -> VectorIndexLoadResult {
        ensureLoaded(
            context: context,
            formatVersion: metadata.formatVersion,
            modelIdentifier: metadata.modelIdentifier,
            dimension: metadata.dimension
        )
    }

    @discardableResult
    func ensureLoadedForTests(
        formatVersion: Int = SemanticEmbeddingText.formatVersion,
        modelIdentifier: String = RAGEmbeddingMetadata.unidentifiedModelIdentifier,
        dimension: Int? = nil,
        fetch: () throws -> [RAGChunk]
    ) -> VectorIndexLoadResult {
        ensureLoaded(formatVersion: formatVersion, modelIdentifier: modelIdentifier, dimension: dimension, fetch: fetch)
    }

    private func ensureLoaded(
        formatVersion: Int,
        modelIdentifier: String,
        dimension requiredDimension: Int?,
        fetch: () throws -> [RAGChunk]
    ) -> VectorIndexLoadResult {
        guard EmbeddingIndexLifecycle.semanticReadsAllowed else {
            invalidate()
            return VectorIndexLoadResult(loadedCount: 0, mode: "unavailable", diagnostic: "embedding_index_reset_pending")
        }
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
        for c in fetched where !c.isReplacementStaging
            && !c.embedding.isEmpty
            && c.embeddingFormatVersion == formatVersion
            && c.embeddingModelIdentifier == modelIdentifier
            && c.embeddingDimension == c.embedding.count
            && (requiredDimension == nil || c.embeddingDimension == requiredDimension) {
            if d == 0 { d = c.embedding.count }
            guard c.embedding.count == d else { continue }
            appendRow(id: c.persistentModelID, bucket: c.sourceType, vector: c.embedding, expectedDim: d)
        }
        dim = requiredDimension ?? d
        return VectorIndexLoadResult(loadedCount: ids.count, mode: "loaded", diagnostic: nil)
    }

    @discardableResult
    func append(
        id: PersistentIdentifier,
        bucket: String,
        vector: [Double],
        formatVersion: Int,
        modelIdentifier: String,
        dimension: Int
    ) -> Bool {
        guard EmbeddingIndexLifecycle.semanticReadsAllowed, loaded else { return false }
        guard loadedFormatVersion == formatVersion,
              loadedModelIdentifier == modelIdentifier,
              dim == dimension,
              !vector.isEmpty,
              vector.count == dimension else {
            invalidate()
            return false
        }
        appendRow(id: id, bucket: bucket, vector: vector, expectedDim: dim)
        return true
    }

    @discardableResult
    func append(
        id: PersistentIdentifier,
        bucket: String,
        vector: [Double],
        metadata: RAGEmbeddingIndexMetadata
    ) -> Bool {
        append(
            id: id,
            bucket: bucket,
            vector: vector,
            formatVersion: metadata.formatVersion,
            modelIdentifier: metadata.modelIdentifier,
            dimension: metadata.dimension
        )
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
        guard EmbeddingIndexLifecycle.semanticReadsAllowed, loaded, !ids.isEmpty, dim > 0, query.count == dim, topK > 0 else { return [] }
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
    private var loadedMetadata: RAGEmbeddingIndexMetadata?
    private var hasStaleEmbeddings = false

    private init() {}

    func invalidate() {
        ids.removeAll(keepingCapacity: false)
        pinned.removeAll(keepingCapacity: false)
        matrix.removeAll(keepingCapacity: false)
        dim = 0
        loaded = false
        loadedMetadata = nil
        hasStaleEmbeddings = false
    }

    @discardableResult
    func ensureLoaded(context: ModelContext, metadata: RAGEmbeddingIndexMetadata) -> VectorIndexLoadResult {
        let readContext = ModelContext(context.container)
        readContext.autosaveEnabled = false
        var descriptor = FetchDescriptor<MemoryItem>()
        descriptor.includePendingChanges = false
        return ensureLoaded(metadata: metadata, fetch: { try readContext.fetch(descriptor) })
    }

    @discardableResult
    func ensureLoadedForTests(
        metadata: RAGEmbeddingIndexMetadata? = nil,
        fetch: () throws -> [MemoryItem]
    ) -> VectorIndexLoadResult {
        ensureLoaded(metadata: metadata, fetch: fetch)
    }

    private func ensureLoaded(
        metadata: RAGEmbeddingIndexMetadata?,
        fetch: () throws -> [MemoryItem]
    ) -> VectorIndexLoadResult {
        guard EmbeddingIndexLifecycle.semanticReadsAllowed else {
            invalidate()
            return VectorIndexLoadResult(loadedCount: 0, mode: "unavailable", diagnostic: "embedding_index_reset_pending")
        }
        if loaded, loadedMetadata != metadata { invalidate() }
        guard !loaded else {
            return VectorIndexLoadResult(loadedCount: ids.count, mode: "already_loaded", diagnostic: hasStaleEmbeddings ? "memory_reindex_required" : nil)
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
        loadedMetadata = metadata
        guard let metadata, metadata.dimension > 0, !metadata.modelIdentifier.isEmpty else {
            hasStaleEmbeddings = !fetched.isEmpty
            return VectorIndexLoadResult(loadedCount: 0, mode: "loaded", diagnostic: hasStaleEmbeddings ? "memory_reindex_required" : nil)
        }
        dim = metadata.dimension
        var staleCount = 0
        for item in fetched {
            guard Self.matches(item, metadata: metadata) else { staleCount += 1; continue }
            append(id: item.persistentModelID, isPinned: item.isPinned, vector: item.embedding, expectedDim: dim)
        }
        hasStaleEmbeddings = staleCount > 0
        return VectorIndexLoadResult(loadedCount: ids.count, mode: "loaded", diagnostic: hasStaleEmbeddings ? "memory_reindex_required" : nil)
    }

    static func matches(_ item: MemoryItem, metadata: RAGEmbeddingIndexMetadata) -> Bool {
        item.embeddingFormatVersion == metadata.formatVersion
            && item.embeddingModelIdentifier == metadata.modelIdentifier
            && item.embeddingDimension == metadata.dimension
            && item.embedding.count == metadata.dimension
            && !item.embedding.isEmpty
            && item.embedding.allSatisfy(\.isFinite)
    }

    @discardableResult
    func append(id: PersistentIdentifier, isPinned: Bool, vector: [Double], metadata: RAGEmbeddingIndexMetadata) -> Bool {
        guard EmbeddingIndexLifecycle.semanticReadsAllowed, loaded, loadedMetadata == metadata, dim == metadata.dimension,
              !vector.isEmpty, vector.count == metadata.dimension, vector.allSatisfy(\.isFinite) else { return false }
        if ids.contains(id) { return true }
        append(id: id, isPinned: isPinned, vector: vector, expectedDim: dim)
        return true
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
        guard EmbeddingIndexLifecycle.semanticReadsAllowed, loaded, !ids.isEmpty, dim > 0, query.count == dim, topK > 0 else { return [] }
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
