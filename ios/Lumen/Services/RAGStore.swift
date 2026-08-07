import Foundation
import SwiftData
import PDFKit
import Photos
import OSLog

@MainActor
enum RAGStore {
    private static let logger = Logger(subsystem: "ai.lumen.app", category: "persistence")

    struct PendingVector {
        let chunk: RAGChunk
        let bucket: String
        let vector: [Double]
        let metadata: RAGEmbeddingIndexMetadata
    }

    struct PersistAndAppendResult: Equatable {
        enum IndexState: Equatable {
            case appended
            case reloaded
            case unavailable
        }

        let persistedCount: Int
        let indexState: IndexState
    }

    enum PersistenceError: Error, Equatable {
        case diskWriteBudgetDenied
    }

    private enum CorpusReplacementCommitResult {
        case committed(PersistAndAppendResult)
        case cancelled
        case maintenanceBudgetDenied
        case diskWriteBudgetDenied
        case failed(String)
    }

    private static func sourceLogID(_ value: String) -> String {
        String(RuntimeFallbackLogger.promptHash(value).prefix(12))
    }

    private static func requirePersistenceBudget() throws {
        guard DiskWriteBudget.shared.canWrite(bytes: 128 * 1024, category: .rag) else {
            throw PersistenceError.diskWriteBudgetDenied
        }
    }

    private static func persistAfterBudgetAuthorization(
        _ context: ModelContext,
        operation: String,
        scope: String
    ) throws {
        do {
            try context.save()
            DiskWriteBudget.shared.recordWrite(bytes: 128 * 1024, category: .rag)
        } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            throw error
        }
    }

    private static func commitStagedCorpusReplacement(
        context: ModelContext,
        replacing sourceTypes: Set<RAGSourceType>,
        operation: String,
        maintenanceReason: String,
        pending: inout [PendingVector],
        save: ((ModelContext, String, String) throws -> Void)? = nil
    ) -> CorpusReplacementCommitResult {
        guard !Task.isCancelled else { return .cancelled }
        guard ResourceBudgetGate.allowsMaintenance(reason: maintenanceReason) else {
            return .maintenanceBudgetDenied
        }
        let metadata = pending.first?.metadata
        guard metadata.map({ expected in pending.allSatisfy { $0.metadata == expected } }) ?? true else {
            return .failed("embedding_identity_changed_during_index")
        }

        let existing: [RAGChunk]
        do {
            existing = try context.fetch(FetchDescriptor<RAGChunk>()).filter {
                sourceTypes.contains($0.kind)
            }
        } catch {
            let diagnostic = "replacement_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            logger.error("rag_fetch_failed op=\(operation, privacy: .public) diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return .failed(diagnostic)
        }

        do {
            try requirePersistenceBudget()
        } catch PersistenceError.diskWriteBudgetDenied {
            return .diskWriteBudgetDenied
        } catch {
            return .failed("replacement_authorization_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
        }

        guard !Task.isCancelled else { return .cancelled }
        for item in pending {
            context.insert(item.chunk)
        }
        for chunk in existing {
            context.delete(chunk)
        }
        if Task.isCancelled {
            context.rollback()
            return .cancelled
        }

        do {
            if let save {
                try save(context, operation, "RAGChunk")
            } else {
                try persistAfterBudgetAuthorization(context, operation: operation, scope: "RAGChunk")
            }
        } catch PersistenceError.diskWriteBudgetDenied {
            context.rollback()
            return .diskWriteBudgetDenied
        } catch {
            context.rollback()
            let diagnostic = "replacement_persist_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            logger.error("persist_failed op=\(operation, privacy: .public) scope=RAGChunk diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return .failed(diagnostic)
        }

        let persistedCount = pending.count
        pending.removeAll(keepingCapacity: true)

        RAGVectorIndex.shared.invalidate()
        let indexState: PersistAndAppendResult.IndexState
        if let metadata {
            let reload = RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata)
            indexState = reload.mode == "failed" ? .unavailable : .reloaded
        } else {
            indexState = .reloaded
        }
        return .committed(PersistAndAppendResult(persistedCount: persistedCount, indexState: indexState))
    }

    static func persistAndAppendVectors(
        context: ModelContext,
        operation: String,
        scope: String = "RAGChunk",
        pending: inout [PendingVector],
        save: ((ModelContext, String, String) throws -> Void)? = nil
    ) -> PersistAndAppendResult? {
        let persistedCount = pending.count
        guard persistedCount > 0 else {
            return PersistAndAppendResult(persistedCount: 0, indexState: .appended)
        }
        if save == nil {
            do {
                try requirePersistenceBudget()
            } catch {
                return nil
            }
        }
        for item in pending {
            context.insert(item.chunk)
        }
        do {
            if let save {
                try save(context, operation, scope)
            } else {
                try persistAfterBudgetAuthorization(context, operation: operation, scope: scope)
            }
        } catch PersistenceError.diskWriteBudgetDenied {
            for item in pending {
                context.delete(item.chunk)
            }
            return nil
        } catch {
            for item in pending {
                context.delete(item.chunk)
            }
            pending.removeAll(keepingCapacity: true)
            return nil
        }
        var appendedAll = true
        for item in pending {
            guard RAGVectorIndex.shared.append(
                id: item.chunk.persistentModelID,
                bucket: item.bucket,
                vector: item.vector,
                metadata: item.metadata
            ) else {
                appendedAll = false
                break
            }
        }
        let indexState: PersistAndAppendResult.IndexState
        if appendedAll {
            indexState = .appended
        } else if let metadata = pending.first?.metadata,
                  pending.allSatisfy({ $0.metadata == metadata }) {
            RAGVectorIndex.shared.invalidate()
            let reload = RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata)
            indexState = reload.mode == "failed" ? .unavailable : .reloaded
        } else {
            RAGVectorIndex.shared.invalidate()
            indexState = .unavailable
        }
        pending.removeAll(keepingCapacity: true)
        return PersistAndAppendResult(persistedCount: persistedCount, indexState: indexState)
    }

    static func auditPersistence(operation: String, scope: String, save: () throws -> Void) -> Bool {
        do {
            try save()
            return true
        } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return false
        }
    }

    static let chunkSize = 600
    static let chunkOverlap = 80
    static let candidatePoolMultiplier = 8
    static let maxCandidatePool = 256
    static let minScore: Float = 0.12
    static let hybridLexicalCandidateMultiplier = 3
    static let hybridLexicalMaxCandidates = 64
    private static let semanticScoreWeight = 0.82
    private static let lexicalScoreWeight = 0.18
    private static let maxLexicalScore = 0.2

    struct SearchResult {
        let matches: [(chunk: RAGChunk, score: Double)]
        let mode: String
        let diagnostic: String?
    }

    struct CountsResult {
        let counts: [RAGSourceType: Int]
        let mode: String
        let diagnostic: String?
    }

    struct ChunkListResult {
        let chunks: [RAGChunk]
        let mode: String
        let diagnostic: String?
    }

    private struct LexicalSearchResult {
        let matches: [(RAGChunk, Double)]
        let diagnostic: String?
    }

    private struct ResolvedVectorCandidatesResult {
        let candidates: [(RAGChunk, Double)]
        let diagnostic: String?
    }

    enum IndexMode: String, Equatable {
        case indexed
        case cleared
        case skipped
        case failed
        case partial
    }

    struct IndexResult: Equatable {
        let indexedCount: Int
        let mode: IndexMode
        let diagnostic: String?

        var didIndexAllChunks: Bool {
            mode == .indexed || mode == .cleared
        }
    }

    static func persistPendingVectorsForEarlyExit(
        context: ModelContext,
        operation: String,
        diagnostic: String,
        pending: inout [PendingVector],
        previouslyPersistedCount: Int = 0,
        save: ((ModelContext, String, String) throws -> Void)? = nil
    ) -> IndexResult {
        guard !pending.isEmpty else {
            return IndexResult(
                indexedCount: previouslyPersistedCount,
                mode: previouslyPersistedCount > 0 ? .partial : .failed,
                diagnostic: diagnostic
            )
        }
        guard let result = persistAndAppendVectors(
            context: context,
            operation: operation,
            pending: &pending,
            save: save
        ) else {
            let persistenceDiagnostic = pending.isEmpty ? ";persist_failed" : ";persistence_deferred"
            return IndexResult(
                indexedCount: previouslyPersistedCount,
                mode: previouslyPersistedCount > 0 ? .partial : .failed,
                diagnostic: diagnostic + persistenceDiagnostic
            )
        }
        let totalPersistedCount = previouslyPersistedCount + result.persistedCount
        let suffix = result.indexState == .unavailable ? ";vector_index_reload_failed" : ""
        return IndexResult(
            indexedCount: totalPersistedCount,
            mode: totalPersistedCount > 0 ? .partial : .failed,
            diagnostic: diagnostic + suffix
        )
    }

    static func prepareEmbeddingMetadata(
        _ metadata: RAGEmbeddingIndexMetadata,
        active: inout RAGEmbeddingIndexMetadata?,
        pending: inout [PendingVector],
        context: ModelContext,
        identityChangeOperation: String,
        previouslyPersistedCount: Int = 0,
        save: ((ModelContext, String, String) throws -> Void)? = nil
    ) -> IndexResult? {
        if let active, active != metadata {
            let result = persistPendingVectorsForEarlyExit(
                context: context,
                operation: identityChangeOperation,
                diagnostic: "embedding_identity_changed_during_index",
                pending: &pending,
                previouslyPersistedCount: previouslyPersistedCount,
                save: save
            )
            RAGVectorIndex.shared.invalidate()
            return result
        }
        if active == nil {
            _ = RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata)
            active = metadata
        }
        return nil
    }

    struct FileExtractionResult: Equatable {
        let text: String?
        let sourceType: RAGSourceType?
        let mode: IndexMode
        let diagnostic: String?
    }

    private enum FileExtractionError: Error, Equatable {
        case pdfOpenFailed
    }

    static func search(
        query: String,
        context: ModelContext,
        limit: Int = 5,
        sourceTypes: Set<RAGSourceType>? = nil
    ) async -> [(chunk: RAGChunk, score: Double)] {
        await searchWithDiagnostics(query: query, context: context, limit: limit, sourceTypes: sourceTypes).matches
    }

    static func searchWithDiagnostics(
        query: String,
        context: ModelContext,
        limit: Int = 5,
        sourceTypes: Set<RAGSourceType>? = nil
    ) async -> SearchResult {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, limit > 0 else {
            return SearchResult(matches: [], mode: "empty_query", diagnostic: "empty_query")
        }
        let allowed: Set<String>? = sourceTypes.map { Set($0.map(\.rawValue)) }

        if let budgetDenial = ResourceBudgetGate.budgetDenialReason(policy: .embedding, reason: "rag.search") {
            logger.error("rag_embedding_budget_denied op=search reason=\(budgetDenial, privacy: .public)")
            let fallback = lexicalSearchResult(query: trimmed, context: context, allowed: allowed, limit: limit)
            return SearchResult(
                matches: fallback.matches,
                mode: "lexical_fallback",
                diagnostic: combinedDiagnostic(primary: budgetDenial, secondary: fallback.diagnostic)
            )
        }

        let queryEmbedding: EmbeddingRuntimeResult
        do {
            queryEmbedding = try await AssistantKernel.runEmbeddingWithIdentity(text: SemanticEmbeddingText.query(trimmed))
        } catch {
            logger.error("rag_embedding_failed op=search error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            let fallback = lexicalSearchResult(query: trimmed, context: context, allowed: allowed, limit: limit)
            return SearchResult(
                matches: fallback.matches,
                mode: "lexical_fallback",
                diagnostic: combinedDiagnostic(
                    primary: "embedding_failed:\(RuntimeMetricErrorSanitizer.code(for: error))",
                    secondary: fallback.diagnostic
                )
            )
        }
        let queryVec = queryEmbedding.vector
        guard !queryVec.isEmpty else {
            logger.error("rag_embedding_empty op=search")
            let fallback = lexicalSearchResult(query: trimmed, context: context, allowed: allowed, limit: limit)
            return SearchResult(
                matches: fallback.matches,
                mode: "lexical_fallback",
                diagnostic: combinedDiagnostic(primary: "embedding_empty", secondary: fallback.diagnostic)
            )
        }

        let embeddingModelIdentifier = queryEmbedding.modelIdentifier
        let vectorLoad = RAGVectorIndex.shared.ensureLoaded(
            context: context,
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: embeddingModelIdentifier,
            dimension: queryVec.count
        )
        var diagnostic = vectorLoad.diagnostic
        if hasStaleEmbeddings(context: context, modelIdentifier: embeddingModelIdentifier, dimension: queryVec.count) {
            diagnostic = combinedDiagnostic(primary: diagnostic ?? "rag_reindex_required", secondary: diagnostic == nil ? nil : "rag_reindex_required")
        }

        let k = min(max(limit * candidatePoolMultiplier, limit + 4), maxCandidatePool)

        let vectorHits = RAGVectorIndex.shared.search(
            query: queryVec,
            topK: k,
            allowedBuckets: allowed,
            minScore: minScore
        )
        let resolvedCandidates = resolvedVectorCandidatesResult(vectorHits: vectorHits, context: context)
        let semanticCandidates = resolvedCandidates.candidates
        if let resolvedDiagnostic = resolvedCandidates.diagnostic {
            diagnostic = diagnostic.map { combinedDiagnostic(primary: $0, secondary: resolvedDiagnostic) } ?? resolvedDiagnostic
        }

        let lexicalLimit = min(
            max(limit * hybridLexicalCandidateMultiplier, limit + 8),
            hybridLexicalMaxCandidates
        )
        let lexical = lexicalScoreResult(query: trimmed, context: context, allowed: allowed, exclude: [], limit: lexicalLimit)
        if let lexicalDiagnostic = lexical.diagnostic {
            diagnostic = diagnostic.map { combinedDiagnostic(primary: $0, secondary: lexicalDiagnostic) } ?? lexicalDiagnostic
        }

        let sorted = hybridMergedCandidates(semantic: semanticCandidates, lexical: lexical.matches, limit: limit)
        let mode = lexical.matches.isEmpty ? "semantic" : "semantic_hybrid"
        return SearchResult(matches: sorted, mode: mode, diagnostic: diagnostic)
    }

    static func lexicalSearch(
        query: String,
        context: ModelContext,
        sourceTypes: Set<RAGSourceType>? = nil,
        limit: Int
    ) -> [(RAGChunk, Double)] {
        let allowed: Set<String>? = sourceTypes.map { Set($0.map(\.rawValue)) }
        return lexicalSearch(query: query, context: context, allowed: allowed, limit: limit)
    }

    private static func lexicalSearch(
        query: String,
        context: ModelContext,
        allowed: Set<String>?,
        limit: Int
    ) -> [(RAGChunk, Double)] {
        lexicalSearchResult(query: query, context: context, allowed: allowed, limit: limit).matches
    }

    private static func lexicalSearchResult(
        query: String,
        context: ModelContext,
        allowed: Set<String>?,
        limit: Int
    ) -> LexicalSearchResult {
        lexicalScoreResult(query: query, context: context, allowed: allowed, exclude: [], limit: limit)
    }

    private static func lexicalScore(
        query: String,
        context: ModelContext,
        allowed: Set<String>?,
        exclude: Set<PersistentIdentifier>,
        limit: Int
    ) -> [(RAGChunk, Double)] {
        lexicalScoreResult(query: query, context: context, allowed: allowed, exclude: exclude, limit: limit).matches
    }

    private static func lexicalScoreResult(
        query: String,
        context: ModelContext,
        allowed: Set<String>?,
        exclude: Set<PersistentIdentifier>,
        limit: Int
    ) -> LexicalSearchResult {
        guard limit > 0 else { return LexicalSearchResult(matches: [], diagnostic: "lexical_limit_empty") }
        let terms = query.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 3 }
        guard !terms.isEmpty else { return LexicalSearchResult(matches: [], diagnostic: "lexical_empty_terms") }
        var descriptor = FetchDescriptor<RAGChunk>()
        descriptor.fetchLimit = 400
        let all: [RAGChunk]
        do {
            all = try context.fetch(descriptor)
        } catch {
            let diagnostic = "lexical_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            logger.error("rag_fetch_failed op=lexicalSearch diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return LexicalSearchResult(matches: [], diagnostic: diagnostic)
        }
        var scored: [(RAGChunk, Double)] = []
        for c in all where !exclude.contains(c.persistentModelID) {
            if let allowed, !allowed.contains(c.sourceType) { continue }
            let haystack = c.content.lowercased()
            var hits = 0
            for t in terms where haystack.contains(t) { hits += 1 }
            if hits > 0 {
                let score = Double(hits) / Double(terms.count) * 0.2
                scored.append((c, score))
            }
        }
        return LexicalSearchResult(matches: scored.sorted { $0.1 > $1.1 }.prefix(limit).map { $0 }, diagnostic: nil)
    }

    private static func mergedVectorHits(
        primary: [(id: PersistentIdentifier, score: Float)],
        secondary: [(id: PersistentIdentifier, score: Float)],
        limit: Int
    ) -> [(id: PersistentIdentifier, score: Float)] {
        guard limit > 0 else { return [] }
        var byID: [PersistentIdentifier: Float] = [:]
        for hit in primary {
            byID[hit.id] = max(byID[hit.id] ?? -Float.infinity, hit.score)
        }
        for hit in secondary {
            byID[hit.id] = max(byID[hit.id] ?? -Float.infinity, hit.score)
        }
        return byID
            .map { (id: $0.key, score: $0.value) }
            .sorted { $0.score > $1.score }
            .prefix(limit)
            .map { $0 }
    }

    static func hybridMergedCandidates(
        semantic: [(RAGChunk, Double)],
        lexical: [(RAGChunk, Double)],
        limit: Int
    ) -> [(RAGChunk, Double)] {
        guard limit > 0 else { return [] }
        var merged: [PersistentIdentifier: (chunk: RAGChunk, semantic: Double?, lexical: Double?)] = [:]
        for (chunk, score) in semantic {
            let id = chunk.persistentModelID
            var current = merged[id] ?? (chunk: chunk, semantic: nil, lexical: nil)
            current.semantic = max(current.semantic ?? -Double.infinity, score)
            merged[id] = current
        }
        for (chunk, score) in lexical {
            let id = chunk.persistentModelID
            var current = merged[id] ?? (chunk: chunk, semantic: nil, lexical: nil)
            current.lexical = max(current.lexical ?? -Double.infinity, score)
            merged[id] = current
        }
        return merged.values
            .map { entry -> (RAGChunk, Double) in
                let semanticScore = max(0, entry.semantic ?? 0)
                let lexicalScore = min(1, max(0, (entry.lexical ?? 0) / maxLexicalScore))
                let score: Double
                if entry.semantic == nil {
                    score = lexicalScore * lexicalScoreWeight
                } else {
                    score = (semanticScore * semanticScoreWeight) + (lexicalScore * lexicalScoreWeight)
                }
                return (entry.chunk, score)
            }
            .sorted { lhs, rhs in
                if lhs.1 == rhs.1 {
                    return lhs.0.createdAt > rhs.0.createdAt
                }
                return lhs.1 > rhs.1
            }
            .prefix(limit)
            .map { $0 }
    }

    private static func combinedDiagnostic(primary: String, secondary: String?) -> String {
        guard let secondary, !secondary.isEmpty else { return primary }
        return "\(primary);\(secondary)"
    }

    private static func hasStaleEmbeddings(context: ModelContext, modelIdentifier: String, dimension: Int) -> Bool {
        guard let chunks = try? context.fetch(FetchDescriptor<RAGChunk>()) else { return false }
        return chunks.contains {
            !$0.embedding.isEmpty && (
                $0.embeddingFormatVersion != SemanticEmbeddingText.formatVersion
                    || $0.embeddingModelIdentifier != modelIdentifier
                    || $0.embeddingDimension != dimension
                    || $0.embeddingDimension != $0.embedding.count
            )
        }
    }

    private static func embeddingMetadata(for result: EmbeddingRuntimeResult) -> RAGEmbeddingIndexMetadata {
        RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: result.modelIdentifier,
            dimension: result.vector.count
        )
    }

    static func resolvedVectorCandidates(
        vectorHits: [(id: PersistentIdentifier, score: Float)],
        context: ModelContext
    ) -> [(RAGChunk, Double)] {
        let chunksByID = fetchedChunksByPersistentIDResult(context: context).chunksByID
        var candidates: [(RAGChunk, Double)] = []
        candidates.reserveCapacity(vectorHits.count)
        for hit in vectorHits {
            if let chunk = chunksByID[hit.id] {
                candidates.append((chunk, Double(hit.score)))
            }
        }
        return candidates
    }

    private static func resolvedVectorCandidatesResult(
        vectorHits: [(id: PersistentIdentifier, score: Float)],
        context: ModelContext
    ) -> ResolvedVectorCandidatesResult {
        let fetch = fetchedChunksByPersistentIDResult(context: context)
        var candidates: [(RAGChunk, Double)] = []
        candidates.reserveCapacity(vectorHits.count)
        for hit in vectorHits {
            if let chunk = fetch.chunksByID[hit.id] {
                candidates.append((chunk, Double(hit.score)))
            }
        }
        return ResolvedVectorCandidatesResult(candidates: candidates, diagnostic: fetch.diagnostic)
    }

    private static func fetchedChunksByPersistentIDResult(context: ModelContext) -> (chunksByID: [PersistentIdentifier: RAGChunk], diagnostic: String?) {
        let chunks: [RAGChunk]
        do {
            chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        } catch {
            let diagnostic = "semantic_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            logger.error("rag_fetch_failed op=resolveVectorCandidates diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return ([:], diagnostic)
        }
        var byID: [PersistentIdentifier: RAGChunk] = [:]
        byID.reserveCapacity(chunks.count)
        for chunk in chunks {
            byID[chunk.persistentModelID] = chunk
        }
        return (byID, nil)
    }

    static func counts(context: ModelContext) -> [RAGSourceType: Int] {
        countsWithDiagnostics(context: context).counts
    }

    static func countsWithDiagnostics(context: ModelContext) -> CountsResult {
        let all: [RAGChunk]
        do {
            all = try context.fetch(FetchDescriptor<RAGChunk>())
        } catch {
            let diagnostic = "counts_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            logger.error("rag_fetch_failed op=counts diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return CountsResult(counts: [:], mode: "failed", diagnostic: diagnostic)
        }
        var out: [RAGSourceType: Int] = [:]
        for c in all { out[c.kind, default: 0] += 1 }
        return CountsResult(counts: out, mode: "loaded", diagnostic: nil)
    }

    private static func wipe(_ types: Set<RAGSourceType>?, context: ModelContext) throws {
        let all = try context.fetch(FetchDescriptor<RAGChunk>())
        try requirePersistenceBudget()
        for c in all {
            if types == nil || types?.contains(c.kind) == true { context.delete(c) }
        }
        try persistAfterBudgetAuthorization(context, operation: "wipe", scope: "RAGChunk")
        if let types {
            for type in types {
                RAGVectorIndex.shared.removeBucket(type.rawValue)
            }
        } else {
            RAGVectorIndex.shared.removeAll()
        }
    }

    static func wipe(_ type: RAGSourceType?, context: ModelContext) throws {
        try wipe(type.map { Set([$0]) }, context: context)
    }

    static func chunks(for type: RAGSourceType, context: ModelContext) -> [RAGChunk] {
        chunksWithDiagnostics(for: type, context: context).chunks
    }

    static func chunksWithDiagnostics(for type: RAGSourceType, context: ModelContext) -> ChunkListResult {
        let raw: [RAGChunk]
        do {
            raw = try context.fetch(FetchDescriptor<RAGChunk>())
        } catch {
            let diagnostic = "chunks_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            logger.error("rag_fetch_failed op=chunks type=\(type.rawValue, privacy: .public) diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return ChunkListResult(chunks: [], mode: "failed", diagnostic: diagnostic)
        }
        return ChunkListResult(
            chunks: raw.filter { $0.kind == type }.sorted { $0.createdAt > $1.createdAt },
            mode: "loaded",
            diagnostic: nil
        )
    }

    // MARK: - File / PDF indexing

    static func indexImportedFiles(context: ModelContext, progress: ((Double) -> Void)? = nil) async -> Int {
        await indexImportedFilesWithDiagnostics(context: context, progress: progress).indexedCount
    }

    static func indexImportedFilesWithDiagnostics(
        context: ModelContext,
        progress: ((Double) -> Void)? = nil,
        importedFilesResult: FileStore.ImportedFilesResult? = nil,
        embed: (String) async throws -> EmbeddingRuntimeResult = { text in
            try await AssistantKernel.runEmbeddingWithIdentity(text: text)
        },
        save: ((ModelContext, String, String) throws -> Void)? = nil
    ) async -> IndexResult {
        let imports = importedFilesResult ?? FileStore.importedFilesWithDiagnostics()
        if imports.mode == "failed" {
            return IndexResult(indexedCount: 0, mode: .failed, diagnostic: imports.diagnostic ?? "imports_list_failed")
        }
        let files = imports.files
        var pendingVectors: [PendingVector] = []
        var activeEmbeddingMetadata: RAGEmbeddingIndexMetadata?
        let cpuToken = CPUWatchdogGuard.shared.begin(category: .rag)
        defer { CPUWatchdogGuard.shared.end(token: cpuToken) }

        for (idx, url) in files.enumerated() {
            let staged = await stageFileVectors(
                url: url,
                pending: &pendingVectors,
                activeEmbeddingMetadata: &activeEmbeddingMetadata,
                embed: embed
            )
            guard staged.didIndexAllChunks else {
                logger.error("rag_index_file_degraded op=indexImportedFiles source_hash=\(Self.sourceLogID(url.lastPathComponent), privacy: .public) status=\(staged.mode.rawValue, privacy: .public) diagnostic=\(staged.diagnostic ?? "none", privacy: .public)")
                return IndexResult(indexedCount: 0, mode: staged.mode, diagnostic: staged.diagnostic)
            }
            progress?(Double(idx + 1) / Double(max(1, files.count)))
        }

        let commit = commitStagedCorpusReplacement(
            context: context,
            replacing: [.file, .pdf],
            operation: "indexImportedFiles.replace",
            maintenanceReason: "rag.indexImportedFiles",
            pending: &pendingVectors,
            save: save
        )
        switch commit {
        case .committed(let result):
            if result.indexState == .unavailable {
                return IndexResult(indexedCount: result.persistedCount, mode: .partial, diagnostic: "vector_index_reload_failed")
            }
            guard !files.isEmpty else {
                return IndexResult(indexedCount: 0, mode: .cleared, diagnostic: imports.diagnostic ?? "no_imported_files")
            }
            return IndexResult(indexedCount: result.persistedCount, mode: .indexed, diagnostic: nil)
        case .cancelled:
            return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "cancelled")
        case .maintenanceBudgetDenied:
            return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "maintenance_budget_denied")
        case .diskWriteBudgetDenied:
            return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "cleanup_deferred:disk_write_budget_denied")
        case .failed(let diagnostic):
            return IndexResult(indexedCount: 0, mode: .failed, diagnostic: diagnostic)
        }
    }

    private static func stageFileVectors(
        url: URL,
        pending: inout [PendingVector],
        activeEmbeddingMetadata: inout RAGEmbeddingIndexMetadata?,
        embed: (String) async throws -> EmbeddingRuntimeResult
    ) async -> IndexResult {
        let name = url.lastPathComponent
        let extracted = extractFileTextWithDiagnostics(url: url)
        guard extracted.mode != .failed, let text = extracted.text, let type = extracted.sourceType else {
            return IndexResult(indexedCount: 0, mode: .failed, diagnostic: extracted.diagnostic ?? "file_extraction_failed")
        }

        let pieces = chunkText(text)
        guard !pieces.isEmpty else {
            return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "empty_text")
        }
        let initialPendingCount = pending.count
        for (index, piece) in pieces.enumerated() {
            if Task.isCancelled {
                return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "cancelled")
            }
            guard !CPUWatchdogGuard.shared.shouldDegrade(category: .rag),
                  ResourceBudgetGate.allowsMaintenance(reason: "rag.indexImportedFiles") else {
                return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "maintenance_budget_denied")
            }

            let embeddingResult: EmbeddingRuntimeResult
            do {
                let embeddingText = SemanticEmbeddingText.document(
                    content: piece,
                    sourceName: name,
                    sourceType: type.rawValue,
                    chunkIndex: index
                )
                embeddingResult = try await embed(embeddingText)
            } catch {
                let diagnostic = "embedding_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
                logger.error("rag_embedding_failed op=indexImportedFiles.stage source_hash=\(Self.sourceLogID(name), privacy: .public) diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
                return IndexResult(indexedCount: 0, mode: .failed, diagnostic: diagnostic)
            }
            let embedding = embeddingResult.vector
            guard !embedding.isEmpty else {
                logger.error("rag_embedding_empty op=indexImportedFiles.stage source_hash=\(Self.sourceLogID(name), privacy: .public)")
                return IndexResult(indexedCount: 0, mode: .failed, diagnostic: "embedding_empty")
            }
            let metadata = embeddingMetadata(for: embeddingResult)
            if let activeEmbeddingMetadata, activeEmbeddingMetadata != metadata {
                return IndexResult(indexedCount: 0, mode: .failed, diagnostic: "embedding_identity_changed_during_index")
            }
            if activeEmbeddingMetadata == nil {
                activeEmbeddingMetadata = metadata
            }
            let chunk = RAGChunk(
                content: piece,
                sourceType: type,
                sourceName: name,
                sourceRef: url.path,
                chunkIndex: index,
                embedding: embedding,
                embeddingFormatVersion: metadata.formatVersion,
                embeddingModelIdentifier: metadata.modelIdentifier,
                embeddingDimension: metadata.dimension
            )
            pending.append(PendingVector(chunk: chunk, bucket: type.rawValue, vector: embedding, metadata: metadata))
        }
        return IndexResult(indexedCount: pending.count - initialPendingCount, mode: .indexed, diagnostic: nil)
    }

    static func indexFile(url: URL, context: ModelContext) async -> Int {
        await indexFileWithDiagnostics(url: url, context: context).indexedCount
    }

    static func indexFileWithDiagnostics(
        url: URL,
        context: ModelContext,
        embed: (String) async throws -> EmbeddingRuntimeResult = { text in
            try await AssistantKernel.runEmbeddingWithIdentity(text: text)
        },
        save: ((ModelContext, String, String) throws -> Void)? = nil
    ) async -> IndexResult {
        let name = url.lastPathComponent
        let extracted = extractFileTextWithDiagnostics(url: url)
        guard extracted.mode != .failed, let text = extracted.text, let type = extracted.sourceType else {
            return IndexResult(indexedCount: 0, mode: .failed, diagnostic: extracted.diagnostic ?? "file_extraction_failed")
        }

        let pieces = chunkText(text)
        guard !pieces.isEmpty else {
            return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "empty_text")
        }
        var count = 0
        var persistedCount = 0
        var pendingVectors: [PendingVector] = []
        var activeEmbeddingMetadata: RAGEmbeddingIndexMetadata?
        let cpuToken = CPUWatchdogGuard.shared.begin(category: .rag)
        defer { CPUWatchdogGuard.shared.end(token: cpuToken) }
        for (i, piece) in pieces.enumerated() {
            if Task.isCancelled || CPUWatchdogGuard.shared.shouldDegrade(category: .rag) || !ResourceBudgetGate.allowsMaintenance(reason: "rag.indexFile") {
                break
            }
            let embeddingResult: EmbeddingRuntimeResult
            do {
                let embeddingText = SemanticEmbeddingText.document(
                    content: piece,
                    sourceName: name,
                    sourceType: type.rawValue,
                    chunkIndex: i
                )
                embeddingResult = try await embed(embeddingText)
            } catch {
                let diagnostic = "embedding_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
                logger.error("rag_embedding_failed op=indexFile source_hash=\(Self.sourceLogID(name), privacy: .public) diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
                return persistPendingVectorsForEarlyExit(
                    context: context,
                    operation: "indexFile.embeddingFailure",
                    diagnostic: diagnostic,
                    pending: &pendingVectors,
                    previouslyPersistedCount: persistedCount,
                    save: save
                )
            }
            let emb = embeddingResult.vector
            guard !emb.isEmpty else {
                logger.error("rag_embedding_empty op=indexFile source_hash=\(Self.sourceLogID(name), privacy: .public)")
                return persistPendingVectorsForEarlyExit(
                    context: context,
                    operation: "indexFile.emptyEmbedding",
                    diagnostic: "embedding_empty",
                    pending: &pendingVectors,
                    previouslyPersistedCount: persistedCount,
                    save: save
                )
            }
            let embeddingMetadata = embeddingMetadata(for: embeddingResult)
            if let earlyExit = prepareEmbeddingMetadata(
                embeddingMetadata,
                active: &activeEmbeddingMetadata,
                pending: &pendingVectors,
                context: context,
                identityChangeOperation: "indexFile.embeddingIdentityChanged",
                previouslyPersistedCount: persistedCount,
                save: save
            ) {
                return earlyExit
            }
            let chunk = RAGChunk(
                content: piece,
                sourceType: type,
                sourceName: name,
                sourceRef: url.path,
                chunkIndex: i,
                embedding: emb,
                embeddingFormatVersion: embeddingMetadata.formatVersion,
                embeddingModelIdentifier: embeddingMetadata.modelIdentifier,
                embeddingDimension: embeddingMetadata.dimension
            )
            pendingVectors.append(PendingVector(chunk: chunk, bucket: type.rawValue, vector: emb, metadata: embeddingMetadata))
            if i % 8 == 7 {
                guard let batchResult = persistAndAppendVectors(context: context, operation: "indexFile.batch", pending: &pendingVectors, save: save) else {
                    logger.error("rag_index_partial_failure op=indexFile source_hash=\(Self.sourceLogID(name), privacy: .public) persisted=\(persistedCount, privacy: .public) attempted=\(count + 1, privacy: .public)")
                    return IndexResult(indexedCount: persistedCount, mode: persistedCount > 0 ? .partial : .failed, diagnostic: "persist_failed")
                }
                persistedCount += batchResult.persistedCount
                if batchResult.indexState == .unavailable {
                    return IndexResult(indexedCount: persistedCount, mode: .partial, diagnostic: "vector_index_reload_failed")
                }
            }
            count += 1
        }
        guard let finalResult = persistAndAppendVectors(context: context, operation: "indexFile.complete", pending: &pendingVectors, save: save) else {
            logger.error("rag_index_partial_failure op=indexFile source_hash=\(Self.sourceLogID(name), privacy: .public) persisted=\(persistedCount, privacy: .public) attempted=\(count, privacy: .public)")
            return IndexResult(indexedCount: persistedCount, mode: persistedCount > 0 ? .partial : .failed, diagnostic: "persist_failed")
        }
        persistedCount += finalResult.persistedCount
        if finalResult.indexState == .unavailable {
            return IndexResult(indexedCount: persistedCount, mode: .partial, diagnostic: "vector_index_reload_failed")
        }
        if persistedCount < pieces.count {
            return IndexResult(indexedCount: persistedCount, mode: persistedCount > 0 ? .partial : .failed, diagnostic: "maintenance_budget_or_cancellation")
        }
        return IndexResult(indexedCount: persistedCount, mode: .indexed, diagnostic: nil)
    }

    static func extractFileTextWithDiagnosticsForTests(
        url: URL,
        readData: (URL) throws -> Data,
        attributedString: (Data, [NSAttributedString.DocumentReadingOptionKey: Any]) throws -> NSAttributedString,
        pdfText: @escaping (URL) throws -> String
    ) -> FileExtractionResult {
        extractFileTextWithDiagnostics(
            url: url,
            readData: readData,
            attributedString: attributedString,
            pdfText: pdfText
        )
    }

    private static func extractFileTextWithDiagnostics(
        url: URL,
        readData: (URL) throws -> Data = { try Data(contentsOf: $0) },
        attributedString: (Data, [NSAttributedString.DocumentReadingOptionKey: Any]) throws -> NSAttributedString = { data, options in
            try NSAttributedString(data: data, options: options, documentAttributes: nil)
        },
        pdfText: ((URL) throws -> String)? = nil
    ) -> FileExtractionResult {
        let ext = url.pathExtension.lowercased()
        if ext == "pdf" {
            do {
                let text = try pdfText?(url) ?? extractPDFText(url: url)
                return FileExtractionResult(text: text, sourceType: .pdf, mode: .indexed, diagnostic: nil)
            } catch {
                return FileExtractionResult(text: nil, sourceType: nil, mode: .failed, diagnostic: "pdf_open_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
            }
        }

        let data: Data
        do {
            data = try readData(url)
        } catch {
            let prefix = (ext == "rtf" || ext == "rtfd") ? "rtf_read_failed" : "file_read_failed"
            return FileExtractionResult(text: nil, sourceType: nil, mode: .failed, diagnostic: "\(prefix):\(RuntimeMetricErrorSanitizer.code(for: error))")
        }

        if ext == "rtf" || ext == "rtfd" {
            do {
                let attr = try attributedString(data, [.documentType: NSAttributedString.DocumentType.rtf])
                return FileExtractionResult(text: attr.string, sourceType: .file, mode: .indexed, diagnostic: nil)
            } catch {
                return FileExtractionResult(text: nil, sourceType: nil, mode: .failed, diagnostic: "rtf_decode_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
            }
        }

        if let utf8 = String(data: data, encoding: .utf8) {
            return FileExtractionResult(text: utf8, sourceType: .file, mode: .indexed, diagnostic: nil)
        }
        if let ascii = String(data: data, encoding: .isoLatin1) {
            return FileExtractionResult(text: ascii, sourceType: .file, mode: .indexed, diagnostic: nil)
        }
        do {
            let attr = try attributedString(data, [:])
            return FileExtractionResult(text: attr.string, sourceType: .file, mode: .indexed, diagnostic: nil)
        } catch {
            return FileExtractionResult(text: nil, sourceType: nil, mode: .failed, diagnostic: "text_decode_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
        }
    }

    private static func extractPDFText(url: URL) throws -> String {
        guard let pdf = PDFDocument(url: url) else {
            throw FileExtractionError.pdfOpenFailed
        }
        var combined = ""
        for i in 0..<pdf.pageCount {
            combined += pdf.page(at: i)?.string ?? ""
            combined += "\n\n"
        }
        return combined
    }

    // MARK: - Photos metadata

    static func indexPhotos(monthsBack: Int = 6, context: ModelContext) async -> Int {
        await indexPhotosWithDiagnostics(monthsBack: monthsBack, context: context).indexedCount
    }

    struct PhotoLibrarySnapshot {
        let authorizationStatus: PHAuthorizationStatus
        let assets: [PHAsset]
    }

    static func indexPhotosWithDiagnostics(
        monthsBack: Int = 6,
        context: ModelContext,
        photoLibrarySnapshot: PhotoLibrarySnapshot? = nil,
        embed: (String) async throws -> EmbeddingRuntimeResult = { text in
            try await AssistantKernel.runEmbeddingWithIdentity(text: text)
        },
        save: ((ModelContext, String, String) throws -> Void)? = nil
    ) async -> IndexResult {
        let snapshot: PhotoLibrarySnapshot
        if let photoLibrarySnapshot {
            snapshot = photoLibrarySnapshot
        } else {
            let status = await withCheckedContinuation { (cont: CheckedContinuation<PHAuthorizationStatus, Never>) in
                PHPhotoLibrary.requestAuthorization(for: .readWrite) { cont.resume(returning: $0) }
            }
            guard status == .authorized || status == .limited else {
                return IndexResult(indexedCount: 0, mode: .failed, diagnostic: "photos_permission_denied:\(photoAuthorizationDiagnostic(status))")
            }
            let start = Calendar.current.date(byAdding: .month, value: -monthsBack, to: Date()) ?? Date()
            let options = PHFetchOptions()
            options.predicate = NSPredicate(format: "creationDate >= %@", start as NSDate)
            options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
            options.fetchLimit = 2000
            let fetch = PHAsset.fetchAssets(with: options)

            var assets: [PHAsset] = []
            fetch.enumerateObjects { asset, _, _ in assets.append(asset) }
            snapshot = PhotoLibrarySnapshot(authorizationStatus: status, assets: assets)
        }
        let status = snapshot.authorizationStatus
        guard status == .authorized || status == .limited else {
            return IndexResult(indexedCount: 0, mode: .failed, diagnostic: "photos_permission_denied:\(photoAuthorizationDiagnostic(status))")
        }
        let assets = snapshot.assets

        var selfieIDs: Set<String> = []
        if !assets.isEmpty {
            let selfieAlbums = PHAssetCollection.fetchAssetCollections(with: .smartAlbum, subtype: .smartAlbumSelfPortraits, options: nil)
            selfieAlbums.enumerateObjects { coll, _, _ in
                let a = PHAsset.fetchAssets(in: coll, options: nil)
                a.enumerateObjects { asset, _, _ in selfieIDs.insert(asset.localIdentifier) }
            }
        }

        var buckets: [String: [PHAsset]] = [:]
        let df = DateFormatter(); df.dateFormat = "yyyy-MM"
        for a in assets {
            let key = df.string(from: a.creationDate ?? Date())
            buckets[key, default: []].append(a)
        }

        var pendingVectors: [PendingVector] = []
        var activeEmbeddingMetadata: RAGEmbeddingIndexMetadata?
        let cpuToken = CPUWatchdogGuard.shared.begin(category: .rag)
        defer { CPUWatchdogGuard.shared.end(token: cpuToken) }
        for (month, items) in buckets {
            if Task.isCancelled {
                return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "cancelled")
            }
            guard !CPUWatchdogGuard.shared.shouldDegrade(category: .rag),
                  ResourceBudgetGate.allowsMaintenance(reason: "rag.indexPhotos") else {
                return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "maintenance_budget_denied")
            }
            let favorites = items.filter(\.isFavorite).count
            let videos = items.filter { $0.mediaType == .video }.count
            let screenshots = items.filter { $0.mediaSubtypes.contains(.photoScreenshot) }.count
            let selfies = items.filter { selfieIDs.contains($0.localIdentifier) }.count
            let livePhotos = items.filter { $0.mediaSubtypes.contains(.photoLive) }.count
            let portraits = items.filter { $0.mediaSubtypes.contains(.photoDepthEffect) }.count
            var geo = 0
            for a in items where a.location != nil { geo += 1 }

            let df2 = DateFormatter(); df2.dateStyle = .medium
            let first = items.last?.creationDate.map { df2.string(from: $0) } ?? "?"
            let last = items.first?.creationDate.map { df2.string(from: $0) } ?? "?"

            let summary = """
            Photos (\(month)): \(items.count) items between \(first) and \(last).
            \(favorites) favorites, \(videos) videos, \(screenshots) screenshots, \(selfies) selfies, \(livePhotos) live photos, \(portraits) portraits, \(geo) with location.
            """

            let embeddingResult: EmbeddingRuntimeResult
            do {
                let embeddingText = SemanticEmbeddingText.document(
                    content: summary,
                    sourceName: "Photos \(month)",
                    sourceType: RAGSourceType.photo.rawValue,
                    chunkIndex: 0
                )
                embeddingResult = try await embed(embeddingText)
            } catch {
                let diagnostic = "embedding_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
                logger.error("rag_embedding_failed op=indexPhotos source=\(month, privacy: .public) diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
                return IndexResult(indexedCount: 0, mode: .failed, diagnostic: diagnostic)
            }
            let emb = embeddingResult.vector
            guard !emb.isEmpty else {
                logger.error("rag_embedding_empty op=indexPhotos source=\(month, privacy: .public)")
                return IndexResult(indexedCount: 0, mode: .failed, diagnostic: "embedding_empty")
            }
            let embeddingMetadata = embeddingMetadata(for: embeddingResult)
            if let activeEmbeddingMetadata, activeEmbeddingMetadata != embeddingMetadata {
                return IndexResult(indexedCount: 0, mode: .failed, diagnostic: "embedding_identity_changed_during_index")
            }
            if activeEmbeddingMetadata == nil {
                activeEmbeddingMetadata = embeddingMetadata
            }
            let chunk = RAGChunk(
                content: summary,
                sourceType: .photo,
                sourceName: "Photos \(month)",
                sourceRef: month,
                chunkIndex: 0,
                embedding: emb,
                embeddingFormatVersion: embeddingMetadata.formatVersion,
                embeddingModelIdentifier: embeddingMetadata.modelIdentifier,
                embeddingDimension: embeddingMetadata.dimension
            )
            pendingVectors.append(PendingVector(chunk: chunk, bucket: RAGSourceType.photo.rawValue, vector: emb, metadata: embeddingMetadata))
        }

        let commit = commitStagedCorpusReplacement(
            context: context,
            replacing: [.photo],
            operation: "indexPhotos.replace",
            maintenanceReason: "rag.indexPhotos",
            pending: &pendingVectors,
            save: save
        )
        switch commit {
        case .committed(let result):
            if result.indexState == .unavailable {
                return IndexResult(indexedCount: result.persistedCount, mode: .partial, diagnostic: "vector_index_reload_failed")
            }
            guard !assets.isEmpty else {
                return IndexResult(indexedCount: 0, mode: .cleared, diagnostic: "empty_photo_library")
            }
            return IndexResult(indexedCount: result.persistedCount, mode: .indexed, diagnostic: nil)
        case .cancelled:
            return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "cancelled")
        case .maintenanceBudgetDenied:
            return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "maintenance_budget_denied")
        case .diskWriteBudgetDenied:
            return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "cleanup_deferred:disk_write_budget_denied")
        case .failed(let diagnostic):
            return IndexResult(indexedCount: 0, mode: .failed, diagnostic: diagnostic)
        }
    }

    // MARK: - Notes (plain text import via share)

    static func indexNote(title: String, body: String, context: ModelContext) async -> Int {
        await indexNoteWithDiagnostics(title: title, body: body, context: context).indexedCount
    }

    static func indexNoteWithDiagnostics(title: String, body: String, context: ModelContext) async -> IndexResult {
        let pieces = chunkText(body)
        guard !pieces.isEmpty else {
            return IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "empty_text")
        }
        var count = 0
        var pendingVectors: [PendingVector] = []
        var activeEmbeddingMetadata: RAGEmbeddingIndexMetadata?
        let cpuToken = CPUWatchdogGuard.shared.begin(category: .rag)
        defer { CPUWatchdogGuard.shared.end(token: cpuToken) }
        for (i, piece) in pieces.enumerated() {
            if Task.isCancelled || CPUWatchdogGuard.shared.shouldDegrade(category: .rag) || !ResourceBudgetGate.allowsMaintenance(reason: "rag.indexNote") { break }
            let embeddingResult: EmbeddingRuntimeResult
            do {
                let embeddingText = SemanticEmbeddingText.document(
                    content: piece,
                    sourceName: title,
                    sourceType: RAGSourceType.note.rawValue,
                    chunkIndex: i
                )
                embeddingResult = try await AssistantKernel.runEmbeddingWithIdentity(text: embeddingText)
            } catch {
                let diagnostic = "embedding_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
                logger.error("rag_embedding_failed op=indexNote source_hash=\(Self.sourceLogID(title), privacy: .public) diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
                return persistPendingVectorsForEarlyExit(
                    context: context,
                    operation: "indexNote.embeddingFailure",
                    diagnostic: diagnostic,
                    pending: &pendingVectors
                )
            }
            let emb = embeddingResult.vector
            guard !emb.isEmpty else {
                logger.error("rag_embedding_empty op=indexNote source_hash=\(Self.sourceLogID(title), privacy: .public)")
                return persistPendingVectorsForEarlyExit(
                    context: context,
                    operation: "indexNote.emptyEmbedding",
                    diagnostic: "embedding_empty",
                    pending: &pendingVectors
                )
            }
            let embeddingMetadata = embeddingMetadata(for: embeddingResult)
            if let earlyExit = prepareEmbeddingMetadata(
                embeddingMetadata,
                active: &activeEmbeddingMetadata,
                pending: &pendingVectors,
                context: context,
                identityChangeOperation: "indexNote.embeddingIdentityChanged"
            ) {
                return earlyExit
            }
            let chunk = RAGChunk(
                content: piece,
                sourceType: .note,
                sourceName: title,
                sourceRef: nil,
                chunkIndex: i,
                embedding: emb,
                embeddingFormatVersion: embeddingMetadata.formatVersion,
                embeddingModelIdentifier: embeddingMetadata.modelIdentifier,
                embeddingDimension: embeddingMetadata.dimension
            )
            pendingVectors.append(PendingVector(chunk: chunk, bucket: RAGSourceType.note.rawValue, vector: emb, metadata: embeddingMetadata))
            count += 1
        }
        guard let persistResult = persistAndAppendVectors(context: context, operation: "indexNote.complete", pending: &pendingVectors) else {
            logger.error("rag_index_partial_failure op=indexNote source_hash=\(Self.sourceLogID(title), privacy: .public) persisted=0 attempted=\(count, privacy: .public)")
            return IndexResult(indexedCount: 0, mode: .failed, diagnostic: "persist_failed")
        }
        if persistResult.indexState == .unavailable {
            return IndexResult(indexedCount: persistResult.persistedCount, mode: .partial, diagnostic: "vector_index_reload_failed")
        }
        if persistResult.persistedCount < pieces.count {
            return IndexResult(indexedCount: persistResult.persistedCount, mode: persistResult.persistedCount > 0 ? .partial : .failed, diagnostic: "maintenance_budget_or_cancellation")
        }
        return IndexResult(indexedCount: persistResult.persistedCount, mode: .indexed, diagnostic: nil)
    }

    static func embeddingRuntimeAvailable() async -> Bool {
        do {
            let probe = try await AssistantKernel.runEmbedding(text: "embedding_readiness_probe")
            if probe.isEmpty {
                logger.error("rag_embedding_empty op=readiness")
                return false
            }
            return true
        } catch {
            logger.error("rag_embedding_failed op=readiness error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return false
        }
    }

    private static func photoAuthorizationDiagnostic(_ status: PHAuthorizationStatus) -> String {
        switch status {
        case .authorized:
            return "authorized"
        case .limited:
            return "limited"
        case .denied:
            return "denied"
        case .restricted:
            return "restricted"
        case .notDetermined:
            return "not_determined"
        @unknown default:
            return "unknown"
        }
    }

    // MARK: - Helpers

    static func chunkText(_ text: String) -> [String] {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        let paragraphs = normalized.components(separatedBy: "\n\n").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        var chunks: [String] = []
        var current = ""
        for p in paragraphs {
            if current.count + p.count + 2 <= chunkSize {
                current += (current.isEmpty ? "" : "\n\n") + p
            } else {
                if !current.isEmpty { chunks.append(current) }
                if p.count > chunkSize {
                    var start = p.startIndex
                    while start < p.endIndex {
                        let end = p.index(start, offsetBy: chunkSize, limitedBy: p.endIndex) ?? p.endIndex
                        chunks.append(String(p[start..<end]))
                        if end == p.endIndex { break }
                        let step = chunkSize - chunkOverlap
                        start = p.index(start, offsetBy: step, limitedBy: p.endIndex) ?? p.endIndex
                    }
                    current = ""
                } else {
                    current = p
                }
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }
}
