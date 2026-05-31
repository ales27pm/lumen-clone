import Foundation
import SwiftData
import PDFKit
import Photos
import OSLog

@MainActor
enum RAGStore {
    private static let logger = Logger(subsystem: "ai.lumen.app", category: "persistence")

    private static func persist(_ context: ModelContext, operation: String, scope: String) throws {
        do { try context.save() } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error=\(String(describing: error), privacy: .public)")
            throw error
        }
    }

    static func persistAndAppendVectors(
        context: ModelContext,
        operation: String,
        scope: String = "RAGChunk",
        pending: inout [(id: PersistentIdentifier, bucket: String, vector: [Double])],
        save: ((ModelContext, String, String) throws -> Void)? = nil
    ) -> Int? {
        let appendedCount = pending.count
        let saveAction = save ?? persist
        do {
            try saveAction(context, operation, scope)
        } catch {
            return nil
        }
        for item in pending {
            RAGVectorIndex.shared.append(id: item.id, bucket: item.bucket, vector: item.vector)
        }
        pending.removeAll(keepingCapacity: true)
        return appendedCount
    }

    static func auditPersistence(operation: String, scope: String, save: () throws -> Void) -> Bool {
        do {
            try save()
            return true
        } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error=\(String(describing: error), privacy: .public)")
            return false
        }
    }

    static let chunkSize = 600
    static let chunkOverlap = 80
    static let candidatePoolMultiplier = 8
    static let maxCandidatePool = 256
    static let minScore: Float = 0.12

    static func search(
        query: String,
        context: ModelContext,
        limit: Int = 5,
        sourceTypes: Set<RAGSourceType>? = nil
    ) async -> [(chunk: RAGChunk, score: Double)] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, limit > 0 else { return [] }
        let queryVec: [Double]
        do {
            queryVec = try await AppLlamaService.shared.embed(trimmed)
        } catch {
            logger.error("rag_embedding_failed op=search error=\(String(describing: error), privacy: .public)")
            return []
        }
        guard !queryVec.isEmpty else {
            logger.error("rag_embedding_empty op=search")
            return []
        }

        RAGVectorIndex.shared.ensureLoaded(context: context)

        let allowed: Set<String>? = sourceTypes.map { Set($0.map(\.rawValue)) }
        let k = min(max(limit * candidatePoolMultiplier, limit + 4), maxCandidatePool)

        let vectorHits = RAGVectorIndex.shared.search(
            query: queryVec,
            topK: k,
            allowedBuckets: allowed,
            minScore: minScore
        )

        var candidates: [(RAGChunk, Double)] = []
        candidates.reserveCapacity(vectorHits.count)
        for hit in vectorHits {
            if let chunk = context.model(for: hit.id) as? RAGChunk {
                candidates.append((chunk, Double(hit.score)))
            }
        }

        if candidates.count < limit {
            // Lexical backfill: keyword overlap as a rescue path when embeddings
            // missed obviously relevant chunks (short queries, OOV vocab, etc.).
            let seenIDs: Set<PersistentIdentifier> = Set(candidates.map { $0.0.persistentModelID })
            let lexical = lexicalScore(query: trimmed, context: context, allowed: allowed, exclude: seenIDs, limit: limit - candidates.count)
            candidates.append(contentsOf: lexical)
        }

        return candidates
            .sorted { $0.1 > $1.1 }
            .prefix(limit)
            .map { ($0.0, $0.1) }
    }

    private static func lexicalScore(
        query: String,
        context: ModelContext,
        allowed: Set<String>?,
        exclude: Set<PersistentIdentifier>,
        limit: Int
    ) -> [(RAGChunk, Double)] {
        guard limit > 0 else { return [] }
        let terms = query.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 3 }
        guard !terms.isEmpty else { return [] }
        var descriptor = FetchDescriptor<RAGChunk>()
        descriptor.fetchLimit = 400
        guard let all = try? context.fetch(descriptor) else { return [] }
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
        return scored.sorted { $0.1 > $1.1 }.prefix(limit).map { $0 }
    }

    static func counts(context: ModelContext) -> [RAGSourceType: Int] {
        guard let all = try? context.fetch(FetchDescriptor<RAGChunk>()) else { return [:] }
        var out: [RAGSourceType: Int] = [:]
        for c in all { out[c.kind, default: 0] += 1 }
        return out
    }

    static func wipe(_ type: RAGSourceType?, context: ModelContext) throws {
        let all = try context.fetch(FetchDescriptor<RAGChunk>())
        for c in all {
            if type == nil || c.kind == type { context.delete(c) }
        }
        try persist(context, operation: "wipe", scope: "RAGChunk")
        if let type {
            RAGVectorIndex.shared.removeBucket(type.rawValue)
        } else {
            RAGVectorIndex.shared.removeAll()
        }
    }

    static func chunks(for type: RAGSourceType, context: ModelContext) -> [RAGChunk] {
        let raw = (try? context.fetch(FetchDescriptor<RAGChunk>())) ?? []
        return raw.filter { $0.kind == type }.sorted { $0.createdAt > $1.createdAt }
    }

    // MARK: - File / PDF indexing

    static func indexImportedFiles(context: ModelContext, progress: ((Double) -> Void)? = nil) async -> Int {
        do {
            try wipe(.file, context: context)
            try wipe(.pdf, context: context)
        } catch {
            logger.error("persist_failed op=indexImportedFiles.cleanup scope=RAGChunk error=\(String(describing: error), privacy: .public)")
            return 0
        }
        let files = FileStore.importedFiles()
        var total = 0
        for (idx, url) in files.enumerated() {
            let added = await indexFile(url: url, context: context)
            total += added
            progress?(Double(idx + 1) / Double(max(1, files.count)))
        }
        return total
    }

    static func indexFile(url: URL, context: ModelContext) async -> Int {
        let name = url.lastPathComponent
        let ext = url.pathExtension.lowercased()
        let text: String
        let type: RAGSourceType

        if ext == "pdf" {
            guard let pdf = PDFDocument(url: url) else { return 0 }
            var combined = ""
            for i in 0..<pdf.pageCount {
                combined += pdf.page(at: i)?.string ?? ""
                combined += "\n\n"
            }
            text = combined
            type = .pdf
        } else if ext == "rtf" || ext == "rtfd" {
            guard let data = try? Data(contentsOf: url),
                  let attr = try? NSAttributedString(data: data, options: [.documentType: NSAttributedString.DocumentType.rtf], documentAttributes: nil) else { return 0 }
            text = attr.string
            type = .file
        } else {
            guard let data = try? Data(contentsOf: url) else { return 0 }
            if let utf8 = String(data: data, encoding: .utf8) {
                text = utf8
            } else if let ascii = String(data: data, encoding: .isoLatin1) {
                text = ascii
            } else if let attr = try? NSAttributedString(data: data, options: [:], documentAttributes: nil) {
                text = attr.string
            } else {
                return 0
            }
            type = .file
        }

        RAGVectorIndex.shared.ensureLoaded(context: context)
        let pieces = chunkText(text)
        var count = 0
        var persistedCount = 0
        var pendingVectors: [(id: PersistentIdentifier, bucket: String, vector: [Double])] = []
        for (i, piece) in pieces.enumerated() {
            let emb: [Double]
            do {
                emb = try await AppLlamaService.shared.embed(piece)
            } catch {
                logger.error("rag_embedding_failed op=indexFile source=\(name, privacy: .public) error=\(String(describing: error), privacy: .public)")
                return 0
            }
            guard !emb.isEmpty else {
                logger.error("rag_embedding_empty op=indexFile source=\(name, privacy: .public)")
                return 0
            }
            let chunk = RAGChunk(content: piece, sourceType: type, sourceName: name, sourceRef: url.path, chunkIndex: i, embedding: emb)
            context.insert(chunk)
            pendingVectors.append((id: chunk.persistentModelID, bucket: type.rawValue, vector: emb))
            if i % 8 == 7 {
                guard let appendedCount = persistAndAppendVectors(context: context, operation: "indexFile.batch", pending: &pendingVectors) else {
                    logger.error("rag_index_partial_failure op=indexFile source=\(name, privacy: .public) persisted=\(persistedCount, privacy: .public) attempted=\(count + 1, privacy: .public)")
                    return persistedCount
                }
                persistedCount += appendedCount
            }
            count += 1
        }
        guard let appendedCount = persistAndAppendVectors(context: context, operation: "indexFile.complete", pending: &pendingVectors) else {
            logger.error("rag_index_partial_failure op=indexFile source=\(name, privacy: .public) persisted=\(persistedCount, privacy: .public) attempted=\(count, privacy: .public)")
            return persistedCount
        }
        persistedCount += appendedCount
        return persistedCount
    }

    // MARK: - Photos metadata

    static func indexPhotos(monthsBack: Int = 6, context: ModelContext) async -> Int {
        let status = await withCheckedContinuation { (cont: CheckedContinuation<PHAuthorizationStatus, Never>) in
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { cont.resume(returning: $0) }
        }
        guard status == .authorized || status == .limited else { return 0 }
        do {
            try wipe(.photo, context: context)
        } catch {
            logger.error("persist_failed op=indexPhotos.cleanup scope=RAGChunk error=\(String(describing: error), privacy: .public)")
            return 0
        }
        RAGVectorIndex.shared.ensureLoaded(context: context)

        let start = Calendar.current.date(byAdding: .month, value: -monthsBack, to: Date()) ?? Date()
        let options = PHFetchOptions()
        options.predicate = NSPredicate(format: "creationDate >= %@", start as NSDate)
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        options.fetchLimit = 2000
        let fetch = PHAsset.fetchAssets(with: options)

        var assets: [PHAsset] = []
        fetch.enumerateObjects { a, _, _ in assets.append(a) }

        var selfieIDs: Set<String> = []
        let selfieAlbums = PHAssetCollection.fetchAssetCollections(with: .smartAlbum, subtype: .smartAlbumSelfPortraits, options: nil)
        selfieAlbums.enumerateObjects { coll, _, _ in
            let a = PHAsset.fetchAssets(in: coll, options: nil)
            a.enumerateObjects { asset, _, _ in selfieIDs.insert(asset.localIdentifier) }
        }

        var buckets: [String: [PHAsset]] = [:]
        let df = DateFormatter(); df.dateFormat = "yyyy-MM"
        for a in assets {
            let key = df.string(from: a.creationDate ?? Date())
            buckets[key, default: []].append(a)
        }

        var count = 0
        var pendingVectors: [(id: PersistentIdentifier, bucket: String, vector: [Double])] = []
        for (month, items) in buckets {
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

            let emb: [Double]
            do {
                emb = try await AppLlamaService.shared.embed(summary)
            } catch {
                logger.error("rag_embedding_failed op=indexPhotos source=\(month, privacy: .public) error=\(String(describing: error), privacy: .public)")
                return 0
            }
            guard !emb.isEmpty else {
                logger.error("rag_embedding_empty op=indexPhotos source=\(month, privacy: .public)")
                return 0
            }
            let chunk = RAGChunk(content: summary, sourceType: .photo, sourceName: "Photos \(month)", sourceRef: month, chunkIndex: 0, embedding: emb)
            context.insert(chunk)
            pendingVectors.append((id: chunk.persistentModelID, bucket: RAGSourceType.photo.rawValue, vector: emb))
            count += 1
        }
        guard let persistedCount = persistAndAppendVectors(context: context, operation: "indexPhotos.complete", pending: &pendingVectors) else {
            logger.error("rag_index_partial_failure op=indexPhotos persisted=0 attempted=\(count, privacy: .public)")
            return 0
        }
        return persistedCount
    }

    // MARK: - Notes (plain text import via share)

    static func indexNote(title: String, body: String, context: ModelContext) async -> Int {
        RAGVectorIndex.shared.ensureLoaded(context: context)
        let pieces = chunkText(body)
        var count = 0
        var pendingVectors: [(id: PersistentIdentifier, bucket: String, vector: [Double])] = []
        for (i, piece) in pieces.enumerated() {
            let emb: [Double]
            do {
                emb = try await AppLlamaService.shared.embed(piece)
            } catch {
                logger.error("rag_embedding_failed op=indexNote source=\(title, privacy: .public) error=\(String(describing: error), privacy: .public)")
                return 0
            }
            guard !emb.isEmpty else {
                logger.error("rag_embedding_empty op=indexNote source=\(title, privacy: .public)")
                return 0
            }
            let chunk = RAGChunk(content: piece, sourceType: .note, sourceName: title, sourceRef: nil, chunkIndex: i, embedding: emb)
            context.insert(chunk)
            pendingVectors.append((id: chunk.persistentModelID, bucket: RAGSourceType.note.rawValue, vector: emb))
            count += 1
        }
        guard let persistedCount = persistAndAppendVectors(context: context, operation: "indexNote.complete", pending: &pendingVectors) else {
            logger.error("rag_index_partial_failure op=indexNote source=\(title, privacy: .public) persisted=0 attempted=\(count, privacy: .public)")
            return 0
        }
        return persistedCount
    }

    static func embeddingRuntimeAvailable() async -> Bool {
        do {
            let probe = try await AppLlamaService.shared.embed("embedding_readiness_probe")
            if probe.isEmpty {
                logger.error("rag_embedding_empty op=readiness")
                return false
            }
            return true
        } catch {
            logger.error("rag_embedding_failed op=readiness error=\(String(describing: error), privacy: .public)")
            return false
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
