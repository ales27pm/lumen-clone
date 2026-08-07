import Foundation
import OSLog
import SwiftData

@MainActor
enum ModelCatalogPersistenceCoordinator {
    enum Failure: LocalizedError {
        case fetchFailed(String)
        case saveFailed(String)

        var errorDescription: String? {
            switch self {
            case .fetchFailed(let code):
                return "The installed-model catalog could not be read (\(code))."
            case .saveFailed(let code):
                return "The installed-model catalog could not be saved (\(code))."
            }
        }
    }

    private static let logger = Logger(subsystem: "ai.lumen.app", category: "model-persistence")

    static func upsertVerifiedCatalogModel(
        _ catalog: CatalogModel,
        localURL: URL,
        context: ModelContext
    ) throws -> StoredModel {
        let models: [StoredModel]
        do {
            models = try context.fetch(FetchDescriptor<StoredModel>())
        } catch {
            throw loggedFailure(.fetchFailed(errorCode(error)), operation: "upsert-fetch")
        }

        if let existing = models.first(where: {
            $0.repoId.caseInsensitiveCompare(catalog.repoId) == .orderedSame
                && $0.fileName.caseInsensitiveCompare(catalog.fileName) == .orderedSame
        }) {
            let previous = Snapshot(existing)
            existing.name = catalog.name
            existing.repoId = catalog.repoId
            existing.fileName = catalog.fileName
            existing.sizeBytes = catalog.sizeBytes
            existing.quantization = catalog.quantization
            existing.parameters = catalog.parameters
            existing.role = catalog.role.rawValue
            existing.downloadedAt = Date()
            existing.localPath = localURL.path
            do {
                try context.save()
                return existing
            } catch {
                previous.restore(into: existing)
                context.rollback()
                throw loggedFailure(.saveFailed(errorCode(error)), operation: "upsert-update")
            }
        }

        let stored = StoredModel(
            name: catalog.name,
            repoId: catalog.repoId,
            fileName: catalog.fileName,
            sizeBytes: catalog.sizeBytes,
            quantization: catalog.quantization,
            parameters: catalog.parameters,
            role: catalog.role,
            localPath: localURL.path
        )
        context.insert(stored)
        do {
            try context.save()
            return stored
        } catch {
            context.rollback()
            throw loggedFailure(.saveFailed(errorCode(error)), operation: "upsert-insert")
        }
    }

    static func insertLocalModel(
        name: String,
        repoID: String,
        fileName: String,
        sizeBytes: Int64,
        role: ModelRole,
        localURL: URL,
        context: ModelContext
    ) throws -> StoredModel {
        let stored = StoredModel(
            name: name,
            repoId: repoID,
            fileName: fileName,
            sizeBytes: sizeBytes,
            quantization: "local",
            parameters: "local",
            role: role,
            localPath: localURL.path
        )
        context.insert(stored)
        do {
            try context.save()
            return stored
        } catch {
            context.rollback()
            throw loggedFailure(.saveFailed(errorCode(error)), operation: "local-insert")
        }
    }

    static func delete(_ model: StoredModel, context: ModelContext) throws {
        context.delete(model)
        do {
            try context.save()
        } catch {
            context.rollback()
            throw loggedFailure(.saveFailed(errorCode(error)), operation: "delete")
        }
    }

    private static func errorCode(_ error: Error) -> String {
        RuntimeMetricErrorSanitizer.code(for: error)
    }

    private static func loggedFailure(_ failure: Failure, operation: String) -> Failure {
        logger.error("model_persistence_failed operation=\(operation, privacy: .public) error=\(failure.localizedDescription, privacy: .public)")
        return failure
    }

    private struct Snapshot {
        let name: String
        let repoID: String
        let fileName: String
        let sizeBytes: Int64
        let quantization: String
        let parameters: String
        let role: String
        let downloadedAt: Date
        let localPath: String

        init(_ model: StoredModel) {
            name = model.name
            repoID = model.repoId
            fileName = model.fileName
            sizeBytes = model.sizeBytes
            quantization = model.quantization
            parameters = model.parameters
            role = model.role
            downloadedAt = model.downloadedAt
            localPath = model.localPath
        }

        func restore(into model: StoredModel) {
            model.name = name
            model.repoId = repoID
            model.fileName = fileName
            model.sizeBytes = sizeBytes
            model.quantization = quantization
            model.parameters = parameters
            model.role = role
            model.downloadedAt = downloadedAt
            model.localPath = localPath
        }
    }
}
