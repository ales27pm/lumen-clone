import CryptoKit
import Foundation

nonisolated enum ModelProvisioningStorageKeys {
    static let receipt = "verifiedModelProvisioningReceipt"
    static let switchJournal = "modelProvisioningSwitchJournalV1"
}

nonisolated enum ModelProvisioningSwitchJournalStore {
    enum Phase: String, Codable, Equatable, Sendable {
        case prepared
        case committed
    }

    private struct Payload: Codable, Equatable, Sendable {
        let schemaVersion: Int
        let phase: Phase
        let targetFamilyID: String
        let previousSelectionData: Data
        let previousReceiptData: Data?
        let committedSelectionData: Data?
        let committedReceiptData: Data?
    }

    static let defaultsKey = ModelProvisioningStorageKeys.switchJournal
    private static let schemaVersion = 1

    static func prepare(
        targetFamily: LumenModelFamily,
        defaults: UserDefaults = .standard
    ) -> Bool {
        // Startup owns stale-journal recovery. A live journal here belongs to an
        // in-flight finalization and must not be displaced by a second switch.
        guard defaults.data(forKey: defaultsKey) == nil else { return false }
        _ = PersistedModelSelectionStore.loadOrMigrate(defaults: defaults)
        guard let previousSelectionData = defaults.data(forKey: PersistedModelSelectionStore.defaultsKey) else {
            return false
        }
        let payload = Payload(
            schemaVersion: schemaVersion,
            phase: .prepared,
            targetFamilyID: targetFamily.rawValue,
            previousSelectionData: previousSelectionData,
            previousReceiptData: defaults.data(forKey: ModelProvisioningStorageKeys.receipt),
            committedSelectionData: nil,
            committedReceiptData: nil
        )
        return write(payload, defaults: defaults)
    }

    static func markCommitted(
        targetFamily: LumenModelFamily,
        chatModelID: String,
        embeddingModelID: String,
        provisioningPlanID: String,
        defaults: UserDefaults = .standard
    ) -> Bool {
        guard let prepared = load(defaults: defaults),
              prepared.phase == .prepared,
              prepared.targetFamilyID == targetFamily.rawValue,
              let selection = PersistedModelSelectionStore.load(defaults: defaults),
              selection.familyID == targetFamily.rawValue,
              selection.chatModelID == chatModelID,
              selection.embeddingModelID == embeddingModelID,
              selection.provisioningPlanID == provisioningPlanID,
              let committedSelectionData = defaults.data(forKey: PersistedModelSelectionStore.defaultsKey),
              let committedReceiptData = defaults.data(forKey: ModelProvisioningStorageKeys.receipt)
        else { return false }

        return write(
            Payload(
                schemaVersion: schemaVersion,
                phase: .committed,
                targetFamilyID: prepared.targetFamilyID,
                previousSelectionData: prepared.previousSelectionData,
                previousReceiptData: prepared.previousReceiptData,
                committedSelectionData: committedSelectionData,
                committedReceiptData: committedReceiptData
            ),
            defaults: defaults
        )
    }

    static func clearCommitted(defaults: UserDefaults = .standard) -> Bool {
        guard let committed = load(defaults: defaults),
              committed.phase == .committed,
              defaults.data(forKey: PersistedModelSelectionStore.defaultsKey) == committed.committedSelectionData,
              defaults.data(forKey: ModelProvisioningStorageKeys.receipt) == committed.committedReceiptData
        else { return false }
        defaults.removeObject(forKey: defaultsKey)
        return defaults.data(forKey: defaultsKey) == nil
    }

    @discardableResult
    static func rollback(defaults: UserDefaults = .standard) -> Bool {
        guard let payload = load(defaults: defaults) else {
            return defaults.data(forKey: defaultsKey) == nil
        }
        return restorePreviousState(from: payload, defaults: defaults)
    }

    @discardableResult
    static func recoverIfNeeded(defaults: UserDefaults = .standard) -> Bool {
        guard let payload = load(defaults: defaults) else {
            return defaults.data(forKey: defaultsKey) == nil
        }
        if payload.phase == .committed,
           defaults.data(forKey: PersistedModelSelectionStore.defaultsKey) == payload.committedSelectionData,
           defaults.data(forKey: ModelProvisioningStorageKeys.receipt) == payload.committedReceiptData {
            defaults.removeObject(forKey: defaultsKey)
            return defaults.data(forKey: defaultsKey) == nil
        }
        return restorePreviousState(from: payload, defaults: defaults)
    }

    private static func load(defaults: UserDefaults) -> Payload? {
        guard let data = defaults.data(forKey: defaultsKey),
              let payload = try? JSONDecoder().decode(Payload.self, from: data),
              payload.schemaVersion == schemaVersion,
              LumenModelFamily(rawValue: payload.targetFamilyID) != nil
        else { return nil }
        return payload
    }

    private static func write(_ payload: Payload, defaults: UserDefaults) -> Bool {
        guard let data = try? JSONEncoder().encode(payload) else { return false }
        defaults.set(data, forKey: defaultsKey)
        return defaults.data(forKey: defaultsKey) == data
    }

    private static func restorePreviousState(
        from payload: Payload,
        defaults: UserDefaults
    ) -> Bool {
        defaults.set(payload.previousSelectionData, forKey: PersistedModelSelectionStore.defaultsKey)
        guard let selection = PersistedModelSelectionStore.load(defaults: defaults) else { return false }
        defaults.set(selection.chatModelID, forKey: PersistedModelSelectionStore.legacyChatKey)
        defaults.set(selection.embeddingModelID, forKey: PersistedModelSelectionStore.legacyEmbeddingKey)
        defaults.set(selection.familyID, forKey: PersistedModelSelectionStore.legacyFamilyKey)

        if let previousReceiptData = payload.previousReceiptData {
            defaults.set(previousReceiptData, forKey: ModelProvisioningStorageKeys.receipt)
        } else {
            defaults.removeObject(forKey: ModelProvisioningStorageKeys.receipt)
        }

        guard defaults.data(forKey: PersistedModelSelectionStore.defaultsKey) == payload.previousSelectionData,
              defaults.data(forKey: ModelProvisioningStorageKeys.receipt) == payload.previousReceiptData
        else { return false }
        defaults.removeObject(forKey: defaultsKey)
        return defaults.data(forKey: defaultsKey) == nil
    }
}

@MainActor
enum ModelProvisioningReceipt {
    enum Status: String, Codable, Equatable, Sendable {
        case deferred
        case consented
        case completed
    }

    private struct Payload: Codable {
        let schemaVersion: Int
        let familyID: String
        let catalogIdentity: String
        let status: Status
        let chatModelID: String?
        let embeddingModelID: String?
        let updatedAt: Date
    }

    static let defaultsKey = ModelProvisioningStorageKeys.receipt
    private static let schemaVersion = 2

    static func status(
        family: LumenModelFamily = LumenModelFamily.persistedSelected,
        defaults: UserDefaults = .standard
    ) -> Status? {
        currentPayload(family: family, defaults: defaults)?.status
    }

    static func isConsented(
        family: LumenModelFamily = LumenModelFamily.persistedSelected,
        defaults: UserDefaults = .standard
    ) -> Bool {
        switch status(family: family, defaults: defaults) {
        case .consented, .completed:
            return true
        case .deferred, .none:
            return false
        }
    }

    static func isCurrent(
        family: LumenModelFamily = LumenModelFamily.persistedSelected,
        chatModelID: String? = nil,
        embeddingModelID: String? = nil,
        defaults: UserDefaults = .standard
    ) -> Bool {
        guard let payload = currentPayload(family: family, defaults: defaults),
              payload.status == .completed,
              let recordedChatID = payload.chatModelID,
              let recordedEmbeddingID = payload.embeddingModelID
        else { return false }

        let persisted = PersistedModelSelectionStore.loadOrMigrate(defaults: defaults)
        let expectedChatID = chatModelID ?? persisted.chatModelID
        let expectedEmbeddingID = embeddingModelID ?? persisted.embeddingModelID
        return recordedChatID == expectedChatID
            && recordedEmbeddingID == expectedEmbeddingID
            && persisted.familyID == family.rawValue
            && persisted.provisioningPlanID == payload.catalogIdentity
    }

    @discardableResult
    static func markDeferred(
        family: LumenModelFamily = LumenModelFamily.persistedSelected,
        defaults: UserDefaults = .standard,
        updatedAt: Date = Date()
    ) -> Bool {
        write(
            status: .deferred,
            family: family,
            chatModelID: nil,
            embeddingModelID: nil,
            defaults: defaults,
            updatedAt: updatedAt
        )
    }

    @discardableResult
    static func markConsented(
        family: LumenModelFamily = LumenModelFamily.persistedSelected,
        defaults: UserDefaults = .standard,
        updatedAt: Date = Date()
    ) -> Bool {
        write(
            status: .consented,
            family: family,
            chatModelID: nil,
            embeddingModelID: nil,
            defaults: defaults,
            updatedAt: updatedAt
        )
    }

    @discardableResult
    static func markCurrent(
        family: LumenModelFamily = LumenModelFamily.persistedSelected,
        chatModelID: String,
        embeddingModelID: String,
        defaults: UserDefaults = .standard,
        completedAt: Date = Date()
    ) -> Bool {
        write(
            status: .completed,
            family: family,
            chatModelID: chatModelID,
            embeddingModelID: embeddingModelID,
            defaults: defaults,
            updatedAt: completedAt
        )
    }

    static func invalidate(defaults: UserDefaults = .standard) {
        defaults.removeObject(forKey: defaultsKey)
    }

    static func catalogIdentity(for family: LumenModelFamily) -> String {
        let records = ModelLaunchBootstrap.provisioningModelsForInstall(family: family)
            .map { model in
                [
                    model.repoId.lowercased(),
                    model.fileName.lowercased(),
                    model.sourceRevision.lowercased(),
                    model.expectedSHA256.lowercased(),
                    String(model.sizeBytes),
                    model.role.rawValue
                ].joined(separator: "|")
            }
            .sorted()
            .joined(separator: "\n")
        return SHA256.hash(data: Data(records.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func currentPayload(
        family: LumenModelFamily,
        defaults: UserDefaults
    ) -> Payload? {
        guard let data = defaults.data(forKey: defaultsKey),
              let payload = try? JSONDecoder().decode(Payload.self, from: data),
              payload.schemaVersion == schemaVersion,
              payload.familyID == family.rawValue,
              payload.catalogIdentity == catalogIdentity(for: family)
        else { return nil }
        return payload
    }

    private static func write(
        status: Status,
        family: LumenModelFamily,
        chatModelID: String?,
        embeddingModelID: String?,
        defaults: UserDefaults,
        updatedAt: Date
    ) -> Bool {
        let payload = Payload(
            schemaVersion: schemaVersion,
            familyID: family.rawValue,
            catalogIdentity: catalogIdentity(for: family),
            status: status,
            chatModelID: chatModelID,
            embeddingModelID: embeddingModelID,
            updatedAt: updatedAt
        )
        guard let data = try? JSONEncoder().encode(payload) else { return false }
        defaults.set(data, forKey: defaultsKey)
        return true
    }
}
