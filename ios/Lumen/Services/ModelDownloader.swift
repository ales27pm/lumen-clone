import Foundation
import Observation
import OSLog

@Observable
final class ModelDownloader: NSObject {
    static let shared = ModelDownloader()
    private static let logger = Logger(subsystem: "ai.lumen.app", category: "model-downloader")

    var progresses: [String: DownloadProgress] = [:]

    @ObservationIgnored private var sessions: [String: URLSessionDownloadTask] = [:]
    @ObservationIgnored private var targets: [Int: (model: CatalogModel, destination: URL, onComplete: (URL) -> Void)] = [:]
    @ObservationIgnored private var resumeData: [String: Data] = [:]
    @ObservationIgnored private var completionHandlers: [String: (URL) -> Void] = [:]
    @ObservationIgnored private var responseStatusCodes: [Int: Int] = [:]
    @ObservationIgnored private var responseMimeTypes: [Int: String] = [:]

    @ObservationIgnored
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 3600
        config.waitsForConnectivity = true
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    static var modelsDirectory: URL { ModelStorage.modelsDirectoryURL() }
    private static var resumeDirectory: URL { ModelStorage.resumeDirectoryURL() }

    func localURL(for model: CatalogModel) -> URL { Self.modelsDirectory.appendingPathComponent(model.fileName) }

    func isDownloaded(_ model: CatalogModel) -> Bool {
        if case .success = ModelFileIntegrity.validateDownloadedCatalogFile(model, at: localURL(for: model)) {
            return true
        }
        return false
    }

    func isDownloading(_ model: CatalogModel) -> Bool { sessions[model.id] != nil }

    @discardableResult
    func start(_ model: CatalogModel, onComplete: @escaping (URL) -> Void) -> Result<Void, CatalogModel.DownloadURLError> {
        if sessions[model.id] != nil {
            NotificationCenter.default.post(name: .modelDownloaderInfo, object: nil, userInfo: ["message": "\(model.name) is already downloading."])
            return .success(())
        }

        if isDownloaded(model) {
            onComplete(localURL(for: model))
            progresses[model.id] = DownloadProgress(fractionCompleted: 1, bytesReceived: model.sizeBytes, totalBytes: model.sizeBytes, state: .completed)
            return .success(())
        }

        completionHandlers[model.id] = onComplete
        let data = resumeData[model.id] ?? loadPersistedResume(for: model)
        let task: URLSessionDownloadTask
        if let data {
            task = session.downloadTask(withResumeData: data)
        } else {
            let urlResult = model.downloadURLResult
            guard case .success(let downloadURL) = urlResult else {
                let error: CatalogModel.DownloadURLError
                if case .failure(let err) = urlResult { error = err } else { error = .invalidURLComponents }
                Self.logger.error("download_start_failed model_id=\(model.id, privacy: .public) reason=\(String(describing: error), privacy: .public)")
                completionHandlers[model.id] = nil
                NotificationCenter.default.post(name: .modelDownloaderInfo, object: nil, userInfo: ["message": "Could not start download for \(model.name): \(error.localizedDescription)"])
                return .failure(error)
            }
            task = session.downloadTask(with: downloadURL)
        }
        progresses[model.id] = DownloadProgress(fractionCompleted: progresses[model.id]?.fractionCompleted ?? 0, bytesReceived: progresses[model.id]?.bytesReceived ?? 0, totalBytes: model.sizeBytes, state: .downloading)
        sessions[model.id] = task
        targets[task.taskIdentifier] = (model, localURL(for: model), onComplete)
        resumeData[model.id] = nil
        responseStatusCodes[task.taskIdentifier] = nil
        responseMimeTypes[task.taskIdentifier] = nil
        task.resume()
        return .success(())
    }

    func pause(_ model: CatalogModel) {
        guard let task = sessions[model.id] else { return }
        let modelID = model.id
        let sizeBytes = model.sizeBytes
        let modelCopy = model
        task.cancel { data in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let data {
                    self.resumeData[modelID] = data
                    self.persistResume(data, for: modelCopy)
                }
                self.sessions[modelID] = nil
                let existing = self.progresses[modelID]
                self.progresses[modelID] = DownloadProgress(fractionCompleted: existing?.fractionCompleted ?? 0, bytesReceived: existing?.bytesReceived ?? 0, totalBytes: sizeBytes, state: .paused)
            }
        }
    }

    func resume(_ model: CatalogModel) {
        guard sessions[model.id] == nil else { return }
        guard let handler = completionHandlers[model.id] else { return }
        start(model, onComplete: handler)
    }

    func cancel(_ model: CatalogModel) {
        sessions[model.id]?.cancel()
        sessions[model.id] = nil
        resumeData[model.id] = nil
        completionHandlers[model.id] = nil
        clearPersistedResume(for: model)
        progresses[model.id] = nil
    }

    func deleteLocal(_ model: CatalogModel) {
        try? FileManager.default.removeItem(at: localURL(for: model))
        clearPersistedResume(for: model)
        progresses[model.id] = nil
    }

    private func persistResume(_ data: Data, for model: CatalogModel) {
        let url = Self.resumeDirectory.appendingPathComponent("\(model.id).resume")
        try? data.write(to: url)
    }

    private func loadPersistedResume(for model: CatalogModel) -> Data? {
        let url = Self.resumeDirectory.appendingPathComponent("\(model.id).resume")
        return try? Data(contentsOf: url)
    }

    private func clearPersistedResume(for model: CatalogModel) {
        let url = Self.resumeDirectory.appendingPathComponent("\(model.id).resume")
        try? FileManager.default.removeItem(at: url)
    }

    private func failDownload(taskID: Int, model: CatalogModel, message: String, cleanupURLs: [URL] = []) {
        for url in cleanupURLs { try? FileManager.default.removeItem(at: url) }
        try? FileManager.default.removeItem(at: localURL(for: model))
        clearPersistedResume(for: model)
        progresses[model.id] = DownloadProgress(fractionCompleted: progresses[model.id]?.fractionCompleted ?? 0, bytesReceived: progresses[model.id]?.bytesReceived ?? 0, totalBytes: model.sizeBytes, state: .failed(message))
        sessions[model.id] = nil
        completionHandlers[model.id] = nil
        targets[taskID] = nil
        responseStatusCodes[taskID] = nil
        responseMimeTypes[taskID] = nil
        NotificationCenter.default.post(name: .modelDownloaderInfo, object: nil, userInfo: ["message": "Download failed for \(model.name): \(message)"])
    }
}

extension Notification.Name {
    static let modelDownloaderInfo = Notification.Name("modelDownloaderInfo")
}

extension ModelDownloader: URLSessionDownloadDelegate {
    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let taskId = downloadTask.taskIdentifier
        Task { @MainActor in
            guard let entry = targets[taskId] else { return }
            let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : entry.model.sizeBytes
            let fraction = total > 0 ? Double(totalBytesWritten) / Double(total) : 0
            progresses[entry.model.id] = DownloadProgress(fractionCompleted: fraction, bytesReceived: totalBytesWritten, totalBytes: total, state: .downloading)
        }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let taskId = downloadTask.taskIdentifier
        let fm = FileManager.default
        let modelsDir = ModelStorage.modelsDirectoryURL(fileManager: fm)
        let staging = modelsDir.appendingPathComponent(".staging-\(UUID().uuidString)")
        try? fm.removeItem(at: staging)
        let movedURL: URL?
        do {
            try fm.moveItem(at: location, to: staging)
            movedURL = staging
        } catch {
            movedURL = nil
        }

        Task { @MainActor in
            guard let entry = targets[taskId] else {
                if let moved = movedURL { try? fm.removeItem(at: moved) }
                return
            }

            guard let moved = movedURL else {
                failDownload(taskID: taskId, model: entry.model, message: "Could not move downloaded temporary file.")
                return
            }

            if let status = responseStatusCodes[taskId], !(200...299).contains(status) {
                failDownload(taskID: taskId, model: entry.model, message: "HTTP status \(status)", cleanupURLs: [moved])
                return
            }

            if let mime = responseMimeTypes[taskId]?.lowercased(), mime.contains("text/html") || mime.contains("application/json") {
                failDownload(taskID: taskId, model: entry.model, message: "Unexpected response type \(mime)", cleanupURLs: [moved])
                return
            }

            switch ModelFileIntegrity.validateDownloadedCatalogFile(entry.model, at: moved) {
            case .success(let actualSize):
                try? fm.removeItem(at: entry.destination)
                do {
                    try fm.moveItem(at: moved, to: entry.destination)
                } catch {
                    do {
                        try fm.copyItem(at: moved, to: entry.destination)
                        try? fm.removeItem(at: moved)
                    } catch {
                        failDownload(taskID: taskId, model: entry.model, message: "Could not install downloaded file: \(error.localizedDescription)", cleanupURLs: [moved])
                        return
                    }
                }

                switch ModelFileIntegrity.validateDownloadedCatalogFile(entry.model, at: entry.destination) {
                case .success:
                    progresses[entry.model.id] = DownloadProgress(fractionCompleted: 1, bytesReceived: actualSize, totalBytes: max(actualSize, entry.model.sizeBytes), state: .completed)
                    clearPersistedResume(for: entry.model)
                    entry.onComplete(entry.destination)
                    sessions[entry.model.id] = nil
                    completionHandlers[entry.model.id] = nil
                    targets[taskId] = nil
                    responseStatusCodes[taskId] = nil
                    responseMimeTypes[taskId] = nil
                case .failure(let failure):
                    failDownload(taskID: taskId, model: entry.model, message: failure.localizedDescription)
                }
            case .failure(let failure):
                failDownload(taskID: taskId, model: entry.model, message: failure.localizedDescription, cleanupURLs: [moved])
            }
        }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didReceive response: URLResponse, completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let taskID = downloadTask.taskIdentifier
        Task { @MainActor in
            if let http = response as? HTTPURLResponse {
                responseStatusCodes[taskID] = http.statusCode
            }
            responseMimeTypes[taskID] = response.mimeType
        }
        completionHandler(.allow)
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let taskId = task.taskIdentifier
        guard let error else { return }
        let nsError = error as NSError
        let resume = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        Task { @MainActor in
            guard let entry = targets[taskId] else { return }
            if let resume {
                self.resumeData[entry.model.id] = resume
                self.persistResume(resume, for: entry.model)
            }
            if nsError.code == NSURLErrorCancelled {
                sessions[entry.model.id] = nil
                targets[taskId] = nil
                responseStatusCodes[taskId] = nil
                responseMimeTypes[taskId] = nil
                return
            }
            failDownload(taskID: taskId, model: entry.model, message: error.localizedDescription)
        }
    }
}
