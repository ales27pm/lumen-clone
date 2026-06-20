import Foundation
import CoreML
import os.log

actor ModelLoader {
    static let shared = ModelLoader()
    
    private var loadedModels: [String: LoadedModel] = [:]
    private let cache = NSCache<NSString, AnyObject>()
    private let manifestStore: AgentManifestStore? = nil // In full project: injected
    private let logger = Logger(subsystem: "com.lumen.model", category: "loader")
    
    private let maxMemoryMB: Int = 512
    private var currentMemoryUsage: Int = 0
    
    struct LoadedModel {
        let model: MLModel
        let config: ModelConfig
        let loadTime: TimeInterval
        let timestamp: Date
    }
    
    struct ModelConfig: Codable {
        let name: String
        let quantizationBits: Int
        let preferredCompute: String
    }
    
    enum LoadPriority: Int {
        case high = 0, normal = 1, background = 2
    }
    
    enum ModelError: Error {
        case modelNotFound(String)
        case compilationFailed(String)
        case memoryLimitExceeded
    }
    
    func ensureChatLoaded(priority: LoadPriority = .high) async throws -> MLModel {
        let key = "chat"
        if let cached = loadedModels[key] {
            logger.info("Cache hit for \(key)")
            return cached.model
        }
        return try await loadModel(key: key, type: .chat, priority: priority)
    }
    
    func ensureEmbedLoaded(priority: LoadPriority = .normal) async throws -> MLModel {
        let key = "embed"
        if let cached = loadedModels[key] { return cached.model }
        return try await loadModel(key: key, type: .embedding, priority: priority)
    }
    
    func ensureVoiceLoaded(priority: LoadPriority = .normal) async throws -> MLModel {
        let key = "voice"
        if let cached = loadedModels[key] { return cached.model }
        return try await loadModel(key: key, type: .voice, priority: priority)
    }
    
    private func loadModel(key: String, type: ModelType, priority: LoadPriority) async throws -> MLModel {
        let start = Date()
        logger.info("Loading \(key) model (priority: \(priority))")
        
        let config = ModelConfig(name: key, quantizationBits: 8, preferredCompute: "cpuAndGPU")
        
        guard let modelURL = Bundle.main.url(forResource: config.name, withExtension: "mlmodelc") ?? 
              URL(string: "file:///generated/models/\(config.name).mlpackage") else {
            throw ModelError.modelNotFound(key)
        }
        
        let compiledURL: URL
        do {
            compiledURL = try await MLModel.compileModel(at: modelURL)
        } catch {
            throw ModelError.compilationFailed(error.localizedDescription)
        }
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        
        let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        
        let duration = Date().timeIntervalSince(start)
        
        let loaded = LoadedModel(model: model, config: config, loadTime: duration, timestamp: Date())
        loadedModels[key] = loaded
        
        currentMemoryUsage += estimateMemoryFootprint(for: model, config: config)
        enforceMemoryLimits()
        
        logger.info("✅ \(key) model loaded in \(String(format: "%.2f", duration))s | Memory: \(currentMemoryUsage)MB")
        
        await notifyConsole(modelType: key, duration: duration)
        
        return model
    }
    
    private func estimateMemoryFootprint(for model: MLModel, config: ModelConfig) -> Int {
        let base = config.quantizationBits == 4 ? 80 : 160
        return base + 50
    }
    
    private func enforceMemoryLimits() {
        if currentMemoryUsage > maxMemoryMB {
            if let oldest = loadedModels.min(by: { $0.value.timestamp < $1.value.timestamp })?.key {
                loadedModels.removeValue(forKey: oldest)
                currentMemoryUsage = max(0, currentMemoryUsage - 120)
                logger.info("Evicted oldest model due to memory pressure")
            }
        }
    }
    
    private func notifyConsole(modelType: String, duration: TimeInterval) async {
        logger.debug("Notified console: \(modelType) loaded")
    }
    
    func preloadForVoiceFusion() async {
        Task.detached(priority: .background) {
            do {
                _ = try await self.ensureEmbedLoaded(priority: .background)
                _ = try await self.ensureVoiceLoaded(priority: .background)
            } catch {
                self.logger.error("Preload failed: \(error)")
            }
        }
    }
    
    func clearCache() {
        loadedModels.removeAll()
        cache.removeAllObjects()
        currentMemoryUsage = 0
    }
}

enum ModelType: String {
    case chat, embedding, voice, gnn
}
