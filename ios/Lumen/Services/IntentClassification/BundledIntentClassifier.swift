import CoreML
import Foundation
import NaturalLanguage
import OSLog

actor BundledIntentClassifier {
    static let shared = BundledIntentClassifier()

    private nonisolated static let logger = Logger(subsystem: "ai.lumen.app", category: "intent-classifier")

    private var cachedNLModel: NLModel?
    private var cachedCoreMLModel: MLModel?
    private var attemptedSourceModelCompilation = false
    private var predictionMainThreadViolationCount = 0

    private init() {}

    func classify(_ text: String) async -> IntentClassificationResult? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let nlModel = loadNLModel(),
           let label = predictedLabel(model: nlModel, text: trimmed),
           let intent = normalizeIntentLabel(label) {
            let probs = predictedLabelHypotheses(model: nlModel, text: trimmed, maximumCount: 5)
            let alternatives = probs.compactMap { pair -> IntentAlternative? in
                guard let intent = normalizeIntentLabel(pair.key) else { return nil }
                return IntentAlternative(intent: intent, confidence: pair.value)
            }
            .sorted { $0.confidence > $1.confidence }
            return IntentClassificationResult(intent: intent, confidence: probs[label] ?? 0.0, alternatives: alternatives, requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: "nlmodel")
        }

        if let model = loadCoreMLModel(), let inferred = inferWithCoreML(model: model, text: trimmed) {
            return inferred
        }

        return nil
    }

    func resetPredictionMainThreadViolationCountForTesting() {
        predictionMainThreadViolationCount = 0
    }

    func predictionMainThreadViolationCountForTesting() -> Int {
        predictionMainThreadViolationCount
    }

    private func loadNLModel() -> NLModel? {
        if let cachedNLModel { return cachedNLModel }
        guard let url = Bundle.main.url(forResource: "IntentClassifier", withExtension: "nlmodel") else { return nil }
        let model = try? NLModel(contentsOf: url)
        cachedNLModel = model
        return model
    }

    private func loadCoreMLModel() -> MLModel? {
        if let cachedCoreMLModel { return cachedCoreMLModel }
        if let compiledURL = Bundle.main.url(forResource: "IntentClassifier", withExtension: "mlmodelc") {
            let model = try? MLModel(contentsOf: compiledURL)
            cachedCoreMLModel = model
            return model
        }

        guard !attemptedSourceModelCompilation else {
            Self.logger.debug("Skipping source IntentClassifier.mlmodel compilation because a previous actor-serialized attempt already ran.")
            return nil
        }
        attemptedSourceModelCompilation = true

        guard let sourceURL = Bundle.main.url(forResource: "IntentClassifier", withExtension: "mlmodel") else {
            Self.logger.debug("Skipping source IntentClassifier.mlmodel compilation because only compiled IntentClassifier.mlmodelc is expected in production.")
            return nil
        }

        Self.logger.warning("Compiled IntentClassifier.mlmodelc was not bundled; compiling source IntentClassifier.mlmodel once inside the classifier actor.")
        guard let compiled = try? MLModel.compileModel(at: sourceURL) else { return nil }
        let model = try? MLModel(contentsOf: compiled)
        cachedCoreMLModel = model
        return model
    }

    private func inferWithCoreML(model: MLModel, text: String) -> IntentClassificationResult? {
        guard let inputName = model.modelDescription.inputDescriptionsByName.keys.first else { return nil }
        let provider = try? MLDictionaryFeatureProvider(dictionary: [inputName: text])
        guard let provider else { return nil }
        noteIfPredictionIsOnMainThread()
        guard let output = try? model.prediction(from: provider) else { return nil }

        let label = (output.featureValue(for: "classLabel")?.stringValue) ?? (output.featureValue(for: "label")?.stringValue)
        guard let label, let intent = normalizeIntentLabel(label) else { return nil }

        let candidates = ["classProbability", "labelProbabilities", "probabilities"]
        var probs: [String: Double] = [:]
        for key in candidates {
            if let dict = stringDoubleDictionary(from: output.featureValue(for: key)?.dictionaryValue) {
                probs = dict
                break
            }
        }
        let alternatives = probs.compactMap { pair -> IntentAlternative? in
            guard let intent = normalizeIntentLabel(pair.key) else { return nil }
            return IntentAlternative(intent: intent, confidence: pair.value)
        }
        .sorted { $0.confidence > $1.confidence }
        let confidence = probs[label] ?? alternatives.first(where: { $0.intent == intent })?.confidence ?? 0.0
        return IntentClassificationResult(intent: intent, confidence: confidence, alternatives: alternatives, requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: "coreml")
    }

    private func predictedLabel(model: NLModel, text: String) -> String? {
        noteIfPredictionIsOnMainThread()
        return model.predictedLabel(for: text)
    }

    private func predictedLabelHypotheses(model: NLModel, text: String, maximumCount: Int) -> [String: Double] {
        noteIfPredictionIsOnMainThread()
        return model.predictedLabelHypotheses(for: text, maximumCount: maximumCount)
    }

    private func noteIfPredictionIsOnMainThread() {
        if Thread.isMainThread {
            predictionMainThreadViolationCount += 1
            Self.logger.fault("Bundled intent model prediction executed on the main thread.")
        }
    }

    private func normalizeIntentLabel(_ raw: String) -> UserIntent? {
        let canonical = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: "_", with: "")
            .lowercased()
        if let exact = UserIntent(rawValue: raw) { return exact }
        return UserIntent.allCases.first { $0.rawValue.lowercased() == canonical }
    }

    private func stringDoubleDictionary(from raw: [AnyHashable: Any]?) -> [String: Double]? {
        guard let raw else { return nil }
        var out: [String: Double] = [:]
        out.reserveCapacity(raw.count)
        for (key, value) in raw {
            guard let stringKey = key as? String else { continue }
            if let number = value as? NSNumber {
                out[stringKey] = number.doubleValue
            } else if let doubleValue = value as? Double {
                out[stringKey] = doubleValue
            } else if let floatValue = value as? Float {
                out[stringKey] = Double(floatValue)
            } else if let intValue = value as? Int {
                out[stringKey] = Double(intValue)
            }
        }
        return out.isEmpty ? nil : out
    }
}
