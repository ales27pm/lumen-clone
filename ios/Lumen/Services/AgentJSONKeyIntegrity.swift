import Foundation

/// JSONSerialization collapses duplicate keys. Check the selected, already parsed
/// JSON before treating its forecast as unambiguous model-authored evidence.
nonisolated enum AgentJSONKeyIntegrity {
    /// The compatibility input form decodes a second JSON document. Inspect the
    /// exact string chosen by parseArgs, not escaped key-like text in ordinary values.
    static func hasAmbiguousEmbeddedInput(in object: [String: Any]) -> Bool {
        guard object["input"] != nil,
              let input = (object["args"] ?? object["arguments"] ?? object["input"]) as? String,
              let data = input.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8),
              let decoded = try? JSONSerialization.jsonObject(with: data),
              decoded is [String: Any] else { return false }
        return hasAmbiguousKeys(in: input)
    }

    static func hasAmbiguousKeys(in json: String) -> Bool {
        let bytes = Array(json.utf8)
        var objects: [Set<String>?] = [] // nil denotes an array, not an object.
        var index = 0
        while index < bytes.count {
            switch bytes[index] {
            case 123: objects.append([]) // {
            case 91: objects.append(nil) // [
            case 125, 93:
                guard !objects.isEmpty else { return true }
                objects.removeLast()
            case 34:
                let start = index
                index += 1
                while index < bytes.count, bytes[index] != 34 {
                    if bytes[index] == 92 { index += 1 }
                    index += 1
                }
                guard index < bytes.count else { return true }
                var next = index + 1
                while next < bytes.count, [9, 10, 13, 32].contains(bytes[next]) { next += 1 }
                if next < bytes.count, bytes[next] == 58 {
                    guard !objects.isEmpty, var keys = objects[objects.count - 1],
                          let key = try? JSONSerialization.jsonObject(
                            with: Data(bytes[start...index]), options: [.fragmentsAllowed]
                          ) as? String else { return true }
                    guard keys.insert(key).inserted else { return true }
                    objects[objects.count - 1] = keys
                }
            default: break
            }
            index += 1
        }
        return !objects.isEmpty
    }
}
