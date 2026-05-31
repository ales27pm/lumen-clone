import Foundation

nonisolated enum AgentJSONValue: Sendable, Hashable, Codable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case array([AgentJSONValue])
    case object([String: AgentJSONValue])
    case null

    static func parse(_ value: Any) -> AgentJSONValue? {
        if value is NSNull { return .null }
        if let str = value as? String { return .string(str) }
        if let bool = value as? Bool { return .bool(bool) }
        if let num = value as? NSNumber {
            if CFGetTypeID(num) == CFBooleanGetTypeID() {
                return .bool(num.boolValue)
            }
            return .number(num.doubleValue)
        }
        if let arr = value as? [Any] {
            let parsed = arr.compactMap(AgentJSONValue.parse)
            return parsed.count == arr.count ? .array(parsed) : nil
        }
        if let obj = value as? [String: Any] {
            var parsed: [String: AgentJSONValue] = [:]
            for (key, innerValue) in obj {
                guard let inner = AgentJSONValue.parse(innerValue) else { return nil }
                parsed[key] = inner
            }
            return .object(parsed)
        }
        return nil
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([AgentJSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: AgentJSONValue].self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .array(let values):
            try container.encode(values)
        case .object(let values):
            try container.encode(values)
        case .null:
            try container.encodeNil()
        }
    }

    var stringValue: String {
        switch self {
        case .string(let value):
            return value
        case .number(let value):
            if let intValue = Int(exactly: value) { return String(intValue) }
            return String(value)
        case .bool(let value):
            return value ? "true" : "false"
        case .array(let values):
            return values.jsonRenderedString ?? "[" + values.map(\.stringValue).joined(separator: ",") + "]"
        case .object(let values):
            if let rendered = values.jsonRenderedString { return rendered }
            let fallback = values.keys.sorted().map { key in
                let value = values[key]?.stringValue ?? "null"
                return "\"\(key)\":\(value)"
            }.joined(separator: ",")
            return "{\(fallback)}"
        case .null:
            return "null"
        }
    }

    var intValue: Int? {
        switch self {
        case .number(let value):
            return Int(exactly: value)
        case .string(let value):
            return Int(value)
        case .bool(let value):
            return value ? 1 : 0
        default:
            return nil
        }
    }

    var boolValue: Bool? {
        switch self {
        case .bool(let value):
            return value
        case .string(let value):
            let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if normalized == "true" { return true }
            if normalized == "false" { return false }
            return nil
        case .number(let value):
            return value != 0
        default:
            return nil
        }
    }
}

typealias AgentJSONArguments = [String: AgentJSONValue]

extension Dictionary where Key == String, Value == AgentJSONValue {
    init(stringDictionary: [String: String]) {
        self = stringDictionary.reduce(into: [:]) { partialResult, element in
            partialResult[element.key] = .string(element.value)
        }
    }

    nonisolated var stringCoerced: [String: String] {
        reduce(into: [String: String]()) { partialResult, element in
            partialResult[element.key] = element.value.stringValue
        }
    }

    nonisolated fileprivate var jsonRenderedString: String? {
        let foundationObject = reduce(into: [String: Any]()) { partialResult, element in
            partialResult[element.key] = element.value.foundationObject
        }
        guard JSONSerialization.isValidJSONObject(foundationObject),
              let data = try? JSONSerialization.data(withJSONObject: foundationObject, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }
        return json
    }
}

extension Array where Element == AgentJSONValue {
    nonisolated fileprivate var jsonRenderedString: String? {
        let foundationObject = map(\.foundationObject)
        guard JSONSerialization.isValidJSONObject(foundationObject),
              let data = try? JSONSerialization.data(withJSONObject: foundationObject, options: []),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }
        return json
    }
}

private extension AgentJSONValue {
    nonisolated var foundationObject: Any {
        switch self {
        case .string(let value):
            return value
        case .number(let value):
            return NSNumber(value: value)
        case .bool(let value):
            return NSNumber(value: value)
        case .array(let values):
            return values.map(\.foundationObject)
        case .object(let value):
            return value.reduce(into: [String: Any]()) { partialResult, element in
                partialResult[element.key] = element.value.foundationObject
            }
        case .null:
            return NSNull()
        }
    }
}
