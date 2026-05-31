import Foundation

enum LLMResponseFormat: Sendable, Codable, Equatable {
    case plainText
    case json
    case toolCallJSON
    case constrainedJSON(schema: String)

    private enum CodingKeys: String, CodingKey {
        case type
        case schema
    }

    private enum Kind: String, Codable {
        case plainText
        case json
        case toolCallJSON
        case constrainedJSON
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(Kind.self, forKey: .type)
        switch type {
        case .plainText:
            self = .plainText
        case .json:
            self = .json
        case .toolCallJSON:
            self = .toolCallJSON
        case .constrainedJSON:
            self = .constrainedJSON(schema: try container.decode(String.self, forKey: .schema))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .plainText:
            try container.encode(Kind.plainText, forKey: .type)
        case .json:
            try container.encode(Kind.json, forKey: .type)
        case .toolCallJSON:
            try container.encode(Kind.toolCallJSON, forKey: .type)
        case .constrainedJSON(let schema):
            try container.encode(Kind.constrainedJSON, forKey: .type)
            try container.encode(schema, forKey: .schema)
        }
    }
}
