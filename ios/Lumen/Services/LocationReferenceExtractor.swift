import Foundation

nonisolated enum LocationReferenceExtractor {
    enum RegexError: Error, Equatable {
        case invalidPattern(String)
    }

    private static let coordinateRegexPattern = #"(-?\d{1,3}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)"#
    private static let coordinateRegexResult: Result<NSRegularExpression, RegexError> = {
        do {
            return .success(try NSRegularExpression(pattern: coordinateRegexPattern))
        } catch {
            return .failure(.invalidPattern(coordinateRegexPattern))
        }
    }()

    static func makeCoordinateRegex(pattern: String = coordinateRegexPattern) -> Result<NSRegularExpression, RegexError> {
        do {
            return .success(try NSRegularExpression(pattern: pattern))
        } catch {
            return .failure(.invalidPattern(pattern))
        }
    }

    static func coordinates(from text: String) -> (latitude: Double, longitude: Double)? {
        guard case let .success(coordinateRegex) = coordinateRegexResult else { return nil }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let match = coordinateRegex.firstMatch(in: text, range: range), match.numberOfRanges >= 3 else { return nil }

        let latText = ns.substring(with: match.range(at: 1))
        let lonText = ns.substring(with: match.range(at: 2))
        guard let latitude = Double(latText), let longitude = Double(lonText) else { return nil }
        guard (-90...90).contains(latitude), (-180...180).contains(longitude) else { return nil }
        return (latitude, longitude)
    }
}
