import Foundation
import CoreLocation

@MainActor
enum WeatherTools {
    private static let retryPolicy = ToolRetryPolicy(maxAttempts: 3, baseDelay: 0.4, maxDelay: 2.0, jitterRatio: 0.2)

    static func currentWeather(location: String? = nil) async -> String {
        let coordinateResult: WeatherCoordinateResolution
        let requestedLocation = (location ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let usesDeviceLocation = isCurrentLocationRequest(requestedLocation)

        if usesDeviceLocation {
            switch await LocationProbe.currentCoordinateResult() {
            case .success(let coordinate):
                coordinateResult = .success(coordinate)
            case .failure(let failure):
                coordinateResult = .failure(weatherLocationFailureMessage(for: failure))
            }
        } else {
            if let coordinate = await geocode(requestedLocation) {
                coordinateResult = .success(coordinate)
            } else {
                coordinateResult = .failure(weatherGeocodingFailureMessage(for: requestedLocation))
            }
        }

        let coordinate: CLLocationCoordinate2D
        switch coordinateResult {
        case .success(let resolvedCoordinate):
            coordinate = resolvedCoordinate
        case .failure(let message):
            return message
        }

        var components = URLComponents(string: "https://api.open-meteo.com/v1/forecast")
        components?.queryItems = [
            URLQueryItem(name: "latitude", value: String(format: "%.5f", coordinate.latitude)),
            URLQueryItem(name: "longitude", value: String(format: "%.5f", coordinate.longitude)),
            URLQueryItem(name: "current", value: "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,wind_speed_10m,wind_gusts_10m"),
            URLQueryItem(name: "timezone", value: "auto")
        ]

        guard let url = components?.url else {
            return "Couldn't build the weather request."
        }

        var request = URLRequest(url: url)
        request.setValue("Lumen iOS", forHTTPHeaderField: "User-Agent")

        let result = await executeRequest(endpoint: "openmeteo.current", request: request, timeout: 10, retryPolicy: retryPolicy, context: "Weather service")
        switch result {
        case .failure(let error):
            return error.localizedDescription
        case .success(let (data, _)):
            guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let current = json["current"] as? [String: Any] else {
                return ToolNetworkResilience.fallbackMessage(for: .parsing, context: "Weather service")
            }

            let temp = currentDouble(current, "temperature_2m")
            let apparent = currentDouble(current, "apparent_temperature")
            let humidity = currentDouble(current, "relative_humidity_2m")
            let wind = currentDouble(current, "wind_speed_10m")
            let gusts = currentDouble(current, "wind_gusts_10m")
            let precipitation = currentDouble(current, "precipitation")
            let cloud = currentDouble(current, "cloud_cover")
            let code = Int(currentDouble(current, "weather_code") ?? -1)
            let time = current["time"] as? String ?? "now"

            var parts: [String] = []
            parts.append("Weather \(usesDeviceLocation ? "at your location" : "for \(requestedLocation)"):")
            if code >= 0 { parts.append(weatherDescription(code)) }
            if let temp { parts.append(String(format: "%.0f°C", temp)) }
            if let apparent { parts.append(String(format: "feels like %.0f°C", apparent)) }
            if let humidity { parts.append(String(format: "humidity %.0f%%", humidity)) }
            if let wind { parts.append(String(format: "wind %.0f km/h", wind)) }
            if let gusts, gusts > (wind ?? 0) + 5 { parts.append(String(format: "gusts %.0f km/h", gusts)) }
            if let precipitation, precipitation > 0 { parts.append(String(format: "precipitation %.1f mm", precipitation)) }
            if let cloud { parts.append(String(format: "cloud cover %.0f%%", cloud)) }
            parts.append("updated \(time)")

            return parts.joined(separator: " · ")
        }
    }

    nonisolated static func isCurrentLocationRequest(_ text: String) -> Bool {
        let normalized = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: #"[\s_\-]+"#, with: " ", options: .regularExpression)
        return normalized.isEmpty
            || normalized == "current"
            || normalized == "here"
            || normalized == "current location"
            || normalized == "my location"
            || normalized == "this location"
            || normalized == "device location"
            || normalized == "near me"
    }

    nonisolated static func weatherLocationFailureMessage(for failure: LocationCoordinateFailure) -> String {
        switch failure {
        case .permissionDenied:
            return "Location access is denied. Enable Location permission for Lumen or ask with a city, for example `weather in Montreal`."
        case .permissionRestricted:
            return "Location access is restricted on this device. Ask with a city, for example `weather in Montreal`."
        case .permissionNotDetermined:
            return "Location permission has not been granted yet. Open Lumen to approve Location access, or ask with a city, for example `weather in Montreal`."
        case .timedOut:
            return "Couldn't get your current location before the GPS timeout. Try again, move somewhere with a clearer signal, or ask with a city."
        case .unavailable(let detail):
            let trimmed = detail.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                return "Couldn't get your current location from Location Services. Try again or ask with a city."
            }
            return "Couldn't get your current location from Location Services: \(trimmed)"
        case .unknownAuthorizationStatus:
            return "Couldn't determine the current Location permission state. Open Lumen to review Location permission, or ask with a city."
        }
    }

    nonisolated static func weatherGeocodingFailureMessage(for requestedLocation: String) -> String {
        let trimmed = requestedLocation.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return "I need location access, or a city name, to check the weather. Try asking `weather in Montreal` or enable Location permission."
        }
        return "Couldn't find weather coordinates for \"\(trimmed)\". Try a more specific city, region, or address."
    }

    private static func executeRequest(endpoint: String, request: URLRequest, timeout: TimeInterval, retryPolicy: ToolRetryPolicy, context: String) async -> Result<(Data, HTTPURLResponse?), any Error> {
        if !(await ToolNetworkResilience.circuitBreaker.allowRequest(endpoint: endpoint)) {
            return .failure(NSError(domain: "WeatherTools", code: 1, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: .circuitOpen, context: context)]))
        }

        var req = request
        req.timeoutInterval = timeout
        var retries = 0
        let started = Date()

        for attempt in 1...retryPolicy.maxAttempts {
            do {
                let (data, response) = try await URLSession.shared.data(for: req)
                let http = response as? HTTPURLResponse
                let errorClass = ToolNetworkResilience.classify(error: nil, response: http)
                if let status = http?.statusCode, !(200..<300).contains(status) {
                    if ToolNetworkResilience.shouldRetry(errorClass: errorClass), attempt < retryPolicy.maxAttempts {
                        retries += 1
                        try? await Task.sleep(nanoseconds: ToolNetworkResilience.backoffDelay(attempt: attempt, policy: retryPolicy))
                        continue
                    }
                    await ToolNetworkResilience.circuitBreaker.record(endpoint: endpoint, success: false)
                    ToolNetworkTelemetry.emit(.init(endpoint: endpoint, latencyMs: Date().timeIntervalSince(started) * 1000, success: false, errorClass: errorClass, retryCount: retries, statusCode: status))
                    return .failure(NSError(domain: "WeatherTools", code: 2, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: errorClass, context: context)]))
                }
                await ToolNetworkResilience.circuitBreaker.record(endpoint: endpoint, success: true)
                ToolNetworkTelemetry.emit(.init(endpoint: endpoint, latencyMs: Date().timeIntervalSince(started) * 1000, success: true, errorClass: nil, retryCount: retries, statusCode: http?.statusCode))
                return .success((data, http))
            } catch {
                let errorClass = ToolNetworkResilience.classify(error: error, response: nil)
                if ToolNetworkResilience.shouldRetry(errorClass: errorClass), attempt < retryPolicy.maxAttempts {
                    retries += 1
                    try? await Task.sleep(nanoseconds: ToolNetworkResilience.backoffDelay(attempt: attempt, policy: retryPolicy))
                    continue
                }
                await ToolNetworkResilience.circuitBreaker.record(endpoint: endpoint, success: false)
                ToolNetworkTelemetry.emit(.init(endpoint: endpoint, latencyMs: Date().timeIntervalSince(started) * 1000, success: false, errorClass: errorClass, retryCount: retries, statusCode: nil))
                return .failure(NSError(domain: "WeatherTools", code: 2, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: errorClass, context: context), NSUnderlyingErrorKey: error]))
            }
        }
        return .failure(NSError(domain: "WeatherTools", code: 3, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: .unknown, context: context)]))
    }

    private static func geocode(_ text: String) async -> CLLocationCoordinate2D? {
        await withCheckedContinuation { continuation in
            CLGeocoder().geocodeAddressString(text) { placemarks, _ in
                continuation.resume(returning: placemarks?.first?.location?.coordinate)
            }
        }
    }

    private static func currentDouble(_ dict: [String: Any], _ key: String) -> Double? {
        if let value = dict[key] as? Double { return value }
        if let value = dict[key] as? Int { return Double(value) }
        if let value = dict[key] as? String { return Double(value) }
        return nil
    }

    private static func weatherDescription(_ code: Int) -> String {
        switch code {
        case 0: return "clear sky"
        case 1: return "mainly clear"
        case 2: return "partly cloudy"
        case 3: return "overcast"
        case 45, 48: return "fog"
        case 51, 53, 55: return "drizzle"
        case 56, 57: return "freezing drizzle"
        case 61, 63, 65: return "rain"
        case 66, 67: return "freezing rain"
        case 71, 73, 75: return "snow"
        case 77: return "snow grains"
        case 80, 81, 82: return "rain showers"
        case 85, 86: return "snow showers"
        case 95: return "thunderstorm"
        case 96, 99: return "thunderstorm with hail"
        default: return "weather code \(code)"
        }
    }
}

private enum WeatherCoordinateResolution {
    case success(CLLocationCoordinate2D)
    case failure(String)
}
