import Foundation

struct LocationMediaHealthLocalTool: LocalTool {
    static let nativeToolIDs: Set<String> = [
        "location.current",
        "weather",
        "maps.directions",
        "maps.search",
        "photos.search",
        "camera.capture",
        "health.summary",
        "motion.activity"
    ]

    @MainActor static var all: [LocationMediaHealthLocalTool] {
        ToolRegistry.all
            .filter { nativeToolIDs.contains(ToolRouteGuard.canonicalToolID($0.id)) }
            .map(LocationMediaHealthLocalTool.init)
    }

    let definition: SecureToolDefinition
    private let toolID: String

    init(_ catalogTool: ToolDefinition) {
        let canonical = ToolRouteGuard.canonicalToolID(catalogTool.id)
        self.toolID = canonical
        self.definition = SecureToolDefinition(
            id: canonical,
            displayName: catalogTool.name,
            description: catalogTool.description,
            category: Self.secureCategory(for: canonical, catalogTool: catalogTool),
            requiredPermissions: [],
            supportsBackgroundExecution: Self.supportsBackgroundExecution(canonical),
            requiresUserApproval: catalogTool.requiresApproval,
            argumentSchemaDescription: Self.argumentSchemaDescription(from: catalogTool.description),
            resultPrivacyLevel: Self.privacyLevel(for: catalogTool.category),
            maxOutputCharacters: 2_400
        )
    }

    func validateArguments(_ arguments: [String: String]) throws {}

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        let approval: ToolExecutionApproval = invocation.source == .userInitiated ? .userApproved : .autonomous
        let args = ToolRouteGuard.normalizedArguments(for: toolID, rawToolID: toolID, arguments: invocation.arguments)

        guard ToolRouteGuard.canExecuteTool(toolID, arguments: args, approval: approval) else {
            return result(invocation: invocation, text: ToolRouteGuard.approvalRequiredMessage(for: toolID), status: .requiresApproval, metricsSummary: "approval_required")
        }
        let text: String
        switch toolID {
        case "location.current":
            if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: toolID, arguments: args) {
                return result(invocation: invocation, text: permissionFailure, status: .denied, metricsSummary: "permission_denied")
            }
            text = await LocationTools.currentLocation()
        case "weather":
            if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: toolID, arguments: args) {
                return result(invocation: invocation, text: permissionFailure, status: .denied, metricsSummary: "permission_denied")
            }
            text = await WeatherTools.currentWeather(location: args["location"] ?? args["city"] ?? args["query"])
        case "maps.directions":
            if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: toolID, arguments: args) {
                return result(invocation: invocation, text: permissionFailure, status: .denied, metricsSummary: "permission_denied")
            }
            text = LocationTools.openDirections(destination: args["destination"] ?? args["query"] ?? "")
        case "maps.search":
            let query = args["query"] ?? args["location"] ?? args["destination"] ?? ""
            if ToolRouteGuard.shouldUseWebSearchInsteadOfNearbySearch(query: query) {
                let networkStatus = await context.permissionRegistry.currentStatus(for: .networkAccess)
                guard networkStatus == .granted else {
                    return result(invocation: invocation, text: "Network tools are disabled.", status: .denied, metricsSummary: "network_denied")
                }
                text = await WebTools.webSearch(query: query)
            } else {
                if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: toolID, arguments: args) {
                    return result(invocation: invocation, text: permissionFailure, status: .denied, metricsSummary: "permission_denied")
                }
                text = await LocationTools.searchNearby(query: query)
            }
        case "photos.search":
            if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: toolID, arguments: args) {
                return result(invocation: invocation, text: permissionFailure, status: .denied, metricsSummary: "permission_denied")
            }
            text = await PhotosTools.searchPhotos(query: args["query"] ?? "")
        case "camera.capture":
            if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: toolID, arguments: args) {
                return result(invocation: invocation, text: permissionFailure, status: .denied, metricsSummary: "permission_denied")
            }
            text = await PhotosTools.captureImage()
        case "health.summary":
            if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: toolID, arguments: args) {
                return result(invocation: invocation, text: permissionFailure, status: .denied, metricsSummary: "permission_denied")
            }
            text = await HealthTools.healthSummary()
        case "motion.activity":
            if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: toolID, arguments: args) {
                return result(invocation: invocation, text: permissionFailure, status: .denied, metricsSummary: "permission_denied")
            }
            text = await MotionTools.shared.motionActivity()
        default:
            text = "Unsupported native location/media/health tool: \(toolID)."
        }
        return result(invocation: invocation, text: text, status: ToolResultStatusClassifier.status(from: text), metricsSummary: "native_location_media_health_tool")
    }

    private func result(invocation: ToolInvocation, text: String, status: ToolResultStatus, metricsSummary: String) -> ToolResult {
        ToolResult(invocationID: invocation.id, status: status, displayText: text, modelText: text, structuredPayload: ["toolID": toolID, "implementation": "LocationMediaHealthLocalTool"], privacyLevel: definition.resultPrivacyLevel, metricsSummary: status == .success ? metricsSummary : "\(metricsSummary)_\(status.rawValue)", errorCode: status == .success ? nil : status.rawValue)
    }

    private static func secureCategory(for canonical: String, catalogTool: ToolDefinition) -> SecureToolCategory {
        if catalogTool.requiresApproval { return .sensitiveAction }
        switch canonical {
        case "maps.directions", "maps.search": return .userVisibleAction
        default: return .readOnly
        }
    }

    private static func supportsBackgroundExecution(_ canonical: String) -> Bool {
        switch canonical {
        case "weather", "motion.activity": return true
        default: return false
        }
    }

    private static func privacyLevel(for category: ToolCategory) -> ToolResultPrivacyLevel {
        switch category {
        case .health, .media: return .sensitive
        case .location: return .moderate
        default: return .low
        }
    }

    private static func argumentSchemaDescription(from description: String) -> String {
        guard let range = description.range(of: "Args:") else { return "{}" }
        return String(description[range.lowerBound...])
    }
}
