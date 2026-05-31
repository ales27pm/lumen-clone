import Foundation
import Contacts

struct ContactsLookupTool: LocalTool {
    protocol Provider { func search(query: String, limit: Int, includeEmails: Bool, includePhones: Bool) throws -> [[String:String]] }
    struct CNProvider: Provider {
        func search(query: String, limit: Int, includeEmails: Bool, includePhones: Bool) throws -> [[String : String]] {
            let keys: [CNKeyDescriptor] = [CNContactGivenNameKey as CNKeyDescriptor, CNContactFamilyNameKey as CNKeyDescriptor, CNContactEmailAddressesKey as CNKeyDescriptor, CNContactPhoneNumbersKey as CNKeyDescriptor]
            let req = CNContactFetchRequest(keysToFetch: keys)
            var out: [[String:String]] = []
            try CNContactStore().enumerateContacts(with: req) { c, stop in
                let name = "\(c.givenName) \(c.familyName)".trimmingCharacters(in: .whitespaces)
                guard name.localizedCaseInsensitiveContains(query) else { return }
                var row = ["name": name]
                if includeEmails { row["emails"] = c.emailAddresses.prefix(3).map{String($0.value)}.joined(separator: ", ") }
                if includePhones { row["phones"] = c.phoneNumbers.prefix(2).map{$0.label ?? "phone"}.joined(separator: ", ") }
                out.append(row)
                if out.count >= limit { stop.pointee = true }
            }
            return out
        }
    }

    let definition = SecureToolDefinition(id: "contacts.lookup", displayName: "Lookup Contacts", description: "Lookup contacts by name", category: .permissionRead, requiredPermissions: [.contacts], supportsBackgroundExecution: false, requiresUserApproval: false, argumentSchemaDescription: "{query:string,limit?:1...10,includeEmails?:bool,includePhones?:bool}", resultPrivacyLevel: .sensitive, maxOutputCharacters: 1400)
    let provider: Provider
    init(provider: Provider = CNProvider()) { self.provider = provider }

    func validateArguments(_ arguments: [String : String]) throws { _ = try parse(arguments) }
    private func parse(_ a:[String:String]) throws -> (String,Int,Bool,Bool) {
        let q = (a["query"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard (1...120).contains(q.count) else { throw ToolExecutionError.invalidArguments("query") }
        let l = Int(a["limit"] ?? "5") ?? 5; guard (1...10).contains(l) else { throw ToolExecutionError.invalidArguments("limit") }
        let e = (a["includeEmails"] ?? "true").lowercased() == "true"
        let p = (a["includePhones"] ?? "false").lowercased() == "true"
        return (q,l,e,p)
    }

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        if !context.isForeground { return .init(invocationID: invocation.id, status: .denied, displayText: "Contacts lookup is unavailable in background.", modelText: "Contacts denied in background.", structuredPayload: nil, privacyLevel: .sensitive, metricsSummary: "bg_denied", errorCode: "bg_denied") }
        let st = await context.permissionRegistry.currentStatus(for: .contacts)
        let gate = PermissionGate.evaluate(domain: .contacts, state: st, isForeground: context.isForeground)
        guard gate.allowed else { return .init(invocationID: invocation.id, status: .denied, displayText: gate.reason ?? "Contacts permission required.", modelText: "Contacts permission required.", structuredPayload: nil, privacyLevel: .sensitive, metricsSummary: "permission_denied", errorCode: "permission") }
        return executeAfterPermissionGranted(invocation: invocation)
    }

    func executeAfterPermissionGranted(invocation: ToolInvocation) -> ToolResult {
        let parsed: (String, Int, Bool, Bool)
        do {
            parsed = try parse(invocation.arguments)
        } catch {
            return .init(invocationID: invocation.id, status: .failed, displayText: "Invalid contacts query.", modelText: "Contacts input invalid.", structuredPayload: nil, privacyLevel: .sensitive, metricsSummary: "invalid_args", errorCode: "invalid")
        }
        do {
            let (q,l,e,p) = parsed
            let rows = try provider.search(query: q, limit: l, includeEmails: e, includePhones: p)
            let text = rows.map { "- \($0["name"] ?? "")\(($0["emails"]?.isEmpty==false) ? " | emails: \($0["emails"]!)" : "")\(($0["phones"]?.isEmpty==false) ? " | phone labels: \($0["phones"]!)" : "")" }.joined(separator: "\n")
            let out = text.isEmpty ? "No contacts matched your query." : text
            return SafeToolOutputLimiter.limit(result: .init(invocationID: invocation.id, status: .success, displayText: out, modelText: out, structuredPayload: ["count":"\(rows.count)"], privacyLevel: .sensitive, metricsSummary: "contacts", errorCode: nil), maxOutput: definition.maxOutputCharacters)
        } catch {
            return .init(invocationID: invocation.id, status: .failed, displayText: "Contacts provider failed.", modelText: "Contacts provider unavailable.", structuredPayload: nil, privacyLevel: .sensitive, metricsSummary: "provider_error", errorCode: "provider_error")
        }
    }
}
