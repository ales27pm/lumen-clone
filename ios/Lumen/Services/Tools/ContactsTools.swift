import Foundation
import Contacts
import UIKit

@MainActor
enum ContactsTools {
    static func searchContacts(query: String) async -> String {
        let store = CNContactStore()
        do {
            let granted = try await store.requestAccess(for: .contacts)
            guard granted else { return "Contacts access was denied." }
            let predicate = CNContact.predicateForContacts(matchingName: query)
            let keys: [CNKeyDescriptor] = [
                CNContactGivenNameKey as CNKeyDescriptor,
                CNContactFamilyNameKey as CNKeyDescriptor,
                CNContactPhoneNumbersKey as CNKeyDescriptor,
            ]
            let results = try store.unifiedContacts(matching: predicate, keysToFetch: keys).prefix(5)
            if results.isEmpty { return "No contacts match \"\(query)\"." }
            return results.map { c in
                let name = [c.givenName, c.familyName].filter { !$0.isEmpty }.joined(separator: " ")
                let phone = c.phoneNumbers.first?.value.stringValue ?? "no phone"
                return "• \(name) — \(phone)"
            }.joined(separator: "\n")
        } catch {
            return "Couldn't search contacts: \(error.localizedDescription)"
        }
    }

    static func call(number: String) async -> String {
        let trimmed = number.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalized = trimmed.replacingOccurrences(of: "[^0-9+]", with: "", options: .regularExpression)
        guard !normalized.isEmpty, let url = URL(string: "tel://\(normalized)") else {
            return "No phone number provided."
        }
        let ok = await UIApplication.shared.open(url)
        return ok ? "Calling \(normalized)…" : "Couldn't start call to \(normalized)."
    }

    static func composeMessage(arguments: [String: String]) async -> String {
        let body = arguments["body"] ?? arguments["message"] ?? arguments["text"] ?? ""
        let toRaw = arguments["to"] ?? arguments["recipient"] ?? arguments["number"] ?? ""
        let recipients = toRaw.split(whereSeparator: { ",;".contains($0) }).map { $0.trimmingCharacters(in: .whitespaces) }
        return await ComposeController.shared.composeMessage(to: recipients, body: body)
    }

    static func composeMail(arguments: [String: String]) async -> String {
        let body = arguments["body"] ?? arguments["message"] ?? arguments["text"] ?? ""
        let subject = arguments["subject"] ?? arguments["title"] ?? ""
        let toRaw = arguments["to"] ?? arguments["recipient"] ?? arguments["email"] ?? ""
        let recipients = toRaw.split(whereSeparator: { ",;".contains($0) }).map { $0.trimmingCharacters(in: .whitespaces) }
        return await ComposeController.shared.composeMail(to: recipients, subject: subject, body: body)
    }
}
