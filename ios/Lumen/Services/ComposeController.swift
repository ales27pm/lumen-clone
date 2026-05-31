import Foundation
import UIKit
import MessageUI

@MainActor
final class ComposeController: NSObject {
    static let shared = ComposeController()

    private var messageContinuation: CheckedContinuation<String, Never>?
    private var mailContinuation: CheckedContinuation<String, Never>?
    private var presentedController: UIViewController?

    func composeMessage(to recipients: [String], body: String) async -> String {
        if MFMessageComposeViewController.canSendText() {
            return await withCheckedContinuation { cont in
                self.messageContinuation = cont
                let vc = MFMessageComposeViewController()
                vc.messageComposeDelegate = self
                vc.recipients = recipients.filter { !$0.isEmpty }
                vc.body = body
                self.present(vc)
            }
        }
        return await fallbackSMS(recipients: recipients, body: body)
    }

    func composeMail(to recipients: [String], subject: String, body: String) async -> String {
        if MFMailComposeViewController.canSendMail() {
            return await withCheckedContinuation { cont in
                self.mailContinuation = cont
                let vc = MFMailComposeViewController()
                vc.mailComposeDelegate = self
                vc.setToRecipients(recipients.filter { !$0.isEmpty })
                vc.setSubject(subject)
                vc.setMessageBody(body, isHTML: false)
                self.present(vc)
            }
        }
        return await fallbackMailto(recipients: recipients, subject: subject, body: body)
    }

    private func present(_ vc: UIViewController) {
        guard let scene = UIApplication.shared.connectedScenes.first(where: { $0.activationState == .foregroundActive }) as? UIWindowScene,
              let root = scene.keyWindow?.rootViewController ?? scene.windows.first?.rootViewController else {
            finishMessage(with: "No window available to present composer.")
            finishMail(with: "No window available to present composer.")
            return
        }
        var top = root
        while let presented = top.presentedViewController { top = presented }
        self.presentedController = vc
        top.present(vc, animated: true)
    }

    private func dismiss() {
        presentedController?.dismiss(animated: true)
        presentedController = nil
    }

    fileprivate func finishMessage(with result: String) {
        let cont = messageContinuation
        messageContinuation = nil
        cont?.resume(returning: result)
    }

    fileprivate func finishMail(with result: String) {
        let cont = mailContinuation
        mailContinuation = nil
        cont?.resume(returning: result)
    }

    private func fallbackSMS(recipients: [String], body: String) async -> String {
        let to = recipients
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: ",")

        var components = URLComponents()
        components.scheme = "sms"
        components.path = to

        if !body.isEmpty {
            components.queryItems = [URLQueryItem(name: "body", value: body)]
        }

        guard let url = components.url ?? URL(string: "sms:\(to)") else {
            return "This device can't send messages."
        }
        let opened = await UIApplication.shared.open(url)
        return opened ? "Opened Messages with draft to \(to.isEmpty ? "recipient" : to)." : "Couldn't open Messages."
    }

    private func fallbackMailto(recipients: [String], subject: String, body: String) async -> String {
        let to = recipients.joined(separator: ",")
        var comps = URLComponents(string: "mailto:\(to)")
        var items: [URLQueryItem] = []
        if !subject.isEmpty { items.append(URLQueryItem(name: "subject", value: subject)) }
        if !body.isEmpty { items.append(URLQueryItem(name: "body", value: body)) }
        if !items.isEmpty { comps?.queryItems = items }
        guard let url = comps?.url else { return "Couldn't build mail URL." }
        let opened = await UIApplication.shared.open(url)
        return opened ? "Opened Mail with draft\(to.isEmpty ? "" : " to \(to)")." : "Couldn't open Mail — no accounts configured."
    }
}

extension ComposeController: MFMessageComposeViewControllerDelegate {
    nonisolated func messageComposeViewController(_ controller: MFMessageComposeViewController, didFinishWith result: MessageComposeResult) {
        let outcome: String
        switch result {
        case .sent: outcome = "Message sent."
        case .cancelled: outcome = "Message cancelled."
        case .failed: outcome = "Message failed to send."
        @unknown default: outcome = "Message composer closed."
        }
        Task { @MainActor in
            self.dismiss()
            self.finishMessage(with: outcome)
        }
    }
}

extension ComposeController: MFMailComposeViewControllerDelegate {
    nonisolated func mailComposeController(_ controller: MFMailComposeViewController, didFinishWith result: MFMailComposeResult, error: Error?) {
        let outcome: String
        switch result {
        case .sent: outcome = "Email sent."
        case .saved: outcome = "Email saved to Drafts."
        case .cancelled: outcome = "Email cancelled."
        case .failed: outcome = "Email failed: \(error?.localizedDescription ?? "unknown error")"
        @unknown default: outcome = "Mail composer closed."
        }
        Task { @MainActor in
            self.dismiss()
            self.finishMail(with: outcome)
        }
    }
}
