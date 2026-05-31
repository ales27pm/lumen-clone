import SwiftUI
import UIKit

struct OutlookMailView: View {
    @State private var auth = MicrosoftGraphAuthManager()
    @State private var viewModel: MicrosoftGraphInboxViewModel?
    @State private var selectedMessage: GraphMailMessage?
    @State private var showingCompose = false
    @State private var composeTo = ""
    @State private var composeSubject = ""
    @State private var composeBody = ""
    @State private var composeError: String?
    @State private var isSending = false
    @State private var microsoftClientID = MicrosoftGraphRuntimeConfig.loadClientIDOverride() ?? ""
    @State private var effectiveConfig = try? MicrosoftGraphConfiguration.load()

    var body: some View {
        Group {
            if auth.isSignedIn {
                inboxContent
            } else {
                signInContent
            }
        }
        .navigationTitle("Outlook")
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                if auth.isSignedIn {
                    Button {
                        showingCompose = true
                    } label: {
                        Label("Compose", systemImage: "square.and.pencil")
                    }
                    Button {
                        Task { await viewModel?.refresh(resetDelta: true) }
                    } label: {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                }
            }
        }
        .task {
            await auth.bootstrap()
            ensureViewModel()
            viewModel?.loadCached()
            if auth.isSignedIn { await viewModel?.refresh() }
        }
        .sheet(isPresented: $showingCompose) { composeSheet }
        .sheet(item: $selectedMessage) { message in
            MailMessageDetailView(message: message, auth: auth)
        }
    }

    private var signInContent: some View {
        VStack(spacing: 20) {
            Image(systemName: "envelope.badge.shield.half.filled")
                .font(.system(size: 52))
                .foregroundStyle(Theme.accent)
            VStack(spacing: 8) {
                Text("Connect Hotmail / Outlook")
                    .font(.title2.weight(.semibold))
                Text("Use Microsoft Graph for Outlook.com, Hotmail, Live, MSN, and Entra ID mailboxes. Auth provider: \(auth.authProviderDescription).")
                    .font(.subheadline)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
            }

            if !auth.canUseMSAL {
                ContentUnavailableView(
                    "Using native OAuth fallback",
                    systemImage: "key.radiowaves.forward",
                    description: Text("MSAL is not linked in this build, so Lumen will sign in with ASWebAuthenticationSession + PKCE and store refresh tokens in the Keychain.")
                )
            }

            if let error = auth.lastError {
                Text(error.localizedDescription)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
            }
            
            debugConfigurationSection

            VStack(alignment: .leading, spacing: 8) {
                Text("Microsoft Entra client ID")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                TextField("Enter app client ID", text: $microsoftClientID)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .textFieldStyle(.roundedBorder)
                HStack {
                    Button("Use this client ID") {
                        MicrosoftGraphRuntimeConfig.saveClientIDOverride(microsoftClientID)
                        effectiveConfig = try? MicrosoftGraphConfiguration.load()
                        auth.registerExternalError(MicrosoftGraphAuthError.invalidConfiguration("Saved client ID for this device. Tap Sign in with Microsoft to continue."))
                    }
                    .buttonStyle(.bordered)

                    if MicrosoftGraphRuntimeConfig.loadClientIDOverride() != nil {
                        Button("Reset", role: .destructive) {
                            microsoftClientID = ""
                            MicrosoftGraphRuntimeConfig.saveClientIDOverride(nil)
                            effectiveConfig = try? MicrosoftGraphConfiguration.load()
                            auth.registerExternalError(MicrosoftGraphAuthError.invalidConfiguration("Cleared saved client ID override."))
                        }
                        .buttonStyle(.bordered)
                    }
                }
                if let effectiveConfig {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Effective client ID: \(effectiveConfig.clientID)")
                        Text("Effective redirect URI: \(effectiveConfig.redirectURI ?? defaultRedirectURI)")
                        Text("Effective authority URL: \(effectiveConfig.authorityURL.absoluteString)")
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                }
            }

            Button {
                Task { await signIn() }
            } label: {
                if auth.isAuthenticating {
                    ProgressView()
                } else {
                    Label("Sign in with Microsoft", systemImage: "person.crop.circle.badge.checkmark")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(auth.isAuthenticating)
        }
        .padding(24)
        .frame(maxWidth: 520)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var defaultRedirectURI: String {
        "msauth.\(Bundle.main.bundleIdentifier ?? "com.27pm.lumen")://auth"
    }

    private var shouldShowDebugConfiguration: Bool {
        #if DEBUG
        return true
        #else
        return Bundle.main.appStoreReceiptURL?.lastPathComponent == "sandboxReceipt"
        #endif
    }

    @ViewBuilder
    private var debugConfigurationSection: some View {
        if shouldShowDebugConfiguration {
            VStack(alignment: .leading, spacing: 6) {
                Text("Debug configuration")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Auth provider path: \(auth.authProviderPath)")
                    Text("Client ID: \(auth.activeClientID)")
                    Text("Redirect URI: \(auth.activeRedirectURI)")
                    Text("Bundle ID: \(auth.bundleIdentifier)")
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            }
        }
    }

    private var inboxContent: some View {
        List {
            Section {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(auth.account?.username ?? auth.account?.name ?? "Microsoft account")
                            .font(.headline)
                        if let lastSync = viewModel?.lastSyncDate {
                            Text("Last sync: \(lastSync.formatted(date: .abbreviated, time: .shortened))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else {
                            Text("Delta sync not completed yet")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    if viewModel?.isLoading == true { ProgressView() }
                }
                Toggle("Unread only", isOn: Binding(
                    get: { viewModel?.unreadOnly ?? false },
                    set: { newValue in
                        viewModel?.unreadOnly = newValue
                        Task { await viewModel?.refresh(resetDelta: true) }
                    }
                ))
            }

            if let error = viewModel?.error {
                Section {
                    Label(error.localizedDescription, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                }
            }

            Section("Inbox") {
                ForEach(viewModel?.messages ?? []) { message in
                    Button {
                        selectedMessage = message
                    } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack(alignment: .firstTextBaseline) {
                                Text(message.subjectText)
                                    .font(.headline)
                                    .lineLimit(1)
                                Spacer()
                                if message.hasAttachments == true {
                                    Image(systemName: "paperclip")
                                        .foregroundStyle(.secondary)
                                }
                            }
                            Text(message.senderLine)
                                .font(.subheadline.weight(.medium))
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                            Text(message.previewLine)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        }
        .refreshable { await viewModel?.refresh() }
        .safeAreaInset(edge: .bottom) {
            HStack {
                Button("Sign Out", role: .destructive) {
                    Task {
                        await auth.signOutCurrentAccount()
                        viewModel = nil
                    }
                }
                Spacer()
                Text("Graph v1.0 · Delta sync · \(auth.authProviderDescription)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial)
        }
    }

    private var composeSheet: some View {
        NavigationStack {
            Form {
                Section("Recipients") {
                    TextField("email@example.com, second@example.com", text: $composeTo)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.emailAddress)
                }
                Section("Message") {
                    TextField("Subject", text: $composeSubject)
                    TextEditor(text: $composeBody)
                        .frame(minHeight: 180)
                }
                if let composeError {
                    Section { Text(composeError).foregroundStyle(.red) }
                }
            }
            .navigationTitle("New Email")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { showingCompose = false }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(isSending ? "Sending…" : "Send") {
                        Task { await sendCompose() }
                    }
                    .disabled(isSending || composeTo.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || composeSubject.isEmpty)
                }
            }
        }
    }

    private func ensureViewModel() {
        if viewModel == nil { viewModel = MicrosoftGraphInboxViewModel(auth: auth) }
    }

    private func signIn() async {
        guard let presenter = MicrosoftGraphPresenter.topViewController() else {
            auth.registerExternalError(MicrosoftGraphAuthError.presentationAnchorUnavailable)
            return
        }
        do {
            try await auth.signIn(presentationViewController: presenter)
            ensureViewModel()
            viewModel?.loadCached()
            await viewModel?.refresh()
        } catch {
            auth.registerExternalError(error)
        }
    }

    private func sendCompose() async {
        ensureViewModel()
        isSending = true
        defer { isSending = false }
        do {
            let recipients = composeTo
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            try await viewModel?.send(subject: composeSubject, body: composeBody, recipients: recipients, sendAsHTML: true)
            composeTo = ""
            composeSubject = ""
            composeBody = ""
            composeError = nil
            showingCompose = false
        } catch {
            composeError = error.localizedDescription
        }
    }
}

private struct MailMessageDetailView: View {
    let message: GraphMailMessage
    let auth: MicrosoftGraphAuthManager
    @State private var loadedMessage: GraphMailMessage?
    @State private var isLoading = false
    @State private var error: Error?
    private let client = MicrosoftGraphMailClient()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text((loadedMessage ?? message).subjectText)
                        .font(.title2.weight(.semibold))
                    VStack(alignment: .leading, spacing: 4) {
                        Text("From: \((loadedMessage ?? message).senderLine)")
                        if let received = (loadedMessage ?? message).receivedDateTime {
                            Text(received)
                        }
                    }
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                    if isLoading { ProgressView("Loading body…") }
                    if let error { Text(error.localizedDescription).foregroundStyle(.red) }

                    MessageBodyContentView(message: loadedMessage ?? message)
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .navigationTitle("Message")
            .navigationBarTitleDisplayMode(.inline)
            .task { await loadBody() }
        }
    }

    private func loadBody() async {
        guard loadedMessage == nil else { return }
        isLoading = true
        defer { isLoading = false }
        do {
            let token = try await auth.acquireToken(scopes: MicrosoftGraphScope.inboxRead, preferredAccountID: auth.account?.id)
            loadedMessage = try await client.fetchMessageBody(messageID: message.id, accessToken: token)
        } catch {
            self.error = error
        }
    }
}

private struct MessageBodyContentView: View {
    let message: GraphMailMessage

    var body: some View {
        let content = message.body?.content ?? message.previewLine
        let bodyType = message.body?.contentType.lowercased() ?? "text"
        Group {
            if bodyType == "html" {
                HTMLTextView(html: content)
            } else {
                Text(content)
                    .font(.body)
                    .textSelection(.enabled)
            }
        }
    }
}

private struct HTMLTextView: UIViewRepresentable {
    let html: String

    func makeUIView(context: Context) -> UITextView {
        let textView = UITextView()
        textView.isEditable = false
        textView.isScrollEnabled = false
        textView.backgroundColor = .clear
        textView.textContainerInset = .zero
        textView.textContainer.lineFragmentPadding = 0
        return textView
    }

    func updateUIView(_ uiView: UITextView, context: Context) {
        guard let data = html.data(using: .utf8) else {
            uiView.text = html
            return
        }
        if let attributed = try? NSAttributedString(
            data: data,
            options: [
                .documentType: NSAttributedString.DocumentType.html,
                .characterEncoding: String.Encoding.utf8.rawValue
            ],
            documentAttributes: nil
        ) {
            uiView.attributedText = attributed
        } else {
            uiView.text = html
        }
    }
}
