import Foundation
import SwiftUI

enum ThirdPartyNoticesDocument {
    static let resourceName = "ThirdPartyNotices"

    static let text: String = {
        guard let url = Bundle.main.url(forResource: resourceName, withExtension: "txt") else {
            return "Open-source license notices are unavailable in this build."
        }

        do {
            return try String(contentsOf: url, encoding: .utf8)
        } catch {
            return "Open-source license notices could not be read in this build."
        }
    }()
}

struct ThirdPartyNoticesView: View {
    var body: some View {
        ScrollView {
            Text(ThirdPartyNoticesDocument.text)
                .font(.footnote.monospaced())
                .foregroundStyle(Theme.textSecondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(20)
        }
        .background(AppBackground())
        .navigationTitle("Open-source licenses")
        .navigationBarTitleDisplayMode(.inline)
    }
}
