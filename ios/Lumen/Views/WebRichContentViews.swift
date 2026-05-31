import SwiftUI
import WebKit

struct WebRichContentStack: View {
    let payloads: [WebRichContentPayload]

    var body: some View {
        if !payloads.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(Array(payloads.enumerated()), id: \.offset) { _, payload in
                    WebRichPayloadCard(payload: payload)
                }
            }
        }
    }
}

private struct WebRichPayloadCard: View {
    let payload: WebRichContentPayload

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header

            if let page = payload.page {
                WebFetchedPageCard(page: page)
            }

            if !payload.results.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(payload.results.prefix(6).enumerated()), id: \.offset) { _, result in
                        WebSearchResultCard(result: result)
                    }
                }
            }

            if !payload.media.isEmpty {
                WebMediaRail(media: payload.media)
            }
        }
        .padding(10)
        .background(Theme.surfaceHigh)
        .clipShape(.rect(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: payload.kind == .searchResults ? "magnifyingglass" : "doc.text.magnifyingglass")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.accent)
            VStack(alignment: .leading, spacing: 1) {
                Text(payload.kind == .searchResults ? "Web results" : "Fetched page")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                if let query = payload.query, !query.isEmpty {
                    Text(query)
                        .font(.caption2)
                        .foregroundStyle(Theme.textSecondary)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 0)
            Text(payload.generatedAt, style: .time)
                .font(.caption2.monospacedDigit())
                .foregroundStyle(Theme.textTertiary)
        }
    }
}

private struct WebSearchResultCard: View {
    let result: WebSearchResultPayload

    var body: some View {
        Group {
            if let urlString = result.url, let url = safeExternalURL(urlString) {
                Link(destination: url) { content(url: url) }
            } else {
                content(url: nil)
            }
        }
        .buttonStyle(.plain)
        .padding(9)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.border.opacity(0.7), lineWidth: 1)
        }
    }

    private func content(url: URL?) -> some View {
        HStack(alignment: .top, spacing: 10) {
            mediaIcon
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.textSecondary)
                .frame(width: 20, height: 20)

            VStack(alignment: .leading, spacing: 4) {
                Text(result.title)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(2)
                if let snippet = result.snippet, !snippet.isEmpty {
                    Text(snippet)
                        .font(.caption2)
                        .foregroundStyle(Theme.textSecondary)
                        .lineLimit(3)
                }
                Text(result.source ?? url?.host() ?? result.url ?? "Web")
                    .font(.caption2.monospaced())
                    .foregroundStyle(Theme.textTertiary)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
            if url != nil {
                Image(systemName: "arrow.up.right")
                    .font(.caption2)
                    .foregroundStyle(Theme.textTertiary)
            }
        }
    }

    private var mediaIcon: Image {
        switch result.mediaKind {
        case .image: Image(systemName: "photo")
        case .video: Image(systemName: "play.rectangle")
        case .pdf: Image(systemName: "doc.richtext")
        case .page, .none: Image(systemName: "globe")
        }
    }
}

private struct WebFetchedPageCard: View {
    let page: WebFetchedPagePayload

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "safari")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.textSecondary)
                    .frame(width: 20, height: 20)
                VStack(alignment: .leading, spacing: 3) {
                    Text(page.title ?? page.siteName ?? URL(string: page.url)?.host() ?? "Fetched page")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                        .lineLimit(2)
                    if let description = page.description, !description.isEmpty {
                        Text(description)
                            .font(.caption2)
                            .foregroundStyle(Theme.textSecondary)
                            .lineLimit(3)
                    }
                    Text(URL(string: page.url)?.host() ?? page.url)
                        .font(.caption2.monospaced())
                        .foregroundStyle(Theme.textTertiary)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
            }

            Text(page.excerpt)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
                .lineLimit(8)
                .textSelection(.enabled)

            if let url = safeExternalURL(page.url) {
                Link(destination: url) {
                    Label("Open page", systemImage: "safari")
                        .font(.caption.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(9)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.border.opacity(0.7), lineWidth: 1)
        }
    }
}

private struct WebMediaRail: View {
    let media: [WebMediaPayload]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Media")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.textPrimary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(media.prefix(10)) { item in
                        WebMediaCard(item: item)
                    }
                }
            }
        }
    }
}

private struct WebMediaCard: View {
    let item: WebMediaPayload

    var body: some View {
        Group {
            if let url = safeExternalURL(item.url) {
                Link(destination: url) { card }
            } else {
                card
            }
        }
        .buttonStyle(.plain)
    }

    private var card: some View {
        VStack(alignment: .leading, spacing: 6) {
            ZStack {
                RoundedRectangle(cornerRadius: 9)
                    .fill(Theme.surface)
                if item.kind == .image, let url = URL(string: item.thumbnailURL ?? item.url) {
                    AsyncImage(url: url) { phase in
                        switch phase {
                        case .success(let image):
                            image.resizable().scaledToFill()
                        case .failure:
                            icon
                        case .empty:
                            ProgressView().scaleEffect(0.7)
                        @unknown default:
                            icon
                        }
                    }
                    .frame(width: 130, height: 82)
                    .clipped()
                    .clipShape(.rect(cornerRadius: 9))
                } else {
                    icon
                }
            }
            .frame(width: 130, height: 82)
            .overlay {
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(Theme.border.opacity(0.7), lineWidth: 1)
            }

            Text(item.title ?? item.kind.rawValue.capitalized)
                .font(.caption2.weight(.medium))
                .foregroundStyle(Theme.textPrimary)
                .lineLimit(1)
                .frame(width: 130, alignment: .leading)
        }
    }

    private var icon: some View {
        Image(systemName: iconName)
            .font(.title3)
            .foregroundStyle(Theme.textSecondary)
    }

    private var iconName: String {
        switch item.kind {
        case .image: "photo"
        case .video: "play.rectangle"
        case .pdf: "doc.richtext"
        case .page: "globe"
        }
    }
}


private func safeExternalURL(_ raw: String) -> URL? {
    guard let url = URL(string: raw), let scheme = url.scheme?.lowercased() else { return nil }
    guard scheme == "http" || scheme == "https" else { return nil }
    return url
}
