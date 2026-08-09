import SwiftUI
import WebKit

struct AlgorithmicPhilosophiesView: View {
    @State private var selectedArtifactID = AlgorithmicPhilosophyCatalog.all.first?.id
    @State private var selectedPanel: ArtifactPanel = .viewer

    private var selectedArtifact: AlgorithmicPhilosophyArtifact? {
        AlgorithmicPhilosophyCatalog.all.first { $0.id == selectedArtifactID } ?? AlgorithmicPhilosophyCatalog.all.first
    }

    var body: some View {
        VStack(spacing: 0) {
            artifactPicker
                .padding(.horizontal, 20)
                .padding(.top, 18)
                .padding(.bottom, 12)

            Divider()
                .overlay(Theme.border.opacity(0.55))

            if let selectedArtifact {
                ArtifactDetailView(artifact: selectedArtifact, selectedPanel: $selectedPanel)
            } else {
                ContentUnavailableView(
                    "No philosophies bundled",
                    systemImage: "sparkles.rectangle.stack",
                    description: Text("Bundled algorithmic philosophy artifacts will appear here when they are included in the app resources.")
                )
            }
        }
        .background(AppBackground())
        .navigationTitle("Philosophies")
    }

    private var artifactPicker: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Algorithmic Philosophies")
                        .font(.title2.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                    Text("Living generative movements bundled as philosophy, viewer, and source code.")
                        .font(.subheadline)
                        .foregroundStyle(Theme.textSecondary)
                }
                Spacer()
                Text("\(AlgorithmicPhilosophyCatalog.all.count) movement")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.accent)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Theme.accent.opacity(0.12), in: Capsule())
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(AlgorithmicPhilosophyCatalog.all) { artifact in
                        Button {
                            selectedArtifactID = artifact.id
                            selectedPanel = .viewer
                        } label: {
                            ArtifactCard(
                                artifact: artifact,
                                isSelected: artifact.id == selectedArtifactID
                            )
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("philosophy.card.\(artifact.id)")
                    }
                }
                .padding(.vertical, 2)
            }
        }
    }
}

private enum ArtifactPanel: String, CaseIterable, Identifiable {
    case viewer = "Viewer"
    case trace = "Trace"
    case philosophy = "Philosophy"
    case algorithm = "Algorithm"
    case reflection = "Reflection"

    var id: String { rawValue }
}

private struct ArtifactCard: View {
    let artifact: AlgorithmicPhilosophyArtifact
    let isSelected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "sparkles")
                    .foregroundStyle(Theme.accent)
                Spacer()
                Text("Seed \(artifact.defaultSeed)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(Theme.textTertiary)
            }

            Text(artifact.title)
                .font(.headline)
                .foregroundStyle(Theme.textPrimary)

            Text(artifact.subtitle)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
                .lineLimit(2)
        }
        .frame(width: 250, alignment: .leading)
        .padding(14)
        .background(isSelected ? Theme.surfaceHigh : Theme.surface, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(isSelected ? Theme.accent.opacity(0.75) : Theme.border, lineWidth: isSelected ? 1.5 : 1)
        )
    }
}

private struct ArtifactDetailView: View {
    let artifact: AlgorithmicPhilosophyArtifact
    @Binding var selectedPanel: ArtifactPanel

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, 20)
                .padding(.vertical, 14)

            Picker("Artifact panel", selection: $selectedPanel) {
                ForEach(ArtifactPanel.allCases) { panel in
                    Text(panel.rawValue).tag(panel)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 20)
            .padding(.bottom, 12)

            Group {
                switch selectedPanel {
                case .viewer:
                    ArtifactViewerPanel(artifact: artifact)
                case .trace:
                    LiveSVGTracerPanel(artifact: artifact)
                case .philosophy:
                    ArtifactTextPanel(
                        title: "Manifesto",
                        systemImage: "doc.text",
                        url: artifact.philosophyURL,
                        rendersMarkdown: true
                    )
                case .algorithm:
                    ArtifactTextPanel(
                        title: "Generative Algorithm",
                        systemImage: "chevron.left.forwardslash.chevron.right",
                        url: artifact.algorithmURL,
                        rendersMarkdown: false
                    )
                case .reflection:
                    ArtifactReflectionPanel(artifact: artifact)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(artifact.title)
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                    Text(artifact.movementSummary)
                        .font(.subheadline)
                        .foregroundStyle(Theme.textSecondary)
                }
                Spacer()
                Label("Self-contained", systemImage: "shippingbox")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.accent)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Theme.accent.opacity(0.12), in: Capsule())
            }

            VStack(alignment: .leading, spacing: 8) {
                Text(artifact.conceptualSeed)
                    .font(.caption)
                    .foregroundStyle(Theme.textTertiary)
                Text(artifact.tracerSignature)
                    .font(.caption2.monospaced())
                    .foregroundStyle(Theme.accent.opacity(0.82))
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))

            ParameterChips(parameters: artifact.parameterNames)
        }
    }
}

private struct ParameterChips: View {
    let parameters: [String]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(parameters, id: \.self) { parameter in
                    Text(parameter)
                        .font(.caption2.weight(.medium))
                        .foregroundStyle(Theme.textSecondary)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(Theme.surface.opacity(0.9), in: Capsule())
                        .overlay(Capsule().stroke(Theme.border.opacity(0.7), lineWidth: 1))
                }
            }
        }
    }
}

private struct LiveSVGTracerPanel: View {
    let artifact: AlgorithmicPhilosophyArtifact

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                Label("Live SVG tracer", systemImage: "point.topleft.down.curvedto.point.bottomright.up")
                    .font(.headline)
                    .foregroundStyle(Theme.textPrimary)
                Spacer()
                Text(artifact.tracerSignature)
                    .font(.caption2.monospaced())
                    .foregroundStyle(Theme.textTertiary)
            }

            LocalHTMLStringView(html: SVGTracerHTMLFactory.html(for: artifact))
                .frame(minHeight: 420)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(Theme.border.opacity(0.65), lineWidth: 1)
                )
                .accessibilityIdentifier("philosophy.svgTracer.\(artifact.id)")

            Text("The tracer is an inline SVG process: hidden clauses orbit by local time, paths reveal themselves through stroke-dash animation, and the same latent grammar from the app background is made inspectable without relying on a bitmap or a remote renderer.")
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        }
        .padding(20)
    }
}

private struct ArtifactReflectionPanel: View {
    let artifact: AlgorithmicPhilosophyArtifact

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Label("Grounded reference", systemImage: "scope")
                    .font(.headline)
                    .foregroundStyle(Theme.textPrimary)

                Text(artifact.internalReference)
                    .font(.body)
                    .foregroundStyle(Theme.textSecondary)
                    .textSelection(.enabled)

                VStack(alignment: .leading, spacing: 10) {
                    Text("Reflection prompts")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                    ForEach(Array(artifact.reflectionPrompts.enumerated()), id: \.offset) { index, prompt in
                        HStack(alignment: .top, spacing: 10) {
                            Text("\(index + 1)")
                                .font(.caption.monospacedDigit().weight(.bold))
                                .foregroundStyle(Theme.accent)
                                .frame(width: 22, height: 22)
                                .background(Theme.accent.opacity(0.12), in: Circle())
                            Text(prompt)
                                .font(.callout)
                                .foregroundStyle(Theme.textSecondary)
                                .textSelection(.enabled)
                        }
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("Runtime mirror")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                    Text("App background: TimelineView → hidden clauses → live Canvas tracer. Artifact view: self-contained HTML Canvas viewer → SVG tracer → manifesto/source/reflection. Both surfaces expose the same computational philosophy through different renderers.")
                        .font(.caption)
                        .foregroundStyle(Theme.textTertiary)
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(18)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Theme.border.opacity(0.65), lineWidth: 1)
            )
            .padding(20)
        }
    }
}

private enum SVGTracerHTMLFactory {
    static func html(for artifact: AlgorithmicPhilosophyArtifact) -> String {
        let title = artifact.title.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
        let signature = artifact.tracerSignature.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
        return """
        <!doctype html>
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
          :root { color-scheme: dark; --bg:#080907; --ink:#f4e7cf; --ember:#d97757; --blue:#6a9bcc; --green:#788c5d; }
          * { box-sizing: border-box; }
          body { margin:0; min-height:100vh; background: radial-gradient(circle at 50% 42%, rgba(217,119,87,.18), transparent 34%), linear-gradient(135deg,#10110f,#030403); font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif; color: var(--ink); overflow:hidden; }
          .wrap { position:relative; height:100vh; min-height:420px; padding:18px; }
          .label { position:absolute; left:18px; top:14px; z-index:2; letter-spacing:.08em; text-transform:uppercase; font-size:11px; color:rgba(244,231,207,.66); }
          .sig { position:absolute; left:18px; bottom:14px; right:18px; z-index:2; font-size:11px; color:rgba(244,231,207,.46); }
          svg { width:100%; height:100%; display:block; filter: drop-shadow(0 0 18px rgba(217,119,87,.18)); }
          .clause { fill:none; stroke:rgba(217,119,87,.34); stroke-width:1.2; }
          .trace { fill:none; stroke-linecap:round; stroke-linejoin:round; stroke-dasharray: var(--length); stroke-dashoffset: var(--length); animation: reveal 7s cubic-bezier(.2,.7,.1,1) infinite; animation-delay: var(--delay); }
          .trace.ink { stroke:rgba(244,231,207,.48); stroke-width:1.25; }
          .trace.ember { stroke:rgba(217,119,87,.54); stroke-width:1.55; }
          .trace.blue { stroke:rgba(106,155,204,.34); stroke-width:1.05; }
          .margin { fill:none; stroke:rgba(106,155,204,.10); stroke-width:1; vector-effect:non-scaling-stroke; }
          @keyframes reveal { 0% { stroke-dashoffset: var(--length); opacity:.0; } 15% { opacity:.72; } 62% { stroke-dashoffset:0; opacity:.74; } 100% { stroke-dashoffset: var(--exit); opacity:.06; } }
        </style>
        </head>
        <body>
          <div class="wrap">
            <div class="label">\(title) · live SVG process</div>
            <svg viewBox="0 0 1000 640" preserveAspectRatio="xMidYMid meet" aria-label="Live SVG tracer for \(title)">
              <g id="margins"></g>
              <g id="clauses"></g>
              <g id="traces"></g>
            </svg>
            <div class="sig">\(signature)</div>
          </div>
        <script>
          const svgNS = 'http://www.w3.org/2000/svg';
          const traces = document.getElementById('traces');
          const clausesGroup = document.getElementById('clauses');
          const margins = document.getElementById('margins');
          const W = 1000, H = 640;
          const golden = Math.PI * (3 - Math.sqrt(5));
          function dayFraction(now) { return (now.getHours()*3600 + now.getMinutes()*60 + now.getSeconds() + now.getMilliseconds()/1000) / 86400; }
          function pseudo(v) { const raw = Math.sin(v * 12.9898) * 43758.5453; return raw - Math.floor(raw); }
          function clauses(day) {
            const out = [];
            for (let i=0; i<7; i++) {
              const n = i / 6;
              const angle = -Math.PI/2 + i * golden + day * Math.PI * 2 * .08;
              const r = 92 + 178 * Math.sqrt(n + .08);
              out.push({ x: W/2 + Math.cos(angle)*r, y: H/2 + Math.sin(angle)*r, r: 34 + 12 * pseudo(i + day) });
            }
            return out;
          }
          function nearest(p, cs) {
            let best = cs[0], bestD = Infinity;
            for (const c of cs) { const dx=c.x-p.x, dy=c.y-p.y, d=dx*dx+dy*dy; if (d<bestD) { best=c; bestD=d; } }
            return best;
          }
          function tracePath(seed, cs, now) {
            const c = cs[seed % cs.length];
            let angle = seed * .43 + now * (.018 + (seed % 7) * .0018);
            let p = { x: c.x + Math.cos(angle) * (45 + pseudo(seed) * 120), y: c.y + Math.sin(angle) * (45 + pseudo(seed + 8.3) * 120) };
            let d = `M ${p.x.toFixed(2)} ${p.y.toFixed(2)}`;
            for (let step=0; step<24; step++) {
              const n = nearest(p, cs);
              const dx = n.x - p.x, dy = n.y - p.y;
              const dist = Math.max(18, Math.hypot(dx,dy));
              const orbital = Math.atan2(dy,dx) + Math.PI/2;
              const center = Math.atan2(H/2-p.y, W/2-p.x);
              const turbulence = Math.sin(step*.55 + seed*12.9898 + now*.11) * .72;
              const a = orbital*.62 + center*.38 + turbulence;
              const stride = (5.2 + 10 * pseudo(seed + step*1.7)) * (1 + 60/dist);
              const nx = p.x + Math.cos(a)*stride, ny = p.y + Math.sin(a)*stride;
              const cx = (p.x+nx)/2 + Math.cos(a+Math.PI/2)*stride*.65;
              const cy = (p.y+ny)/2 + Math.sin(a+Math.PI/2)*stride*.65;
              d += ` Q ${cx.toFixed(2)} ${cy.toFixed(2)} ${nx.toFixed(2)} ${ny.toFixed(2)}`;
              p = {x:nx,y:ny};
            }
            return d;
          }
          function render() {
            const now = new Date();
            const t = now.getTime()/1000;
            const cs = clauses(dayFraction(now));
            clausesGroup.replaceChildren(); traces.replaceChildren(); margins.replaceChildren();
            for (let i=0; i<8; i++) {
              const r = document.createElementNS(svgNS,'rect');
              r.setAttribute('class','margin'); r.setAttribute('x', 16+i*18); r.setAttribute('y', 16+i*12); r.setAttribute('width', W-32-i*36); r.setAttribute('height', H-32-i*24); r.setAttribute('rx', 24+i*2);
              margins.appendChild(r);
            }
            cs.forEach((c,i) => { const e = document.createElementNS(svgNS,'circle'); e.setAttribute('class','clause'); e.setAttribute('cx',c.x); e.setAttribute('cy',c.y); e.setAttribute('r',c.r + 8*Math.sin(t*.45+i*.72)); clausesGroup.appendChild(e); });
            for (let i=0; i<34; i++) {
              const p = document.createElementNS(svgNS,'path');
              p.setAttribute('class', 'trace ' + (i%5===0 ? 'blue' : (i%3===0 ? 'ember' : 'ink')));
              p.setAttribute('d', tracePath(i, cs, t));
              const length = 480 + pseudo(i)*420;
              p.style.setProperty('--length', length);
              p.style.setProperty('--exit', -length * .35);
              p.style.setProperty('--delay', `${-pseudo(i+4)*7}s`);
              traces.appendChild(p);
            }
            setTimeout(() => requestAnimationFrame(render), 1000 / 24);
          }
          render();
        </script>
        </body>
        </html>
        """
    }
}

private struct LocalHTMLStringView: UIViewRepresentable {
    let html: String

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.isOpaque = false
        webView.scrollView.backgroundColor = .clear
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        guard context.coordinator.lastHTML != html else { return }
        context.coordinator.lastHTML = html
        webView.loadHTMLString(html, baseURL: nil)
    }

    final class Coordinator {
        var lastHTML: String?
    }
}

private struct ArtifactViewerPanel: View {
    let artifact: AlgorithmicPhilosophyArtifact

    var body: some View {
        if let viewerURL = artifact.viewerURL {
            LocalHTMLArtifactView(url: viewerURL)
                .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .stroke(Theme.border.opacity(0.65), lineWidth: 1)
                )
                .padding(.horizontal, 20)
                .padding(.bottom, 20)
                .accessibilityIdentifier("philosophy.viewer.\(artifact.id)")
        } else {
            ContentUnavailableView(
                "Viewer missing",
                systemImage: "exclamationmark.triangle",
                description: Text("The bundled HTML viewer could not be found for \(artifact.title).")
            )
        }
    }
}

private struct ArtifactTextPanel: View {
    let title: String
    let systemImage: String
    let url: URL?
    let rendersMarkdown: Bool

    private var bodyText: String {
        guard let url else { return "Missing bundled resource." }
        return (try? String(contentsOf: url, encoding: .utf8)) ?? "Could not read \(url.lastPathComponent)."
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Label(title, systemImage: systemImage)
                    .font(.headline)
                    .foregroundStyle(Theme.textPrimary)

                if rendersMarkdown, let attributed = try? AttributedString(markdown: bodyText) {
                    Text(attributed)
                        .font(.body)
                        .foregroundStyle(Theme.textSecondary)
                        .textSelection(.enabled)
                } else {
                    Text(bodyText)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(Theme.textSecondary)
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(18)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Theme.border.opacity(0.65), lineWidth: 1)
            )
            .padding(20)
        }
    }
}

private struct LocalHTMLArtifactView: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.allowsBackForwardNavigationGestures = false
        webView.scrollView.keyboardDismissMode = .onDrag
        webView.scrollView.backgroundColor = .clear
        webView.isOpaque = false
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        guard webView.url != url else { return }
        webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent().deletingLastPathComponent())
    }
}
