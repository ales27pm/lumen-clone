import SwiftUI

struct VoicePickerList: View {
    @Environment(AppState.self) private var appState
    @Environment(VoiceService.self) private var voice
    @State private var voices: [VoiceCatalog.Entry] = []

    var body: some View {
        List {
            Section {
                Button {
                    appState.voiceID = nil
                } label: {
                    HStack {
                        Text("System default")
                        Spacer()
                        if appState.voiceID == nil {
                            Image(systemName: "checkmark").foregroundStyle(Theme.accent)
                        }
                    }
                }
            }
            Section("Installed voices") {
                ForEach(voices) { entry in
                    Button {
                        appState.voiceID = entry.id
                        voice.speak("Hi, I'm \(entry.name). I'll speak for you.", voiceID: entry.id, rate: appState.speakingRate)
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(entry.name)
                                Text("\(entry.language) · \(entry.quality)")
                                    .font(.caption).foregroundStyle(Theme.textSecondary)
                            }
                            Spacer()
                            if appState.voiceID == entry.id {
                                Image(systemName: "checkmark").foregroundStyle(Theme.accent)
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("Voice")
        .onAppear { voices = VoiceCatalog.available() }
    }
}
