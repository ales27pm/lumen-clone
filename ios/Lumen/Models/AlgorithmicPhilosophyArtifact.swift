import Foundation

struct AlgorithmicPhilosophyArtifact: Identifiable, Hashable {
    let id: String
    let title: String
    let subtitle: String
    let movementSummary: String
    let folderName: String
    let philosophyFileName: String
    let viewerFileName: String
    let algorithmFileName: String
    let defaultSeed: Int
    let parameterNames: [String]
    let conceptualSeed: String
    let internalReference: String
    let reflectionPrompts: [String]
    let tracerSignature: String

    var resourceSubdirectory: String {
        "Resources/AlgorithmicPhilosophies/\(folderName)"
    }

    var philosophyURL: URL? {
        bundledURL(forResource: philosophyFileName, withExtension: "md")
    }

    var viewerURL: URL? {
        bundledURL(forResource: viewerFileName, withExtension: "html")
    }

    var algorithmURL: URL? {
        bundledURL(forResource: algorithmFileName, withExtension: "js")
    }

    private func bundledURL(forResource name: String, withExtension fileExtension: String) -> URL? {
        let candidateSubdirectories: [String?] = [
            resourceSubdirectory,
            "AlgorithmicPhilosophies/\(folderName)",
            folderName,
            nil
        ]

        for subdirectory in candidateSubdirectories {
            if let url = Bundle.main.url(
                forResource: name,
                withExtension: fileExtension,
                subdirectory: subdirectory
            ) {
                return url
            }
        }

        return nil
    }
}

enum AlgorithmicPhilosophyCatalog {
    static let all: [AlgorithmicPhilosophyArtifact] = [
        AlgorithmicPhilosophyArtifact(
            id: "latent-liturgy",
            title: "Latent Liturgy",
            subtitle: "Hidden clauses guide living particles into ceremonial traces.",
            movementSummary: "A seeded field system where invisible attractors, margins, and turbulence shape particle readers into luminous evidence of hidden instruction.",
            folderName: "latent_liturgy",
            philosophyFileName: "latent_liturgy",
            viewerFileName: "latent_liturgy",
            algorithmFileName: "latent_liturgy",
            defaultSeed: 12345,
            parameterNames: [
                "Hidden Clauses",
                "Particle Readers",
                "Turbulence",
                "Clause Gravity",
                "Procession Speed",
                "Trail Patience",
                "Symmetry Pressure",
                "Margin Bend"
            ],
            conceptualSeed: "Latent instruction systems: prompts and constraints are never shown directly, but their grammar bends every particle path.",
            internalReference: "The app shell now mirrors the artwork's hidden-clause grammar: real time, local state, and seeded constants drive visible traces without exposing the invisible instructions directly.",
            reflectionPrompts: [
                "Which invisible constraints are shaping the current surface?",
                "Where does a repeated path become evidence rather than decoration?",
                "How does local time alter the same seeded system without breaking reproducibility?"
            ],
            tracerSignature: "7 clauses · golden-angle placement · day-fraction drift · SVG path revelation"
        )
    ]
}
