import Foundation

enum PromptGroundingRenderer {
    static func render(memories: MemoryContextResult, rag: RAGContextResult, tools: [SecureToolDefinition], lowPower: Bool, thermal: DeviceThermalState, selfModel: PromptGroundingSection? = nil) -> [PromptGroundingSection] {
        let renderedMemories = Array(memories.selected.prefix(8))
        let memLines = renderedMemories.map { "- [\($0.id.uuidString.prefix(8))] \(String($0.content.prefix(120)))" }
        let renderedRAG = Array(rag.selected.prefix(8))
        let ragLines = renderedRAG.map { "- [\($0.chunkID.uuidString.prefix(8))] \($0.source.title) (\($0.retrievalMode), \(String(format: "%.2f", $0.score)))" }
        let renderedTools = Array(tools.prefix(20))
        let canonicalToolIDs = renderedTools.map { ToolRouteGuard.canonicalToolID($0.id) }
        let toolLines = zip(renderedTools, canonicalToolIDs).map { tool, canonicalID in
            "- \(canonicalID): \(tool.description)"
        }
        let runtimePolicy = "lowPower=\(lowPower), thermal=\(thermal.rawValue)"
        let memoryContent = memLines.joined(separator: "\n")
        let ragContent = ragLines.joined(separator: "\n")
        let toolContent = toolLines.joined(separator: "\n")
        let memorySection = PromptGroundingSection(title: "Relevant memories", content: memoryContent, estimatedChars: memoryContent.count, sourceIDs: renderedMemories.map { $0.id.uuidString }, privacyLevel: .moderate)
        let ragSection = PromptGroundingSection(title: "Retrieved sources", content: ragContent, estimatedChars: ragContent.count, sourceIDs: renderedRAG.map { $0.chunkID.uuidString }, privacyLevel: .moderate)
        let toolSection = PromptGroundingSection(title: "Available tools", content: toolContent, estimatedChars: toolContent.count, sourceIDs: canonicalToolIDs, privacyLevel: .low)
        let runtimeSection = PromptGroundingSection(title: "Runtime policy", content: runtimePolicy, estimatedChars: runtimePolicy.count, sourceIDs: [], privacyLevel: .low)
        var sections = [memorySection, ragSection, toolSection, runtimeSection].filter { !$0.content.isEmpty }
        if let selfModel, !selfModel.content.isEmpty {
            sections.append(selfModel)
        }
        return sections
    }

    static func renderForPrompt(_ sections: [PromptGroundingSection], maxChars: Int) -> String {
        var out = ""; var chars = 0
        for s in sections {
            let block = "\n[\(s.title)]\n\(s.content)\n"
            if chars + block.count > maxChars { continue }
            out += block; chars += block.count
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
