import Foundation

struct ContextBudgetSections: Sendable {
    let system: Int
    let history: Int
    let memories: Int
    let rag: Int
    let tools: Int
    let runtime: Int
}

enum ContextBudgetAllocator {
    static func allocate(maxChars: Int) -> ContextBudgetSections {
        let bounded = max(0, maxChars)
        let system = Int(Double(bounded) * 0.18)
        let history = Int(Double(bounded) * 0.34)
        let memories = Int(Double(bounded) * 0.18)
        let rag = Int(Double(bounded) * 0.20)
        let tools = Int(Double(bounded) * 0.06)
        let runtime = max(0, bounded - (system + history + memories + rag + tools))
        return ContextBudgetSections(system: system, history: history, memories: memories, rag: rag, tools: tools, runtime: runtime)
    }
}
