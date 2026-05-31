import Foundation

/// Coarse token ≈ char conversion factor used for llama.cpp-style prompt budgeting.
/// SentencePiece/BPE tokenizers average ~3.5–4.2 chars/token on English text; we pick
/// a conservative value so budgets err on the side of fitting, not overflowing.
nonisolated enum PromptBudgetConstants {
    static let approxCharsPerToken: Double = 3.8
    static let safetyTokens: Int = 256
    static let hardAttachmentCeiling: Int = 80_000
    static let hardMemoriesCeiling: Int = 4_000
    static let minUserCharCap: Int = 4_000
    static let attachmentShareFraction: Double = 0.55
    static let memoriesShareFraction: Double = 0.20
    static let minHistoryChars: Int = 512
    static let minPerAttachmentChars: Int = 800
}

/// Total per-section character budget for a single generation request.
/// Attachments get the largest dedicated share because they are typically the
/// biggest non-mandatory block. The user message and system prompt are always
/// fit in full (bounded by a generous cap to avoid pathological inputs).
nonisolated struct PromptBudget: Sendable {
    let totalChars: Int
    let attachmentsShare: Int
    let memoriesShare: Int
    let historyShare: Int

    static func make(
        contextSize: Int,
        maxTokens: Int,
        systemPromptChars: Int,
        userMessageChars: Int,
        hasAttachments: Bool,
        hasMemories: Bool
    ) -> PromptBudget {
        let tokenBudget = max(512, contextSize - maxTokens - PromptBudgetConstants.safetyTokens)
        let totalChars = max(2048, Int(Double(tokenBudget) * PromptBudgetConstants.approxCharsPerToken))

        // Fixed cost: system prompt + a reasonable cap on user text + chat template overhead.
        let boundedUser = min(userMessageChars, max(PromptBudgetConstants.minUserCharCap, totalChars / 4))
        let templateOverhead = 512
        let fixed = min(systemPromptChars + boundedUser + templateOverhead, totalChars - 512)
        var remaining = max(1024, totalChars - fixed)

        let attachmentsShare: Int
        if hasAttachments {
            attachmentsShare = min(
                Int(Double(remaining) * PromptBudgetConstants.attachmentShareFraction),
                PromptBudgetConstants.hardAttachmentCeiling
            )
        } else {
            attachmentsShare = 0
        }
        remaining -= attachmentsShare

        let memoriesShare: Int
        if hasMemories {
            memoriesShare = min(
                Int(Double(remaining) * PromptBudgetConstants.memoriesShareFraction),
                PromptBudgetConstants.hardMemoriesCeiling
            )
        } else {
            memoriesShare = 0
        }
        remaining -= memoriesShare

        let historyShare = max(PromptBudgetConstants.minHistoryChars, remaining)

        return PromptBudget(
            totalChars: totalChars,
            attachmentsShare: attachmentsShare,
            memoriesShare: memoriesShare,
            historyShare: historyShare
        )
    }
}

/// Per-attachment outcome after the budget is applied. Surfaced to the UI so
/// the user can see when a file was trimmed.
nonisolated struct AttachmentRenderState: Sendable, Hashable {
    let id: UUID
    let name: String
    let includedChars: Int
    let totalChars: Int
    var truncated: Bool { includedChars < totalChars }
}

/// Result of running the prompt assembler on a request.
nonisolated struct PromptAssembly: Sendable {
    let systemPrompt: String
    let history: [(role: MessageRole, content: String)]
    let userMessage: String
    let attachmentStates: [AttachmentRenderState]
    let usedChars: Int
    let budgetChars: Int

    var historyTuples: [(String, String)] {
        history.map { ($0.role.rawValue, $0.content) }
    }
}

nonisolated enum AttachmentNormalizationMode: Sendable {
    case preserveRaw
    case agentRouting
}

/// Deterministic, priority-based prompt assembly.
///
/// Truncation priority (first to shrink, in order):
/// 1. Attachments — each file is head-truncated to its per-file cap and annotated
///    with a visible "[... truncated ...]" marker.
/// 2. Conversation history — oldest turns are dropped until the share fits.
///    The most recent turns are preserved because they're most relevant.
/// 3. Memories — the list is prefix-truncated to fit its share.
///
/// Never shrunk:
/// - The raw system prompt (agent rules, tool list).
/// - The current user message, except for a final safety cap at roughly
///   ¼ of the total budget (protects against multi-MB paste dumps).
nonisolated enum PromptAssembler {

    static func assemble(
        systemPrompt: String,
        history: [(role: MessageRole, content: String)],
        userMessage: String,
        memories: [MemoryContextItem],
        attachments: [ChatAttachment],
        budget: PromptBudget,
        attachmentNormalization: AttachmentNormalizationMode = .preserveRaw
    ) -> PromptAssembly {
        let userCap = min(userMessage.count, max(PromptBudgetConstants.minUserCharCap, budget.totalChars / 4))
        let boundedUser = truncateMiddle(userMessage, maxChars: userCap)

        let memoriesBlock = buildMemoriesBlock(memories: memories, share: budget.memoriesShare)
        let (attachmentsBlock, states) = buildAttachmentsBlock(
            attachments: attachments,
            share: budget.attachmentsShare,
            normalization: attachmentNormalization
        )
        let finalSystem = systemPrompt + memoriesBlock + attachmentsBlock

        let keptHistory = fitHistory(history: history, share: budget.historyShare)

        let historyChars = keptHistory.reduce(0) { $0 + $1.content.count + 16 }
        let totalUsed = finalSystem.count + boundedUser.count + historyChars

        return PromptAssembly(
            systemPrompt: finalSystem,
            history: keptHistory,
            userMessage: boundedUser,
            attachmentStates: states,
            usedChars: totalUsed,
            budgetChars: budget.totalChars
        )
    }

    /// Lightweight preview used by the chat UI to show whether attachments
    /// will be truncated at send time. Uses the same budget math as `assemble`.
    static func previewAttachmentStates(
        attachments: [ChatAttachment],
        contextSize: Int,
        maxTokens: Int,
        systemPromptChars: Int,
        userMessageChars: Int,
        hasMemories: Bool
    ) -> [AttachmentRenderState] {
        guard !attachments.isEmpty else { return [] }
        let budget = PromptBudget.make(
            contextSize: contextSize,
            maxTokens: maxTokens,
            systemPromptChars: systemPromptChars,
            userMessageChars: userMessageChars,
            hasAttachments: true,
            hasMemories: hasMemories
        )
        let (_, states) = buildAttachmentsBlock(
            attachments: attachments,
            share: budget.attachmentsShare,
            normalization: .preserveRaw
        )
        return states
    }

    // MARK: - Sections

    private static func buildMemoriesBlock(memories: [MemoryContextItem], share: Int) -> String {
        guard !memories.isEmpty, share > 0 else { return "" }
        var used = 0
        var items: [MemoryContextItem] = []
        for m in memories.prefix(10) {
            let cleaned = m.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if cleaned.isEmpty { continue }
            let line = "• [\(m.scope.rawValue) | \(m.authority.rawValue)] " + cleaned
            let cost = line.count + 1
            if used + cost > share { break }
            items.append(m)
            used += cost
        }
        return PromptContextBuilder.renderMemoryBlock(items)
    }

    private static func buildAttachmentsBlock(
        attachments: [ChatAttachment],
        share: Int,
        normalization: AttachmentNormalizationMode
    ) -> (String, [AttachmentRenderState]) {
        guard !attachments.isEmpty else { return ("", []) }

        struct Loaded { let attachment: ChatAttachment; let raw: String }
        let loaded: [Loaded] = attachments.map {
            let raw = AttachmentResolver.rawExtractText($0)
            let prepared: String
            switch normalization {
            case .preserveRaw:
                prepared = raw
            case .agentRouting:
                prepared = normalizeForAgentRouting(raw)
            }
            return Loaded(attachment: $0, raw: prepared)
        }

        if share <= 0 {
            // No room at all — still report states so UI can show "truncated".
            let states = loaded.map {
                AttachmentRenderState(
                    id: $0.attachment.id,
                    name: $0.attachment.name,
                    includedChars: 0,
                    totalChars: $0.raw.count
                )
            }
            return ("", states)
        }

        let perFileCap = max(PromptBudgetConstants.minPerAttachmentChars, share / max(1, loaded.count))
        var out = "\nThe user attached the following file(s) to this message. Treat them as authoritative context. Do NOT call files.read for them — their content is provided below.\n"
        var states: [AttachmentRenderState] = []
        states.reserveCapacity(loaded.count)

        for (i, item) in loaded.enumerated() {
            let totalCount = item.raw.count
            let included = truncateHead(item.raw, maxChars: perFileCap)
            let trimmed = included.trimmingCharacters(in: .whitespacesAndNewlines)

            out += "\n--- Attachment \(i + 1): \(item.attachment.name) (\(item.attachment.kind.rawValue)) ---\n"
            if trimmed.isEmpty {
                out += "[Could not extract text from this file.]\n"
            } else {
                out += trimmed
                if included.count < totalCount {
                    let omitted = totalCount - included.count
                    out += "\n[... truncated: \(omitted) more characters omitted ...]\n"
                } else {
                    out += "\n"
                }
            }

            states.append(AttachmentRenderState(
                id: item.attachment.id,
                name: item.attachment.name,
                includedChars: min(included.count, totalCount),
                totalChars: totalCount
            ))
        }
        out += "--- End attachments ---\n"
        return (out, states)
    }

    private static func fitHistory(
        history: [(role: MessageRole, content: String)],
        share: Int
    ) -> [(role: MessageRole, content: String)] {
        guard share > 0, !history.isEmpty else { return [] }
        var kept: [(role: MessageRole, content: String)] = []
        var used = 0
        // Walk newest → oldest; prepend so original order is preserved.
        for h in history.reversed() {
            let content = h.content.count > share
                ? truncateMiddle(h.content, maxChars: max(256, share / 2))
                : h.content
            let cost = content.count + 16
            if used + cost > share { break }
            kept.insert((role: h.role, content: content), at: 0)
            used += cost
        }
        return kept
    }

    // MARK: - Truncation primitives

    static func truncateHead(_ s: String, maxChars: Int) -> String {
        if s.count <= maxChars { return s }
        if maxChars <= 0 { return "" }
        return String(s.prefix(maxChars))
    }

    static func truncateMiddle(_ s: String, maxChars: Int) -> String {
        if s.count <= maxChars { return s }
        if maxChars <= 64 { return String(s.prefix(maxChars)) }
        let half = (maxChars - 32) / 2
        return String(s.prefix(half)) + "\n[... truncated ...]\n" + String(s.suffix(half))
    }

    private static func normalizeForAgentRouting(_ text: String) -> String {
        var normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        normalized = stripCodeFenceWrapperIfPresent(normalized)
        normalized = collapseNoisyStructuralRuns(normalized)
        normalized = normalized.replacingOccurrences(of: #"[ \t]{2,}"#, with: " ", options: .regularExpression)
        normalized = normalized.replacingOccurrences(of: #"\n{3,}"#, with: "\n\n", options: .regularExpression)
        return normalized.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func stripCodeFenceWrapperIfPresent(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("```"), trimmed.hasSuffix("```") else { return text }
        let lines = trimmed.split(separator: "\n", omittingEmptySubsequences: false)
        guard lines.count >= 2 else { return text }
        guard String(lines[0]).hasPrefix("```"), String(lines[lines.count - 1]) == "```" else { return text }
        return lines.dropFirst().dropLast().joined(separator: "\n")
    }

    private static func collapseNoisyStructuralRuns(_ text: String) -> String {
        var out = text
        out = out.replacingOccurrences(of: #"`{4,}"#, with: "```", options: .regularExpression)
        out = out.replacingOccurrences(of: #"[=_\-\*#]{8,}"#, with: "-----", options: .regularExpression)
        out = out.replacingOccurrences(of: #"[|]{4,}"#, with: "|||", options: .regularExpression)
        out = out.replacingOccurrences(of: #"[<>]{4,}"#, with: "<<<>>>", options: .regularExpression)
        out = out.replacingOccurrences(of: #"[\\/\[\]\(\)\{\}:;,+]{12,}"#, with: " [structural-run-collapsed] ", options: .regularExpression)
        return out
    }
}
