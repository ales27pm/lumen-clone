import Foundation
import os.log

actor KnowledgeGraphService {
    static let shared = KnowledgeGraphService()

    private var nodes: [String: KGNode] = [:]
    private var edges: [KGEdge] = []
    private let logger = Logger(subsystem: "com.lumen.kg", category: "service")

    struct KGNode {
        let id: String
        let type: String
        let content: String
        var embedding: [Double]?
        var gnnScore: Double = 0.0
    }

    struct KGEdge {
        let from: String
        let to: String
        let relation: String
        let weight: Double
    }

    func buildFromManifestAndAudits() async {
        let manifestResult = RuntimeManifestAuditor(
            registryProvider: LiveRuntimeToolRegistryProvider()
        ).loadManifestFromStoreBundleOrRuntimeFallback()
        buildFromManifestAndAudits(manifest: manifestResult.manifest, auditFailures: [])
        logger.info("KG built with \(self.nodes.count) nodes and \(self.edges.count) edges")
    }

    func buildFromManifestAndAudits(
        manifest: AgentBehaviorManifest,
        auditFailures: [RuntimeManifestFailure]
    ) {
        nodes.removeAll(keepingCapacity: true)
        edges.removeAll(keepingCapacity: true)

        upsertNode(
            id: "app:\(manifest.app.name)",
            type: "app",
            content: [
                manifest.app.name,
                manifest.app.bundleIdentifier,
                manifest.app.buildVersion,
                manifest.fleet.contractVersion
            ].compactMap { $0 }.joined(separator: " ")
        )

        for slot in manifest.fleet.slots {
            let slotID = nodeID("slot", slot.id)
            upsertNode(
                id: slotID,
                type: "slot",
                content: ([slot.id, slot.role] + [slot.modelFamily].compactMap { $0 } + slot.responsibilities)
                    .joined(separator: " ")
            )
            addEdge("app:\(manifest.app.name)", slotID, relation: "has_slot", weight: 0.95)
        }

        for intent in manifest.intents {
            let intentID = nodeID("intent", intent.id)
            upsertNode(
                id: intentID,
                type: "intent",
                content: ([intent.id] + intent.allowedToolIDs).joined(separator: " ")
            )
            addEdge(nodeID("slot", "cortex"), intentID, relation: "routes_intent", weight: 0.9)
            for toolID in intent.allowedToolIDs {
                addToolNodeIfNeeded(toolID, manifest: manifest)
                addEdge(intentID, nodeID("tool", toolID), relation: "allows_tool", weight: 0.85)
            }
        }

        for route in manifest.routingMatrix {
            let intentID = nodeID("intent", route.intent)
            upsertNode(
                id: intentID,
                type: "intent",
                content: ([route.intent] + route.allowedTools).joined(separator: " ")
            )
            for toolID in route.allowedTools {
                addToolNodeIfNeeded(toolID, manifest: manifest)
                addEdge(intentID, nodeID("tool", toolID), relation: "routes_to_tool", weight: 0.8)
                addEdge(nodeID("slot", "executor"), nodeID("tool", toolID), relation: "executes_tool", weight: 0.7)
            }
        }

        for tool in manifest.tools {
            addToolNode(tool)
            if let permissionKey = tool.permissionKey, !permissionKey.isEmpty {
                let permissionID = nodeID("permission", permissionKey)
                upsertNode(id: permissionID, type: "permission", content: permissionKey)
                addEdge(nodeID("tool", tool.id), permissionID, relation: "requires_permission", weight: 0.65)
            }
        }

        if let memory = manifest.memory {
            for scope in memory.scopes {
                let scopeID = nodeID("memory", scope)
                upsertNode(id: scopeID, type: "memory_scope", content: scope)
                addEdge(nodeID("slot", "rem"), scopeID, relation: "curates_memory_scope", weight: 0.75)
            }
            for freshness in memory.freshnessClasses {
                let freshnessID = nodeID("freshness", freshness.id)
                upsertNode(
                    id: freshnessID,
                    type: "freshness",
                    content: "\(freshness.id) durable:\(freshness.durable) ttl:\(freshness.ttlSeconds.map(String.init) ?? "none")"
                )
                addEdge(nodeID("slot", "rem"), freshnessID, relation: "classifies_memory", weight: 0.7)
            }
        }

        for failure in auditFailures {
            let failureID = nodeID("audit", failure.id)
            upsertNode(
                id: failureID,
                type: "audit_failure",
                content: (
                    [failure.type, failure.problem]
                    + [failure.agent, failure.actual, failure.scenario].compactMap { $0 }
                    + failure.expected
                ).joined(separator: " ")
            )
            addEdge(nodeID("slot", "rem"), failureID, relation: "tracks_failure", weight: 0.9)
            if let agent = failure.agent, !agent.isEmpty {
                let slotID = nodeID("slot", agent)
                if nodes[slotID] != nil {
                    addEdge(failureID, slotID, relation: "affects_agent", weight: 0.8)
                }
            }
        }
    }

    func multiHopTraverse(startId: String, maxHops: Int = 3) async -> [TraversalPath] {
        let start = resolveNodeID(startId)
        guard nodes[start] != nil, maxHops > 0 else { return [] }

        let adjacency = adjacencyBySource()
        var paths: [TraversalPath] = []
        var queue: [(path: [String], score: Double)] = [([start], 1.0)]

        while !queue.isEmpty {
            let current = queue.removeFirst()
            guard let last = current.path.last, current.path.count <= maxHops + 1 else { continue }

            for edge in adjacency[last, default: []] {
                guard nodes[edge.to] != nil, !current.path.contains(edge.to) else { continue }
                let nextPath = current.path + [edge.to]
                let nextScore = current.score * edge.weight
                paths.append(TraversalPath(nodes: nextPath, score: nextScore))
                if nextPath.count <= maxHops {
                    queue.append((nextPath, nextScore))
                }
            }
        }

        return paths.sorted {
            if $0.score == $1.score { return $0.nodes.count < $1.nodes.count }
            return $0.score > $1.score
        }
    }

    func queryWithGNN(query: String) async -> [KGNode] {
        logger.info("GNN reasoning for query: \(query)")
        let terms = Self.queryTerms(query)
        guard !terms.isEmpty else { return [] }

        let adjacency = adjacencyBySource()
        return nodes.values.compactMap { node in
            let lexicalScore = Self.lexicalScore(terms: terms, node: node)
            guard lexicalScore > 0 else { return nil }
            let neighborScore = adjacency[node.id, default: []].reduce(0.0) { partial, edge in
                guard let neighbor = nodes[edge.to] else { return partial }
                return partial + (Self.lexicalScore(terms: terms, node: neighbor) * edge.weight * 0.25)
            }
            var scored = node
            scored.gnnScore = lexicalScore + neighborScore + centralityBonus(for: node.id)
            return scored
        }
        .sorted {
            if $0.gnnScore == $1.gnnScore { return $0.id < $1.id }
            return $0.gnnScore > $1.gnnScore
        }
        .prefix(5)
        .map { $0 }
    }

    private func addToolNodeIfNeeded(_ toolID: String, manifest: AgentBehaviorManifest) {
        if let tool = manifest.tools.first(where: { $0.id == toolID }) {
            addToolNode(tool)
        } else {
            upsertNode(id: nodeID("tool", toolID), type: "tool", content: toolID)
        }
    }

    private func addToolNode(_ tool: RuntimeToolDefinition) {
        upsertNode(
            id: nodeID("tool", tool.id),
            type: "tool",
            content: [
                tool.id,
                tool.displayName,
                tool.description,
                tool.permissionKey,
                tool.requiresApproval ? "requires approval" : "no approval"
            ].compactMap { $0 }.joined(separator: " ")
        )
    }

    private func upsertNode(id: String, type: String, content: String) {
        nodes[id] = KGNode(id: id, type: type, content: content)
    }

    private func addEdge(_ from: String, _ to: String, relation: String, weight: Double) {
        guard from != to else { return }
        let edge = KGEdge(from: from, to: to, relation: relation, weight: weight)
        if !edges.contains(where: { $0.from == edge.from && $0.to == edge.to && $0.relation == edge.relation }) {
            edges.append(edge)
        }
    }

    private func adjacencyBySource() -> [String: [KGEdge]] {
        Dictionary(grouping: edges, by: \.from)
    }

    private func centralityBonus(for nodeID: String) -> Double {
        let degree = edges.reduce(0) { partial, edge in
            partial + (edge.from == nodeID || edge.to == nodeID ? 1 : 0)
        }
        return min(Double(degree) * 0.02, 0.2)
    }

    private func resolveNodeID(_ id: String) -> String {
        if nodes[id] != nil { return id }
        for prefix in ["slot", "intent", "tool", "memory", "permission", "audit", "freshness"] {
            let candidate = nodeID(prefix, id)
            if nodes[candidate] != nil { return candidate }
        }
        return id
    }

    private func nodeID(_ type: String, _ rawID: String) -> String {
        "\(type):\(rawID)"
    }

    private nonisolated static func queryTerms(_ query: String) -> [String] {
        let stopwords: Set<String> = ["a", "an", "and", "for", "in", "is", "of", "on", "or", "the", "to", "with"]
        return query
            .lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
            .filter { $0.count > 1 && !stopwords.contains($0) }
    }

    private nonisolated static func lexicalScore(terms: [String], node: KGNode) -> Double {
        let haystack = "\(node.id) \(node.type) \(node.content)".lowercased()
        return terms.reduce(0.0) { partial, term in
            partial + (haystack.contains(term) ? 1.0 : 0.0)
        }
    }
}

struct TraversalPath {
    let nodes: [String]
    let score: Double
}
