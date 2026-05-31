import SwiftUI

struct ToolSecurityView: View {
    let tools: ToolSecuritySnapshot
    var body: some View {
        List(tools.tools, id: \.id) { tool in
            VStack(alignment: .leading) {
                Text(tool.id).font(.headline)
                Text("Category: \(tool.category)")
                Text("Permissions: \(tool.requiredPermissions.joined(separator: ", "))")
                Text("Background-safe: \(tool.supportsBackground ? "yes" : "no")")
                Text("Approval: \(tool.requiresApproval ? "required" : "not required")")
            }
        }.navigationTitle("Tool Security")
    }
}
