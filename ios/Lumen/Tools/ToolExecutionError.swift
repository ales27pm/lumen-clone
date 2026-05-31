import Foundation

enum ToolExecutionError: Error, Sendable { case invalidArguments(String), denied(String), unavailable(String), failed(String) }
