import Foundation
import os

enum SystemMemoryLimit {
    static func availableMemoryBytes() -> UInt64 {
        UInt64(os_proc_available_memory())
    }
}
