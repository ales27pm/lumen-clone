import Foundation
import CoreMotion

@MainActor
final class MotionTools {
    static let shared = MotionTools()

    private let pedometer = CMPedometer()
    private let activityManager = CMMotionActivityManager()

    func motionActivity() async -> String {
        guard CMMotionActivityManager.isActivityAvailable() else {
            return "Motion activity isn't available on this device."
        }

        let end = Date()
        let start = Calendar.current.startOfDay(for: end)

        async let ped = pedometerData(start: start, end: end)
        async let activities = activitySegments(start: start, end: end)

        let (p, acts) = await (ped, activities)

        var parts: [String] = []
        if let p {
            parts.append("\(p.steps) steps")
            if let dist = p.distance { parts.append(String(format: "%.2f km", dist / 1000)) }
            if let floors = p.floors { parts.append("\(floors) floors") }
        }
        if !acts.isEmpty {
            let summary = acts.map { "\($0.minutes)m \($0.label)" }.joined(separator: ", ")
            parts.append("activity: \(summary)")
        }

        if parts.isEmpty {
            return "No motion data today. Grant Motion permission and carry the device."
        }
        return "Today's motion — " + parts.joined(separator: " · ")
    }

    private func pedometerData(start: Date, end: Date) async -> (steps: Int, distance: Double?, floors: Int?)? {
        guard CMPedometer.isStepCountingAvailable() else { return nil }
        return await withCheckedContinuation { cont in
            pedometer.queryPedometerData(from: start, to: end) { data, _ in
                guard let data else { cont.resume(returning: nil); return }
                cont.resume(returning: (data.numberOfSteps.intValue, data.distance?.doubleValue, data.floorsAscended?.intValue))
            }
        }
    }

    private func activitySegments(start: Date, end: Date) async -> [(label: String, minutes: Int)] {
        let status = CMMotionActivityManager.authorizationStatus()
        if status == .denied || status == .restricted { return [] }
        return await withCheckedContinuation { cont in
            activityManager.queryActivityStarting(from: start, to: end, to: .main) { activities, _ in
                guard let activities else { cont.resume(returning: []); return }
                var totals: [String: TimeInterval] = [:]
                for i in 0..<activities.count {
                    let a = activities[i]
                    let next = i + 1 < activities.count ? activities[i + 1].startDate : end
                    let duration = next.timeIntervalSince(a.startDate)
                    guard duration > 0 else { continue }
                    let label: String
                    if a.walking { label = "walking" }
                    else if a.running { label = "running" }
                    else if a.cycling { label = "cycling" }
                    else if a.automotive { label = "driving" }
                    else if a.stationary { label = "stationary" }
                    else { continue }
                    totals[label, default: 0] += duration
                }
                let sorted = totals
                    .map { (label: $0.key, minutes: Int($0.value / 60)) }
                    .filter { $0.minutes > 0 }
                    .sorted { $0.minutes > $1.minutes }
                cont.resume(returning: sorted)
            }
        }
    }
}
