import Foundation
import HealthKit

@MainActor
enum HealthTools {
    private static let healthStore = HKHealthStore()

    static func healthSummary() async -> String {
        guard HKHealthStore.isHealthDataAvailable() else {
            return "Health data isn't available on this device."
        }
        let stepType = HKQuantityType(.stepCount)
        let hrType = HKQuantityType(.heartRate)
        let sleepType = HKCategoryType(.sleepAnalysis)
        let energyType = HKQuantityType(.activeEnergyBurned)
        let distanceType = HKQuantityType(.distanceWalkingRunning)

        let readTypes: Set<HKObjectType> = [stepType, hrType, sleepType, energyType, distanceType]

        do {
            try await healthStore.requestAuthorization(toShare: [], read: readTypes)
        } catch {
            return "Couldn't request Health access: \(error.localizedDescription)"
        }

        let cal = Calendar.current
        let start = cal.startOfDay(for: Date())
        let end = Date()

        async let steps = sumQuantity(type: stepType, unit: .count(), start: start, end: end)
        async let distance = sumQuantity(type: distanceType, unit: .meter(), start: start, end: end)
        async let energy = sumQuantity(type: energyType, unit: .kilocalorie(), start: start, end: end)
        async let hr = averageQuantity(type: hrType, unit: HKUnit.count().unitDivided(by: .minute()), start: cal.date(byAdding: .hour, value: -24, to: end)!, end: end)
        async let sleep = sleepHours(start: cal.date(byAdding: .hour, value: -36, to: end)!, end: end)

        let (s, d, e, h, sl) = await (steps, distance, energy, hr, sleep)

        var parts: [String] = []
        if let s { parts.append("\(Int(s).formatted()) steps") }
        if let d { parts.append(String(format: "%.2f km", d / 1000)) }
        if let e { parts.append("\(Int(e)) kcal") }
        if let h { parts.append(String(format: "%.0f bpm avg HR", h)) }
        if let sl { parts.append(String(format: "%.1fh sleep", sl)) }

        if parts.isEmpty {
            return "No Health data available yet today (or access denied). Open the Health app and grant permission."
        }
        return "Today's Health: " + parts.joined(separator: " · ")
    }

    private static func sumQuantity(type: HKQuantityType, unit: HKUnit, start: Date, end: Date) async -> Double? {
        await withCheckedContinuation { cont in
            let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)
            let q = HKStatisticsQuery(quantityType: type, quantitySamplePredicate: predicate, options: .cumulativeSum) { _, stats, _ in
                cont.resume(returning: stats?.sumQuantity()?.doubleValue(for: unit))
            }
            healthStore.execute(q)
        }
    }

    private static func averageQuantity(type: HKQuantityType, unit: HKUnit, start: Date, end: Date) async -> Double? {
        await withCheckedContinuation { cont in
            let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)
            let q = HKStatisticsQuery(quantityType: type, quantitySamplePredicate: predicate, options: .discreteAverage) { _, stats, _ in
                cont.resume(returning: stats?.averageQuantity()?.doubleValue(for: unit))
            }
            healthStore.execute(q)
        }
    }

    private static func sleepHours(start: Date, end: Date) async -> Double? {
        await withCheckedContinuation { cont in
            let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictEndDate)
            let q = HKSampleQuery(sampleType: HKCategoryType(.sleepAnalysis), predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, _ in
                guard let samples = samples as? [HKCategorySample] else {
                    cont.resume(returning: nil); return
                }
                let asleepValues: Set<Int> = [
                    HKCategoryValueSleepAnalysis.asleepUnspecified.rawValue,
                    HKCategoryValueSleepAnalysis.asleepCore.rawValue,
                    HKCategoryValueSleepAnalysis.asleepDeep.rawValue,
                    HKCategoryValueSleepAnalysis.asleepREM.rawValue,
                ]
                let total = samples
                    .filter { asleepValues.contains($0.value) }
                    .reduce(0.0) { $0 + $1.endDate.timeIntervalSince($1.startDate) }
                cont.resume(returning: total > 0 ? total / 3600 : nil)
            }
            healthStore.execute(q)
        }
    }
}
