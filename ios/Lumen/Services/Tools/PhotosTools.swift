import Foundation
import Photos
import AVFoundation
import OSLog

@MainActor
enum PhotosTools {
    private static let logger = Logger(subsystem: "ai.lumen.app", category: "PhotosTools")

    static func searchPhotos(query: String) async -> String {
        let status = await withCheckedContinuation { (cont: CheckedContinuation<PHAuthorizationStatus, Never>) in
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { cont.resume(returning: $0) }
        }
        guard status == .authorized || status == .limited else {
            return "Photo library access was denied."
        }

        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        options.fetchLimit = 500
        let assets = PHAsset.fetchAssets(with: .image, options: options)

        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let now = Date()
        let cal = Calendar.current

        var dateRange: (Date, Date)? = nil
        if trimmed.contains("today") {
            let start = cal.startOfDay(for: now)
            dateRange = (start, now)
        } else if trimmed.contains("yesterday") {
            dateRange = previousDayRange(now: now, calendar: cal)
        } else if trimmed.contains("week") {
            if let weekStart = cal.date(byAdding: .day, value: -7, to: now) {
                dateRange = (weekStart, now)
            }
        } else if trimmed.contains("month") {
            if let monthStart = cal.date(byAdding: .month, value: -1, to: now) {
                dateRange = (monthStart, now)
            }
        } else if trimmed.contains("year") {
            if let yearStart = cal.date(byAdding: .year, value: -1, to: now) {
                dateRange = (yearStart, now)
            }
        }

        let wantFavorites = trimmed.contains("favorite") || trimmed.contains("favourite")
        let wantSelfies = trimmed.contains("selfie")
        let wantVideos = trimmed.contains("video")
        let wantScreenshots = trimmed.contains("screenshot")
        let wantLivePhotos = trimmed.contains("live photo") || trimmed.contains("live")
        let wantPortraits = trimmed.contains("portrait")

        var selfieIDs: Set<String> = []
        if wantSelfies {
            let collections = PHAssetCollection.fetchAssetCollections(with: .smartAlbum, subtype: .smartAlbumSelfPortraits, options: nil)
            collections.enumerateObjects { coll, _, _ in
                let assetsInAlbum = PHAsset.fetchAssets(in: coll, options: nil)
                assetsInAlbum.enumerateObjects { a, _, _ in selfieIDs.insert(a.localIdentifier) }
            }
        }

        var matches: [PHAsset] = []
        assets.enumerateObjects { asset, _, _ in
            if let range = dateRange, let created = asset.creationDate {
                if created < range.0 || created > range.1 { return }
            }
            if wantFavorites && !asset.isFavorite { return }
            if wantScreenshots && !asset.mediaSubtypes.contains(.photoScreenshot) { return }
            if wantLivePhotos && !asset.mediaSubtypes.contains(.photoLive) { return }
            if wantPortraits && !asset.mediaSubtypes.contains(.photoDepthEffect) { return }
            if wantVideos && asset.mediaType != .video { return }
            if wantSelfies && !selfieIDs.contains(asset.localIdentifier) { return }
            matches.append(asset)
        }

        let total = matches.count
        let totalInLibrary = assets.count
        if trimmed.isEmpty {
            return "Photo library has \(totalInLibrary) images. Most recent: \(formatAssetDate(assets.firstObject?.creationDate))."
        }
        if total == 0 {
            if wantSelfies {
                return "I searched your photo library and found no selfies."
            }
            return "No photos match \"\(query)\"."
        }
        if let first = matches.first, trimmed.contains("latest") || trimmed.contains("newest") || trimmed.contains("recent") {
            let noun = wantSelfies ? "selfie" : "photo"
            return "Found latest \(noun) from \(formatAssetDate(first.creationDate)).\n\(assetPayloadJSON(first, queryMatched: query))"
        }
        let sample = matches.prefix(5).map { formatAssetDate($0.creationDate) }.joined(separator: ", ")
        let payloads = matches.prefix(3).map { assetPayloadDictionary($0, queryMatched: query) }
        let payload = jsonString(["matches": payloads]) ?? "{}"
        return "Found \(total) photos matching \"\(query)\". Recent dates: \(sample).\n\(payload)"
    }

    static func captureImage() async -> String {
        #if targetEnvironment(simulator)
        return "Camera is unavailable in the simulator. Install on a real device to capture images."
        #else
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        let granted: Bool
        switch status {
        case .authorized:
            granted = true
        case .notDetermined:
            granted = await AVCaptureDevice.requestAccess(for: .video)
        default:
            granted = false
        }
        guard granted else { return "Camera access was denied." }
        guard AVCaptureDevice.default(for: .video) != nil else {
            return "No camera device available."
        }
        return await CameraCaptureController.shared.capture()
        #endif
    }

    private static func formatAssetDate(_ date: Date?) -> String {
        guard let date else { return "unknown date" }
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    private static func assetPayloadJSON(_ asset: PHAsset, queryMatched: String) -> String {
        jsonString(["match": assetPayloadDictionary(asset, queryMatched: queryMatched)]) ?? "{}"
    }

    private static func assetPayloadDictionary(_ asset: PHAsset, queryMatched: String) -> [String: Any] {
        [
            "localIdentifier": asset.localIdentifier,
            "creationDate": asset.creationDate.map { ISO8601DateFormatter().string(from: $0) } ?? "",
            "mediaSubtypes": mediaSubtypeNames(asset.mediaSubtypes),
            "isFavorite": asset.isFavorite,
            "pixelWidth": asset.pixelWidth,
            "pixelHeight": asset.pixelHeight,
            "queryMatched": queryMatched,
            "displayToken": "photos://asset/\(asset.localIdentifier)"
        ]
    }

    private static func mediaSubtypeNames(_ subtypes: PHAssetMediaSubtype) -> [String] {
        var names: [String] = []
        if subtypes.contains(.photoPanorama) { names.append("photoPanorama") }
        if subtypes.contains(.photoHDR) { names.append("photoHDR") }
        if subtypes.contains(.photoScreenshot) { names.append("photoScreenshot") }
        if subtypes.contains(.photoLive) { names.append("photoLive") }
        if subtypes.contains(.photoDepthEffect) { names.append("photoDepthEffect") }
        return names
    }

    private static func jsonString(_ object: Any) -> String? {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let string = String(data: data, encoding: .utf8) else {
            return nil
        }
        return string
    }

    static func previousDayRange(
        now: Date,
        calendar: Calendar,
        previousDayProvider: (Calendar, Date) -> Date? = { cal, reference in
            cal.date(byAdding: .day, value: -1, to: reference)
        }
    ) -> (Date, Date) {
        if let previousDay = previousDayProvider(calendar, now) {
            return (calendar.startOfDay(for: previousDay), calendar.startOfDay(for: now))
        }

        let fallbackStart = now.addingTimeInterval(-86_400)
        logger.warning(
            "date_math_fallback tool=photos reason=previous_day_addition_failed now=\(now.formatted(date: .numeric, time: .standard), privacy: .public) fallback_start=\(fallbackStart.formatted(date: .numeric, time: .standard), privacy: .public)"
        )
        return (fallbackStart, now)
    }
}
