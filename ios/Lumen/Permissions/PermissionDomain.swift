import Foundation

enum PermissionDomain: String, CaseIterable, Codable, Sendable {
    case microphone, speechRecognition, camera, photoLibrary, locationWhenInUse, calendars, reminders, contacts, notifications, localNetwork, motion, appIntents, filesUserSelected, networkAccess
}
