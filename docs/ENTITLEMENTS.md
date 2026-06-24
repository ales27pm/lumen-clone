# Entitlements and Usage Descriptions

Required background task identifiers:
- `com.27pm.lumenclone.agent.refresh`
- `com.27pm.lumenclone.agent.process`
- `com.27pm.lumenclone.agent.continued-processing.*`

Generated Info.plist usage descriptions are configured in `ios/Lumen.xcodeproj/project.pbxproj` for microphone, speech recognition, contacts, location, photos, AlarmKit, calendar full access, reminders full access, motion, and background modes.

The entitlement validator accepts either `NSCalendarsUsageDescription` or `NSCalendarsFullAccessUsageDescription` for calendar access to match modern generated Info.plist keys.

AppIntents/Shortcuts added in this phase do not require additional entitlements; sensitive actions return an open-app approval message instead of executing directly.

CarPlay support is intentionally limited to voice-based conversation. `ios/Lumen/Lumen.entitlements` and `ios/Lumen/LumenAppStore.entitlements` may include `com.apple.developer.carplay-voice-based-conversation`, and generated Info.plist settings may declare the `CPTemplateApplicationSceneSessionRoleApplication` scene for `CarPlayVoiceSceneDelegate`. The validator still rejects stale generic CarPlay switches such as `UIApplicationSupportsCarPlay` and unsupported CarPlay entitlement keys. The Release/App Store Xcode configuration signs with `ios/Lumen/LumenAppStore.entitlements` so App Store archives do not accidentally pick up development-only capabilities.
