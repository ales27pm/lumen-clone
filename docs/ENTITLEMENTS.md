# Entitlements and Usage Descriptions

Required background task identifiers:
- `com.27pm.lumenclone.agent.refresh`
- `com.27pm.lumenclone.agent.process`
- `com.27pm.lumenclone.agent.continued-processing.*`

Generated Info.plist usage descriptions are configured in `ios/Lumen.xcodeproj/project.pbxproj` for microphone, speech recognition, contacts, location, photos, AlarmKit, calendar full access, reminders full access, motion, and background modes.

The entitlement validator accepts either `NSCalendarsUsageDescription` or `NSCalendarsFullAccessUsageDescription` for calendar access to match modern generated Info.plist keys.

AppIntents/Shortcuts added in this phase do not require additional entitlements; sensitive actions return an open-app approval message instead of executing directly.

CarPlay support is intentionally limited to voice-based conversation. `ios/Lumen/Lumen.entitlements` and `ios/Lumen/LumenAppStore.entitlements` may include `com.apple.developer.carplay-voice-based-conversation`, and generated Info.plist settings may declare the `CPTemplateApplicationSceneSessionRoleApplication` scene for `CarPlayVoiceSceneDelegate`. The validator still rejects stale generic CarPlay switches such as `UIApplicationSupportsCarPlay` and unsupported CarPlay entitlement keys. The Release/App Store Xcode configuration signs with `ios/Lumen/LumenAppStore.entitlements` so App Store archives do not accidentally pick up development-only capabilities.

`scripts/validate_ios_signing_capabilities.py` now treats `ios/Lumen/LumenAppStore.entitlements` as the sanitized App Store profile for `ios/Lumen/Lumen.entitlements`. The default validation fails if the App Store file drifts from the development file after removing known development-only keys such as `com.apple.developer.kernel.increased-debugging-memory-limit` and `com.apple.security.hardened-process.checked-allocations.soft-mode`.

Final archives and exported IPAs should also be validated with signed entitlements:

```bash
python3 scripts/validate_ios_signing_capabilities.py --signed-app-path build/export-Lumen-YYYYMMDD-HHMMSS/Lumen.ipa
```

`scripts/archive_lumen_stable.sh`, `scripts/build_and_submit_appstoreconnect.sh`, and `scripts/check-ios-build-readiness.sh` run this check automatically when a signed `.xcarchive` or `.ipa` is available.

`scripts/archive_lumen_stable.sh` also checks for a local signing identity before starting the expensive archive step. Release/App Store archives require a matching Apple Distribution/iOS Distribution certificate with a private key in the keychain, unless authenticated automatic provisioning is enabled through App Store Connect API key arguments and `LUMEN_IOS_ALLOW_PROVISIONING_UPDATES=1`.
