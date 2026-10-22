# Lumen Release Workflow

Use this runbook only when a release upload is explicitly requested. Validation and upload evidence must remain separate: compile/archive/export success is not the same as App Store Connect accepting the IPA.

## Native EAS TestFlight lane

`eas.json` now defines separate `testflight` build and submit profiles for the
existing native app. The linked EAS project is `@ales27/lumen-clone`
(`1b2304fb-7f12-401b-813c-9a331d8b265d`); the submit destination is the existing
App Store Connect app `6775079037`. The native bundle remains
`com.27pm.lumenclone`, team `52T7P32J34`, scheme `Lumen`, configuration `Release`,
with iOS 27.1 as its minimum. EAS runs Xcode on its macOS builder; this adds no
React Native, Expo runtime, JavaScript bundle, CocoaPods or prebuild step.

Commit the reviewed source and generated scope, resolve and review the complete
Swift package lock, check the current App Store Connect build number, and commit
the newly allocated native build number before dispatch. `cli.requireCommit`
keeps the uploaded shallow clone Git-clean. Do not exclude `.git` or tracked
release inputs with `.easignore`. The dependency-free npm lock serves the EAS
CLI's metadata requirement; native dependencies remain in `Package.resolved`.

The custom workflow `.eas/build/ios-testflight.yml` first runs
`bash scripts/eas-native-ios.sh --preflight`. This verifies the exact source
revision, source hashes, all direct package pins, Xcode/SDK versions, and an
ARM64 iOS 27.1 compiler import of the CoreAI types used by Lumen. The SDK must
also support the committed deployment target before credentials are installed.
These checks do not compile Lumen or execute XCTest.

The named image
`macos-tahoe-26.6-xcode-27.1` is the newest image in the
[EAS infrastructure list](https://docs.expo.dev/build-reference/infrastructure/).
Build `f02f7edb-237c-4682-97f1-01da5646b97f` confirmed Xcode 27.1 / SDK 27.1
and failed during archive planning with the previous iOS 27.2 target and a
profile incorrectly applied to SwiftPM resource bundles. No archive, IPA or
submission was produced. The user authorized the iOS 27.1 minimum for the next
candidate, matching this SDK; its actual app compilation and export must still
pass. The available local Mac has Xcode 26.3 / SDK 26.2 and cannot compile this
CoreAI candidate.

The EAS credentials resolver installs the supplied certificate and provisioning
profile without editing the Xcode project. The wrapper selects exactly one
installed, unexpired App Store profile/certificate pair for Lumen's identity and
capabilities, then supplies a private target-scoped signing configuration to
`archive_lumen_stable.sh`. The Lumen profile must not propagate to SwiftPM
resource bundles, which cannot accept provisioning profiles. The committed
Xcode project remains untouched. It preserves the native lane's source-provenance,
resource, privacy, entitlement and dSYM checks. Export uses the committed
version/build number and cannot upload or silently renumber the app. No App
Store Connect API key is required merely to archive/export with installed
signing credentials.

`EXPO_NO_CAPABILITY_SYNC=1` preserves the existing Apple Developer capability
configuration. The EAS capability mapper does not represent every native Lumen
capability and must not disable existing portal entries while preparing
credentials. This flag does not relax profile or signed-artifact validation;
an incompatible profile still fails the release.

Build and submit are separate operations:

```bash
npx --yes eas-cli@25.1.1 build --platform ios --profile testflight --non-interactive --freeze-credentials --no-wait
```

Record the returned immutable build ID. After that exact build finishes,
download and independently inspect its IPA and `build/eas-evidence` artifacts.
Then submit that ID:

```bash
npx --yes eas-cli@25.1.1 submit --platform ios --profile testflight --id <EXACT_BUILD_ID> --non-interactive --no-auto-testflight-setup
```

`build/eas-evidence/candidate.json` binds source hashes and the full commit.
`artifact.json` binds the validated IPA SHA256, original dSYM hashes and their
exported `Lumen.dSYMs.zip`; toolchain and archive/export/validation logs are
alongside them. Signing context stays outside this uploaded evidence directory.
On failure, the always-run recovery collector retains any existing IPA and
original dSYMs with their hashes in a separate, explicitly unverified evidence
directory. This does not create a successful artifact receipt or authorize
submission. Retain the build ID and partial evidence; inspect any exported IPA
and resolve each failed gate before starting another cloud build. The wrapper
never submits a binary. Submission keeps the existing internal TestFlight
audience; it does not create a new group.

An EAS `FINISHED` build and successful submission are separate from Apple
`VALID` processing and the exact TestFlight build's internal group availability.
Monitor those states independently. None establishes physical-iPhone model
execution, embedding parity, search quality, latency or Neural Engine placement.

## Preflight

1. Check the worktree and preserve unrelated changes.
2. Confirm `CURRENT_PROJECT_VERSION` in `ios/Lumen.xcodeproj/project.pbxproj` is higher than the latest uploaded App Store Connect build number. If it is stale, bump it before archiving.
3. Run the deterministic local checkpoint that fits the change:

```bash
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' \
  build-for-testing \
  CODE_SIGNING_ALLOWED=NO
```

For simulator execution proof, reuse the build-for-testing output with a focused `.xctestrun` and `test-without-building`. Do not recompile just to rerun a narrow simulator suite.

## Upload

Run the repo-native lane through `bash`:

```bash
bash scripts/build_and_submit_appstoreconnect.sh
```

The lane archives with `scripts/archive_lumen_stable.sh`, exports an IPA, validates Info.plist values, validates signed entitlements, and uploads with App Store Connect credentials from the saved local config.

## Success Criteria

Only report a completed submission when all of these are present:

- `** ARCHIVE SUCCEEDED **`
- `** EXPORT SUCCEEDED **`
- archived and exported `CFBundleVersion` match the intended new build number
- archive and IPA entitlement checks pass
- upload output says `UPLOAD SUCCEEDED with no errors`
- upload output includes a `Delivery UUID`

If the upload log contains `ERROR:`, `Failed to upload`, `ENTITY_ERROR`, `must be higher than`, or another validation rejection, the upload failed. Fix the root cause, usually a stale build number or signing/profile issue, and rerun the lane.

Post-upload processing can lag. A successful upload plus `Delivery UUID` is the handoff proof unless the task explicitly asks to wait for App Store Connect processing status.
