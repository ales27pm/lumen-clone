# Lumen Release Workflow

Use this runbook only when a release upload is explicitly requested. Validation and upload evidence must remain separate: compile/archive/export success is not the same as App Store Connect accepting the IPA.

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
