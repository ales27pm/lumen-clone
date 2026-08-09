#!/bin/sh
set -eu

plist="${TARGET_BUILD_DIR}/${INFOPLIST_PATH}"
if [ ! -f "$plist" ]; then
  if [ "${CONFIGURATION:-}" = "Debug" ]; then
    echo "warning: Debug build has no generated Info.plist to patch: $plist" >&2
    exit 0
  fi
  echo "error: ${CONFIGURATION:-non-Debug} build is missing generated Info.plist: $plist" >&2
  exit 1
fi

plistbuddy=/usr/libexec/PlistBuddy

set_bool() {
  key="$1"
  value="$2"
  "$plistbuddy" -c "Set :${key} ${value}" "$plist" 2>/dev/null || \
    "$plistbuddy" -c "Add :${key} bool ${value}" "$plist"
}

set_string() {
  key="$1"
  value="$2"
  if [ -z "$value" ]; then
    return
  fi
  escaped="$(printf '%s' "$value" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  "$plistbuddy" -c "Set :${key} \"${escaped}\"" "$plist" 2>/dev/null || \
    "$plistbuddy" -c "Add :${key} string \"${escaped}\"" "$plist"
}

set_background_modes() {
  "$plistbuddy" -c "Delete :UIBackgroundModes" "$plist" 2>/dev/null || true
  "$plistbuddy" -c "Add :UIBackgroundModes array" "$plist"
  "$plistbuddy" -c "Add :UIBackgroundModes:0 string audio" "$plist"
  "$plistbuddy" -c "Add :UIBackgroundModes:1 string fetch" "$plist"
  "$plistbuddy" -c "Add :UIBackgroundModes:2 string location" "$plist"
  "$plistbuddy" -c "Add :UIBackgroundModes:3 string processing" "$plist"
}

set_background_task_identifiers_from_build_setting() {
  identifiers="${INFOPLIST_KEY_BGTaskSchedulerPermittedIdentifiers:-}"
  if [ -z "$identifiers" ]; then
    echo "error: INFOPLIST_KEY_BGTaskSchedulerPermittedIdentifiers is empty" >&2
    exit 1
  fi

  "$plistbuddy" -c "Delete :BGTaskSchedulerPermittedIdentifiers" "$plist" 2>/dev/null || true
  "$plistbuddy" -c "Add :BGTaskSchedulerPermittedIdentifiers array" "$plist"

  index=0
  set -f
  for identifier in $identifiers; do
    "$plistbuddy" -c "Add :BGTaskSchedulerPermittedIdentifiers:${index} string ${identifier}" "$plist"
    index=$((index + 1))
  done
  set +f
}

set_msal_url_registration() {
  bundle_identifier="${PRODUCT_BUNDLE_IDENTIFIER:-}"
  if [ -z "$bundle_identifier" ]; then
    echo "error: PRODUCT_BUNDLE_IDENTIFIER is empty; cannot register the MSAL callback URL" >&2
    exit 1
  fi

  callback_scheme="msauth.${bundle_identifier}"
  "$plistbuddy" -c "Delete :CFBundleURLTypes" "$plist" 2>/dev/null || true
  "$plistbuddy" -c "Add :CFBundleURLTypes array" "$plist"
  "$plistbuddy" -c "Add :CFBundleURLTypes:0 dict" "$plist"
  "$plistbuddy" -c "Add :CFBundleURLTypes:0:CFBundleTypeRole string Editor" "$plist"
  "$plistbuddy" -c "Add :CFBundleURLTypes:0:CFBundleURLName string ${bundle_identifier}" "$plist"
  "$plistbuddy" -c "Add :CFBundleURLTypes:0:CFBundleURLSchemes array" "$plist"
  "$plistbuddy" -c "Add :CFBundleURLTypes:0:CFBundleURLSchemes:0 string ${callback_scheme}" "$plist"

  "$plistbuddy" -c "Delete :LSApplicationQueriesSchemes" "$plist" 2>/dev/null || true
  "$plistbuddy" -c "Add :LSApplicationQueriesSchemes array" "$plist"
  "$plistbuddy" -c "Add :LSApplicationQueriesSchemes:0 string msauth" "$plist"
  "$plistbuddy" -c "Add :LSApplicationQueriesSchemes:1 string msauthv2" "$plist"
  "$plistbuddy" -c "Add :LSApplicationQueriesSchemes:2 string msauthv3" "$plist"
}

set_carplay_voice_scene() {
  module="${PRODUCT_MODULE_NAME:-Lumen}"
  "$plistbuddy" -c "Add :UIApplicationSceneManifest dict" "$plist" 2>/dev/null || true
  "$plistbuddy" -c "Set :UIApplicationSceneManifest:UIApplicationSupportsMultipleScenes true" "$plist" 2>/dev/null || \
    "$plistbuddy" -c "Add :UIApplicationSceneManifest:UIApplicationSupportsMultipleScenes bool true" "$plist"
  "$plistbuddy" -c "Add :UIApplicationSceneManifest:UISceneConfigurations dict" "$plist" 2>/dev/null || true
  "$plistbuddy" -c "Delete :UIApplicationSceneManifest:UISceneConfigurations:CPTemplateApplicationSceneSessionRoleApplication" "$plist" 2>/dev/null || true
  "$plistbuddy" -c "Add :UIApplicationSceneManifest:UISceneConfigurations:CPTemplateApplicationSceneSessionRoleApplication array" "$plist"
  "$plistbuddy" -c "Add :UIApplicationSceneManifest:UISceneConfigurations:CPTemplateApplicationSceneSessionRoleApplication:0 dict" "$plist"
  "$plistbuddy" -c "Add :UIApplicationSceneManifest:UISceneConfigurations:CPTemplateApplicationSceneSessionRoleApplication:0:UISceneConfigurationName string Lumen CarPlay Voice" "$plist"
  "$plistbuddy" -c "Add :UIApplicationSceneManifest:UISceneConfigurations:CPTemplateApplicationSceneSessionRoleApplication:0:UISceneDelegateClassName string ${module}.CarPlayVoiceSceneDelegate" "$plist"
  "$plistbuddy" -c "Add :UIApplicationSceneManifest:UISceneConfigurations:CPTemplateApplicationSceneSessionRoleApplication:0:UISceneClassName string CPTemplateApplicationScene" "$plist"
}

resolve_source_revision() {
  requested_revision="${LUMEN_GIT_SHA:-}"

  # Debug builds intentionally permit developer labels such as "unknown" or a
  # short local revision. Every other configuration must stamp an immutable,
  # full Git object ID into the built artifact.
  if [ "${CONFIGURATION:-}" = "Debug" ]; then
    printf '%s' "${requested_revision:-unknown}"
    return 0
  fi

  case "$requested_revision" in
    ""|unknown)
      requested_revision=HEAD
      ;;
  esac

  source_root="${SRCROOT:-}"
  project_root="${PROJECT_DIR:-}"
  for candidate in "${source_root:+${source_root}/..}" "${project_root:+${project_root}/..}" "$(pwd)"; do
    [ -n "$candidate" ] || continue
    if resolved_revision="$(git -C "$candidate" rev-parse --verify "${requested_revision}^{commit}" 2>/dev/null)"; then
      if printf '%s' "$resolved_revision" | grep -Eq '^[0-9a-f]{40}([0-9a-f]{24})?$'; then
        printf '%s' "$resolved_revision"
        return 0
      fi
    fi
  done

  echo "error: ${CONFIGURATION:-non-Debug} build requires a resolvable full Git source revision; got ${requested_revision}" >&2
  exit 1
}

if ! source_revision="$(resolve_source_revision)"; then
  exit 1
fi

if [ "${CONFIGURATION:-}" = "Debug" ]; then
  set_bool UIFileSharingEnabled true
else
  set_bool UIFileSharingEnabled false
fi
set_bool LSSupportsOpeningDocumentsInPlace true
set_bool UISupportsDocumentBrowser true
set_string NSAlarmKitUsageDescription "${INFOPLIST_KEY_NSAlarmKitUsageDescription:-Lumen uses AlarmKit to schedule prominent alarms and countdowns when you ask.}"
set_string LumenBuildSourceIdentifier "${CURRENT_PROJECT_VERSION:-}"
set_string LumenBuildConfiguration "${CONFIGURATION:-}"
set_string LumenBuildScheme "${LUMEN_BUILD_SCHEME:-${TARGET_NAME:-}}"
set_string LumenGitSHA "$source_revision"
set_msal_url_registration
set_background_modes
set_background_task_identifiers_from_build_setting
set_carplay_voice_scene

alarm_usage="$("$plistbuddy" -c "Print :NSAlarmKitUsageDescription" "$plist" 2>/dev/null || true)"
trimmed_alarm_usage="$(printf '%s' "$alarm_usage" | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
if [ -z "$trimmed_alarm_usage" ]; then
  echo "error: generated Info.plist missing NSAlarmKitUsageDescription after capability patching" >&2
  exit 1
fi
