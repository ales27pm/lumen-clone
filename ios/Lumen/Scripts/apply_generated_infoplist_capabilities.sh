#!/bin/sh
set -eu

plist="${TARGET_BUILD_DIR}/${INFOPLIST_PATH}"
if [ ! -f "$plist" ]; then
  exit 0
fi

plistbuddy=/usr/libexec/PlistBuddy

set_bool() {
  key="$1"
  "$plistbuddy" -c "Set :${key} true" "$plist" 2>/dev/null || \
    "$plistbuddy" -c "Add :${key} bool true" "$plist"
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

set_bool UIFileSharingEnabled
set_bool LSSupportsOpeningDocumentsInPlace
set_bool UISupportsDocumentBrowser
set_background_modes
set_background_task_identifiers_from_build_setting
set_carplay_voice_scene
