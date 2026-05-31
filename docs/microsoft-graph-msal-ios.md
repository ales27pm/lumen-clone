# Microsoft Graph MSAL iOS Configuration

Lumen uses `GENERATE_INFOPLIST_FILE=YES`, so the app does not have a checked-in `Info.plist` file. The required MSAL callback keys must live in the Lumen target build settings inside:

```text
ios/Lumen.xcodeproj/project.pbxproj
```

## Required values

```text
Bundle ID: com.27pm.lumen
Redirect URI: msauth.com.27pm.lumen://auth
Client ID: 51aa8fd9-16b2-4f8e-8b97-b8618ceb6c40
Authority: https://login.microsoftonline.com/common
Keychain group: $(AppIdentifierPrefix)com.microsoft.adalcache
```

## Patch generated Info.plist settings

Run from repo root:

```bash
python3 scripts/patch-msal-ios-infoplist.py
```

This idempotently inserts these build settings into both Lumen Debug and Release configurations:

```pbxproj
INFOPLIST_KEY_CFBundleURLTypes = (
	{
		CFBundleURLName = "com.27pm.lumen";
		CFBundleURLSchemes = (
			"msauth.com.27pm.lumen",
		);
	},
);
INFOPLIST_KEY_LSApplicationQueriesSchemes = (
	msauth,
	msauthv2,
	msauthv3,
);
```

## CI / pre-release validation

Run from repo root:

```bash
python3 scripts/validate-msal-ios-release-config.py
```

This validates the documented source-of-truth values for:

- `MSAL_CLIENT_ID` / `MSALClientID` presence + expected ID.
- `MSALRedirectURI` format (`msauth.<bundle-id>://auth`) and expected value.
- App bundle identifier alignment (`PRODUCT_BUNDLE_IDENTIFIER` ↔ redirect URI bundle id).

## Callback handling

SwiftUI already forwards URLs in `LumenApp.swift`:

```swift
.onOpenURL { url in
    MicrosoftGraphURLHandler.handle(url)
}
```

`LumenAppDelegate` also forwards classic app delegate URL callbacks:

```swift
func application(
    _ app: UIApplication,
    open url: URL,
    options: [UIApplication.OpenURLOptionsKey: Any] = [:]
) -> Bool
```

Both paths call:

```swift
MSALPublicClientApplication.handleMSALResponse(...)
```

## Entitlements

`ios/Lumen/Lumen.entitlements` must include:

```xml
<key>keychain-access-groups</key>
<array>
	<string>$(AppIdentifierPrefix)com.microsoft.adalcache</string>
</array>
```

## Verification

After running the patch script, clean build and inspect the generated app plist. It should contain:

```xml
<key>CFBundleURLTypes</key>
<array>
	<dict>
		<key>CFBundleURLName</key>
		<string>com.27pm.lumen</string>
		<key>CFBundleURLSchemes</key>
		<array>
			<string>msauth.com.27pm.lumen</string>
		</array>
	</dict>
</array>

<key>LSApplicationQueriesSchemes</key>
<array>
	<string>msauth</string>
	<string>msauthv2</string>
	<string>msauthv3</string>
</array>
```

Then test Microsoft sign-in on device or simulator.

## TestFlight release handoff checklist

- [ ] Run `python3 scripts/validate-msal-ios-release-config.py` and confirm pass output.
- [ ] Confirm `ios/Lumen/MicrosoftGraphConfig.plist` has expected `MSALClientID` and `MSALRedirectURI`.
- [ ] Confirm Lumen target bundle identifier is `com.27pm.lumen` in `ios/Lumen.xcodeproj/project.pbxproj`.
- [ ] Confirm Entra app registration redirect URI is `msauth.com.27pm.lumen://auth`.
- [ ] Perform sign-in smoke test on a TestFlight candidate build before handoff.
