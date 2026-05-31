# Microsoft Graph / MSAL setup for Lumen

This branch adds a native Microsoft Graph mail surface to Lumen under **Outlook** in the sidebar.

## Implemented

- MSAL-backed delegated authentication for personal Microsoft accounts and Entra ID accounts.
- `common` authority support for Hotmail, Outlook.com, Live, MSN, and work/school accounts.
- Silent token acquisition with proactive refresh checks.
- Microsoft Graph v1.0 mail client using `URLSession` and async/await.
- Inbox delta sync with `@odata.nextLink` and `@odata.deltaLink` support.
- Protected local JSON cache with complete file protection.
- Send mail through `/me/sendMail`.
- Full message body fetch on demand.
- Large attachment upload-session support in the client layer.
- Graph error parsing, throttling, `Retry-After`, and exponential backoff with jitter.
- URL callback hook through SwiftUI `.onOpenURL`.
- Keychain Sharing entitlement for the MSAL cache group.
- Compile-safe fallback when the MSAL package has not been linked yet.

## Required Xcode setup

1. Add Swift Package dependency:

```text
https://github.com/AzureAD/microsoft-authentication-library-for-objc.git
```

Use the latest stable MSAL 2.x release (for example, 2.10.0) and link the `MSAL` product to the **Lumen** target.

2. In Microsoft Entra admin center, register the app:

- Supported account types: **Accounts in any organizational directory and personal Microsoft accounts**.
- Platform: **iOS/macOS**.
- Redirect URI: `msauth.com.27pm.lumen://auth`.
- Public client: enabled.

3. Add delegated Graph permissions:

```text
User.Read
Mail.Read
Mail.Send
offline_access
```

Add `Mail.ReadWrite` only when destructive actions like delete/move/flag are added.

4. Configure the client ID.

Either replace the placeholder in:

```text
ios/Lumen/MicrosoftGraphConfig.plist
```

or set `MSALClientID` / `MSAL_CLIENT_ID` in the target build settings and Info.plist generation flow.

5. Ensure the Lumen target Info.plist includes the URL scheme and Authenticator query schemes. The ready-to-merge fragment is in:

```text
ios/Lumen/MicrosoftGraphInfo.plist.fragment
```

With generated Info.plist targets, add equivalent generated keys in Build Settings or switch the target to an explicit Info.plist.

## Security notes

- Access tokens are not stored in UserDefaults or the local mail cache.
- MSAL owns token and refresh-token persistence through the iOS Keychain.
- The local message cache is written with complete file protection.
- Cache is purged on Microsoft sign-out.
- The app requests minimal scopes for read and send flows.

## Production notes

- Graph webhooks require a server-side relay and APNs; this branch intentionally keeps the iOS app client-only.
- Kiota can be added later for generated typed clients. The current implementation keeps binary size low and avoids the abandoned monolithic ObjC Graph SDK.

## CI / pre-release validation

Before TestFlight handoff, run:

```bash
python3 scripts/validate-msal-ios-release-config.py
```

The validator enforces:

- Expected `MSAL_CLIENT_ID` / `MSALClientID` value presence.
- `MSALRedirectURI` format: `msauth.<bundle-id>://auth`.
- Bundle identifier alignment between redirect URI and `PRODUCT_BUNDLE_IDENTIFIER` (`com.27pm.lumen`).

### TestFlight release handoff checklist

- [ ] Validation script passes from repo root.
- [ ] Entra registration still includes `msauth.com.27pm.lumen://auth` under iOS/macOS platform.
- [ ] Outlook sign-in + inbox fetch smoke-tested on TestFlight candidate.
