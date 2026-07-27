# AGENTS.md

## Scope

Governs `ios/Lumen/Services/MicrosoftGraph/`: authentication, native OAuth fallback, Graph mail transport, protected cache, and Outlook tool adapters. Parent rules: [`../AGENTS.md`](../AGENTS.md).

## Role In The System

This subtree is Lumen's optional network/account boundary for Microsoft Graph. Local assistant behavior must remain available when Graph is unconfigured, signed out, offline, denied, or failing.

## Key Files And Entry Points

- `MicrosoftGraphAuthManager.swift`: main-actor account/auth state and MSAL/native flow selection.
- `NativeMicrosoftOAuthClient.swift`: AuthenticationServices PKCE/state flow and Keychain material.
- `MicrosoftGraphMailClient.swift`: URLSession mail requests, upload sessions, and retry handling.
- `OutlookTools.swift`: secure-tool-facing availability and operation boundary.
- `ios/Lumen/LumenAppDelegate.swift`: authentication callback handoff.
- `ios/Lumen/Views/OutlookMailView.swift`: account/mail UI consumer.

## Public Interfaces

Sign-in/sign-out state, callback handling, typed availability/errors, mail operations, cache behavior, and Outlook tool IDs/results are consumed by views, secure tools, diagnostics, and tests.

## Internal Structure

Auth manager chooses linked MSAL when available or the native PKCE client, stores sensitive material in Keychain-backed state, and supplies access tokens to the mail client. Mail transport performs bounded retries/upload-session operations. Outlook tools convert typed auth/network outcomes into tool results.

## Incoming Dependencies

App delegate URL callbacks, Outlook views, secure tool registry, and user-initiated sign-in/out call this subsystem.

## Outgoing Dependencies

MSAL when linked, AuthenticationServices, Security/Keychain, CryptoKit, URLSession, protected local file cache, and Microsoft Graph endpoints.

## Data And Control Flow

Explicit sign-in -> authorization request with state/PKCE -> callback validation -> token/account state -> authorized Graph request -> bounded response/cache -> typed tool/UI result. Sign-out -> token/account clear -> protected cache purge.

## Local Invariants

- Never log authorization codes, access/refresh tokens, PKCE verifier, raw mail, recipients, or attachment content.
- Validate callback state and redirect ownership before token exchange.
- Store sensitive state in Keychain/protected storage; cache uses atomic writes and complete file protection.
- Sign-out purges account-scoped cache and credentials.
- Retries are bounded and safe for the operation; large upload sessions preserve server-provided state.
- Graph unavailability is typed and cannot break local chat/memory/RAG.

## Coordinated Changes

Auth callback/config changes require `LumenAppDelegate.swift`, Xcode URL/Info settings, MSAL linkage workflow, views, and tests. Tool changes require ToolRegistry/schema/catalog/generated manifest updates. Cache changes require file-protection/sign-out review.

## Safe Editing Rules

Keep authentication user-initiated and main-actor state explicit. Do not embed client secrets or tokens. Do not convert auth/network errors into an empty mailbox. Treat Graph response bodies as untrusted and bound tool output.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' build-for-testing CODE_SIGNING_ALLOWED=NO
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/OutlookToolAvailabilityTests -only-testing:LumenTests/ToolNetworkResilienceTests
```

The hosted MSAL linkage workflow is not run unless explicitly requested. Real sign-in and Graph calls require manual/live evidence; compilation is not authentication proof.

## Common Failure Modes

- Callback routing compiles but URL configuration/product linkage is absent.
- Sign-out leaves protected cache or token material behind.
- Retry repeats a non-idempotent mail action.
- Network/auth failure appears as no messages.
- Sensitive mail/token data enters diagnostics.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). Tool security also follows [`../../Tools/AGENTS.md`](../../Tools/AGENTS.md); workflow linkage follows [`../../../../.github/AGENTS.md`](../../../../.github/AGENTS.md).
