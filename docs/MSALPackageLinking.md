# MSAL package linking

The Outlook integration must compile with the `MSAL` Swift package product linked into the **Lumen** app target. If this is missing, `#if canImport(MSAL)` evaluates to false and the app shows the missing-package screen.

## One-command fix

From the repository root:

```bash
python3 scripts/link-msal-package.py
```

Then commit:

```bash
git add ios/Lumen.xcodeproj/project.pbxproj
git commit -m "Link MSAL package to Lumen target"
```

## What the script adds

- Package repository:

```text
https://github.com/AzureAD/microsoft-authentication-library-for-objc.git
```

- Package requirement:

```text
upToNextMajorVersion from 1.7.0
```

- Product dependency:

```text
MSAL
```

- Lumen app target linkage:

```text
MSAL in Frameworks
```

## Validation

The workflow `.github/workflows/msal-link-validation.yml` runs the same patcher and fails if the resulting `project.pbxproj` diff was not committed. That prevents another TestFlight build from shipping without MSAL linked.
