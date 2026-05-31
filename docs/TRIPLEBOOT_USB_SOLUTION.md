# TripleBoot USB Creation Solution

Lumen now includes a safe, end-to-end **TripleBoot AIO** command-line workflow for creating a UEFI/GPT USB that boots Ubuntu, Windows, and a lawful macOS rescue/OpenCore scaffold through Ventoy.

The implementation lives in `scripts/tripleboot_aio.sh` and turns the earlier design report into an executable pipeline with guardrails, repeatable staging, and dry-run support.

## What the solution does

- Runs host readiness diagnostics for download, checksum, firmware, and disk-preparation tooling.
- Creates a reproducible local workspace under `build/tripleboot` by default.
- Registers user-supplied Ubuntu, Windows, and macOS/OpenCore assets with SHA-256 sidecars.
- Downloads Ubuntu release ISOs and validates them against `SHA256SUMS`.
- Downloads explicit Ventoy release archives and verifies a caller-provided SHA-256 before extraction.
- Installs Ventoy to a target USB only after destructive-operation confirmation.
- Stages payloads into a Ventoy-friendly folder layout.
- Provides a one-command orchestration path plus smaller subcommands for auditable operation.

## Safety model

The script is safe by default:

1. Destructive USB preparation requires `--yes-destroy`.
2. Non-interactive destructive runs must set `TRIPLEBOOT_CONFIRM='ERASE /dev/sdX'`.
3. The active root disk is rejected.
4. A target containing a partition labeled `DATA` is rejected unless `--allow-data-label` is passed after backup verification.
5. Every disk-writing path supports `--dry-run` so the intended command can be reviewed first.
6. Ventoy extraction is blocked unless the archive checksum matches the expected `--sha256` value.

## USB folder layout

After `stage-payloads`, the Ventoy data partition contains:

```text
/ISO/Ubuntu/        # Ubuntu ISO and checksum sidecars
/ISO/Windows/       # Windows ISO and checksum sidecars
/macOS/OSX-KVM/     # Optional user-supplied macOS/OpenCore rescue assets
/TripleBoot/        # README and TripleBoot metadata
```

## Recommended workflow

```bash
# 1. Check the host.
scripts/tripleboot_aio.sh installer-doctor

# 2. Prepare a workspace.
scripts/tripleboot_aio.sh init-workspace --workdir build/tripleboot

# 3. Register local installer media.
scripts/tripleboot_aio.sh register-iso --kind windows --source ~/Downloads/Win.iso
scripts/tripleboot_aio.sh register-iso --kind ubuntu --source ~/Downloads/ubuntu.iso

# 4. Download and verify Ventoy. Use the SHA-256 published for the exact release.
scripts/tripleboot_aio.sh download-ventoy --version 1.x.y --sha256 '<published-sha256>'

# 5. Review the destructive command first.
scripts/tripleboot_aio.sh prepare-usb-ventoy \
  --usb-disk /dev/sdX \
  --ventoy-dir build/tripleboot/ventoy \
  --secure-boot \
  --yes-destroy \
  --dry-run

# 6. Execute after replacing /dev/sdX with the real USB disk.
sudo env TRIPLEBOOT_CONFIRM='ERASE /dev/sdX' scripts/tripleboot_aio.sh prepare-usb-ventoy \
  --usb-disk /dev/sdX \
  --ventoy-dir build/tripleboot/ventoy \
  --secure-boot \
  --yes-destroy

# 7. Mount the Ventoy data partition and stage payloads.
scripts/tripleboot_aio.sh stage-payloads --usb-mount /media/$USER/Ventoy

# 8. Audit the final content.
scripts/tripleboot_aio.sh status --workdir build/tripleboot --usb-mount /media/$USER/Ventoy
```

## One-command orchestration

For controlled automation, use the full pipeline with local ISO inputs:

```bash
scripts/tripleboot_aio.sh build-tripleboot-usb \
  --usb-disk /dev/sdX \
  --windows-iso ~/Downloads/Win.iso \
  --ubuntu-iso ~/Downloads/ubuntu.iso \
  --include-osx-kvm \
  --secure-boot \
  --yes-destroy \
  --dry-run
```

Remove `--dry-run`, provide `TRIPLEBOOT_CONFIRM='ERASE /dev/sdX'`, and add `--usb-mount /path/to/Ventoy` when you are ready to perform the real write-and-stage flow.

## macOS constraints

macOS installer creation is intentionally constrained. The script does **not** download or redistribute Apple proprietary installers. For Apple hardware, use Apple's official media creation flow on macOS. On Linux, the TripleBoot workspace can stage only user-supplied recovery/OpenCore/OSX-KVM resources that the user is legally allowed to use.
