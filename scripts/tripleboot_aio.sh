#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_WORKDIR="$ROOT/build/tripleboot"
TRIPLEBOOT_VERSION="0.1.0"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
info() { printf '==> %s\n' "$1"; }
warn() { printf 'warning: %s\n' "$1" >&2; }
fail() { printf 'error: %s\n' "$1" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || fail "$1 requires a value"; }

usage() {
  cat <<'USAGE'
TripleBoot AIO - end-to-end Ventoy USB builder for Ubuntu, Windows, and macOS rescue workflows.

Safe-by-default commands:
  installer-doctor                         Check host tooling and firmware readiness.
  init-workspace [--workdir DIR]           Create the local TripleBoot workspace.
  register-iso --kind KIND --source FILE   Copy a user-provided ISO/DMG/QCOW into the workspace.
  download-ubuntu [options]                Download and verify an Ubuntu ISO from releases.ubuntu.com.
  download-ventoy [options]                Download and verify a Ventoy release archive.
  stage-payloads --usb-mount DIR           Copy staged payloads to an already-mounted Ventoy data partition.
  status [--workdir DIR] [--usb-mount DIR] Show workspace and mounted USB contents.

Destructive command (blocked without --yes-destroy and confirmation):
  prepare-usb-ventoy --usb-disk /dev/sdX --ventoy-dir DIR [--secure-boot] --yes-destroy

Full pipeline:
  build-tripleboot-usb --usb-disk /dev/sdX --windows-iso FILE [--ubuntu-iso FILE]
                       [--include-osx-kvm] [--secure-boot] [--yes-destroy] [--dry-run]

Common options:
  --workdir DIR           Workspace root (default: build/tripleboot).
  --dry-run               Print actions without writing disks or downloading files.
  --yes-destroy           Acknowledge that the USB disk may be erased.
  --allow-data-label      Do not block target disks containing a partition labeled DATA.
  -h, --help              Show this help.

Confirmation for destructive commands:
  Interactive shells prompt for: ERASE /dev/sdX
  Non-interactive usage can set: TRIPLEBOOT_CONFIRM='ERASE /dev/sdX'
USAGE
}

mkdirs() {
  local workdir="$1"
  mkdir -p \
    "$workdir/downloads" \
    "$workdir/isos/ubuntu" \
    "$workdir/isos/windows" \
    "$workdir/macos" \
    "$workdir/ventoy" \
    "$workdir/staging/TripleBoot"
}

json_escape() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }

write_manifest() {
  local workdir="$1"
  mkdirs "$workdir"
  local manifest="$workdir/tripleboot-manifest.json"
  cat >"$manifest" <<EOF_MANIFEST
{
  "schema": "lumen.tripleboot.manifest.v1",
  "created_by": "scripts/tripleboot_aio.sh",
  "version": "$TRIPLEBOOT_VERSION",
  "workspace": $(json_escape "$workdir"),
  "payload_roots": {
    "ubuntu": "isos/ubuntu",
    "windows": "isos/windows",
    "macos": "macos",
    "ventoy": "ventoy"
  },
  "legal_notice": "macOS assets are not redistributed. Use official Apple media on Apple hardware, or provide your own lawful recovery/OpenCore resources."
}
EOF_MANIFEST
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else fail "sha256sum or shasum is required"; fi
}

verify_sha256sums_entry() {
  local sums_file="$1" target="$2"
  [[ -f "$sums_file" ]] || fail "checksum file not found: $sums_file"
  [[ -f "$target" ]] || fail "payload file not found for checksum verification: $target"
  python3 - "$sums_file" "$target" <<'PY_VERIFY'
import hashlib
import pathlib
import sys

sums_path = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
want = None
for raw in sums_path.read_text(encoding="utf-8", errors="replace").splitlines():
    parts = raw.strip().split()
    if len(parts) < 2:
        continue
    digest, name = parts[0], parts[-1].lstrip("*")
    if pathlib.Path(name).name == target.name:
        want = digest.lower()
        break
if want is None:
    raise SystemExit(f"no checksum entry for {target.name} in {sums_path}")
h = hashlib.sha256()
with target.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(chunk)
got = h.hexdigest()
if got != want:
    raise SystemExit(f"checksum mismatch for {target.name}: expected {want} got {got}")
print(f"{target.name}: OK")
PY_VERIFY
}

copy_payload() {
  local src="$1" dest_dir="$2" dry_run="$3"
  [[ -f "$src" ]] || fail "source file not found: $src"
  mkdir -p "$dest_dir"
  local dest="$dest_dir/$(basename "$src")"
  if [[ "$dry_run" == "1" ]]; then
    info "dry-run: would copy $src -> $dest"
  else
    cp -f "$src" "$dest"
    sha256_file "$dest" >"$dest.sha256"
    info "registered $(basename "$dest") with sha256 $(cat "$dest.sha256")"
  fi
}

require_commands() {
  local missing=0
  for cmd in "$@"; do
    if command -v "$cmd" >/dev/null 2>&1; then
      printf 'ok: %s\n' "$cmd"
    else
      printf 'missing: %s\n' "$cmd"
      missing=1
    fi
  done
  return "$missing"
}

root_parent_disk() {
  local root_source pk
  root_source="$(findmnt -n -o SOURCE / 2>/dev/null || true)"
  [[ -n "$root_source" ]] || return 0
  case "$root_source" in
    /dev/*)
      pk="$(lsblk -no PKNAME "$root_source" 2>/dev/null | head -n 1 || true)"
      if [[ -n "$pk" ]]; then printf '/dev/%s\n' "$pk"; else printf '%s\n' "$root_source"; fi
      ;;
  esac
}

normalize_disk() {
  local disk="$1"
  [[ "$disk" == /dev/* ]] || fail "USB disk must be a block device path like /dev/sdX: $disk"
  printf '%s\n' "$disk"
}

assert_safe_target() {
  local disk="$1" allow_data_label="$2"
  [[ -b "$disk" ]] || fail "target disk is not a block device: $disk"
  local root_disk
  root_disk="$(root_parent_disk || true)"
  if [[ -n "$root_disk" && "$disk" == "$root_disk" ]]; then
    fail "refusing to operate on the active root disk ($disk)"
  fi
  if [[ "$allow_data_label" != "1" ]] && lsblk -no LABEL "$disk" 2>/dev/null | awk 'toupper($0)=="DATA" {found=1} END {exit !found}'; then
    fail "target disk contains a partition labeled DATA; pass --allow-data-label only after backup verification"
  fi
}

confirm_destroy() {
  local disk="$1" yes_destroy="$2" dry_run="$3"
  [[ "$dry_run" == "1" ]] && return 0
  [[ "$yes_destroy" == "1" ]] || fail "destructive USB preparation requires --yes-destroy"
  local expected="ERASE $disk"
  if [[ "${TRIPLEBOOT_CONFIRM:-}" == "$expected" ]]; then return 0; fi
  if [[ -t 0 ]]; then
    printf 'Type "%s" to continue: ' "$expected" >&2
    local answer
    read -r answer
    [[ "$answer" == "$expected" ]] || fail "confirmation mismatch"
  else
    fail "non-interactive destructive run requires TRIPLEBOOT_CONFIRM='$expected'"
  fi
}

cmd_installer_doctor() {
  bold "TripleBoot AIO host diagnostics"
  local status=0
  require_commands bash curl awk sed tar findmnt lsblk python3 || status=1
  if command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1; then echo "ok: sha256 tool"; else echo "missing: sha256sum or shasum"; status=1; fi
  for cmd in sgdisk wipefs mkfs.vfat mount umount; do
    if command -v "$cmd" >/dev/null 2>&1; then echo "ok: $cmd"; else warn "optional for prepare-usb-ventoy: $cmd"; fi
  done
  [[ -d /sys/firmware/efi ]] && echo "ok: UEFI firmware detected" || warn "UEFI firmware directory not found; target systems should boot UEFI/GPT"
  if command -v mokutil >/dev/null 2>&1; then mokutil --sb-state || true; else warn "mokutil unavailable; Secure Boot state not detected"; fi
  return "$status"
}

cmd_init_workspace() {
  local workdir="$DEFAULT_WORKDIR"
  while [[ $# -gt 0 ]]; do
    case "$1" in --workdir) need_value "$1" "${2:-}"; workdir="$2"; shift 2;; -h|--help) usage; return 0;; *) fail "unknown option for init-workspace: $1";; esac
  done
  mkdirs "$workdir"
  write_manifest "$workdir"
  cat >"$workdir/staging/TripleBoot/README.txt" <<'EOF_README'
TripleBoot USB

Boot this USB in UEFI mode. Ubuntu and Windows ISO files are staged for Ventoy.
macOS content, when present, is a lawful user-supplied rescue/OpenCore scaffold only;
this kit does not redistribute Apple installers or bypass Apple licensing.
EOF_README
  info "workspace ready: $workdir"
}

cmd_register_iso() {
  local workdir="$DEFAULT_WORKDIR" kind="" source="" dry_run=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --workdir) need_value "$1" "${2:-}"; workdir="$2"; shift 2;;
      --kind) need_value "$1" "${2:-}"; kind="$2"; shift 2;;
      --source) need_value "$1" "${2:-}"; source="$2"; shift 2;;
      --dry-run) dry_run=1; shift;;
      *) fail "unknown option for register-iso: $1";;
    esac
  done
  [[ -n "$kind" && -n "$source" ]] || fail "register-iso requires --kind and --source"
  mkdirs "$workdir"
  case "$kind" in
    ubuntu) copy_payload "$source" "$workdir/isos/ubuntu" "$dry_run";;
    windows) copy_payload "$source" "$workdir/isos/windows" "$dry_run";;
    macos|osx-kvm) copy_payload "$source" "$workdir/macos" "$dry_run";;
    *) fail "kind must be ubuntu, windows, or macos";;
  esac
}

fetch() {
  local url="$1" dest="$2" dry_run="$3"
  mkdir -p "$(dirname "$dest")"
  if [[ "$dry_run" == "1" ]]; then
    info "dry-run: would download $url -> $dest"
  else
    curl -fL --retry 3 --connect-timeout 20 -o "$dest" "$url"
  fi
}

cmd_download_ubuntu() {
  local workdir="$DEFAULT_WORKDIR" version="26.04" edition="desktop" arch="amd64" dry_run=0 url="" checksum_url=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --workdir) need_value "$1" "${2:-}"; workdir="$2"; shift 2;;
      --version) need_value "$1" "${2:-}"; version="$2"; shift 2;;
      --edition) need_value "$1" "${2:-}"; edition="$2"; shift 2;;
      --arch) need_value "$1" "${2:-}"; arch="$2"; shift 2;;
      --url) need_value "$1" "${2:-}"; url="$2"; shift 2;;
      --checksum-url) need_value "$1" "${2:-}"; checksum_url="$2"; shift 2;;
      --dry-run) dry_run=1; shift;;
      *) fail "unknown option for download-ubuntu: $1";;
    esac
  done
  local name="ubuntu-${version}-${edition}-${arch}.iso"
  [[ -n "$url" ]] || url="https://releases.ubuntu.com/${version}/${name}"
  [[ -n "$checksum_url" ]] || checksum_url="https://releases.ubuntu.com/${version}/SHA256SUMS"
  mkdirs "$workdir"
  fetch "$url" "$workdir/isos/ubuntu/$name" "$dry_run"
  fetch "$checksum_url" "$workdir/isos/ubuntu/SHA256SUMS" "$dry_run"
  if [[ "$dry_run" != "1" ]]; then
    verify_sha256sums_entry "$workdir/isos/ubuntu/SHA256SUMS" "$workdir/isos/ubuntu/$name"
  fi
}

cmd_download_ventoy() {
  local workdir="$DEFAULT_WORKDIR" version="" url="" sha256="" dry_run=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --workdir) need_value "$1" "${2:-}"; workdir="$2"; shift 2;;
      --version) need_value "$1" "${2:-}"; version="$2"; shift 2;;
      --url) need_value "$1" "${2:-}"; url="$2"; shift 2;;
      --sha256) need_value "$1" "${2:-}"; sha256="$2"; shift 2;;
      --dry-run) dry_run=1; shift;;
      *) fail "unknown option for download-ventoy: $1";;
    esac
  done
  [[ -n "$version" || -n "$url" ]] || fail "download-ventoy requires --version or --url so the release is explicit"
  [[ -n "$url" ]] || url="https://github.com/ventoy/Ventoy/releases/download/v${version}/ventoy-${version}-linux.tar.gz"
  local archive="$workdir/downloads/$(basename "$url")"
  mkdirs "$workdir"
  fetch "$url" "$archive" "$dry_run"
  if [[ "$dry_run" != "1" ]]; then
    if [[ -n "$sha256" ]]; then
      local actual
      actual="$(sha256_file "$archive")"
      [[ "$actual" == "$sha256" ]] || fail "Ventoy checksum mismatch: expected $sha256 got $actual"
    else
      warn "no --sha256 provided; archive downloaded but not extracted"
      return 0
    fi
    tar -xzf "$archive" -C "$workdir/ventoy" --strip-components=1
    [[ -x "$workdir/ventoy/Ventoy2Disk.sh" ]] || fail "Ventoy2Disk.sh not found after extraction"
  fi
}

cmd_prepare_usb_ventoy() {
  local usb_disk="" ventoy_dir="" secure_boot=0 yes_destroy=0 dry_run=0 allow_data_label=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --usb-disk) need_value "$1" "${2:-}"; usb_disk="$(normalize_disk "$2")"; shift 2;;
      --ventoy-dir) need_value "$1" "${2:-}"; ventoy_dir="$2"; shift 2;;
      --secure-boot) secure_boot=1; shift;;
      --yes-destroy) yes_destroy=1; shift;;
      --dry-run) dry_run=1; shift;;
      --allow-data-label) allow_data_label=1; shift;;
      *) fail "unknown option for prepare-usb-ventoy: $1";;
    esac
  done
  [[ -n "$usb_disk" && -n "$ventoy_dir" ]] || fail "prepare-usb-ventoy requires --usb-disk and --ventoy-dir"
  if [[ "$dry_run" != "1" ]]; then
    assert_safe_target "$usb_disk" "$allow_data_label"
    [[ -x "$ventoy_dir/Ventoy2Disk.sh" ]] || fail "missing executable: $ventoy_dir/Ventoy2Disk.sh"
  fi
  confirm_destroy "$usb_disk" "$yes_destroy" "$dry_run"
  local args=(-I -g)
  [[ "$secure_boot" == "1" ]] && args+=(-s)
  args+=("$usb_disk")
  if [[ "$dry_run" == "1" ]]; then
    info "dry-run: would run sudo $ventoy_dir/Ventoy2Disk.sh ${args[*]}"
  else
    sudo "$ventoy_dir/Ventoy2Disk.sh" "${args[@]}"
  fi
}

cmd_stage_payloads() {
  local workdir="$DEFAULT_WORKDIR" usb_mount="" dry_run=0 include_osx_kvm=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --workdir) need_value "$1" "${2:-}"; workdir="$2"; shift 2;;
      --usb-mount) need_value "$1" "${2:-}"; usb_mount="$2"; shift 2;;
      --include-osx-kvm) include_osx_kvm=1; shift;;
      --dry-run) dry_run=1; shift;;
      *) fail "unknown option for stage-payloads: $1";;
    esac
  done
  [[ -n "$usb_mount" ]] || fail "stage-payloads requires --usb-mount"
  [[ "$dry_run" == "1" || -d "$usb_mount" ]] || fail "USB mount path not found: $usb_mount"
  local dirs=("ISO/Ubuntu" "ISO/Windows" "TripleBoot")
  [[ "$include_osx_kvm" == "1" ]] && dirs+=("macOS/OSX-KVM")
  for d in "${dirs[@]}"; do
    if [[ "$dry_run" == "1" ]]; then info "dry-run: would create $usb_mount/$d"; else mkdir -p "$usb_mount/$d"; fi
  done
  local pair src dest
  for pair in "$workdir/isos/ubuntu:$usb_mount/ISO/Ubuntu" "$workdir/isos/windows:$usb_mount/ISO/Windows" "$workdir/staging/TripleBoot:$usb_mount/TripleBoot"; do
    src="${pair%%:*}"; dest="${pair#*:}"
    if [[ -d "$src" ]]; then
      if [[ "$dry_run" == "1" ]]; then info "dry-run: would sync $src/ -> $dest/"; else cp -a "$src"/. "$dest"/; fi
    fi
  done
  if [[ "$include_osx_kvm" == "1" && -d "$workdir/macos" ]]; then
    if [[ "$dry_run" == "1" ]]; then info "dry-run: would sync $workdir/macos/ -> $usb_mount/macOS/OSX-KVM/"; else cp -a "$workdir/macos"/. "$usb_mount/macOS/OSX-KVM"/; fi
  fi
}

cmd_status() {
  local workdir="$DEFAULT_WORKDIR" usb_mount=""
  while [[ $# -gt 0 ]]; do
    case "$1" in --workdir) need_value "$1" "${2:-}"; workdir="$2"; shift 2;; --usb-mount) need_value "$1" "${2:-}"; usb_mount="$2"; shift 2;; *) fail "unknown option for status: $1";; esac
  done
  bold "TripleBoot workspace status"
  [[ -f "$workdir/tripleboot-manifest.json" ]] && echo "manifest: $workdir/tripleboot-manifest.json" || warn "manifest missing: run init-workspace"
  find "$workdir" -maxdepth 3 -type f \( -name '*.iso' -o -name '*.img' -o -name '*.qcow2' -o -name '*.dmg' -o -name 'README.txt' \) -print 2>/dev/null | sort || true
  if [[ -n "$usb_mount" ]]; then
    bold "Mounted USB status"
    find "$usb_mount" -maxdepth 4 -type f -print 2>/dev/null | sort || true
  fi
}

cmd_build_tripleboot_usb() {
  local workdir="$DEFAULT_WORKDIR" usb_disk="" windows_iso="" ubuntu_iso="" macos_asset="" include_osx_kvm=0 secure_boot=0 yes_destroy=0 dry_run=0 allow_data_label=0 usb_mount=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --workdir) need_value "$1" "${2:-}"; workdir="$2"; shift 2;;
      --usb-disk) need_value "$1" "${2:-}"; usb_disk="$(normalize_disk "$2")"; shift 2;;
      --windows-iso) need_value "$1" "${2:-}"; windows_iso="$2"; shift 2;;
      --ubuntu-iso) need_value "$1" "${2:-}"; ubuntu_iso="$2"; shift 2;;
      --macos-asset) need_value "$1" "${2:-}"; macos_asset="$2"; include_osx_kvm=1; shift 2;;
      --include-osx-kvm) include_osx_kvm=1; shift;;
      --secure-boot) secure_boot=1; shift;;
      --yes-destroy) yes_destroy=1; shift;;
      --dry-run) dry_run=1; shift;;
      --allow-data-label) allow_data_label=1; shift;;
      --usb-mount) need_value "$1" "${2:-}"; usb_mount="$2"; shift 2;;
      *) fail "unknown option for build-tripleboot-usb: $1";;
    esac
  done
  [[ -n "$usb_disk" ]] || fail "build-tripleboot-usb requires --usb-disk"
  cmd_init_workspace --workdir "$workdir"
  local dry_args=() secure_args=() destroy_args=() data_label_args=() osx_args=()
  [[ "$dry_run" == "1" ]] && dry_args+=(--dry-run)
  [[ "$secure_boot" == "1" ]] && secure_args+=(--secure-boot)
  [[ "$yes_destroy" == "1" ]] && destroy_args+=(--yes-destroy)
  [[ "$allow_data_label" == "1" ]] && data_label_args+=(--allow-data-label)
  [[ "$include_osx_kvm" == "1" ]] && osx_args+=(--include-osx-kvm)
  [[ -n "$windows_iso" ]] && cmd_register_iso --workdir "$workdir" --kind windows --source "$windows_iso" "${dry_args[@]}"
  [[ -n "$ubuntu_iso" ]] && cmd_register_iso --workdir "$workdir" --kind ubuntu --source "$ubuntu_iso" "${dry_args[@]}"
  [[ -n "$macos_asset" ]] && cmd_register_iso --workdir "$workdir" --kind macos --source "$macos_asset" "${dry_args[@]}"
  if [[ "$dry_run" == "1" || -x "$workdir/ventoy/Ventoy2Disk.sh" ]]; then
    cmd_prepare_usb_ventoy --usb-disk "$usb_disk" --ventoy-dir "$workdir/ventoy" "${secure_args[@]}" "${destroy_args[@]}" "${dry_args[@]}" "${data_label_args[@]}"
  else
    warn "Ventoy2Disk.sh missing at $workdir/ventoy; run download-ventoy and retry prepare-usb-ventoy"
  fi
  if [[ -n "$usb_mount" ]]; then
    cmd_stage_payloads --workdir "$workdir" --usb-mount "$usb_mount" "${osx_args[@]}" "${dry_args[@]}"
  else
    warn "USB mount not provided; after Ventoy formats the disk, mount its data partition and run stage-payloads"
  fi
  if [[ -n "$usb_mount" ]]; then
    cmd_status --workdir "$workdir" --usb-mount "$usb_mount"
  else
    cmd_status --workdir "$workdir"
  fi
}

main() {
  local cmd="${1:-}"
  [[ -n "$cmd" ]] || { usage; exit 0; }
  shift || true
  case "$cmd" in
    -h|--help|help) usage;;
    installer-doctor) cmd_installer_doctor "$@";;
    init-workspace) cmd_init_workspace "$@";;
    register-iso) cmd_register_iso "$@";;
    download-ubuntu) cmd_download_ubuntu "$@";;
    download-ventoy) cmd_download_ventoy "$@";;
    prepare-usb-ventoy) cmd_prepare_usb_ventoy "$@";;
    stage-payloads) cmd_stage_payloads "$@";;
    status) cmd_status "$@";;
    build-tripleboot-usb) cmd_build_tripleboot_usb "$@";;
    *) fail "unknown command: $cmd";;
  esac
}

main "$@"
