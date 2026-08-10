#!/usr/bin/env python3
"""Portable, fail-closed serialization helpers for tracked pipeline artifacts.

Operational commands need absolute paths while they execute.  Tracked summaries
do not: repository-local paths are made relative to the repository and external
input files are represented by their complete SHA-256 identity.  Any local path
which has not been explicitly classified makes serialization fail instead of
leaking a machine-specific location.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_REF_PREFIX_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FILE_URI_RE = re.compile(r"file:///")
_ABSOLUTE_LOCAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9+.:/\-])/(?!/)[^\s\"'<>`]+"
)


class PortableArtifactError(ValueError):
    """Raised when an artifact cannot be serialized without a local path."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_prefix(prefix: str) -> str:
    if not _REF_PREFIX_RE.fullmatch(prefix):
        raise ValueError("content-reference prefix must use lowercase slug syntax")
    return prefix


def file_content_reference(path: Path, *, prefix: str) -> str:
    """Return a collision-resistant external-file reference.

    Symlinks are rejected so the bytes being named cannot silently come from a
    different location between inspection and reference generation.
    """

    prefix = _validated_prefix(prefix)
    if path.is_symlink() or not path.is_file():
        raise PortableArtifactError(
            "external path is not a regular, non-symlink file and cannot be published"
        )
    return f"{prefix}-sha256-{sha256_file(path)}"


def portable_path_reference(
    root: Path,
    value: str | Path,
    *,
    external_prefix: str,
) -> str:
    """Make a path repository-relative or replace an external file by content ID."""

    root = root.resolve()
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        if candidate.is_symlink() and external_prefix != "external-executable":
            raise PortableArtifactError(
                "external symlink cannot be published as a content reference"
            )
        return file_content_reference(candidate.resolve(), prefix=external_prefix)
    return relative.as_posix() if relative.parts else "."


def path_replacements(
    root: Path,
    paths: Iterable[str | Path],
    *,
    external_prefix: str,
) -> dict[str, str]:
    """Build exact replacements, ordered later by longest source first.

    Both the spelling supplied by the caller and the canonical absolute spelling
    are registered.  Distinct files are named by a full content digest rather
    than a basename, preventing basename and prefix collisions.
    """

    root = root.resolve()
    replacements: dict[str, str] = {str(root): "."}
    for value in paths:
        raw = Path(value).expanduser()
        candidate = raw if raw.is_absolute() else root / raw
        reference = portable_path_reference(
            root,
            candidate,
            external_prefix=external_prefix,
        )
        lexical = Path(os.path.abspath(candidate))
        absolute_spellings = {str(lexical)}
        if not candidate.is_symlink():
            absolute_spellings.add(str(candidate.resolve()))
        spellings = {str(value), *absolute_spellings}
        spellings.update(f"file://{spelling}" for spelling in absolute_spellings)
        for spelling in spellings:
            previous = replacements.get(spelling)
            if previous is not None and previous != reference:
                raise PortableArtifactError(
                    "one local path spelling resolved to conflicting portable references"
                )
            replacements[spelling] = reference
    return replacements


def audit_spec_replacements(
    root: Path,
    specs: Iterable[str | Path],
    *,
    prefix: str = "runtime-audit",
) -> dict[str, str]:
    """Map audit file/glob arguments to content-derived file or set identities."""

    root = root.resolve()
    replacements: dict[str, str] = {}
    for value in specs:
        raw = Path(value).expanduser()
        candidate = raw if raw.is_absolute() else root / raw
        matches = sorted(
            {
                Path(match).resolve()
                for match in glob.glob(str(candidate), recursive=True)
                if Path(match).is_file()
            }
        )
        if not matches and candidate.is_file():
            matches = [candidate.resolve()]
        if not matches:
            raise PortableArtifactError(
                "runtime-audit argument did not resolve to a regular file; refusing to publish its path"
            )
        refs = [
            {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in matches
        ]
        if len(refs) == 1:
            reference = f"{_validated_prefix(prefix)}-sha256-{refs[0]['sha256']}"
        else:
            digest = hashlib.sha256(
                json.dumps(refs, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            reference = f"{_validated_prefix(prefix)}-set-sha256-{digest}"
        for spelling in {str(value), str(candidate)}:
            replacements[spelling] = reference
        replacements.update(
            path_replacements(root, matches, external_prefix=prefix)
        )
    return replacements


def _sanitize_text(text: str, replacements: Mapping[str, str]) -> str:
    sanitized = text
    for source, reference in sorted(
        replacements.items(), key=lambda item: (-len(item[0]), item[0])
    ):
        if source:
            sanitized = re.sub(
                rf"(?<![A-Za-z0-9._~\-/]){re.escape(source)}(?![A-Za-z0-9._~\-/])",
                lambda _match: reference,
                sanitized,
            )
    if _FILE_URI_RE.search(sanitized) or _ABSOLUTE_LOCAL_PATH_RE.search(sanitized):
        raise PortableArtifactError(
            "artifact still contains an unclassified absolute local path"
        )
    return sanitized


def sanitize_payload(
    value: Any,
    *,
    replacements: Mapping[str, str],
) -> Any:
    """Recursively sanitize JSON-compatible data and reject residual local paths."""

    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        return _sanitize_text(value, replacements)
    if isinstance(value, list):
        return [sanitize_payload(item, replacements=replacements) for item in value]
    if isinstance(value, tuple):
        return [sanitize_payload(item, replacements=replacements) for item in value]
    if isinstance(value, dict):
        return {
            str(key): sanitize_payload(item, replacements=replacements)
            for key, item in value.items()
        }
    return value


def portable_command(
    root: Path,
    command: Sequence[str],
    *,
    replacements: Mapping[str, str] | None = None,
) -> list[str]:
    """Serialize argv with repo-relative paths and content-named executables."""

    root = root.resolve()
    merged: dict[str, str] = {str(root): "."}
    if replacements:
        merged.update(replacements)
    for argument in command:
        candidate = Path(argument).expanduser()
        if not candidate.is_absolute():
            continue
        if argument in merged:
            continue
        merged.update(
            path_replacements(
                root,
                [candidate],
                external_prefix="external-executable",
            )
        )
    sanitized = sanitize_payload(list(command), replacements=merged)
    assert isinstance(sanitized, list)
    return [str(item) for item in sanitized]
