from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SwiftFile:
    path: Path
    relpath: str
    text: str


class SwiftExtractor:
    target_names: tuple[str, ...] = ()

    def accepts(self, file: SwiftFile) -> bool:
        return file.path.name in self.target_names

    def extract(self, file: SwiftFile, manifest):  # pragma: no cover - interface
        raise NotImplementedError


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def string_literals(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r'"((?:\\.|[^"\\])*)"', text)]


def enum_cases(text: str, enum_name: str) -> list[str]:
    m = re.search(rf"enum\s+{re.escape(enum_name)}\b[^{{]*{{(?P<body>.*?)\n}}", text, flags=re.S)
    if not m:
        return []
    body = m.group("body")
    cases: list[str] = []
    for cm in re.finditer(r"\bcase\s+([^\n]+)", body):
        line = cm.group(1).split("//", 1)[0]
        for part in line.split(","):
            name = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", part)
            if name:
                cases.append(name.group(1))
    return sorted(dict.fromkeys(cases))


def balanced_call_blocks(text: str, callee: str) -> list[str]:
    blocks: list[str] = []
    token = callee + "("
    start = 0
    while True:
        idx = text.find(token, start)
        if idx == -1:
            break
        pos = idx + len(callee)
        depth = 0
        in_string = False
        escaped = False
        end = pos
        while end < len(text):
            ch = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        blocks.append(text[idx:end + 1])
                        break
            end += 1
        start = max(end + 1, idx + len(token))
    return blocks


def argument_value(block: str, label: str) -> str | None:
    match = re.search(rf"\b{re.escape(label)}\s*:\s*", block)
    if not match:
        return None

    start = match.end()
    depth = 0
    in_string = False
    escaped = False
    end = start
    while end < len(block):
        ch = block[end]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch in "([{":
                depth += 1
            elif ch in ")]}" and depth > 0:
                depth -= 1
            elif depth == 0 and ch in ",\n)":
                break
        end += 1

    value = block[start:end].strip()
    return value or None


def bool_value(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    token = raw.strip().strip(",)").lower()
    if token in {"true", ".true"}:
        return True
    if token in {"false", ".false"}:
        return False
    return default


def clean_swift_string(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip().strip(",")
    if raw in {"nil", "NULL", "Null", "null", "Optional.none", ".none"}:
        return None
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("."):
        return raw[1:]
    return raw
