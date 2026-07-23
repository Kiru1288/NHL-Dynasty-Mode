"""Repair split-module files where block bodies lost indentation."""

from __future__ import annotations

import ast
from pathlib import Path

FRANCHISE = Path(__file__).resolve().parent


def repair_source(source: str) -> str:
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip("\n\r")
        if not stripped.strip():
            out.append(line if line.endswith("\n") else line + "\n")
            continue
        content = stripped.lstrip()
        cur = len(stripped) - len(content)
        if out:
            prev = out[-1]
            prev_stripped = prev.lstrip()
            prev_content = prev_stripped.lstrip()
            prev_indent = len(prev_stripped) - len(prev_content)
            if prev_content.rstrip().endswith(":"):
                needed = prev_indent + 4
                if cur < needed and content and not content.startswith(("def ", "class ", "@")):
                    content = content  # noqa
                    stripped = (" " * needed) + content
        if not stripped.endswith("\n"):
            stripped += "\n"
        out.append(stripped)
    return "".join(out)


def main() -> None:
    skip = {
        "engine.py",
        "_split_engine.py",
        "_repair_indent.py",
        "__init__.py",
        "paths.py",
        "session.py",
        "calendar.py",
        "offseason.py",
        "retirement.py",
        "scouting.py",
        "trade_service.py",
    }
    for fp in FRANCHISE.glob("*.py"):
        if fp.name in skip:
            continue
        raw = fp.read_text(encoding="utf-8")
        fixed = raw
        for _ in range(40):
            nxt = repair_source(fixed)
            if nxt == fixed:
                break
            fixed = nxt
        try:
            ast.parse(fixed)
            fp.write_text(fixed, encoding="utf-8")
            print("repaired OK", fp.name)
        except SyntaxError as e:
            print("still broken", fp.name, e)


if __name__ == "__main__":
    main()
