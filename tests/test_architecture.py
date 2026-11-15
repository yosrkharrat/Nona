"""Architecture rules, enforced by tests rather than by convention.

`nona.core` must stay pure: no network, no subprocesses (docs/design/03-architecture.md, section 2).
This keeps it fast to test, reusable as a library, and unable to send data anywhere by itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1] / "src" / "nona" / "core"

FORBIDDEN = {
    "aiohttp", "asyncio.subprocess", "ftplib", "http", "httpx", "requests", "smtplib",
    "socket", "ssl", "subprocess", "urllib", "urllib3", "websockets",
    "nona.proxy",  # core must not depend on the layer above it
}  # fmt: skip


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _is_forbidden(module: str) -> bool:
    return any(module == f or module.startswith(f + ".") for f in FORBIDDEN)


CORE_FILES = sorted(CORE.rglob("*.py"))


def test_core_package_exists() -> None:
    assert CORE_FILES, f"no modules found under {CORE}"


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: str(p.relative_to(CORE)))
def test_core_does_no_network_or_process_io(path: Path) -> None:
    bad = sorted(m for m in _imports(path) if _is_forbidden(m))
    assert not bad, f"{path.name} imports {bad}; nona.core must stay free of network and process I/O"
