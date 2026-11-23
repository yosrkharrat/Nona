"""Suite-wide pytest hooks."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Real-model tests (`model`) run only when the marker expression names them (`-m model`).

    They load ~1.2 GB and switch the process offline for good (ADR-0008), so no other selection, such as
    `-m "not e2e"`, may pull them in by accident.
    """
    if "model" in (config.getoption("markexpr") or ""):
        return
    kept, dropped = [], []
    for item in items:
        (dropped if item.get_closest_marker("model") else kept).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
