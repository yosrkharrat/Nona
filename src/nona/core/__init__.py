"""Core library: pure Python, no network or file I/O (docs/design/03-architecture.md, section 2)."""

from nona.core.normalize import NormalizedDoc, normalize
from nona.core.types import Entity, EntityType, Span

__all__ = ["Entity", "EntityType", "NormalizedDoc", "Span", "normalize"]
