"""Core library: pure Python, no network or file I/O (docs/design/03-architecture.md, section 2)."""

from nona.core.normalize import NormalizedDoc, normalize
from nona.core.pipeline import DetectorError, Nona
from nona.core.pseudonymize import Anonymized, Mapping, Policy, restore
from nona.core.types import Entity, EntityType, Span

__all__ = [
    "Anonymized",
    "DetectorError",
    "Entity",
    "EntityType",
    "Mapping",
    "Nona",
    "NormalizedDoc",
    "Policy",
    "Span",
    "normalize",
    "restore",
]
