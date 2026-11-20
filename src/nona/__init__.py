"""Nona: pseudonymize personal data before it reaches an LLM, and restore it in the answer.

Nona reduces exposure. It does not guarantee anonymity. See docs/design/01-requirements.md, section 6.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nona-proxy")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
