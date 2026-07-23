"""Public package facade for :mod:`qsvt`.

Importing :mod:`qsvt` loads only package metadata and the API-status registry.
Public objects are imported from their implementation modules on first access.
The compact, frozen surface remains available from :mod:`qsvt.stable`.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

from . import api as _api

API_STATUS_COMPATIBILITY = _api.API_STATUS_COMPATIBILITY
API_STATUS_EXPERIMENTAL = _api.API_STATUS_EXPERIMENTAL
API_STATUS_STABLE = _api.API_STATUS_STABLE
COMPATIBILITY_API_NAMES = _api.COMPATIBILITY_API_NAMES
DEPRECATION_POLICY = _api.DEPRECATION_POLICY
STABLE_API_NAMES = _api.STABLE_API_NAMES
__api_statuses__ = _api.__api_statuses__
api_status = _api.api_status

try:
    __version__ = _pkg_version("qsvt-pennylane")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__api_status__ = "alpha"
__public_api_policy__ = (
    "qsvt.stable is the frozen public facade for the remainder of the 0.x "
    "series. Stable and compatibility names remain lazily importable from the "
    "qsvt package root and comprise qsvt.__all__; experimental interfaces "
    "should be imported from their documented submodules. "
    "qsvt.api_status(name) identifies the tier. See qsvt.DEPRECATION_POLICY "
    "for the required deprecation window."
)

# Ordered from focused public facades to lower-level compatibility modules.
# A requested object is cached in this module after the first successful lookup.
_EXPORT_MODULES = (
    "stable",
    "algorithms",
    "approximation",
    "benchmarks",
    "block_encoding",
    "degree",
    "design",
    "diagnostics",
    "execution",
    "flagship",
    "hamiltonians",
    "hardware",
    "comparisons",
    "matrices",
    "operators",
    "pde",
    "planning",
    "polynomials",
    "presets",
    "qsvt",
    "reports",
    "rescaling",
    "research",
    "research_frontier",
    "resources",
    "spectral",
    "synthesis",
    "templates",
    "workflow",
)

_METADATA_EXPORTS = (
    "__version__",
    "__api_status__",
    "__api_statuses__",
    "__public_api_policy__",
    "API_STATUS_COMPATIBILITY",
    "API_STATUS_EXPERIMENTAL",
    "API_STATUS_STABLE",
    "COMPATIBILITY_API_NAMES",
    "DEPRECATION_POLICY",
    "STABLE_API_NAMES",
    "api_status",
)

# Star imports intentionally contain only governed stable/compatibility names.
# Explicit legacy and experimental imports continue to resolve through
# ``__getattr__`` and their documented submodules.
__all__ = [
    *_METADATA_EXPORTS,
    *sorted(set(STABLE_API_NAMES) | set(COMPATIBILITY_API_NAMES)),
]


def __getattr__(name: str) -> Any:
    """Resolve a public object without importing the full package eagerly."""
    for module_name in _EXPORT_MODULES:
        module = import_module(f".{module_name}", __name__)
        if name not in getattr(module, "__all__", ()):
            continue
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return metadata and governed root exports without eager imports."""
    return sorted(set(globals()) | set(__all__))
