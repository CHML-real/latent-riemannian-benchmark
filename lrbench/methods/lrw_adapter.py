from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import pkgutil
from dataclasses import dataclass, asdict
from types import ModuleType
from typing import Any, Iterable


DEFAULT_MODULE_CANDIDATES = [
    "latent_riemannian_world",
    "latent_riemannian_worlds",
    "lrw",
    "riemannian_world",
]

DEFAULT_DISTRIBUTION_CANDIDATES = [
    "latent-riemannian-world",
    "latent_riemannian_world",
    "lrw",
]

KEYWORD_SUBMODULES = (
    "geodesic",
    "metric",
    "pullback",
    "transport",
    "solver",
    "sampling",
    "sampler",
    "svgd",
    "sgld",
    "world",
    "latent",
)


@dataclass
class LRWProbeResult:
    available: bool
    module_name: str | None
    version: str | None
    module_file: str | None
    public_symbols: list[str]
    candidate_submodules: list[str]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_version(module: ModuleType, module_name: str) -> str | None:
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version:
        return version

    candidates = [module_name, *DEFAULT_DISTRIBUTION_CANDIDATES]
    for name in candidates:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return None


def _public_symbols(module: ModuleType, limit: int = 80) -> list[str]:
    try:
        symbols = [name for name in dir(module) if not name.startswith("_")]
    except Exception:
        return []
    return sorted(symbols)[:limit]


def _candidate_submodules(module: ModuleType, limit: int = 80) -> list[str]:
    module_path = getattr(module, "__path__", None)
    if module_path is None:
        return []

    found: list[str] = []
    try:
        for item in pkgutil.walk_packages(module_path, prefix=f"{module.__name__}."):
            lower = item.name.lower()
            if any(keyword in lower for keyword in KEYWORD_SUBMODULES):
                found.append(item.name)
                if len(found) >= limit:
                    break
    except Exception:
        return found
    return found


def probe_lrw_package(module_candidates: Iterable[str] | None = None) -> dict[str, Any]:
    """Probe whether the LRW package is importable.

    This function intentionally does not require LRW as a dependency. It returns
    a structured result that benchmark runners can save as `success` or
    `skipped` instead of crashing when LRW is not installed.
    """
    candidates = list(module_candidates or DEFAULT_MODULE_CANDIDATES)
    last_error: str | None = None

    for module_name in candidates:
        try:
            spec = importlib.util.find_spec(module_name)
        except Exception as exc:
            last_error = f"find_spec({module_name!r}) failed: {exc}"
            continue

        if spec is None:
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            last_error = f"import {module_name!r} failed: {exc}"
            continue

        result = LRWProbeResult(
            available=True,
            module_name=module_name,
            version=_safe_version(module, module_name),
            module_file=getattr(module, "__file__", None),
            public_symbols=_public_symbols(module),
            candidate_submodules=_candidate_submodules(module),
            error=None,
        )
        return result.to_dict()

    result = LRWProbeResult(
        available=False,
        module_name=None,
        version=None,
        module_file=None,
        public_symbols=[],
        candidate_submodules=[],
        error=last_error or "No LRW candidate module was importable.",
    )
    return result.to_dict()
