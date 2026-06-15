from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, asdict
from typing import Any, Iterable


DEFAULT_LRW_CLASS_PATHS = [
    "lrw.geodesic.bvp.BVPSolver",
    "lrw.geodesic.solver.GeodesicSolver",
    "lrw.metric.pullback.PullbackMetric",
    "lrw.metric.base.RiemannianMetric",
]


@dataclass
class MethodInfo:
    name: str
    signature: str | None
    kind: str
    error: str | None = None


@dataclass
class ClassInfo:
    class_path: str
    importable: bool
    module_name: str | None
    class_name: str | None
    qualname: str | None
    module_file: str | None
    init_signature: str | None
    public_attributes: list[str]
    methods: list[dict[str, Any]]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def import_object(dotted_path: str) -> Any:
    """Import an object from a full dotted path such as pkg.mod.Class."""
    if "." not in dotted_path:
        raise ValueError(f"dotted object path must contain a module and object name: {dotted_path!r}")
    module_name, object_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def safe_signature(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return None


def _method_kind(obj: Any) -> str:
    if isinstance(obj, staticmethod):
        return "staticmethod"
    if isinstance(obj, classmethod):
        return "classmethod"
    if isinstance(obj, property):
        return "property"
    if inspect.isfunction(obj):
        return "function"
    if inspect.ismethoddescriptor(obj):
        return "method_descriptor"
    if callable(obj):
        return "callable"
    return type(obj).__name__


def inspect_class(class_path: str) -> dict[str, Any]:
    try:
        obj = import_object(class_path)
    except Exception as exc:
        return ClassInfo(
            class_path=class_path,
            importable=False,
            module_name=None,
            class_name=None,
            qualname=None,
            module_file=None,
            init_signature=None,
            public_attributes=[],
            methods=[],
            error=f"import failed: {exc}",
        ).to_dict()

    module_name = getattr(obj, "__module__", None)
    module_file = None
    if module_name:
        try:
            module = importlib.import_module(module_name)
            module_file = getattr(module, "__file__", None)
        except Exception:
            module_file = None

    public_attributes = sorted([name for name in dir(obj) if not name.startswith("_")])

    methods: list[dict[str, Any]] = []
    for name, raw_obj in getattr(obj, "__dict__", {}).items():
        if name.startswith("_"):
            continue
        if isinstance(raw_obj, property):
            methods.append(asdict(MethodInfo(name=name, signature=None, kind="property")))
            continue
        target = raw_obj
        if isinstance(raw_obj, (staticmethod, classmethod)):
            target = raw_obj.__func__
        if callable(target):
            methods.append(
                asdict(
                    MethodInfo(
                        name=name,
                        signature=safe_signature(target),
                        kind=_method_kind(raw_obj),
                    )
                )
            )

    methods = sorted(methods, key=lambda x: x["name"])

    return ClassInfo(
        class_path=class_path,
        importable=True,
        module_name=module_name,
        class_name=getattr(obj, "__name__", None),
        qualname=getattr(obj, "__qualname__", None),
        module_file=module_file,
        init_signature=safe_signature(getattr(obj, "__init__", None)),
        public_attributes=public_attributes,
        methods=methods,
        error=None,
    ).to_dict()


def introspect_lrw_api(class_paths: Iterable[str] | None = None) -> dict[str, Any]:
    paths = list(class_paths or DEFAULT_LRW_CLASS_PATHS)
    classes = [inspect_class(path) for path in paths]
    importable_count = sum(1 for item in classes if item.get("importable"))
    method_count = sum(len(item.get("methods", [])) for item in classes if item.get("importable"))
    signature_count = sum(
        1
        for item in classes
        for method in item.get("methods", [])
        if method.get("signature")
    )
    return {
        "class_paths": paths,
        "classes": classes,
        "class_count": len(paths),
        "importable_class_count": importable_count,
        "method_count": method_count,
        "signature_count": signature_count,
        "all_importable": importable_count == len(paths) and len(paths) > 0,
    }
