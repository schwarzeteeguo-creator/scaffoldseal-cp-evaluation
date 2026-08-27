"""Executable split boundary for ScaffoldSeal-CP v2 R1.

The boundary deliberately exposes no unrestricted ``**fit_kwargs`` path.  It
materializes split-local predictor tables, validates every training and
validation identity before invoking model code, rejects outcome/group metadata
as predictors, isolates mutable run state, and mints stopping histories only
from predictions evaluated on a contract-issued inner-validation batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import secrets
from typing import Callable, Iterable, Mapping, NamedTuple, Sequence
import weakref

import numpy as np
import pandas as pd


ID_COLUMN = "curated_id"
FIT_ROLES = frozenset({"outer_train", "inner_train"})
PREDICT_ROLES = frozenset({"outer_train", "inner_train", "inner_validation", "outer_test"})
FORBIDDEN_FIT_ROLES = frozenset({"test", "outer_test", "final_test", "validation", "inner_validation"})
SAFE_SCALAR_OPTION_KEYS = frozenset({"verbose"})
OUTCOME_NAME_FRAGMENTS = (
    "permeab",
    "papp",
    "logpapp",
    "label",
    "target",
    "outcome",
    "replicate",
)
RESERVED_PREDICTOR_NAMES = frozenset(
    {
        "curatedid",
        "moleculeid",
        "source",
        "analoguecomponentid",
        "sealedblockid",
        "outertestblock",
        "outerfold",
        "innerbasket",
        "partition",
        "role",
        "outerrole",
        "sampleweight",
    }
)
RISKY_OPTION_NAME_FRAGMENTS = (
    "evalset",
    "validationdata",
    "validationframe",
    "callback",
    "dataset",
    "dataloader",
    "loader",
    "sampler",
    "collate",
    "worker",
    "sampleweight",
    "checkpoint",
    "restore",
)
METRICS = frozenset({"mean_absolute_error", "mean_squared_error"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CLAIMED_MUTABLE_OBJECTS: list[Callable[[], object | None]] = []


class _StrongReference:
    """Fallback only for rare mutable objects that do not support weak refs."""

    def __init__(self, value: object) -> None:
        self.value = value

    def __call__(self) -> object:
        return self.value


class SplitViolation(ValueError):
    """Raised before unsafe data or state can reach model code."""


def canonical_id_hash(ids: Iterable[str]) -> str:
    values = sorted({str(value) for value in ids})
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def canonical_array_hash(values: Sequence[float] | np.ndarray) -> str:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise SplitViolation("Audited arrays must be one-dimensional and finite")
    payload = json.dumps(
        [format(float(value), ".17g") for value in array],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def feature_schema_hash(feature_columns: Sequence[str]) -> str:
    payload = json.dumps(list(map(str, feature_columns)), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_frame_hash(frame: pd.DataFrame) -> str:
    """Hash exact frame values, dtypes, column order and index identity."""
    if not isinstance(frame, pd.DataFrame):
        raise SplitViolation("Canonical frame hashing requires a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise SplitViolation("Canonical frame hashing rejects duplicate columns")
    digest = hashlib.sha256(b"scaffoldseal-canonical-frame-v1\0")
    schema = {
        "columns": [(str(column), str(frame[column].dtype)) for column in frame.columns],
        "index_names": [None if name is None else str(name) for name in frame.index.names],
        "index_dtypes": [str(frame.index.dtype)],
        "shape": list(frame.shape),
    }
    digest.update(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0index\0")
    index_hashes = pd.util.hash_pandas_object(frame.index, index=False).to_numpy(
        dtype="uint64", copy=False
    )
    digest.update(index_hashes.tobytes(order="C"))
    for position, column in enumerate(frame.columns):
        digest.update(f"\0column:{position}:{column}\0".encode("utf-8"))
        value_hashes = pd.util.hash_pandas_object(
            frame[column], index=False, categorize=False
        ).to_numpy(dtype="uint64", copy=False)
        digest.update(value_hashes.tobytes(order="C"))
    return digest.hexdigest()


def _detached_read_only_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a deep, non-aliasing frame whose existing arrays are read-only."""
    detached = frame.copy(deep=True)
    for column in detached.columns:
        values = detached[column].to_numpy(copy=False)
        try:
            values.flags.writeable = False
        except (AttributeError, ValueError):
            pass
    return detached


def _checkpoint_namespace_hash(checkpoint_dir: str | Path) -> str:
    """Hash the exact regular-file contents in a completed run namespace."""
    root = Path(checkpoint_dir).resolve()
    if not root.is_dir():
        raise SplitViolation("Run checkpoint namespace is missing")
    digest = hashlib.sha256(b"scaffoldseal-checkpoint-namespace-v1\0")
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise SplitViolation("Checkpoint namespace must not contain symbolic links")
        if path.is_dir():
            digest.update(f"D\0{relative}\0".encode("utf-8"))
            continue
        if not path.is_file():
            raise SplitViolation("Checkpoint namespace contains an unsupported filesystem entry")
        digest.update(f"F\0{relative}\0{path.stat().st_size}\0".encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _content_digest_update(
    digest: "hashlib._Hash", value: object, *, path: str, seen: set[int]
) -> None:
    """Deterministically hash model/optimizer state without caller callbacks."""
    if value is None or isinstance(value, (bool, int, str)):
        digest.update(f"{path}:{type(value).__name__}:{value!r}\0".encode("utf-8"))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SplitViolation(f"{path} contains non-finite state")
        digest.update(f"{path}:float:{value.hex()}\0".encode("utf-8"))
        return
    if isinstance(value, bytes):
        digest.update(f"{path}:bytes:{len(value)}\0".encode("utf-8"))
        digest.update(value)
        return
    if isinstance(value, np.generic):
        _content_digest_update(digest, value.item(), path=path, seen=seen)
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(
            f"{path}:ndarray:{array.dtype}:{array.shape}\0".encode("utf-8")
        )
        digest.update(array.tobytes(order="C"))
        return
    if isinstance(value, Path):
        digest.update(f"{path}:path:{value.resolve()}\0".encode("utf-8"))
        return
    identity = id(value)
    if identity in seen:
        digest.update(f"{path}:cycle\0".encode("utf-8"))
        return
    seen.add(identity)
    if isinstance(value, Mapping):
        digest.update(f"{path}:mapping:{len(value)}\0".encode("utf-8"))
        for key in sorted(value, key=lambda item: repr(item)):
            _content_digest_update(digest, key, path=f"{path}.key", seen=seen)
            _content_digest_update(digest, value[key], path=f"{path}[{key!r}]", seen=seen)
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"{path}:{type(value).__name__}:{len(value)}\0".encode("utf-8"))
        for index, item in enumerate(value):
            _content_digest_update(digest, item, path=f"{path}[{index}]", seen=seen)
        return
    if isinstance(value, (set, frozenset)):
        digest.update(f"{path}:{type(value).__name__}:{len(value)}\0".encode("utf-8"))
        for index, item in enumerate(sorted(value, key=repr)):
            _content_digest_update(digest, item, path=f"{path}[{index}]", seen=seen)
        return

    qualified = f"{type(value).__module__}.{type(value).__qualname__}"
    # Torch tensors include parameters and registered buffers.  This avoids a
    # hard torch dependency for the non-DMPNN boundary tests.
    if qualified.startswith("torch.") and all(
        hasattr(value, name) for name in ("detach", "cpu", "contiguous", "numpy")
    ):
        tensor = value.detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(
            f"{path}:tensor:{tensor.dtype}:{tuple(tensor.shape)}\0".encode("utf-8")
        )
        digest.update(array.tobytes(order="C"))
        return

    state_dict = getattr(value, "state_dict", None)
    if callable(state_dict):
        digest.update(f"{path}:state_dict:{qualified}\0".encode("utf-8"))
        _content_digest_update(digest, state_dict(), path=f"{path}.state", seen=seen)
        return

    # DeepChem TorchModel wrappers keep the prediction-bearing torch module in
    # ``.model``. Hash its full state_dict plus the effective fit/predict code.
    nested_model = getattr(value, "model", None)
    nested_state_dict = getattr(nested_model, "state_dict", None)
    if callable(nested_state_dict):
        digest.update(f"{path}:wrapped_model:{qualified}\0".encode("utf-8"))
        _content_digest_update(
            digest, nested_state_dict(), path=f"{path}.model_state", seen=seen
        )
        for method_name in ("fit", "predict"):
            method = getattr(value, method_name, None)
            digest.update(
                f"{path}.{method_name}:{_callable_identity(method)}\0".encode("utf-8")
            )
        return

    digest.update(f"{path}:object:{qualified}\0".encode("utf-8"))
    if hasattr(value, "__dict__"):
        for key, item in sorted(vars(value).items()):
            if callable(item):
                digest.update(
                    f"{path}.{key}:callable:{_callable_identity(item)}\0".encode("utf-8")
                )
            else:
                _content_digest_update(digest, item, path=f"{path}.{key}", seen=seen)
        for key, item in sorted(vars(type(value)).items()):
            if key.startswith("__") or callable(item) or isinstance(
                item, (property, classmethod, staticmethod)
            ):
                continue
            _content_digest_update(
                digest, item, path=f"{path}.class.{key}", seen=seen
            )


def _object_content_hash(value: object, *, label: str) -> str:
    digest = hashlib.sha256(b"scaffoldseal-object-content-v1\0")
    _content_digest_update(digest, value, path=label, seen=set())
    return digest.hexdigest()


def _training_state_content_hash(state: "FreshTrainingState") -> str:
    return _object_content_hash(
        {"model": state.model, "optimizer": state.optimizer, "scheduler": state.scheduler},
        label="training_state",
    )


def _object_reaches_any(value: object, targets: set[int], seen: set[int] | None = None) -> bool:
    """Return whether ordinary instance/container state aliases a sealed target."""
    if seen is None:
        seen = set()
    if id(value) in targets:
        return True
    if value is None or isinstance(value, (bool, int, float, str, bytes, Path)) or callable(value):
        return False
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, Mapping):
        return any(
            _object_reaches_any(item, targets, seen)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_object_reaches_any(item, targets, seen) for item in value)
    if hasattr(value, "__dict__"):
        return any(_object_reaches_any(item, targets, seen) for item in vars(value).values())
    return False


def _is_data_runtime_object(value: object) -> bool:
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        return True
    cls = type(value)
    qualified = f"{cls.__module__}.{cls.__qualname__}".lower()
    markers = (
        "torch.tensor",
        "torch.utils.data.dataset",
        "torch.utils.data.dataloader",
        "deepchem.data",
        ".subset",
        ".dataset",
        ".dataloader",
    )
    return any(marker in qualified for marker in markers)


def _scan_object_graph(
    value: object,
    *,
    path: str,
    forbidden_ids: frozenset[str] = frozenset(),
    reject_container: bool = False,
    seen: set[int] | None = None,
) -> None:
    """Reject data, callbacks and hidden parent-dataset reachability recursively."""
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and value in forbidden_ids:
            raise SplitViolation(f"{path} captures a forbidden outer-test ID")
        if isinstance(value, float) and not math.isfinite(value):
            raise SplitViolation(f"{path} contains a non-finite scalar")
        return
    if isinstance(value, Path):
        raise SplitViolation(f"{path} contains a path payload")
    if _is_data_runtime_object(value):
        raise SplitViolation(f"{path} contains a frame/array/tensor/dataset/loader payload")
    if callable(value):
        raise SplitViolation(f"{path} contains a callable payload")
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _scan_object_graph(
                key, path=f"{path}.key", forbidden_ids=forbidden_ids,
                reject_container=reject_container, seen=seen,
            )
            _scan_object_graph(
                item, path=f"{path}[{key!r}]", forbidden_ids=forbidden_ids,
                reject_container=reject_container, seen=seen,
            )
        if reject_container:
            raise SplitViolation(f"{path} contains a container; scalar options only")
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            _scan_object_graph(
                item, path=f"{path}[{index}]", forbidden_ids=forbidden_ids,
                reject_container=reject_container, seen=seen,
            )
        if reject_container:
            raise SplitViolation(f"{path} contains a container; scalar options only")
        return
    if hasattr(value, "dataset") or hasattr(value, "sampler") or hasattr(value, "collate_fn"):
        raise SplitViolation(f"{path} exposes parent dataset/loader reachability")
    if hasattr(value, "__dict__"):
        for key, item in vars(value).items():
            _scan_object_graph(
                item, path=f"{path}.{key}", forbidden_ids=forbidden_ids,
                reject_container=False, seen=seen,
            )
        return
    raise SplitViolation(f"{path} contains unsupported non-scalar state: {type(value).__name__}")


def _callable_identity(value: object) -> str:
    cls = value if inspect.isclass(value) else type(value)
    module = getattr(value, "__module__", cls.__module__)
    qualname = getattr(value, "__qualname__", cls.__qualname__)
    try:
        source = inspect.getsource(value if inspect.isfunction(value) else cls)
    except (OSError, TypeError):
        source = f"{module}.{qualname}"
    return f"{module}.{qualname}:{hashlib.sha256(source.encode('utf-8')).hexdigest()}"


def _callable_content_hash(value: object, *, label: str) -> str:
    """Hash effective code plus mutable defaults, attributes and closures."""
    if not callable(value):
        raise SplitViolation(f"{label} must be callable")
    digest = hashlib.sha256(b"scaffoldseal-callable-content-v1\0")
    seen: set[int] = set()

    def update_item(item: object, path: str) -> None:
        if callable(item):
            update_callable(item, path)
        else:
            _content_digest_update(digest, item, path=path, seen=seen)

    def update_callable(candidate: object, path: str) -> None:
        identity = id(candidate)
        if identity in seen:
            digest.update(f"{path}:callable-cycle\0".encode("utf-8"))
            return
        seen.add(identity)
        digest.update(
            f"{path}:identity:{_callable_identity(candidate)}\0".encode("utf-8")
        )
        if isinstance(candidate, functools.partial):
            update_callable(candidate.func, f"{path}.func")
            update_item(candidate.args, f"{path}.args")
            if candidate.keywords is None:
                digest.update(f"{path}.keywords:none\0".encode("utf-8"))
            else:
                update_item(candidate.keywords, f"{path}.keywords")
            return
        if inspect.ismethod(candidate):
            update_callable(candidate.__func__, f"{path}.__func__")
            return
        if inspect.isfunction(candidate):
            if candidate.__defaults__ is None:
                digest.update(f"{path}.__defaults__:none\0".encode("utf-8"))
            else:
                update_item(candidate.__defaults__, f"{path}.__defaults__")
            if candidate.__kwdefaults__ is None:
                digest.update(f"{path}.__kwdefaults__:none\0".encode("utf-8"))
            else:
                update_item(candidate.__kwdefaults__, f"{path}.__kwdefaults__")
            update_item(vars(candidate), f"{path}.__dict__")
            for index, cell in enumerate(candidate.__closure__ or ()):
                try:
                    contents = cell.cell_contents
                except ValueError:
                    contents = "<empty>"
                update_item(contents, f"{path}.closure[{index}]")
            return
        digest.update(
            f"{path}:callable-object:{_object_content_hash(candidate, label=path)}\0".encode(
                "utf-8"
            )
        )

    update_callable(value, label)
    return digest.hexdigest()


_RUNTIME_CLASS_METADATA = frozenset(
    {
        "__annotations__",
        "__classcell__",
        "__dict__",
        "__doc__",
        "__module__",
        "__slots__",
        "__weakref__",
    }
)


def _frame_adapter_content_hash(value: object, *, label: str) -> str:
    """Hash instance state plus effective callable/config state across its MRO.

    ``_object_content_hash`` deliberately treats most callables as identities and
    only sees non-callable state on the concrete class.  A prediction adapter can
    also execute inherited helpers, descriptors, or callables stored directly on
    the instance.  Those carriers are part of the effective predictor and must be
    sealed with their defaults, attributes, closures, and nested class state.
    """
    digest = hashlib.sha256(b"scaffoldseal-frame-adapter-content-v1\0")
    digest.update(_object_content_hash(value, label=label).encode("ascii"))
    seen: set[int] = set()

    def update_callable(candidate: object, path: str) -> None:
        digest.update(f"\0{path}:callable\0".encode("utf-8"))
        digest.update(_callable_content_hash(candidate, label=path).encode("ascii"))

    for key, item in sorted(vars(value).items()):
        if callable(item):
            update_callable(item, f"{label}.{key}")

    for depth, owner in enumerate(type(value).__mro__):
        if owner is object:
            continue
        owner_path = f"{label}.mro[{depth}].{owner.__module__}.{owner.__qualname__}"
        digest.update(f"\0{owner_path}\0".encode("utf-8"))
        for key, item in sorted(vars(owner).items()):
            if key in _RUNTIME_CLASS_METADATA:
                continue
            path = f"{owner_path}.{key}"
            if isinstance(item, (staticmethod, classmethod)):
                update_callable(item.__func__, path)
            elif isinstance(item, functools.partialmethod):
                update_callable(item.func, f"{path}.func")
                _content_digest_update(
                    digest, item.args, path=f"{path}.args", seen=seen
                )
                if item.keywords is None:
                    digest.update(f"{path}.keywords:none\0".encode("utf-8"))
                else:
                    _content_digest_update(
                        digest,
                        item.keywords,
                        path=f"{path}.keywords",
                        seen=seen,
                    )
            elif isinstance(item, property):
                for accessor_name in ("fget", "fset", "fdel"):
                    accessor = getattr(item, accessor_name)
                    if accessor is not None:
                        update_callable(accessor, f"{path}.{accessor_name}")
            elif callable(item):
                update_callable(item, path)
            else:
                _content_digest_update(digest, item, path=path, seen=seen)

        slots = vars(owner).get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in {"__dict__", "__weakref__"} or not hasattr(value, slot):
                continue
            item = getattr(value, slot)
            path = f"{owner_path}.slot.{slot}"
            if callable(item):
                update_callable(item, path)
            else:
                _content_digest_update(digest, item, path=path, seen=seen)
    return digest.hexdigest()


def _sealed_run_context(run: "RunContext") -> "RunContext":
    """Create an owned tuple snapshot containing only canonical immutable fields."""
    if type(run) is not RunContext:
        raise SplitViolation("Run context must be the canonical immutable RunContext type")
    return RunContext(*tuple(run))


def _run_context_content_hash(run: "RunContext") -> str:
    snapshot = _sealed_run_context(run)
    return _object_content_hash(snapshot, label="run_context")


def _scan_boundary_state(value: object, forbidden_ids: frozenset[str], path: str) -> None:
    """Inspect instance state and non-callable class state used by an official adapter."""
    if not inspect.isclass(value):
        for key, item in vars(value).items():
            _scan_object_graph(item, path=f"{path}.{key}", forbidden_ids=forbidden_ids)
        cls = type(value)
    else:
        cls = value

    def scan_class_payload(item: object, item_path: str, seen: set[int] | None = None) -> None:
        if seen is None:
            seen = set()
        if item is None or isinstance(item, (bool, int, float, str)):
            if isinstance(item, str) and item in forbidden_ids:
                raise SplitViolation(f"{item_path} captures a forbidden outer-test ID")
            return
        if _is_data_runtime_object(item):
            raise SplitViolation(f"{item_path} contains a class-level data payload")
        if callable(item):
            return
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, Mapping):
            for key, nested in item.items():
                scan_class_payload(key, f"{item_path}.key", seen)
                scan_class_payload(nested, f"{item_path}[{key!r}]", seen)
        elif isinstance(item, (list, tuple, set, frozenset)):
            for index, nested in enumerate(item):
                scan_class_payload(nested, f"{item_path}[{index}]", seen)
        elif hasattr(item, "dataset") or hasattr(item, "sampler") or hasattr(item, "collate_fn"):
            raise SplitViolation(f"{item_path} exposes class-level dataset reachability")

    for owner in cls.__mro__:
        if owner is object:
            continue
        for key, item in vars(owner).items():
            if key.startswith("__") or callable(item) or isinstance(
                item, (property, classmethod, staticmethod, functools.partialmethod)
            ):
                continue
            scan_class_payload(
                item,
                f"{path}.class.{owner.__module__}.{owner.__qualname__}.{key}",
            )


def _validate_typed_callable(value: object, forbidden_ids: frozenset[str], path: str) -> str:
    if not callable(value):
        raise SplitViolation(f"{path} must be callable")

    seen: set[int] = set()

    def inspect_signature_defaults(candidate: object, candidate_path: str) -> None:
        try:
            signature = inspect.signature(candidate)
        except (TypeError, ValueError):
            return
        for name, parameter in signature.parameters.items():
            if parameter.default is inspect.Parameter.empty:
                continue
            _scan_object_graph(
                parameter.default,
                path=f"{candidate_path}.signature.{name}",
                forbidden_ids=forbidden_ids,
            )

    def inspect_effective_constructor(cls: type, method_name: str, candidate_path: str) -> None:
        owner = next((base for base in cls.__mro__ if method_name in vars(base)), None)
        if owner is None:
            return
        method = vars(owner)[method_name]
        method_path = f"{candidate_path}.{owner.__qualname__}.{method_name}"
        if isinstance(method, functools.partialmethod):
            _scan_object_graph(
                method.args,
                path=f"{method_path}.partial.args",
                forbidden_ids=forbidden_ids,
            )
            _scan_object_graph(
                method.keywords or {},
                path=f"{method_path}.partial.keywords",
                forbidden_ids=forbidden_ids,
            )
            inspect_callable(method.func, f"{method_path}.partial.func")
            return
        if isinstance(method, (staticmethod, classmethod)):
            method = method.__func__
        if inspect.isfunction(method) or inspect.ismethod(method) or isinstance(
            method, functools.partial
        ) or hasattr(method, "__dict__"):
            inspect_callable(method, method_path)
        elif callable(method):
            inspect_signature_defaults(method, method_path)

    def inspect_callable(candidate: object, candidate_path: str) -> None:
        identity = id(candidate)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(candidate, functools.partial):
            _scan_boundary_state(candidate, forbidden_ids, candidate_path)
            _scan_object_graph(
                candidate.args,
                path=f"{candidate_path}.partial.args",
                forbidden_ids=forbidden_ids,
            )
            _scan_object_graph(
                candidate.keywords or {},
                path=f"{candidate_path}.partial.keywords",
                forbidden_ids=forbidden_ids,
            )
            inspect_callable(candidate.func, f"{candidate_path}.partial.func")
            return
        if inspect.ismethod(candidate):
            if candidate.__self__ is not None and not inspect.isclass(candidate.__self__):
                _scan_boundary_state(candidate.__self__, forbidden_ids, f"{candidate_path}.__self__")
            inspect_callable(candidate.__func__, f"{candidate_path}.__func__")
            return
        if inspect.isfunction(candidate):
            for key, item in vars(candidate).items():
                _scan_object_graph(
                    item,
                    path=f"{candidate_path}.{key}",
                    forbidden_ids=forbidden_ids,
                )
            _scan_object_graph(
                candidate.__defaults__ or (),
                path=f"{candidate_path}.__defaults__",
                forbidden_ids=forbidden_ids,
            )
            _scan_object_graph(
                candidate.__kwdefaults__ or {},
                path=f"{candidate_path}.__kwdefaults__",
                forbidden_ids=forbidden_ids,
            )
            for index, cell in enumerate(candidate.__closure__ or ()):
                try:
                    contents = cell.cell_contents
                except ValueError:
                    continue
                _scan_object_graph(
                    contents,
                    path=f"{candidate_path}.closure[{index}]",
                    forbidden_ids=forbidden_ids,
                )
            return
        _scan_boundary_state(candidate, forbidden_ids, candidate_path)
        if inspect.isclass(candidate):
            for method_name in ("__new__", "__init__"):
                inspect_effective_constructor(candidate, method_name, candidate_path)
        else:
            method = getattr(type(candidate), "__call__", None)
            if callable(method):
                inspect_callable(method, f"{candidate_path}.__call__")

    inspect_callable(value, path)
    return _callable_identity(value)


def _claim_fresh_object(value: object, label: str) -> None:
    if value is None:
        raise SplitViolation(f"Fresh {label} object is required")
    live: list[Callable[[], object | None]] = []
    for reference in _CLAIMED_MUTABLE_OBJECTS:
        prior = reference()
        if prior is value:
            raise SplitViolation(f"{label} mutable state was reused across runs")
        if prior is not None:
            live.append(reference)
    try:
        live.append(weakref.ref(value))
    except TypeError:
        live.append(_StrongReference(value))
    _CLAIMED_MUTABLE_OBJECTS[:] = live


def _validate_sha256(value: str, label: str) -> str:
    normalized = str(value).lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise SplitViolation(f"{label} must be an explicit SHA-256")
    return normalized


def _plain_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SplitViolation(f"{label} must be a positive built-in integer")
    return value


def _plain_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise SplitViolation(f"{label} must be a built-in boolean")
    return value


def _freeze_prediction_telemetry(
    telemetry: Mapping[str, object] | Sequence[tuple[str, object]] | None,
) -> tuple[tuple[str, object], ...]:
    """Freeze scalar-only non-scientific telemetry outside adapter state."""
    if telemetry is None:
        return ()
    items = list(telemetry.items()) if isinstance(telemetry, Mapping) else list(telemetry)
    if len(items) > 32:
        raise SplitViolation("Prediction telemetry may contain at most 32 scalar fields")
    frozen: list[tuple[str, object]] = []
    seen_keys: set[str] = set()

    def freeze_value(value: object, path: str) -> object:
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or type(value) in (bool, int):
            return value
        if type(value) is str:
            if len(value) > 1024:
                raise SplitViolation(f"{path} contains an overlong telemetry string")
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise SplitViolation(f"{path} contains non-finite telemetry")
            return value
        raise SplitViolation(f"{path} must be a built-in scalar telemetry value")

    for raw_key, value in items:
        if type(raw_key) is not str:
            raise SplitViolation("Prediction telemetry keys must be built-in strings")
        key = raw_key
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key)
            or key in seen_keys
        ):
            raise SplitViolation("Prediction telemetry keys must be unique snake_case names")
        seen_keys.add(key)
        frozen.append((key, freeze_value(value, f"prediction_telemetry.{key}")))
    return tuple(sorted(frozen))


@dataclass(frozen=True)
class SafeFitOptions:
    """Closed scalar option schema; no data, callback or container can enter."""

    verbose: bool | int | None = None

    @classmethod
    def coerce(cls, value: "SafeFitOptions | Mapping[str, object] | None") -> "SafeFitOptions":
        if value is None:
            return cls()
        if isinstance(value, cls):
            _scan_object_graph(value.verbose, path="options.verbose", reject_container=True)
            if value.verbose is not None and not isinstance(value.verbose, (bool, int)):
                raise SplitViolation("verbose must be bool, int or None")
            return value
        if not isinstance(value, Mapping):
            raise SplitViolation("Options must be SafeFitOptions or a scalar mapping")
        unknown = {normalize_column_name(key) for key in value} - SAFE_SCALAR_OPTION_KEYS
        risky = {
            key for key in value
            if any(fragment in normalize_column_name(key) for fragment in RISKY_OPTION_NAME_FRAGMENTS)
        }
        if unknown or risky:
            raise SplitViolation(f"Unsafe or unrecognized fit option keys: {sorted(map(str, unknown | risky))}")
        for key, item in value.items():
            _scan_object_graph(item, path=f"options.{key}", reject_container=True)
        if value.get("verbose") is not None and not isinstance(value.get("verbose"), (bool, int)):
            raise SplitViolation("verbose must be bool, int or None")
        return cls(verbose=value.get("verbose"))

    def estimator_kwargs(self) -> dict[str, object]:
        return {} if self.verbose is None else {"verbose": self.verbose}


@dataclass(frozen=True)
class SafeBatch:
    outer_fold: int
    role: str
    ids: tuple[str, ...]
    frame: pd.DataFrame = field(repr=False, compare=False)
    contract_token: str = field(repr=False)
    inner_basket: int | None = None
    issuance_token: str | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class SafeTrainingWeights:
    outer_fold: int
    role: str
    inner_basket: int | None
    ids: tuple[str, ...]
    values: tuple[float, ...]
    ids_sha256: str
    values_sha256: str
    contract_token: str = field(repr=False)


class RunContext(NamedTuple):
    """Tuple-backed run identity; even ``object.__setattr__`` cannot mutate it."""

    outer_fold: int
    inner_basket: int | None
    config_id: str
    seed: int
    checkpoint_dir: str
    restore: bool
    pretrained_checkpoint: str | None
    pretrained_checkpoint_sha256: str | None
    run_token: str
    contract_token: str

    def __repr__(self) -> str:
        return (
            "RunContext("
            f"outer_fold={self.outer_fold!r}, inner_basket={self.inner_basket!r}, "
            f"config_id={self.config_id!r}, seed={self.seed!r}, "
            f"checkpoint_dir={self.checkpoint_dir!r}, restore={self.restore!r}, "
            f"pretrained_checkpoint={self.pretrained_checkpoint!r}, "
            f"pretrained_checkpoint_sha256={self.pretrained_checkpoint_sha256!r})"
        )


@dataclass(frozen=True)
class FreshTrainingState:
    model: object = field(repr=False, compare=False)
    optimizer: object = field(repr=False, compare=False)
    scheduler: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class FramePredictionOutput:
    """Adapter output with telemetry detached from sealed scientific state."""

    predictions: object = field(repr=False, compare=False)
    telemetry: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "telemetry", _freeze_prediction_telemetry(self.telemetry))


@dataclass(frozen=True)
class InnerEvaluationEvent:
    epoch: int
    loss: float
    metric_identity: str
    validation_ids_sha256: str
    prediction_sha256: str
    target_sha256: str
    evaluator_identity: str
    feature_schema_sha256: str
    transform_sha256: str
    model_config_sha256: str
    checkpoint_sha256: str
    target_identity: str
    fitted_transform_sha256: str
    execution_identity_sha256: str


@dataclass(frozen=True)
class GuardedInnerHistory:
    outer_fold: int
    inner_basket: int
    seed: int
    config_id: str
    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    metric_identity: str
    feature_columns: tuple[str, ...]
    feature_schema_sha256: str
    target_identity: str
    transform_sha256: str
    model_config_sha256: str
    evaluator_identity: str
    execution_identity_sha256: str
    events: tuple[InnerEvaluationEvent, ...]
    history_token: str = field(repr=False)
    contract_token: str = field(repr=False)


@dataclass(frozen=True)
class InnerExecutionIdentity:
    contract_token: str
    outer_fold: int
    inner_basket: int
    run_token: str
    run_namespace_sha256: str
    training_ids_sha256: str
    validation_ids_sha256: str
    feature_columns: tuple[str, ...]
    feature_schema_sha256: str
    target_identity: str
    transform_sha256: str
    fitted_transform_sha256: str
    model_config_sha256: str
    evaluator_identity: str

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class OuterFrameFitIdentity:
    """Immutable identity minted only after a guarded outer frame fit completes."""

    contract_token: str = field(repr=False)
    outer_fold: int
    run_token: str = field(repr=False)
    run_namespace_sha256: str
    run_context_sha256: str
    config_id: str
    seed: int
    fixed_epoch: int
    outer_train_ids_sha256: str
    outer_test_ids_sha256: str
    authoritative_records_sha256: str
    outer_test_frame_sha256: str
    feature_columns: tuple[str, ...]
    feature_schema_sha256: str
    target_identity: str
    transform_sha256: str
    fitted_transform_sha256: str
    model_config_sha256: str
    checkpoint_sha256: str
    training_state_sha256: str
    adapter_state_sha256: str
    adapter_identity: str
    predictor_identity: str
    predictor_state_sha256: str
    model_identity: str
    optimizer_identity: str
    scheduler_identity: str | None

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def audit_payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"contract_token", "run_token"}
        }


@dataclass(frozen=True)
class SealedOuterFrameFitHandle:
    """Opaque, single-use capability for one executor-owned completed fit."""

    outer_fold: int
    outer_fit_identity_sha256: str
    handle_token: str = field(repr=False)


@dataclass(frozen=True)
class _SealedOuterFrameState:
    handle: SealedOuterFrameFitHandle
    adapter: object = field(repr=False, compare=False)
    state: FreshTrainingState = field(repr=False, compare=False)
    run_context: RunContext = field(repr=False, compare=False)
    identity: OuterFrameFitIdentity = field(repr=False, compare=False)
    outer_prediction_frame: pd.DataFrame = field(repr=False, compare=False)


@dataclass(frozen=True)
class GuardedPredictionResult:
    outer_fold: int
    inner_basket: int | None
    role: str
    ids: tuple[str, ...]
    predictions: tuple[float, ...]
    ids_sha256: str
    prediction_sha256: str
    evaluator_identity: str
    feature_schema_sha256: str
    target_identity: str
    run_namespace_sha256: str
    contract_token: str = field(repr=False)


@dataclass(frozen=True)
class GuardedOuterFramePredictionResult:
    outer_fold: int
    role: str
    ids: tuple[str, ...]
    predictions: tuple[float, ...]
    ids_sha256: str
    prediction_sha256: str
    outer_fit_identity_sha256: str
    run_namespace_sha256: str
    feature_schema_sha256: str
    target_identity: str
    model_config_sha256: str
    transform_sha256: str
    fitted_transform_sha256: str
    checkpoint_sha256: str
    adapter_identity: str
    model_identity: str
    adapter_telemetry: tuple[tuple[str, object], ...]
    contract_token: str = field(repr=False)


class FitAuditTrail:
    """In-memory records that R1 must serialize beside each run."""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def record_fit(
        self,
        operation: str,
        batch: SafeBatch,
        feature_columns: Sequence[str],
        target_column: str,
        run_context: RunContext | None = None,
        weights: SafeTrainingWeights | None = None,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "operation": str(operation),
            "outer_fold": int(batch.outer_fold),
            "inner_basket": batch.inner_basket,
            "role": str(batch.role),
            "n_fit_ids": len(batch.ids),
            "fit_ids_sha256": canonical_id_hash(batch.ids),
            "feature_columns": list(map(str, feature_columns)),
            "feature_schema_sha256": feature_schema_hash(feature_columns),
            "target_column": str(target_column),
        }
        if run_context is not None:
            record.update(
                {
                    "config_id": run_context.config_id,
                    "seed": run_context.seed,
                    "checkpoint_dir": run_context.checkpoint_dir,
                    "restore": run_context.restore,
                }
            )
        if weights is not None:
            record["sample_weight_sha256"] = weights.values_sha256
        self.records.append(record)
        return record

    def record_evaluation(self, event: InnerEvaluationEvent, run: RunContext) -> None:
        self.records.append(
            {
                "operation": "guarded_inner_evaluation",
                "outer_fold": run.outer_fold,
                "inner_basket": run.inner_basket,
                "seed": run.seed,
                **event.__dict__,
            }
        )

    def record_prediction(self, result: GuardedPredictionResult) -> None:
        self.records.append(
            {
                "operation": "guarded_prediction",
                "outer_fold": result.outer_fold,
                "inner_basket": result.inner_basket,
                "role": result.role,
                "n_prediction_ids": len(result.ids),
                "prediction_ids_sha256": result.ids_sha256,
                "prediction_sha256": result.prediction_sha256,
                "evaluator_identity": result.evaluator_identity,
                "feature_schema_sha256": result.feature_schema_sha256,
                "target_identity": result.target_identity,
                "run_namespace_sha256": result.run_namespace_sha256,
            }
        )

    def record_outer_frame_prediction(
        self,
        result: GuardedOuterFramePredictionResult,
        fit_identity: OuterFrameFitIdentity,
    ) -> None:
        if result.outer_fit_identity_sha256 != fit_identity.sha256():
            raise SplitViolation("Outer frame prediction does not match its fit identity")
        self.records.append(
            {
                "operation": "guarded_outer_frame_prediction",
                "outer_fold": result.outer_fold,
                "role": result.role,
                "n_prediction_ids": len(result.ids),
                "prediction_ids_sha256": result.ids_sha256,
                "prediction_sha256": result.prediction_sha256,
                "outer_fit_identity_sha256": result.outer_fit_identity_sha256,
                "run_namespace_sha256": result.run_namespace_sha256,
                "feature_schema_sha256": result.feature_schema_sha256,
                "target_identity": result.target_identity,
                "model_config_sha256": result.model_config_sha256,
                "transform_sha256": result.transform_sha256,
                "fitted_transform_sha256": result.fitted_transform_sha256,
                "checkpoint_sha256": result.checkpoint_sha256,
                "adapter_identity": result.adapter_identity,
                "model_identity": result.model_identity,
                "adapter_telemetry": dict(result.adapter_telemetry),
                "outer_fit_identity": fit_identity.audit_payload(),
            }
        )

    def record_outer_frame_prediction_failure(
        self,
        fit_identity: OuterFrameFitIdentity,
        error: BaseException,
    ) -> None:
        """Record a consumed sealed attempt without accepting prediction output."""
        reason = str(error) or type(error).__name__
        self.records.append(
            {
                "operation": "guarded_outer_frame_prediction_failure",
                "outer_fold": fit_identity.outer_fold,
                "outer_fit_identity_sha256": fit_identity.sha256(),
                "run_namespace_sha256": fit_identity.run_namespace_sha256,
                "failure_type": type(error).__name__,
                "failure_reason": reason,
                "failure_sha256": hashlib.sha256(reason.encode("utf-8")).hexdigest(),
                "handle_consumed": True,
            }
        )

    def record_stopping(
        self,
        outer_fold: int,
        histories: Sequence[GuardedInnerHistory],
        selected_epoch: int,
    ) -> None:
        self.records.append(
            {
                "operation": "select_stopping_epoch",
                "outer_fold": int(outer_fold),
                "selected_epoch": int(selected_epoch),
                "histories": [
                    {
                        "inner_basket": history.inner_basket,
                        "training_ids_sha256": canonical_id_hash(history.training_ids),
                        "validation_ids_sha256": canonical_id_hash(history.validation_ids),
                        "metric_identity": history.metric_identity,
                        "best_epoch": min(history.events, key=lambda event: (event.loss, event.epoch)).epoch,
                    }
                    for history in sorted(histories, key=lambda item: item.inner_basket)
                ],
            }
        )

    def write_json(self, path: Path) -> None:
        payload = json.dumps(self.records, indent=2, sort_keys=True) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8", newline="\n")


def validate_feature_schema(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[str, ...]:
    columns = tuple(map(str, feature_columns))
    if not columns or len(set(columns)) != len(columns):
        raise SplitViolation("Feature columns must be non-empty and unique")
    missing = [column for column in (*columns, str(target_column)) if column not in frame.columns]
    if missing:
        raise SplitViolation(f"Missing feature/target columns: {missing}")
    normalized_target = normalize_column_name(target_column)
    normalized_features = [normalize_column_name(column) for column in columns]
    if len(set(normalized_features)) != len(normalized_features):
        raise SplitViolation("Feature columns collide after name normalization")
    forbidden: list[str] = []
    for column, normalized in zip(columns, normalized_features):
        if (
            normalized == normalized_target
            or normalized in RESERVED_PREDICTOR_NAMES
            or any(fragment in normalized for fragment in OUTCOME_NAME_FRAGMENTS)
        ):
            forbidden.append(column)
    if forbidden:
        raise SplitViolation(f"Outcome/group/identity columns are forbidden predictors: {forbidden}")
    return columns


class OuterFoldContract:
    """Identity, state and history issuer for one outer fold."""

    def __init__(
        self,
        outer_fold: int,
        outer_train_ids: Iterable[str],
        outer_test_ids: Iterable[str],
        inner_basket_by_id: dict[str, int],
    ) -> None:
        self.outer_fold = int(outer_fold)
        self.outer_train_ids = frozenset(map(str, outer_train_ids))
        self.outer_test_ids = frozenset(map(str, outer_test_ids))
        self.inner_basket_by_id = {str(key): int(value) for key, value in inner_basket_by_id.items()}
        if not self.outer_train_ids or not self.outer_test_ids:
            raise SplitViolation("Outer train and test ID sets must both be non-empty")
        if self.outer_train_ids & self.outer_test_ids:
            raise SplitViolation("Outer train and test IDs overlap")
        if set(self.inner_basket_by_id) != set(self.outer_train_ids):
            raise SplitViolation("Every outer-training ID must have exactly one inner basket")
        if set(self.inner_basket_by_id.values()) != {1, 2, 3, 4}:
            raise SplitViolation("Inner basket assignments must use exactly baskets 1-4")
        token_payload = (
            f"fold={self.outer_fold}\ntrain={canonical_id_hash(self.outer_train_ids)}\n"
            f"test={canonical_id_hash(self.outer_test_ids)}\n"
        ).encode("utf-8")
        self._token = hashlib.sha256(token_payload).hexdigest()
        self._run_contexts: dict[str, RunContext] = {}
        self._consumed_runs: set[str] = set()
        self._run_models: dict[str, object] = {}
        self._run_frame_states: dict[str, tuple[object, FreshTrainingState]] = {}
        self._authoritative_record_issuances: dict[str, tuple[pd.DataFrame, str]] = {}
        self._issued_histories: dict[str, str] = {}
        self._consumed_histories: set[str] = set()

    def _validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise SplitViolation("Safe batches must be materialized pandas DataFrames")
        if ID_COLUMN not in frame.columns:
            raise SplitViolation(f"Missing required ID column: {ID_COLUMN}")
        if frame[ID_COLUMN].astype(str).duplicated().any():
            raise SplitViolation("Input frame contains duplicate curated IDs")
        return frame.assign(**{ID_COLUMN: frame[ID_COLUMN].astype(str)})

    def _batch(
        self,
        frame: pd.DataFrame,
        ids: Iterable[str],
        role: str,
        basket: int | None = None,
        issuance_token: str | None = None,
    ) -> SafeBatch:
        work = self._validate_frame(frame)
        requested = frozenset(map(str, ids))
        missing = requested - set(work[ID_COLUMN])
        if missing:
            raise SplitViolation(f"Input frame is missing {len(missing)} required IDs")
        subset = work.loc[work[ID_COLUMN].isin(requested)].copy()
        subset = subset.sort_values(ID_COLUMN, kind="stable").reset_index(drop=True)
        if len(subset) != len(requested):
            raise SplitViolation("Safe-batch row count does not match requested IDs")
        return SafeBatch(
            self.outer_fold,
            role,
            tuple(subset[ID_COLUMN]),
            subset,
            self._token,
            basket,
            issuance_token,
        )

    def outer_training_batch(self, frame: pd.DataFrame) -> SafeBatch:
        return self._batch(frame, self.outer_train_ids, "outer_train")

    def outer_frame_training_batch(self, frame: pd.DataFrame) -> SafeBatch:
        """Issue outer training plus the authoritative records needed by frame prediction."""
        issuance_token = self._issue_authoritative_records(frame)
        return self._batch(
            frame,
            self.outer_train_ids,
            "outer_train",
            issuance_token=issuance_token,
        )

    def outer_test_batch(self, frame: pd.DataFrame) -> SafeBatch:
        return self._batch(frame, self.outer_test_ids, "outer_test")

    def _issue_authoritative_records(self, frame: pd.DataFrame) -> str:
        """Seal exact train/test records for one outer-training batch issuance."""
        work = self._validate_frame(frame)
        required = self.outer_train_ids | self.outer_test_ids
        missing = required - set(work[ID_COLUMN])
        if missing:
            raise SplitViolation(
                "Outer-training issuance requires all authoritative train/test records"
            )
        snapshot = work.loc[work[ID_COLUMN].isin(required)].copy(deep=True)
        snapshot = snapshot.sort_values(ID_COLUMN, kind="stable").reset_index(drop=True)
        if len(snapshot) != len(required):
            raise SplitViolation("Authoritative outer record set has an invalid row count")
        observed_hash = canonical_frame_hash(snapshot)
        issuance_token = secrets.token_hex(32)
        self._authoritative_record_issuances[issuance_token] = (snapshot, observed_hash)
        return issuance_token

    def authoritative_outer_prediction_frame(
        self,
        feature_columns: Sequence[str],
        target_column: str,
        issuance_token: str,
    ) -> tuple[pd.DataFrame, str, str]:
        """Reconstruct the exact outcome-free outer predictor frame internally."""
        issued = self._authoritative_record_issuances.get(str(issuance_token))
        if issued is None:
            raise SplitViolation("Outer prediction records lack a valid batch issuance")
        authoritative_records, authoritative_records_sha256 = issued
        if canonical_frame_hash(authoritative_records) != authoritative_records_sha256:
            raise SplitViolation("Sealed authoritative outer records were mutated")
        features = validate_feature_schema(
            authoritative_records, feature_columns, target_column
        )
        projection = authoritative_records.loc[
            authoritative_records[ID_COLUMN].isin(self.outer_test_ids),
            [ID_COLUMN, *features],
        ].copy(deep=True)
        projection = projection.sort_values(ID_COLUMN, kind="stable").reset_index(drop=True)
        expected_ids = tuple(sorted(self.outer_test_ids))
        if tuple(projection[ID_COLUMN].astype(str)) != expected_ids:
            raise SplitViolation("Sealed outer predictor frame has incorrect IDs/order")
        return (
            projection,
            authoritative_records_sha256,
            canonical_frame_hash(projection),
        )

    def validate_authoritative_outer_training_batch(self, batch: SafeBatch) -> str:
        """Bind outer fitting to the exact record values sealed at issuance."""
        if batch.issuance_token is None:
            raise SplitViolation("Outer training records were not sealed at batch issuance")
        issued = self._authoritative_record_issuances.get(batch.issuance_token)
        if issued is None:
            raise SplitViolation("Outer training record issuance is forged or consumed")
        authoritative_records, authoritative_records_sha256 = issued
        if canonical_frame_hash(authoritative_records) != authoritative_records_sha256:
            raise SplitViolation("Sealed authoritative outer records were mutated")
        expected = authoritative_records.loc[
            authoritative_records[ID_COLUMN].isin(self.outer_train_ids)
        ].copy(deep=True).reset_index(drop=True)
        if canonical_frame_hash(batch.frame) != canonical_frame_hash(expected):
            raise SplitViolation("Outer training batch values differ from the sealed records")
        return batch.issuance_token

    def inner_training_batch(self, frame: pd.DataFrame, basket: int) -> SafeBatch:
        train, _ = self.expected_inner_ids(basket)
        return self._batch(frame, train, "inner_train", int(basket))

    def inner_validation_batch(self, frame: pd.DataFrame, basket: int) -> SafeBatch:
        _, validation = self.expected_inner_ids(basket)
        return self._batch(frame, validation, "inner_validation", int(basket))

    def _validate_contract_batch(self, batch: SafeBatch) -> frozenset[str]:
        if batch.contract_token != self._token or batch.outer_fold != self.outer_fold:
            raise SplitViolation("Safe batch belongs to a different outer-fold contract")
        ordered_ids = tuple(map(str, batch.ids))
        ids = frozenset(ordered_ids)
        frame_ids = (
            tuple(batch.frame[ID_COLUMN].astype(str))
            if ID_COLUMN in batch.frame.columns
            else ()
        )
        if frame_ids != ordered_ids or len(ids) != len(ordered_ids):
            raise SplitViolation("Safe batch IDs and frame IDs differ")
        return ids

    def _validate_role_columns(self, batch: SafeBatch) -> None:
        for role_column in ("role", "outer_role", "partition"):
            if role_column in batch.frame.columns:
                values = {str(value).strip().lower() for value in batch.frame[role_column]}
                if values & FORBIDDEN_FIT_ROLES:
                    raise SplitViolation(f"Fitting input contains forbidden {role_column} role values")

    def validate_fit_batch(self, batch: SafeBatch) -> None:
        ids = self._validate_contract_batch(batch)
        if batch.role not in FIT_ROLES:
            raise SplitViolation(f"Role {batch.role!r} is forbidden for fitting")
        if ids & self.outer_test_ids or not ids <= self.outer_train_ids:
            raise SplitViolation("Fitting input contains IDs outside outer training")
        self._validate_role_columns(batch)

    def validate_exact_fit_batch(self, batch: SafeBatch) -> None:
        self.validate_fit_batch(batch)
        ids = frozenset(batch.ids)
        if batch.role == "outer_train":
            if batch.inner_basket is not None or ids != self.outer_train_ids:
                raise SplitViolation("Outer fit requires the exact full outer-training SafeBatch")
        elif batch.role == "inner_train":
            if batch.inner_basket not in {1, 2, 3, 4}:
                raise SplitViolation("Inner-training batch lacks a valid basket")
            expected, _ = self.expected_inner_ids(batch.inner_basket)
            if ids != expected:
                raise SplitViolation("Inner fit requires the exact frozen inner-training IDs")

    def validate_exact_outer_test_batch(self, batch: SafeBatch) -> None:
        ids = self._validate_contract_batch(batch)
        expected_order = tuple(sorted(self.outer_test_ids))
        if (
            batch.role != "outer_test"
            or batch.inner_basket is not None
            or ids != self.outer_test_ids
            or batch.ids != expected_order
        ):
            raise SplitViolation("Outer prediction requires the exact contract-issued outer-test batch")

    def validate_transform_batch(self, batch: SafeBatch) -> None:
        self._validate_contract_batch(batch)
        if batch.role not in PREDICT_ROLES:
            raise SplitViolation(f"Unknown transform/predict role: {batch.role}")

    def validate_inner_pair(self, train: SafeBatch, validation: SafeBatch, basket: int) -> None:
        basket = int(basket)
        self.validate_exact_fit_batch(train)
        validation_ids = self._validate_contract_batch(validation)
        expected_train, expected_validation = self.expected_inner_ids(basket)
        if (
            train.role != "inner_train"
            or validation.role != "inner_validation"
            or train.inner_basket != basket
            or validation.inner_basket != basket
            or frozenset(train.ids) != expected_train
            or validation_ids != expected_validation
        ):
            raise SplitViolation("Inner fit requires exact same-contract train/validation batches")
        if (frozenset(train.ids) | validation_ids) & self.outer_test_ids:
            raise SplitViolation("Outer-test IDs are forbidden in inner fit/evaluation")

    def expected_inner_ids(self, basket: int) -> tuple[frozenset[str], frozenset[str]]:
        basket = int(basket)
        if basket not in {1, 2, 3, 4}:
            raise SplitViolation("Inner basket must be 1-4")
        validation = frozenset(
            record_id for record_id, assigned in self.inner_basket_by_id.items() if assigned == basket
        )
        return self.outer_train_ids - validation, validation

    def training_weights(self, batch: SafeBatch, column: str) -> SafeTrainingWeights:
        self.validate_exact_fit_batch(batch)
        if column not in batch.frame.columns:
            raise SplitViolation(f"Missing training-weight column: {column}")
        normalized = normalize_column_name(column)
        if any(fragment in normalized for fragment in OUTCOME_NAME_FRAGMENTS):
            raise SplitViolation("Outcome-derived sample-weight columns are forbidden")
        values = pd.to_numeric(batch.frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
            raise SplitViolation("Training weights must be finite, nonnegative and nonzero")
        return SafeTrainingWeights(
            self.outer_fold,
            batch.role,
            batch.inner_basket,
            batch.ids,
            tuple(map(float, values)),
            canonical_id_hash(batch.ids),
            canonical_array_hash(values),
            self._token,
        )

    def validate_training_weights(self, weights: SafeTrainingWeights, batch: SafeBatch) -> None:
        if (
            weights.contract_token != self._token
            or weights.outer_fold != self.outer_fold
            or weights.role != batch.role
            or weights.inner_basket != batch.inner_basket
            or weights.ids != batch.ids
            or weights.ids_sha256 != canonical_id_hash(batch.ids)
            or weights.values_sha256 != canonical_array_hash(weights.values)
        ):
            raise SplitViolation("Training-weight payload does not match the exact fit batch")

    def mint_run_context(
        self,
        checkpoint_root: Path,
        *,
        config_id: str,
        seed: int,
        inner_basket: int | None,
        pretrained_checkpoint: Path | None = None,
        pretrained_checkpoint_sha256: str | None = None,
    ) -> RunContext:
        if inner_basket is not None and (
            isinstance(inner_basket, bool) or not isinstance(inner_basket, int)
        ):
            raise SplitViolation("Run context inner basket must be a built-in integer")
        basket = inner_basket
        if basket is not None and basket not in {1, 2, 3, 4}:
            raise SplitViolation("Run context inner basket must be 1-4 or None")
        if not isinstance(config_id, str):
            raise SplitViolation("Run config_id must be a string")
        normalized_config = re.sub(r"[^A-Za-z0-9_.-]+", "_", config_id).strip("_.")
        if not normalized_config:
            raise SplitViolation("Run config_id must be non-empty")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SplitViolation("Run seed must be a built-in integer")
        scope = "outer_refit" if basket is None else f"inner_{basket:02d}"
        checkpoint_dir = Path(checkpoint_root).resolve() / f"outer_{self.outer_fold:02d}" / scope / normalized_config / f"seed_{seed}"
        if checkpoint_dir.exists():
            raise SplitViolation("Checkpoint namespace already exists; fresh empty namespace required")
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        pretrained_path: str | None = None
        pretrained_hash: str | None = None
        if pretrained_checkpoint is not None:
            path = Path(pretrained_checkpoint).resolve()
            if pretrained_checkpoint_sha256 is None or not path.is_file():
                raise SplitViolation("Pretrained checkpoint requires an existing file and frozen hash")
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = _validate_sha256(pretrained_checkpoint_sha256, "pretrained checkpoint hash")
            if observed != expected:
                raise SplitViolation("Pretrained checkpoint hash mismatch")
            pretrained_path, pretrained_hash = str(path), observed
        elif pretrained_checkpoint_sha256 is not None:
            raise SplitViolation("Checkpoint hash supplied without an explicit checkpoint")
        run = RunContext(
            self.outer_fold,
            basket,
            normalized_config,
            seed,
            str(checkpoint_dir),
            False,
            pretrained_path,
            pretrained_hash,
            secrets.token_hex(32),
            self._token,
        )
        self._run_contexts[run.run_token] = run
        return run

    def validate_run_context(self, run: RunContext, basket: int | None, *, allow_consumed: bool = False) -> None:
        if (
            type(run) is not RunContext
            or run.contract_token != self._token
            or run.outer_fold != self.outer_fold
            or self._run_contexts.get(run.run_token) != run
            or run.inner_basket != basket
            or run.restore is not False
        ):
            raise SplitViolation("Run context does not match this fold/basket or requests restore")
        if not allow_consumed and run.run_token in self._consumed_runs:
            raise SplitViolation("Run context has already been consumed")
        directory = Path(run.checkpoint_dir)
        if not directory.is_dir():
            raise SplitViolation("Run checkpoint namespace is missing")
        if not allow_consumed and any(directory.iterdir()):
            raise SplitViolation("Run checkpoint namespace must be empty before fit")

    def register_estimator_run(self, run: RunContext, model: object) -> None:
        self._run_models[run.run_token] = model
        self._consumed_runs.add(run.run_token)

    def register_frame_run(self, run: RunContext, adapter: object, state: FreshTrainingState) -> None:
        self._run_frame_states[run.run_token] = (adapter, state)
        self._consumed_runs.add(run.run_token)

    def create_inner_evaluation_recorder(
        self,
        train: SafeBatch,
        validation: SafeBatch,
        *,
        basket: int,
        feature_columns: Sequence[str],
        target_column: str,
        metric_identity: str,
        run_context: RunContext,
        transform_sha256: str,
        model_config_sha256: str,
        checkpoint_sha256: str,
        audit: FitAuditTrail,
    ) -> "GuardedInnerEvaluationRecorder":
        self.validate_inner_pair(train, validation, basket)
        self.validate_run_context(run_context, int(basket), allow_consumed=True)
        if run_context.seed != 0:
            raise SplitViolation("Stopping evaluation uses frozen seed 0 only")
        features = validate_feature_schema(validation.frame, feature_columns, target_column)
        metric = str(metric_identity)
        if metric not in METRICS:
            raise SplitViolation(f"Unsupported stopping metric: {metric}")
        return GuardedInnerEvaluationRecorder(
            self,
            train,
            validation,
            features,
            str(target_column),
            metric,
            run_context,
            _validate_sha256(transform_sha256, "transform hash"),
            _validate_sha256(model_config_sha256, "model/config hash"),
            _validate_sha256(checkpoint_sha256, "checkpoint hash"),
            audit,
        )

    def _history_fingerprint(self, history: GuardedInnerHistory) -> str:
        payload = {
            "outer_fold": history.outer_fold,
            "inner_basket": history.inner_basket,
            "seed": history.seed,
            "config_id": history.config_id,
            "training_ids": list(history.training_ids),
            "validation_ids": list(history.validation_ids),
            "metric_identity": history.metric_identity,
            "feature_columns": list(history.feature_columns),
            "feature_schema_sha256": history.feature_schema_sha256,
            "target_identity": history.target_identity,
            "transform_sha256": history.transform_sha256,
            "model_config_sha256": history.model_config_sha256,
            "evaluator_identity": history.evaluator_identity,
            "execution_identity_sha256": history.execution_identity_sha256,
            "events": [event.__dict__ for event in history.events],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _issue_history(self, history: GuardedInnerHistory) -> None:
        self._issued_histories[history.history_token] = self._history_fingerprint(history)

    def select_stopping_epoch(
        self,
        histories: Sequence[GuardedInnerHistory],
        audit: FitAuditTrail,
    ) -> int:
        if len(histories) != 4 or not all(
            isinstance(history, GuardedInnerHistory) for history in histories
        ):
            raise SplitViolation("Stopping selection requires four contract-issued histories")
        if {history.inner_basket for history in histories} != {1, 2, 3, 4}:
            raise SplitViolation("Stopping selection requires exactly four guarded inner histories")
        if {history.seed for history in histories} != {0}:
            raise SplitViolation("Stopping histories must be generated with frozen seed 0")
        if len({history.config_id for history in histories}) != 1:
            raise SplitViolation("Stopping histories must use one frozen config identity")
        if len({history.metric_identity for history in histories}) != 1:
            raise SplitViolation("Stopping histories must use one frozen metric identity")
        if any(not history.events for history in histories):
            raise SplitViolation("Stopping histories must contain guarded evaluation events")
        if len({history.model_config_sha256 for history in histories}) != 1:
            raise SplitViolation("Stopping histories must use one frozen model/config hash")
        if len({history.evaluator_identity for history in histories}) != 1:
            raise SplitViolation("Stopping histories must use one evaluator identity")
        if len({history.feature_columns for history in histories}) != 1 or len(
            {history.feature_schema_sha256 for history in histories}
        ) != 1:
            raise SplitViolation("Stopping histories must use one ordered feature schema")
        if len({history.target_identity for history in histories}) != 1:
            raise SplitViolation("Stopping histories must use one target identity")
        if len({history.transform_sha256 for history in histories}) != 1:
            raise SplitViolation("Stopping histories must use one transform identity")
        best_epochs: list[int] = []
        for history in histories:
            expected_train, expected_validation = self.expected_inner_ids(history.inner_basket)
            expected_fingerprint = self._issued_histories.get(history.history_token)
            if (
                history.contract_token != self._token
                or history.outer_fold != self.outer_fold
                or frozenset(history.training_ids) != expected_train
                or frozenset(history.validation_ids) != expected_validation
                or not history.events
                or expected_fingerprint != self._history_fingerprint(history)
                or history.history_token in self._consumed_histories
            ):
                raise SplitViolation("Stopping history is unissued, altered, replayed or cross-contract")
            epochs = [event.epoch for event in history.events]
            if epochs != list(range(1, len(epochs) + 1)):
                raise SplitViolation("Guarded stopping history must contain complete consecutive epochs")
            if len({event.evaluator_identity for event in history.events}) != 1:
                raise SplitViolation("A stopping history may not change evaluator identity")
            if any(
                event.evaluator_identity != history.evaluator_identity
                or event.feature_schema_sha256 != history.feature_schema_sha256
                or event.target_identity != history.target_identity
                or event.transform_sha256 != history.transform_sha256
                or event.model_config_sha256 != history.model_config_sha256
                or event.execution_identity_sha256 != history.execution_identity_sha256
                for event in history.events
            ):
                raise SplitViolation("Stopping event identity differs from its bound history")
            best_epochs.append(min(history.events, key=lambda event: (event.loss, event.epoch)).epoch)
        selected = int(math.ceil(float(np.median(best_epochs))))
        self._consumed_histories.update(history.history_token for history in histories)
        audit.record_stopping(self.outer_fold, histories, selected)
        return selected


class SplitSafePreprocessor:
    """Train-only median imputation, variance filtering and standardization."""

    def __init__(
        self,
        contract: OuterFoldContract,
        audit: FitAuditTrail,
        *,
        max_missing_fraction: float = 0.50,
        min_variance: float = 0.0,
    ) -> None:
        self.contract = contract
        self.audit = audit
        self.max_missing_fraction = float(max_missing_fraction)
        self.min_variance = float(min_variance)
        transform_payload = {
            "implementation": _callable_identity(type(self)),
            "max_missing_fraction": self.max_missing_fraction,
            "min_variance": self.min_variance,
        }
        self.transform_sha256_ = hashlib.sha256(
            json.dumps(transform_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.is_fitted = False

    def fit(
        self,
        batch: SafeBatch,
        feature_columns: Sequence[str],
        *,
        target_column: str,
    ) -> "SplitSafePreprocessor":
        self.contract.validate_exact_fit_batch(batch)
        columns = validate_feature_schema(batch.frame, feature_columns, target_column)
        numeric = batch.frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        missing_fraction = numeric.isna().mean()
        candidate = [
            column for column in columns
            if float(missing_fraction[column]) <= self.max_missing_fraction and numeric[column].notna().any()
        ]
        if not candidate:
            raise SplitViolation("No features survive the training-only missingness filter")
        medians = numeric[candidate].median(axis=0)
        imputed = numeric[candidate].fillna(medians)
        variances = imputed.var(axis=0, ddof=0)
        kept = [column for column in candidate if float(variances[column]) > self.min_variance]
        if not kept:
            raise SplitViolation("No features survive the training-only variance filter")
        means = imputed[kept].mean(axis=0)
        scales = imputed[kept].std(axis=0, ddof=0).replace(0.0, 1.0)
        self.requested_features_ = columns
        self.kept_features_ = tuple(kept)
        self.medians_ = {column: float(medians[column]) for column in kept}
        self.means_ = {column: float(means[column]) for column in kept}
        self.scales_ = {column: float(scales[column]) for column in kept}
        self.fit_ids_sha256_ = canonical_id_hash(batch.ids)
        self.fit_id_count_ = len(batch.ids)
        self.fit_role_ = batch.role
        self.fit_inner_basket_ = batch.inner_basket
        self.target_column_ = str(target_column)
        stats_payload = {
            "features": self.kept_features_, "medians": self.medians_,
            "means": self.means_, "scales": self.scales_,
            "fit_ids_sha256": self.fit_ids_sha256_,
        }
        self.statistics_sha256_ = hashlib.sha256(
            json.dumps(stats_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.is_fitted = True
        self.audit.record_fit("preprocessor.fit", batch, columns, target_column)
        return self

    def transform(self, batch: SafeBatch) -> pd.DataFrame:
        if not self.is_fitted:
            raise SplitViolation("Preprocessor must be fitted before transform")
        self.contract.validate_transform_batch(batch)
        numeric = batch.frame.loc[:, self.kept_features_].apply(pd.to_numeric, errors="coerce")
        for column in self.kept_features_:
            numeric[column] = numeric[column].fillna(self.medians_[column])
            numeric[column] = (numeric[column] - self.means_[column]) / self.scales_[column]
        result = numeric.copy()
        result.insert(0, ID_COLUMN, list(batch.frame[ID_COLUMN].astype(str)))
        return result

    def fit_transform(
        self,
        batch: SafeBatch,
        feature_columns: Sequence[str],
        *,
        target_column: str,
    ) -> pd.DataFrame:
        return self.fit(batch, feature_columns, target_column=target_column).transform(batch)


class SplitSafeMixedPreprocessor(SplitSafePreprocessor):
    """Train-only scaling for continuous columns with finite binary passthrough."""

    def __init__(
        self,
        contract: OuterFoldContract,
        audit: FitAuditTrail,
        *,
        passthrough_columns: Sequence[str],
        max_missing_fraction: float = 0.50,
        min_variance: float = 0.0,
    ) -> None:
        super().__init__(
            contract,
            audit,
            max_missing_fraction=max_missing_fraction,
            min_variance=min_variance,
        )
        passthrough = tuple(map(str, passthrough_columns))
        if len(set(passthrough)) != len(passthrough):
            raise SplitViolation("Passthrough columns must be unique")
        self.passthrough_columns = passthrough
        transform_payload = {
            "implementation": _callable_identity(type(self)),
            "max_missing_fraction": self.max_missing_fraction,
            "min_variance": self.min_variance,
            "passthrough_columns": self.passthrough_columns,
        }
        self.transform_sha256_ = hashlib.sha256(
            json.dumps(transform_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def fit(
        self,
        batch: SafeBatch,
        feature_columns: Sequence[str],
        *,
        target_column: str,
    ) -> "SplitSafeMixedPreprocessor":
        self.contract.validate_exact_fit_batch(batch)
        columns = validate_feature_schema(batch.frame, feature_columns, target_column)
        passthrough = set(self.passthrough_columns)
        if not passthrough <= set(columns):
            raise SplitViolation("Passthrough columns must be part of the requested schema")
        numeric = batch.frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.loc[:, self.passthrough_columns].to_numpy(float)).all():
            raise SplitViolation("Binary passthrough features must be finite before preprocessing")
        continuous = tuple(column for column in columns if column not in passthrough)
        missing_fraction = numeric.loc[:, continuous].isna().mean(axis=0)
        candidates = [
            column
            for column in continuous
            if float(missing_fraction[column]) <= self.max_missing_fraction
        ]
        if not candidates:
            raise SplitViolation("No continuous features survive the training-only missingness filter")
        medians = numeric.loc[:, candidates].median(axis=0)
        imputed = numeric.loc[:, candidates].fillna(medians)
        variances = imputed.var(axis=0, ddof=0)
        kept_continuous = [
            column for column in candidates if float(variances[column]) > self.min_variance
        ]
        if not kept_continuous:
            raise SplitViolation("No continuous features survive the training-only variance filter")
        means = imputed.loc[:, kept_continuous].mean(axis=0)
        scales = imputed.loc[:, kept_continuous].std(axis=0, ddof=0).replace(0.0, 1.0)
        kept_set = passthrough | set(kept_continuous)
        self.requested_features_ = columns
        self.kept_features_ = tuple(column for column in columns if column in kept_set)
        self.continuous_features_ = tuple(kept_continuous)
        self.medians_ = {column: float(medians[column]) for column in kept_continuous}
        self.means_ = {column: float(means[column]) for column in kept_continuous}
        self.scales_ = {column: float(scales[column]) for column in kept_continuous}
        self.fit_ids_sha256_ = canonical_id_hash(batch.ids)
        self.fit_id_count_ = len(batch.ids)
        self.fit_role_ = batch.role
        self.fit_inner_basket_ = batch.inner_basket
        self.target_column_ = str(target_column)
        stats_payload = {
            "requested_features": self.requested_features_,
            "kept_features": self.kept_features_,
            "passthrough_columns": self.passthrough_columns,
            "continuous_features": self.continuous_features_,
            "medians": self.medians_,
            "means": self.means_,
            "scales": self.scales_,
            "fit_ids_sha256": self.fit_ids_sha256_,
        }
        self.statistics_sha256_ = hashlib.sha256(
            json.dumps(stats_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.is_fitted = True
        self.audit.record_fit("mixed_preprocessor.fit", batch, columns, target_column)
        return self

    def transform(self, batch: SafeBatch) -> pd.DataFrame:
        if not self.is_fitted:
            raise SplitViolation("Preprocessor must be fitted before transform")
        self.contract.validate_transform_batch(batch)
        numeric = batch.frame.loc[:, self.kept_features_].apply(pd.to_numeric, errors="coerce")
        result = numeric.copy()
        for column in self.continuous_features_:
            result[column] = result[column].fillna(self.medians_[column])
            result[column] = (result[column] - self.means_[column]) / self.scales_[column]
        if not np.isfinite(result.to_numpy(float)).all():
            raise SplitViolation("Mixed preprocessor produced non-finite features")
        result.insert(0, ID_COLUMN, list(batch.frame[ID_COLUMN].astype(str)))
        return result


class SplitSafeFitExecutor:
    """Typed outer/inner estimator and locked-DMPNN adapter entry points."""

    def __init__(self, contract: OuterFoldContract, audit: FitAuditTrail) -> None:
        self.contract = contract
        self.audit = audit
        self.__sealed_outer_frame_states: dict[str, _SealedOuterFrameState] = {}

    def _project(
        self,
        batch: SafeBatch,
        feature_columns: Sequence[str],
        target_column: str,
        *,
        preprocessor: SplitSafePreprocessor | None = None,
        preprocessor_fit_batch: SafeBatch | None = None,
    ) -> tuple[tuple[str, ...], pd.DataFrame, pd.Series, pd.DataFrame]:
        features = validate_feature_schema(batch.frame, feature_columns, target_column)
        index = pd.Index(batch.frame[ID_COLUMN].astype(str), name=ID_COLUMN)
        if preprocessor is None:
            X = batch.frame.loc[:, features].copy()
        else:
            if preprocessor_fit_batch is None:
                raise SplitViolation("Preprocessor fit batch is required")
            self.contract.validate_exact_fit_batch(preprocessor_fit_batch)
            if (
                preprocessor.contract is not self.contract
                or not preprocessor.is_fitted
                or preprocessor.requested_features_ != features
                or preprocessor.target_column_ != str(target_column)
                or preprocessor.fit_ids_sha256_ != canonical_id_hash(preprocessor_fit_batch.ids)
                or preprocessor.fit_role_ != preprocessor_fit_batch.role
                or preprocessor.fit_inner_basket_ != preprocessor_fit_batch.inner_basket
            ):
                raise SplitViolation(
                    "Preprocessor was not fitted on the exact applicable training batch/schema"
                )
            transformed = preprocessor.transform(batch)
            X = transformed.loc[:, preprocessor.kept_features_].copy()
        y = batch.frame[target_column].copy()
        X.index, y.index = index, index
        projected = batch.frame.loc[:, [ID_COLUMN, *features, target_column]].copy()
        return features, X, y, projected

    def _weights(
        self, batch: SafeBatch, weights: SafeTrainingWeights | None
    ) -> pd.Series | None:
        if weights is None:
            return None
        self.contract.validate_training_weights(weights, batch)
        return pd.Series(weights.values, index=pd.Index(batch.ids, name=ID_COLUMN), name="sample_weight")

    def fit_preprocessor(
        self,
        preprocessor: SplitSafePreprocessor,
        batch: SafeBatch,
        feature_columns: Sequence[str],
        *,
        target_column: str,
    ) -> SplitSafePreprocessor:
        if preprocessor.contract is not self.contract:
            raise SplitViolation("Preprocessor belongs to a different contract")
        return preprocessor.fit(batch, feature_columns, target_column=target_column)

    def _transform_hashes(
        self, preprocessor: SplitSafePreprocessor | None
    ) -> tuple[str, str]:
        if preprocessor is None:
            return "0" * 64, "0" * 64
        return (
            _validate_sha256(preprocessor.transform_sha256_, "transform definition hash"),
            _validate_sha256(preprocessor.statistics_sha256_, "fitted transform hash"),
        )

    def _new_estimator(self, factory: Callable[[], object], run: RunContext) -> object:
        _validate_typed_callable(factory, self.contract.outer_test_ids, "model_factory")
        model = factory()
        _claim_fresh_object(model, "model")
        _scan_boundary_state(model, self.contract.outer_test_ids, "fresh_model")
        if hasattr(model, "get_params"):
            params = model.get_params(deep=True)
            for key, value in params.items():
                _scan_object_graph(value, path=f"model.params.{key}", forbidden_ids=self.contract.outer_test_ids)
        self.contract.register_estimator_run(run, model)
        return model

    def fit_outer_estimator(
        self,
        model_factory: Callable[[], object],
        outer_train: SafeBatch,
        feature_columns: Sequence[str],
        target_column: str,
        *,
        run_context: RunContext,
        options: SafeFitOptions | Mapping[str, object] | None = None,
        sample_weight: SafeTrainingWeights | None = None,
        fixed_iterations: int | None = None,
        preprocessor: SplitSafePreprocessor | None = None,
    ) -> object:
        self.contract.validate_exact_fit_batch(outer_train)
        if outer_train.role != "outer_train":
            raise SplitViolation("Outer estimator path accepts only outer_train")
        self.contract.validate_run_context(run_context, None)
        safe_options = SafeFitOptions.coerce(options)
        features, X, y, _ = self._project(
            outer_train,
            feature_columns,
            target_column,
            preprocessor=preprocessor,
            preprocessor_fit_batch=outer_train,
        )
        weights = self._weights(outer_train, sample_weight)
        if fixed_iterations is not None:
            _plain_positive_int(fixed_iterations, "fixed_iterations")
        model = self._new_estimator(model_factory, run_context)
        if fixed_iterations is not None:
            if not hasattr(model, "get_params") or int(
                model.get_params(deep=True).get("n_estimators", -1)
            ) != fixed_iterations:
                raise SplitViolation(
                    "Outer estimator factory must encode the authenticated fixed n_estimators"
                )
        kwargs = safe_options.estimator_kwargs()
        if weights is not None:
            kwargs["sample_weight"] = weights
        self.audit.record_fit(
            "outer_estimator.fit", outer_train, features, target_column, run_context, sample_weight
        )
        model.fit(X, y, **kwargs)
        return model

    def predict_outer_estimator(
        self,
        estimator: object,
        outer_test: SafeBatch,
        *,
        feature_columns: Sequence[str],
        target_column: str,
        run_context: RunContext,
        preprocessor: SplitSafePreprocessor | None = None,
    ) -> GuardedPredictionResult:
        self.contract.validate_exact_outer_test_batch(outer_test)
        self.contract.validate_run_context(run_context, None, allow_consumed=True)
        if self.contract._run_models.get(run_context.run_token) is not estimator:
            raise SplitViolation("Estimator was not minted for this guarded outer run")
        features = validate_feature_schema(outer_test.frame, feature_columns, target_column)
        index = pd.Index(outer_test.frame[ID_COLUMN].astype(str), name=ID_COLUMN)
        if preprocessor is None:
            X = outer_test.frame.loc[:, features].copy()
        else:
            if (
                preprocessor.contract is not self.contract
                or not preprocessor.is_fitted
                or preprocessor.requested_features_ != features
                or preprocessor.target_column_ != str(target_column)
                or preprocessor.fit_ids_sha256_ != canonical_id_hash(self.contract.outer_train_ids)
                or preprocessor.fit_role_ != "outer_train"
                or preprocessor.fit_inner_basket_ is not None
            ):
                raise SplitViolation(
                    "Outer prediction preprocessor was not fitted on exact outer training/schema"
                )
            transformed = preprocessor.transform(outer_test)
            X = transformed.loc[:, preprocessor.kept_features_].copy()
        X.index = index
        array = np.asarray(estimator.predict(X), dtype=float)
        if array.ndim != 1 or len(array) != len(outer_test.ids) or not np.isfinite(array).all():
            raise SplitViolation("Outer predictions must be finite and align exactly to outer-test IDs")
        result = GuardedPredictionResult(
            self.contract.outer_fold,
            None,
            "outer_test",
            outer_test.ids,
            tuple(map(float, array)),
            canonical_id_hash(outer_test.ids),
            canonical_array_hash(array),
            _callable_identity(type(estimator)),
            feature_schema_hash(features),
            str(target_column),
            hashlib.sha256(
                str(Path(run_context.checkpoint_dir).resolve()).encode("utf-8")
            ).hexdigest(),
            self.contract._token,
        )
        self.audit.record_prediction(result)
        self.contract._run_models.pop(run_context.run_token, None)
        return result

    def fit_inner_estimator(
        self,
        model_factory: Callable[[], object],
        inner_train: SafeBatch,
        inner_validation: SafeBatch,
        *,
        basket: int,
        feature_columns: Sequence[str],
        target_column: str,
        run_context: RunContext,
        recorder: "GuardedInnerEvaluationRecorder | None" = None,
        options: SafeFitOptions | Mapping[str, object] | None = None,
        sample_weight: SafeTrainingWeights | None = None,
        use_internal_eval_set: bool = False,
        preprocessor: SplitSafePreprocessor | None = None,
    ) -> object:
        self.contract.validate_inner_pair(inner_train, inner_validation, basket)
        self.contract.validate_run_context(run_context, int(basket))
        if recorder is None:
            raise SplitViolation("Every inner estimator fit requires its bound evaluation recorder")
        safe_options = SafeFitOptions.coerce(options)
        features, X_train, y_train, _ = self._project(
            inner_train,
            feature_columns,
            target_column,
            preprocessor=preprocessor,
            preprocessor_fit_batch=inner_train,
        )
        _, X_validation, y_validation, _ = self._project(
            inner_validation,
            feature_columns,
            target_column,
            preprocessor=preprocessor,
            preprocessor_fit_batch=inner_train,
        )
        weights = self._weights(inner_train, sample_weight)
        transform_sha256, fitted_transform_sha256 = self._transform_hashes(preprocessor)
        recorder.bind_execution(
            train=inner_train,
            validation=inner_validation,
            basket=int(basket),
            feature_columns=features,
            target_column=target_column,
            run_context=run_context,
            evaluator=model_factory,
            evaluator_is_callable=True,
            transform_sha256=transform_sha256,
            fitted_transform_sha256=fitted_transform_sha256,
            model_config_sha256=_validate_sha256(
                getattr(model_factory, "model_config_sha256", ""),
                "estimator factory model/config hash",
            ),
        )
        recorder._bind_estimator_validation_projection(X_validation)
        model = self._new_estimator(model_factory, run_context)
        kwargs = safe_options.estimator_kwargs()
        if weights is not None:
            kwargs["sample_weight"] = weights
        use_eval = _plain_bool(use_internal_eval_set, "use_internal_eval_set")
        if use_eval:
            kwargs["eval_set"] = [(X_validation, y_validation)]
        self.audit.record_fit(
            "inner_estimator.fit", inner_train, features, target_column, run_context, sample_weight
        )
        model.fit(X_train, y_train, **kwargs)
        return model

    def _new_frame_state(self, adapter: object, run: RunContext) -> FreshTrainingState:
        _scan_boundary_state(adapter, self.contract.outer_test_ids, "frame_adapter")
        required = (
            "create_fresh_state",
            "fit_inner",
            "fit_outer",
            "predict_validation",
            "predict_outer",
        )
        if any(not callable(getattr(adapter, name, None)) for name in required):
            raise SplitViolation(f"Frame adapter must implement {required}")
        state = adapter.create_fresh_state(run_context=run)
        if not isinstance(state, FreshTrainingState):
            raise SplitViolation("Frame adapter must return FreshTrainingState")
        _claim_fresh_object(state.model, "model")
        _claim_fresh_object(state.optimizer, "optimizer")
        if state.scheduler is not None:
            _claim_fresh_object(state.scheduler, "scheduler")
        self.contract.register_frame_run(run, adapter, state)
        return state

    def _frame_adapter_hashes(
        self,
        adapter: object,
    ) -> tuple[str, str, str]:
        transform = _validate_sha256(
            getattr(adapter, "transform_sha256", ""), "frame adapter transform hash"
        )
        fitted_transform = _validate_sha256(
            getattr(adapter, "fitted_transform_sha256", ""),
            "frame adapter fitted transform hash",
        )
        model_config = _validate_sha256(
            getattr(adapter, "model_config_sha256", ""),
            "frame adapter model/config hash",
        )
        return transform, fitted_transform, model_config

    def _frame_prediction_runtime_identity(
        self,
        adapter: object,
        state: FreshTrainingState,
        run_context: RunContext,
        identity: OuterFrameFitIdentity,
        *,
        prediction_frame_sha256: str,
    ) -> tuple[dict[str, object], Callable[..., object], str]:
        """Recompute every sealed adapter, predictor, run and model identity."""
        run_snapshot = _sealed_run_context(run_context)
        run_namespace = hashlib.sha256(
            str(Path(run_snapshot.checkpoint_dir).resolve()).encode("utf-8")
        ).hexdigest()
        transform, fitted_transform, model_config = self._frame_adapter_hashes(adapter)
        predictor = getattr(adapter, "predict_outer", None)
        predictor_identity = _validate_typed_callable(
            predictor, self.contract.outer_test_ids, "frame_adapter.predict_outer"
        )
        runtime_identity: dict[str, object] = {
            "run_namespace": run_namespace,
            "run_context": _run_context_content_hash(run_snapshot),
            "records": identity.authoritative_records_sha256,
            "prediction_frame": prediction_frame_sha256,
            "features": tuple(identity.feature_columns),
            "feature_schema": feature_schema_hash(identity.feature_columns),
            "target": identity.target_identity,
            "transform": transform,
            "fitted_transform": fitted_transform,
            "model_config": model_config,
            "checkpoint": _checkpoint_namespace_hash(run_snapshot.checkpoint_dir),
            "training_state": _training_state_content_hash(state),
            "adapter_state": _frame_adapter_content_hash(
                adapter, label="frame_adapter"
            ),
            "adapter": _callable_identity(type(adapter)),
            "predictor": predictor_identity,
            "predictor_state": _callable_content_hash(
                predictor, label="frame_adapter.predict_outer"
            ),
            "model": _callable_identity(type(state.model)),
            "optimizer": _callable_identity(type(state.optimizer)),
            "scheduler": None
            if state.scheduler is None
            else _callable_identity(type(state.scheduler)),
        }
        return runtime_identity, predictor, run_namespace

    @staticmethod
    def _expected_frame_prediction_runtime_identity(
        identity: OuterFrameFitIdentity,
    ) -> dict[str, object]:
        return {
            "run_namespace": identity.run_namespace_sha256,
            "run_context": identity.run_context_sha256,
            "records": identity.authoritative_records_sha256,
            "prediction_frame": identity.outer_test_frame_sha256,
            "features": identity.feature_columns,
            "feature_schema": identity.feature_schema_sha256,
            "target": identity.target_identity,
            "transform": identity.transform_sha256,
            "fitted_transform": identity.fitted_transform_sha256,
            "model_config": identity.model_config_sha256,
            "checkpoint": identity.checkpoint_sha256,
            "training_state": identity.training_state_sha256,
            "adapter_state": identity.adapter_state_sha256,
            "adapter": identity.adapter_identity,
            "predictor": identity.predictor_identity,
            "predictor_state": identity.predictor_state_sha256,
            "model": identity.model_identity,
            "optimizer": identity.optimizer_identity,
            "scheduler": identity.scheduler_identity,
        }

    def _reject_outer_frame_prediction(
        self,
        identity: OuterFrameFitIdentity,
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> "NoReturn":
        error = SplitViolation(message)
        self.audit.record_outer_frame_prediction_failure(identity, error)
        if cause is None:
            raise error
        raise error from cause

    def fit_inner_frame(
        self,
        adapter: object,
        inner_train: SafeBatch,
        inner_validation: SafeBatch,
        *,
        basket: int,
        feature_columns: Sequence[str],
        target_column: str,
        run_context: RunContext,
        recorder: "GuardedInnerEvaluationRecorder",
        options: SafeFitOptions | Mapping[str, object] | None = None,
        sample_weight: SafeTrainingWeights | None = None,
        maximum_epochs: int,
    ) -> FreshTrainingState:
        self.contract.validate_inner_pair(inner_train, inner_validation, basket)
        self.contract.validate_run_context(run_context, int(basket))
        if recorder.contract is not self.contract or recorder.run_context != run_context:
            raise SplitViolation("Inner trainer recorder does not match its contract/run")
        safe_options = SafeFitOptions.coerce(options)
        features, _, _, train_frame = self._project(inner_train, feature_columns, target_column)
        _, _, _, validation_frame = self._project(inner_validation, feature_columns, target_column)
        weights = self._weights(inner_train, sample_weight)
        epochs = _plain_positive_int(maximum_epochs, "maximum_epochs")
        _scan_boundary_state(adapter, self.contract.outer_test_ids, "frame_adapter")
        recorder.bind_execution(
            train=inner_train,
            validation=inner_validation,
            basket=int(basket),
            feature_columns=features,
            target_column=target_column,
            run_context=run_context,
            evaluator=adapter,
            evaluator_is_callable=False,
            transform_sha256=_validate_sha256(
                getattr(adapter, "transform_sha256", ""), "frame adapter transform hash"
            ),
            fitted_transform_sha256=_validate_sha256(
                getattr(adapter, "fitted_transform_sha256", ""),
                "frame adapter fitted transform hash",
            ),
            model_config_sha256=_validate_sha256(
                getattr(adapter, "model_config_sha256", ""),
                "frame adapter model/config hash",
            ),
        )
        state = self._new_frame_state(adapter, run_context)
        self.audit.record_fit(
            "inner_frame_trainer.fit", inner_train, features, target_column, run_context, sample_weight
        )
        adapter.fit_inner(
            state=state,
            train_frame=train_frame,
            validation_frame=validation_frame,
            target_column=target_column,
            feature_columns=features,
            sample_weight=weights,
            maximum_epochs=epochs,
            options=safe_options,
            run_context=run_context,
            recorder=recorder,
        )
        return state

    def fit_outer_frame(
        self,
        adapter: object,
        outer_train: SafeBatch,
        *,
        feature_columns: Sequence[str],
        target_column: str,
        run_context: RunContext,
        fixed_epoch: int,
        options: SafeFitOptions | Mapping[str, object] | None = None,
        sample_weight: SafeTrainingWeights | None = None,
    ) -> SealedOuterFrameFitHandle:
        self.contract.validate_exact_fit_batch(outer_train)
        if outer_train.role != "outer_train":
            raise SplitViolation("Outer frame path accepts only outer_train")
        issuance_token = self.contract.validate_authoritative_outer_training_batch(outer_train)
        self.contract.validate_run_context(run_context, None)
        safe_options = SafeFitOptions.coerce(options)
        features, _, _, train_frame = self._project(outer_train, feature_columns, target_column)
        (
            _outer_prediction_frame,
            authoritative_records_sha256,
            outer_test_frame_sha256,
        ) = self.contract.authoritative_outer_prediction_frame(
            features, target_column, issuance_token
        )
        weights = self._weights(outer_train, sample_weight)
        epoch = _plain_positive_int(fixed_epoch, "fixed_epoch")
        before_hashes = self._frame_adapter_hashes(adapter)
        state = self._new_frame_state(adapter, run_context)
        self.audit.record_fit(
            "outer_frame_trainer.fit", outer_train, features, target_column, run_context, sample_weight
        )
        adapter.fit_outer(
            state=state,
            train_frame=train_frame,
            target_column=target_column,
            feature_columns=features,
            sample_weight=weights,
            fixed_epoch=epoch,
            options=safe_options,
            run_context=run_context,
        )
        transform, fitted_transform, model_config = self._frame_adapter_hashes(adapter)
        if (transform, fitted_transform, model_config) != before_hashes:
            raise SplitViolation("Frame adapter identity changed during the guarded outer fit")
        sealed_target_ids = {id(state), id(state.model), id(state.optimizer)}
        if state.scheduler is not None:
            sealed_target_ids.add(id(state.scheduler))
        if _object_reaches_any(adapter, sealed_target_ids):
            raise SplitViolation("Frame adapter retained an alias to sealed fitted state")
        predictor = getattr(adapter, "predict_outer", None)
        predictor_identity = _validate_typed_callable(
            predictor, self.contract.outer_test_ids, "frame_adapter.predict_outer"
        )
        predictor_state_sha256 = _callable_content_hash(
            predictor, label="frame_adapter.predict_outer"
        )
        training_state_sha256 = _training_state_content_hash(state)
        checkpoint_sha256 = _checkpoint_namespace_hash(run_context.checkpoint_dir)
        adapter_state_sha256 = _frame_adapter_content_hash(
            adapter, label="frame_adapter"
        )
        run_namespace_sha256 = hashlib.sha256(
            str(Path(run_context.checkpoint_dir).resolve()).encode("utf-8")
        ).hexdigest()
        sealed_run_context = _sealed_run_context(run_context)
        run_context_sha256 = _run_context_content_hash(sealed_run_context)
        identity = OuterFrameFitIdentity(
            self.contract._token,
            self.contract.outer_fold,
            run_context.run_token,
            run_namespace_sha256,
            run_context_sha256,
            run_context.config_id,
            run_context.seed,
            epoch,
            canonical_id_hash(self.contract.outer_train_ids),
            canonical_id_hash(self.contract.outer_test_ids),
            authoritative_records_sha256,
            outer_test_frame_sha256,
            features,
            feature_schema_hash(features),
            str(target_column),
            transform,
            fitted_transform,
            model_config,
            checkpoint_sha256,
            training_state_sha256,
            adapter_state_sha256,
            _callable_identity(type(adapter)),
            predictor_identity,
            predictor_state_sha256,
            _callable_identity(type(state.model)),
            _callable_identity(type(state.optimizer)),
            None if state.scheduler is None else _callable_identity(type(state.scheduler)),
        )
        handle = SealedOuterFrameFitHandle(
            self.contract.outer_fold, identity.sha256(), secrets.token_hex(32)
        )
        self.__sealed_outer_frame_states[handle.handle_token] = _SealedOuterFrameState(
            handle,
            adapter,
            state,
            sealed_run_context,
            identity,
            _outer_prediction_frame.copy(deep=True),
        )
        # The contract no longer owns or exposes the prediction model.  Its
        # temporary fit registry is cleared as soon as executor sealing succeeds.
        self.contract._run_frame_states.pop(run_context.run_token, None)
        self.contract._authoritative_record_issuances.pop(issuance_token, None)
        return handle

    def predict_outer_frame(
        self,
        handle: SealedOuterFrameFitHandle,
    ) -> GuardedOuterFramePredictionResult:
        """Predict once with only an executor-minted sealed capability."""
        if not isinstance(handle, SealedOuterFrameFitHandle):
            raise SplitViolation("Outer frame prediction requires a sealed completed-fit handle")
        sealed = self.__sealed_outer_frame_states.get(handle.handle_token)
        if sealed is None or sealed.handle != handle:
            raise SplitViolation("Outer frame fit handle is forged, foreign or already consumed")
        # Consume before any adapter invocation. All validation failures and partial
        # predictions require a new fit, so mutable state can never be replayed.
        self.__sealed_outer_frame_states.pop(handle.handle_token, None)
        adapter, state, run_context, identity = (
            sealed.adapter,
            sealed.state,
            sealed.run_context,
            sealed.identity,
        )
        try:
            self.contract.validate_run_context(run_context, None, allow_consumed=True)
        except Exception as error:
            self.audit.record_outer_frame_prediction_failure(identity, error)
            raise
        if (
            handle.outer_fold != self.contract.outer_fold
            or handle.outer_fit_identity_sha256 != identity.sha256()
            or identity.contract_token != self.contract._token
            or identity.run_token != run_context.run_token
            or identity.config_id != run_context.config_id
            or identity.seed != run_context.seed
        ):
            self._reject_outer_frame_prediction(
                identity, "Sealed outer fit identity does not match contract/run"
            )
        prediction_frame = sealed.outer_prediction_frame
        prediction_frame_hash = canonical_frame_hash(prediction_frame)
        try:
            current_identity, predictor, run_namespace = (
                self._frame_prediction_runtime_identity(
                    adapter,
                    state,
                    run_context,
                    identity,
                    prediction_frame_sha256=prediction_frame_hash,
                )
            )
        except Exception as error:
            self.audit.record_outer_frame_prediction_failure(identity, error)
            raise
        expected_identity = self._expected_frame_prediction_runtime_identity(identity)
        if current_identity != expected_identity:
            self._reject_outer_frame_prediction(
                identity, "Outer prediction content/state differs from the sealed fit"
            )
        expected_ids = tuple(sorted(self.contract.outer_test_ids))
        if (
            tuple(prediction_frame.columns) != (ID_COLUMN, *identity.feature_columns)
            or identity.target_identity in prediction_frame.columns
            or tuple(prediction_frame[ID_COLUMN].astype(str)) != expected_ids
        ):
            self._reject_outer_frame_prediction(
                identity, "Outer prediction projection is not exact and outcome-free"
            )
        # The adapter receives a detached copy, never the authoritative registry
        # frame. A post-call hash also rejects adapter-side value/schema mutation.
        prediction_frame = _detached_read_only_frame(prediction_frame)
        disposable_run_context = _sealed_run_context(run_context)

        def post_call_runtime_is_exact() -> bool:
            """Recheck all sealed state, including both predictor-frame copies."""
            try:
                detached_frame_sha256 = canonical_frame_hash(prediction_frame)
                sealed_frame_sha256 = canonical_frame_hash(
                    sealed.outer_prediction_frame
                )
                post_identity, _post_predictor, post_run_namespace = (
                    self._frame_prediction_runtime_identity(
                        adapter,
                        state,
                        run_context,
                        identity,
                        prediction_frame_sha256=sealed_frame_sha256,
                    )
                )
                disposable_run_sha256 = _run_context_content_hash(
                    disposable_run_context
                )
            except Exception:
                return False
            return bool(
                post_identity == expected_identity
                and post_run_namespace == run_namespace
                and detached_frame_sha256 == identity.outer_test_frame_sha256
                and sealed_frame_sha256 == identity.outer_test_frame_sha256
                and disposable_run_sha256 == identity.run_context_sha256
                and disposable_run_context == run_context
            )

        try:
            adapter_output = predictor(
                state=state,
                outer_test_frame=prediction_frame,
                feature_columns=identity.feature_columns,
                run_context=disposable_run_context,
            )
        except Exception as error:
            message = "Outer prediction adapter failed inside the sealed call"
            if not post_call_runtime_is_exact():
                message = (
                    "Outer prediction adapter failed and mutated or invalidated "
                    "sealed adapter/predictor/config/run/model/checkpoint state"
                )
            self._reject_outer_frame_prediction(
                identity,
                message,
                cause=error,
            )
        if not post_call_runtime_is_exact():
            self._reject_outer_frame_prediction(
                identity,
                "Prediction mutated sealed adapter/predictor/config/run/model/checkpoint state",
            )
        try:
            if type(adapter_output) is not FramePredictionOutput:
                raise SplitViolation(
                    "Outer frame adapter must return the canonical "
                    "FramePredictionOutput type"
                )
            predictions = adapter_output.predictions
            telemetry = _freeze_prediction_telemetry(adapter_output.telemetry)
            if type(predictions) is not pd.Series:
                raise SplitViolation(
                    "Outer frame predictions must be a canonical ID-indexed pandas Series"
                )
            prediction_ids = tuple(predictions.index.astype(str))
            array = pd.to_numeric(predictions, errors="coerce").to_numpy(float)
            if (
                predictions.index.name != ID_COLUMN
                or prediction_ids != expected_ids
                or predictions.index.has_duplicates
                or array.ndim != 1
                or len(array) != len(expected_ids)
                or not np.isfinite(array).all()
            ):
                raise SplitViolation(
                    "Outer frame predictions must be finite and align in exact "
                    "outer-test ID order"
                )
            # Freeze the only accepted output into built-in immutable scalars
            # before the final scientific-state check and success audit.
            prediction_values = tuple(map(float, array))
            prediction_sha256 = canonical_array_hash(array)
        except Exception as error:
            self._reject_outer_frame_prediction(
                identity,
                "Outer frame adapter returned an invalid prediction/telemetry payload",
                cause=error,
            )
        # Output coercion is executable for third-party pandas/object dtypes.  A
        # second post-call verification closes mutation after the first check.
        if not post_call_runtime_is_exact():
            self._reject_outer_frame_prediction(
                identity,
                "Prediction output processing mutated sealed adapter/predictor/config/"
                "run/model/checkpoint state",
            )
        result = GuardedOuterFramePredictionResult(
            self.contract.outer_fold,
            "outer_test",
            expected_ids,
            prediction_values,
            canonical_id_hash(expected_ids),
            prediction_sha256,
            identity.sha256(),
            run_namespace,
            identity.feature_schema_sha256,
            identity.target_identity,
            identity.model_config_sha256,
            identity.transform_sha256,
            identity.fitted_transform_sha256,
            identity.checkpoint_sha256,
            identity.adapter_identity,
            identity.model_identity,
            telemetry,
            self.contract._token,
        )
        self.audit.record_outer_frame_prediction(result, identity)
        return result


class GuardedInnerEvaluationRecorder:
    """Mint immutable losses only from exact inner-validation predictions."""

    def __init__(
        self,
        contract: OuterFoldContract,
        train: SafeBatch,
        validation: SafeBatch,
        features: tuple[str, ...],
        target_column: str,
        metric_identity: str,
        run_context: RunContext,
        transform_sha256: str,
        model_config_sha256: str,
        checkpoint_sha256: str,
        audit: FitAuditTrail,
    ) -> None:
        self.contract = contract
        self.train = train
        self.validation = validation
        self.features = features
        self.target_column = target_column
        self.metric_identity = metric_identity
        self.run_context = run_context
        self.transform_sha256 = transform_sha256
        self.model_config_sha256 = model_config_sha256
        self.checkpoint_sha256 = checkpoint_sha256
        self.audit = audit
        self._events: list[InnerEvaluationEvent] = []
        self._finalized = False
        self._execution: InnerExecutionIdentity | None = None
        self._estimator_X: pd.DataFrame | None = None
        index = pd.Index(validation.frame[ID_COLUMN].astype(str), name=ID_COLUMN)
        self._X = validation.frame.loc[:, features].copy()
        self._X.index = index
        self._y = pd.to_numeric(validation.frame[target_column], errors="coerce").to_numpy(float)
        if not np.isfinite(self._y).all():
            raise SplitViolation("Inner-validation targets must be finite")
        self._target_hash = canonical_array_hash(self._y)

    def bind_execution(
        self,
        *,
        train: SafeBatch,
        validation: SafeBatch,
        basket: int,
        feature_columns: Sequence[str],
        target_column: str,
        run_context: RunContext,
        evaluator: object,
        evaluator_is_callable: bool,
        transform_sha256: str,
        fitted_transform_sha256: str,
        model_config_sha256: str,
    ) -> InnerExecutionIdentity:
        if self._execution is not None or self._events or self._finalized:
            raise SplitViolation("Inner evaluation recorder may bind exactly one execution")
        self.contract.validate_inner_pair(train, validation, int(basket))
        self.contract.validate_run_context(run_context, int(basket), allow_consumed=True)
        features = validate_feature_schema(validation.frame, feature_columns, target_column)
        if evaluator_is_callable:
            evaluator_identity = _validate_typed_callable(
                evaluator, self.contract.outer_test_ids, "inner_evaluator"
            )
        else:
            _scan_boundary_state(evaluator, self.contract.outer_test_ids, "inner_evaluator")
            evaluator_identity = _callable_identity(type(evaluator))
        transform = _validate_sha256(transform_sha256, "bound transform hash")
        fitted_transform = _validate_sha256(
            fitted_transform_sha256, "bound fitted transform hash"
        )
        model_config = _validate_sha256(model_config_sha256, "bound model/config hash")
        if (
            train != self.train
            or validation != self.validation
            or run_context != self.run_context
            or int(basket) != int(self.validation.inner_basket)
            or features != self.features
            or str(target_column) != self.target_column
            or transform != self.transform_sha256
            or model_config != self.model_config_sha256
        ):
            raise SplitViolation(
                "Inner fit identity does not match recorder train/validation/schema/target/transform/config"
            )
        execution = InnerExecutionIdentity(
            self.contract._token,
            self.contract.outer_fold,
            int(basket),
            run_context.run_token,
            hashlib.sha256(str(Path(run_context.checkpoint_dir).resolve()).encode("utf-8")).hexdigest(),
            canonical_id_hash(train.ids),
            canonical_id_hash(validation.ids),
            features,
            feature_schema_hash(features),
            str(target_column),
            transform,
            fitted_transform,
            model_config,
            evaluator_identity,
        )
        self._execution = execution
        return execution

    def _bind_estimator_validation_projection(self, X: pd.DataFrame) -> None:
        if self._execution is None or self._estimator_X is not None or self._events:
            raise SplitViolation("Estimator validation projection may bind once after execution identity")
        if not isinstance(X, pd.DataFrame) or tuple(X.index.astype(str)) != self.validation.ids:
            raise SplitViolation("Estimator validation projection must align to exact validation IDs")
        numeric = X.apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise SplitViolation("Estimator validation projection must be finite")
        self._estimator_X = numeric.copy()

    def _record(self, epoch: int, predictions: object) -> InnerEvaluationEvent:
        if self._finalized:
            raise SplitViolation("Evaluation recorder is already finalized")
        if self._execution is None:
            raise SplitViolation("Evaluation recorder is not bound to an inner fit")
        epoch = int(epoch)
        if epoch != len(self._events) + 1:
            raise SplitViolation("Evaluation epochs must be complete and consecutive from 1")
        array = np.asarray(predictions, dtype=float)
        if array.ndim != 1 or len(array) != len(self.validation.ids) or not np.isfinite(array).all():
            raise SplitViolation("Predictions must be finite and align exactly to validation IDs")
        if self.metric_identity == "mean_absolute_error":
            loss = float(np.mean(np.abs(array - self._y)))
        else:
            loss = float(np.mean((array - self._y) ** 2))
        event = InnerEvaluationEvent(
            epoch,
            loss,
            self.metric_identity,
            canonical_id_hash(self.validation.ids),
            canonical_array_hash(array),
            self._target_hash,
            self._execution.evaluator_identity,
            feature_schema_hash(self.features),
            self.transform_sha256,
            self.model_config_sha256,
            self.checkpoint_sha256,
            self.target_column,
            self._execution.fitted_transform_sha256,
            self._execution.sha256(),
        )
        self._events.append(event)
        self.audit.record_evaluation(event, self.run_context)
        return event

    def evaluate_estimator_epoch(self, epoch: int, estimator: object) -> InnerEvaluationEvent:
        event, _ = self.evaluate_estimator_predictions(epoch, estimator)
        return event

    def evaluate_estimator_predictions(
        self, epoch: int, estimator: object
    ) -> tuple[InnerEvaluationEvent, GuardedPredictionResult]:
        if self.contract._run_models.get(self.run_context.run_token) is not estimator:
            raise SplitViolation("Estimator was not minted for this guarded inner run")
        if self._estimator_X is None:
            raise SplitViolation("Estimator validation projection was not bound by the fit executor")
        predictions = estimator.predict(self._estimator_X.copy())
        event = self._record(epoch, predictions)
        if self._execution is None:
            raise SplitViolation("Evaluation recorder lacks a bound execution")
        array = np.asarray(predictions, dtype=float)
        result = GuardedPredictionResult(
            self.contract.outer_fold,
            int(self.validation.inner_basket),
            "inner_validation",
            self.validation.ids,
            tuple(map(float, array)),
            canonical_id_hash(self.validation.ids),
            canonical_array_hash(array),
            self._execution.evaluator_identity,
            feature_schema_hash(self.features),
            self.target_column,
            self._execution.run_namespace_sha256,
            self.contract._token,
        )
        self.audit.record_prediction(result)
        self.contract._run_models.pop(self.run_context.run_token, None)
        return event, result

    def evaluate_frame_epoch(
        self, epoch: int, adapter: object, state: FreshTrainingState
    ) -> InnerEvaluationEvent:
        registered = self.contract._run_frame_states.get(self.run_context.run_token)
        if registered is None or registered[0] is not adapter or registered[1] is not state:
            raise SplitViolation("Frame adapter/state was not minted for this guarded inner run")
        predictions = adapter.predict_validation(
            state=state,
            validation_frame=self.validation.frame.loc[
                :, [ID_COLUMN, *self.features]
            ].copy(),
            feature_columns=self.features,
            run_context=self.run_context,
        )
        return self._record(epoch, predictions)

    def finalize(self) -> GuardedInnerHistory:
        if self._finalized or not self._events or self._execution is None:
            raise SplitViolation("Recorder must contain events and may be finalized only once")
        self._finalized = True
        history = GuardedInnerHistory(
            self.contract.outer_fold,
            int(self.validation.inner_basket),
            self.run_context.seed,
            self.run_context.config_id,
            self.train.ids,
            self.validation.ids,
            self.metric_identity,
            self.features,
            feature_schema_hash(self.features),
            self.target_column,
            self.transform_sha256,
            self.model_config_sha256,
            self._execution.evaluator_identity,
            self._execution.sha256(),
            tuple(self._events),
            secrets.token_hex(32),
            self.contract._token,
        )
        self.contract._issue_history(history)
        return history


def contracts_from_manifests(
    records: pd.DataFrame,
    outer_records: pd.DataFrame,
    inner_manifest: pd.DataFrame,
) -> dict[int, OuterFoldContract]:
    required_records = {ID_COLUMN, "sealed_block_id"}
    if not required_records <= set(records.columns):
        raise SplitViolation("Record mapping lacks curated_id or sealed_block_id")
    if records[ID_COLUMN].astype(str).duplicated().any():
        raise SplitViolation("Record mapping contains duplicate curated IDs")
    record_ids = set(records[ID_COLUMN].astype(str))
    if set(outer_records[ID_COLUMN].astype(str)) != record_ids:
        raise SplitViolation("Outer-record assignments do not cover the frozen records")
    block_by_id = records.assign(**{ID_COLUMN: records[ID_COLUMN].astype(str)}).set_index(ID_COLUMN)[
        "sealed_block_id"
    ].astype(str)
    contracts: dict[int, OuterFoldContract] = {}
    for outer_fold in sorted(outer_records["outer_fold"].astype(int).unique()):
        test_ids = set(
            outer_records.loc[
                outer_records["outer_fold"].astype(int) == outer_fold, ID_COLUMN
            ].astype(str)
        )
        train_ids = record_ids - test_ids
        basket_rows = inner_manifest.loc[
            inner_manifest["outer_fold"].astype(int) == outer_fold,
            ["sealed_block_id", "inner_basket"],
        ]
        if len(basket_rows) != 17 or basket_rows["sealed_block_id"].duplicated().any():
            raise SplitViolation(f"Outer fold {outer_fold} lacks 17 unique inner blocks")
        basket_by_block = {
            str(row.sealed_block_id): int(row.inner_basket)
            for row in basket_rows.itertuples(index=False)
        }
        inner_by_id = {record_id: basket_by_block[block_by_id[record_id]] for record_id in train_ids}
        contracts[outer_fold] = OuterFoldContract(outer_fold, train_ids, test_ids, inner_by_id)
    if set(contracts) != set(range(1, 19)):
        raise SplitViolation("Expected contracts for outer folds 1-18")
    return contracts


def contract_manifest(contracts: dict[int, OuterFoldContract]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outer_fold, contract in sorted(contracts.items()):
        for basket in range(1, 5):
            training_ids, validation_ids = contract.expected_inner_ids(basket)
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_basket": basket,
                    "n_fit_ids": len(training_ids),
                    "fit_ids_sha256": canonical_id_hash(training_ids),
                    "n_inner_validation_ids": len(validation_ids),
                    "inner_validation_ids_sha256": canonical_id_hash(validation_ids),
                    "n_outer_test_ids": len(contract.outer_test_ids),
                    "outer_test_ids_sha256": canonical_id_hash(contract.outer_test_ids),
                }
            )
    return pd.DataFrame(rows)


def fit_boundary_policy() -> dict[str, object]:
    """Machine-readable R0 policy emitted by the deterministic builder."""
    return {
        "version": "r0-split-safe-repair7",
        "r1_authorized": False,
        "safe_scalar_option_keys": sorted(SAFE_SCALAR_OPTION_KEYS),
        "outcome_name_fragments": list(OUTCOME_NAME_FRAGMENTS),
        "reserved_predictor_names": sorted(RESERVED_PREDICTOR_NAMES),
        "typed_fit_paths": [
            "fit_outer_estimator",
            "fit_inner_estimator",
            "fit_outer_frame",
            "fit_inner_frame",
        ],
        "typed_prediction_paths": [
            "predict_outer_estimator",
            "predict_outer_frame",
        ],
        "generic_fit_kwargs": False,
        "caller_supplied_loss_curves": False,
        "split_local_materialized_frames": True,
        "parent_dataset_reachability_forbidden": True,
        "fresh_model_optimizer_scheduler_per_run": True,
        "fresh_checkpoint_namespace_per_run": True,
        "restore_latest_forbidden": True,
        "outer_frame_validation_payload": False,
        "outer_frame_prediction_outcome_free_projection": True,
        "outer_frame_prediction_id_indexed": True,
        "outer_frame_prediction_single_use": True,
        "outer_frame_prediction_caller_supplied_frame": False,
        "outer_frame_prediction_authoritative_content_hash": [
            "values",
            "dtypes",
            "column_order",
            "index",
        ],
        "outer_frame_fit_returns_mutable_state": False,
        "outer_frame_executor_private_registry": True,
        "outer_frame_prediction_handle_only": True,
        "outer_frame_model_content_digest_internal": True,
        "outer_frame_checkpoint_content_digest_internal": True,
        "outer_frame_prediction_pre_post_state_verification": True,
        "outer_frame_run_context_tuple_immutable": True,
        "outer_frame_run_context_registry_snapshot": True,
        "outer_frame_run_context_disposable_adapter_copy": True,
        "outer_frame_run_context_pre_post_content_digest": True,
        "outer_frame_adapter_pre_post_deep_content_digest": True,
        "outer_frame_adapter_mro_slots_helpers_content_digest": True,
        "outer_frame_predictor_defaults_closures_attributes_digest": True,
        "outer_frame_prediction_telemetry_detached": True,
        "outer_frame_prediction_telemetry_scalar_only": True,
        "outer_frame_prediction_final_post_output_state_verification": True,
        "outer_frame_prediction_failure_audited": True,
        "caller_controlled_state_digest": False,
        "outer_frame_prediction_identity_fields": [
            "contract",
            "outer_fold",
            "run_namespace",
            "run_context_content",
            "config",
            "seed",
            "fixed_epoch",
            "outer_train_ids",
            "outer_test_ids",
            "authoritative_record_values_dtypes_columns_index",
            "outer_test_predictor_values_dtypes_columns_index",
            "ordered_feature_schema",
            "target",
            "transform",
            "fitted_transform",
            "model_config",
            "checkpoint",
            "training_state_content",
            "adapter_state_content",
            "adapter",
            "predictor",
            "predictor_defaults_closures_attributes",
            "model",
            "optimizer",
            "scheduler",
        ],
        "inner_validation_issued_by_same_contract": True,
        "stopping_history_single_use": True,
        "typed_callable_recursive_carriers": [
            "defaults",
            "kwdefaults",
            "partial_args_keywords",
            "bound_instance_state",
            "closures",
            "callable_objects",
        ],
        "inner_fit_recorder_identity_binding_required": True,
        "inner_fit_identity_fields": [
            "contract",
            "outer_fold",
            "inner_basket",
            "run_namespace",
            "training_ids",
            "validation_ids",
            "ordered_feature_schema",
            "target",
            "transform",
            "model_config",
            "evaluator",
        ],
        "stopping_cross_history_identity_required": True,
        "pinned_dmpnn_no_learning_smoke_required": True,
    }
