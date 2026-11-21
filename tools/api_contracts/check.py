"""Validate the declared Python contract and enforce it on an installed package."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import inspect
import json
import math
import re
import sys
import tomllib
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python" / "crc_framework"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KINDS = {"class", "function", "enum", "protocol"}
_PARAMETER_KINDS = {"positional_only", "positional_or_keyword", "keyword_only"}
_BUILTIN_TYPES = {
    "None",
    "bool",
    "int",
    "float",
    "str",
    "bytes",
    "tuple",
    "list",
    "dict",
    "type",
    "Sequence",
    "Mapping",
    "numpy",
}


class ContractError(ValueError):
    """The declared or installed interface contradicts the release contract."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def load_contracts(
    package_root: Path = PACKAGE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = package_root / "contracts" / "api"
    return _load(directory / "public-api.json"), _load(directory / "release.json")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _keys(
    value: Mapping[str, Any], allowed: set[str], required: set[str], path: str
) -> None:
    _require(
        set(value) <= allowed, f"{path}: unknown fields {sorted(set(value) - allowed)}"
    )
    _require(
        required <= set(value),
        f"{path}: missing fields {sorted(required - set(value))}",
    )


def _annotation(value: Any, names: set[str], path: str) -> None:
    _require(
        isinstance(value, str) and bool(value), f"{path}: expected an explicit type"
    )
    try:
        expression = ast.parse(value, mode="eval")
    except SyntaxError as error:
        raise ContractError(f"{path}: invalid type expression {value!r}") from error
    referenced = {
        node.id for node in ast.walk(expression) if isinstance(node, ast.Name)
    }
    _require("Any" not in referenced, f"{path}: Any cannot hide a public contract")
    _require(referenced <= names, f"{path}: unknown types {sorted(referenced - names)}")
    allowed_nodes = (
        ast.Expression,
        ast.Name,
        ast.Load,
        ast.Subscript,
        ast.Tuple,
        ast.Constant,
        ast.BinOp,
        ast.BitOr,
        ast.Attribute,
    )
    _require(
        all(isinstance(node, allowed_nodes) for node in ast.walk(expression)),
        f"{path}: unsupported type expression",
    )
    for node in ast.walk(expression):
        if isinstance(node, ast.Constant):
            _require(
                node.value is None or node.value is Ellipsis,
                f"{path}: type expressions cannot contain values",
            )
        if isinstance(node, ast.Attribute):
            _require(
                ast.unparse(node).startswith("numpy."),
                f"{path}: canonical public type names must not use module aliases",
            )


def _default(
    value: Any, annotation: str, exports: Mapping[str, Any], path: str
) -> None:
    if isinstance(value, dict):
        if set(value) == {"enum"}:
            reference = value["enum"]
            _require(
                isinstance(reference, str) and reference.count(".") == 1,
                f"{path}: invalid enum default",
            )
            owner, member = reference.split(".")
            _require(
                owner in exports and exports[owner]["kind"] == "enum",
                f"{path}: unknown enum {owner}",
            )
            _require(
                member in exports[owner]["members"],
                f"{path}: unknown member {reference}",
            )
            _require(
                owner
                in {
                    n.id
                    for n in ast.walk(ast.parse(annotation, mode="eval"))
                    if isinstance(n, ast.Name)
                },
                f"{path}: enum default contradicts annotation",
            )
            return
        if set(value) == {"tuple"}:
            items = value["tuple"]
            _require(
                isinstance(items, list),
                f"{path}: tuple default must contain a list",
            )
            if annotation.startswith("tuple[") and not items:
                return
            _require(
                annotation.startswith("Sequence[") and annotation.endswith("]"),
                f"{path}: tuple default contradicts annotation",
            )
            item_type = annotation[len("Sequence[") : -1]
            for index, item in enumerate(items):
                _default(item, item_type, exports, f"{path}/{index}")
            return
        raise ContractError(f"{path}: unsupported default record")
    _require(
        value is None or isinstance(value, (str, bool, int, float)),
        f"{path}: default must be an immutable literal or enum",
    )
    if value is None:
        _require("None" in annotation, f"{path}: None default needs an optional type")
    elif isinstance(value, bool):
        _require(
            annotation == "bool", f"{path}: boolean default contradicts annotation"
        )
    elif isinstance(value, (int, float)):
        _require(math.isfinite(value), f"{path}: non-finite default")
        _require(
            annotation in {"int", "float"},
            f"{path}: numeric default contradicts annotation",
        )
        _require(
            annotation != "int" or isinstance(value, int),
            f"{path}: integer default required",
        )
    elif isinstance(value, str):
        _require(
            annotation == "str", f"{path}: closed choices must use an enum default"
        )


def _call(value: Any, names: set[str], exports: Mapping[str, Any], path: str) -> None:
    _require(isinstance(value, dict), f"{path}: expected a callable contract")
    _keys(value, {"parameters", "returns", "binding"}, {"parameters", "returns"}, path)
    if "binding" in value:
        _require(
            value["binding"] in {"instance", "class", "static"},
            f"{path}: invalid method binding",
        )
    _annotation(value["returns"], names, f"{path}/returns")
    _require(
        isinstance(value["parameters"], list), f"{path}/parameters: expected a list"
    )
    seen: set[str] = set()
    keyword_only = False
    optional_positional = False
    for parameter in value["parameters"]:
        _require(isinstance(parameter, dict), f"{path}: expected a parameter object")
        _keys(
            parameter,
            {"name", "kind", "annotation", "default"},
            {"name", "kind", "annotation"},
            path,
        )
        name = parameter["name"]
        _require(
            isinstance(name, str) and bool(_IDENTIFIER.fullmatch(name)),
            f"{path}: invalid parameter name",
        )
        _require(
            name not in {"self", "cls"} and name not in seen,
            f"{path}: repeated or implicit parameter {name}",
        )
        seen.add(name)
        _require(
            parameter["kind"] in _PARAMETER_KINDS,
            f"{path}/{name}: unsupported parameter kind",
        )
        _annotation(parameter["annotation"], names, f"{path}/{name}")
        if parameter["kind"] == "keyword_only":
            keyword_only = True
        else:
            _require(
                not keyword_only,
                f"{path}/{name}: positional parameter follows keyword-only parameter",
            )
            _require(
                not optional_positional or "default" in parameter,
                f"{path}/{name}: required positional follows a default",
            )
            optional_positional = "default" in parameter
        if "default" in parameter:
            _default(
                parameter["default"],
                parameter["annotation"],
                exports,
                f"{path}/{name}/default",
            )


def validate_contract(contract: Mapping[str, Any], release: Mapping[str, Any]) -> None:
    """Check declarations without importing an implementation or retaining old APIs."""
    _keys(
        contract,
        {
            "schema_version",
            "canonical_module",
            "package",
            "type_aliases",
            "exports",
            "semantics",
            "removals",
        },
        {
            "schema_version",
            "canonical_module",
            "package",
            "type_aliases",
            "exports",
            "semantics",
            "removals",
        },
        "/",
    )
    _require(
        contract["schema_version"] == "1.0",
        "/schema_version: unsupported contract format",
    )
    _require(
        contract["canonical_module"] == "crc_framework",
        "/canonical_module: canonical import must be crc_framework",
    )
    _require(contract["package"] == "crc-framework", "/package: expected crc-framework")
    exports = contract["exports"]
    aliases = contract["type_aliases"]
    _require(
        isinstance(exports, dict) and bool(exports),
        "/exports: expected a nonempty object",
    )
    _require(isinstance(aliases, dict), "/type_aliases: expected an object")
    _require(
        not set(exports) & set(aliases),
        "/type_aliases: annotation aliases are not extra public exports",
    )
    names = _BUILTIN_TYPES | set(exports) | set(aliases)
    for name, annotation in aliases.items():
        _require(
            bool(_IDENTIFIER.fullmatch(name)),
            f"/type_aliases/{name}: invalid identifier",
        )
        _annotation(annotation, names, f"/type_aliases/{name}")
    for name, definition in exports.items():
        path = f"/exports/{name}"
        _require(bool(_IDENTIFIER.fullmatch(name)), f"{path}: invalid public name")
        _require(isinstance(definition, dict), f"{path}: expected an object")
        _require(definition.get("kind") in _KINDS, f"{path}: unsupported symbol kind")
        kind = definition["kind"]
        if kind == "enum":
            _keys(definition, {"kind", "members"}, {"kind", "members"}, path)
            members = definition["members"]
            _require(
                isinstance(members, dict) and bool(members),
                f"{path}/members: expected nonempty object",
            )
            _require(
                all(
                    isinstance(k, str)
                    and bool(_IDENTIFIER.fullmatch(k))
                    and k.isupper()
                    for k in members
                ),
                f"{path}: enum names must be uppercase identifiers",
            )
            _require(
                all(isinstance(v, str) and bool(v) for v in members.values()),
                f"{path}: enum values must be nonempty strings",
            )
            _require(
                len(set(members.values())) == len(members),
                f"{path}: enum aliases are forbidden",
            )
        elif kind == "function":
            _keys(
                definition,
                {"kind", "parameters", "returns"},
                {"kind", "parameters", "returns"},
                path,
            )
            _call(
                {k: v for k, v in definition.items() if k != "kind"},
                names,
                exports,
                path,
            )
        else:
            _keys(
                definition,
                {"kind", "constructor", "methods", "properties", "generic_parameters"},
                {"kind"},
                path,
            )
            parameters = definition.get("generic_parameters", [])
            _require(
                isinstance(parameters, list)
                and all(
                    isinstance(parameter, str) and _IDENTIFIER.fullmatch(parameter)
                    for parameter in parameters
                )
                and len(parameters) == len(set(parameters))
                and not set(parameters) & names,
                f"{path}: invalid type parameters",
            )
            class_names = names | set(parameters)
            if "constructor" in definition:
                _call(
                    definition["constructor"],
                    class_names,
                    exports,
                    f"{path}/constructor",
                )
                _require(
                    definition["constructor"]["returns"] == name,
                    f"{path}: constructor must return its own type",
                )
            methods = definition.get("methods", {})
            properties = definition.get("properties", {})
            _require(
                isinstance(methods, dict) and isinstance(properties, dict),
                f"{path}: methods and properties must be objects",
            )
            _require(
                not set(methods) & set(properties), f"{path}: method/property collision"
            )
            for method, signature in methods.items():
                _require(
                    bool(_IDENTIFIER.fullmatch(method)), f"{path}: invalid method name"
                )
                _call(signature, class_names, exports, f"{path}/methods/{method}")
            for field, annotation in properties.items():
                _require(
                    bool(_IDENTIFIER.fullmatch(field)), f"{path}: invalid property name"
                )
                _annotation(annotation, class_names, f"{path}/properties/{field}")
    _require(
        isinstance(contract["semantics"], dict)
        and bool(contract["semantics"])
        and all(isinstance(k, str) and bool(k) for k in contract["semantics"]),
        "/semantics: expected named semantic requirements",
    )
    removals = contract["removals"]
    _require(isinstance(removals, dict), "/removals: expected an object")
    _keys(
        removals,
        {
            "symbols",
            "modules",
            "methods",
            "schema_fields",
            "wire_formats",
        },
        {
            "symbols",
            "modules",
            "methods",
            "schema_fields",
            "wire_formats",
        },
        "/removals",
    )
    for category in ("symbols", "modules", "schema_fields", "wire_formats"):
        entries = removals[category]
        _require(
            isinstance(entries, list)
            and all(isinstance(v, str) and bool(v) for v in entries),
            f"/removals/{category}: expected string list",
        )
        _require(
            len(entries) == len(set(entries)),
            f"/removals/{category}: duplicate removal",
        )
    _require(
        not set(removals["symbols"]) & {f"crc_framework.{name}" for name in exports},
        "/removals: a public symbol cannot also be removed",
    )
    for owner, methods in removals["methods"].items():
        _require(
            owner == "*" or owner in exports,
            f"/removals/methods: unknown owner {owner}",
        )
        _require(
            isinstance(methods, list)
            and all(isinstance(name, str) for name in methods),
            f"/removals/methods/{owner}: expected method names",
        )
        _require(
            owner == "*" or not set(methods) & set(exports[owner].get("methods", {})),
            f"/removals/methods/{owner}: method is both declared and removed",
        )
    validate_release(release)


def validate_release(release: Mapping[str, Any]) -> None:
    _keys(
        release,
        {"schema_version", "packages", "formats"},
        {"schema_version", "packages", "formats"},
        "/release",
    )
    _require(
        release["schema_version"] == "1.0",
        "/release/schema_version: unsupported format",
    )
    packages = {"framework", "sdk", "framework_requirement"}
    formats = {
        "dataset",
        "evaluation",
        "collection",
        "taxonomy",
        "numerical",
        "impact",
        "evidence",
    }
    _keys(release["packages"], packages, packages, "/release/packages")
    _keys(release["formats"], formats, formats, "/release/formats")
    _require(
        all(
            isinstance(version, str) and version
            for version in release["packages"].values()
        ),
        "/release/packages: expected package version declarations",
    )
    _require(
        all(
            isinstance(version, str) and re.fullmatch(r"[0-9]+\.[0-9]+", version)
            for version in release["formats"].values()
        ),
        "/release/formats: expected major.minor format versions",
    )


def validate_source_versions(root: Path, release: Mapping[str, Any]) -> None:
    cargo = tomllib.loads((root / "Cargo.toml").read_text())
    _require(
        cargo["workspace"]["package"]["version"] == release["packages"]["framework"],
        "Cargo.toml version contradicts release.json",
    )
    for relative in ("Cargo.toml", "crates/core/Cargo.toml", "crates/wasm/Cargo.toml"):
        package = tomllib.loads((root / relative).read_text())["package"]
        _require(
            package["version"] == {"workspace": True},
            f"{relative}: crate must inherit the workspace version",
        )
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    _require(
        "version" in pyproject["project"].get("dynamic", []),
        "Python version must be derived from Cargo",
    )
    javascript = _load(root / "js" / "package.json")
    _require(
        javascript.get("private") is True and "version" not in javascript,
        "JavaScript development manifest must remain private and derive its public version from Cargo",
    )
    resources = root / "python" / "crc_framework" / "contracts"
    for name, directory in (
        ("dataset", "dataset"),
        ("taxonomy", "taxonomy"),
        ("numerical", "numerical"),
        ("impact", "scientific"),
    ):
        version = release["formats"][name]
        schemas = {
            path.name
            for path in (resources / directory).glob(f"{name}-[0-9]*.schema.json")
        }
        _require(
            schemas == {f"{name}-{version}.schema.json"},
            f"{name} schema does not match release ledger version {version}: {sorted(schemas)}",
        )


def _render_annotation(value: Any) -> str:
    if value is inspect.Signature.empty:
        return "<missing>"
    if value is None or value is type(None):
        return "None"
    if isinstance(value, str):
        rendered = value
    elif getattr(value, "__module__", None) == "builtins":
        rendered = getattr(value, "__qualname__", str(value))
    else:
        rendered = str(value)
        if rendered.startswith("<class '"):
            rendered = getattr(value, "__qualname__", rendered)
    rendered = (
        rendered.replace("typing.", "")
        .replace("collections.abc.", "")
        .replace("crc_framework.", "")
    )
    try:
        return ast.unparse(ast.parse(rendered, mode="eval"))
    except SyntaxError:
        return rendered


def _runtime_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return {"enum": f"{type(value).__name__}.{value.name}"}
    if isinstance(value, tuple):
        return {"tuple": [_runtime_default(item) for item in value]}
    return value


def _check_call(
    value: Any,
    signature: Mapping[str, Any],
    path: str,
    errors: list[str],
    *,
    constructor: bool = False,
) -> None:
    try:
        actual = inspect.signature(value)
    except (TypeError, ValueError) as error:
        errors.append(f"{path}: cannot inspect call signature: {error}")
        return
    parameters = [
        p for p in actual.parameters.values() if p.name not in {"self", "cls"}
    ]
    expected = signature["parameters"]
    if [p.name for p in parameters] != [p["name"] for p in expected]:
        errors.append(f"{path}: parameter names differ from the declared signature")
        return
    for parameter, required in zip(parameters, expected):
        where = f"{path}/{parameter.name}"
        if parameter.kind.name.lower() != required["kind"]:
            errors.append(f"{where}: parameter kind differs")
        has_default = parameter.default is not inspect.Parameter.empty
        if has_default != ("default" in required) or (
            has_default and _runtime_default(parameter.default) != required["default"]
        ):
            errors.append(f"{where}: default differs")
        if _render_annotation(parameter.annotation) != _render_annotation(
            required["annotation"]
        ):
            errors.append(
                f"{where}: annotation differs; expected {required['annotation']}"
            )
    if not constructor and _render_annotation(
        actual.return_annotation
    ) != _render_annotation(signature["returns"]):
        errors.append(
            f"{path}: return annotation differs; expected {signature['returns']}"
        )


def runtime_errors(contract: Mapping[str, Any], module: ModuleType) -> list[str]:
    """Return runtime surface mismatches, including hidden removed attributes."""
    errors: list[str] = []
    exports = contract["exports"]
    actual = getattr(module, "__all__", None)
    if (
        not isinstance(actual, (list, tuple))
        or len(actual) != len(set(actual))
        or set(actual) != set(exports)
    ):
        errors.append(
            "crc_framework.__all__ differs from the exact public export contract"
        )
    undeclared = sorted(
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and name not in exports
        and not isinstance(value, ModuleType)
    )
    if undeclared:
        errors.append(f"crc_framework has undeclared public attributes: {undeclared}")
    for name, definition in exports.items():
        if not hasattr(module, name):
            errors.append(f"crc_framework.{name}: missing public symbol")
            continue
        value = getattr(module, name)
        kind = definition["kind"]
        if kind == "enum":
            if not inspect.isclass(value) or not issubclass(value, Enum):
                errors.append(f"crc_framework.{name}: expected an enum")
            elif {
                key: member.value for key, member in value.__members__.items()
            } != definition["members"]:
                errors.append(f"crc_framework.{name}: enum members or aliases differ")
        elif kind == "function":
            _check_call(value, definition, f"crc_framework.{name}", errors)
        else:
            if not inspect.isclass(value):
                errors.append(f"crc_framework.{name}: expected a class or protocol")
                continue
            if kind == "protocol" and not getattr(value, "_is_protocol", False):
                errors.append(f"crc_framework.{name}: expected a Protocol")
            if "constructor" in definition:
                _check_call(
                    value,
                    definition["constructor"],
                    f"crc_framework.{name}",
                    errors,
                    constructor=True,
                )
            for method, signature in definition.get("methods", {}).items():
                member = getattr(value, method, None)
                if not callable(member):
                    errors.append(f"crc_framework.{name}.{method}: missing callable")
                else:
                    _check_call(
                        member, signature, f"crc_framework.{name}.{method}", errors
                    )
                    declared_binding = signature.get("binding")
                    descriptor = inspect.getattr_static(value, method)
                    actual_binding = (
                        "class"
                        if isinstance(descriptor, classmethod)
                        else "static"
                        if isinstance(descriptor, staticmethod)
                        else "instance"
                    )
                    if (
                        declared_binding is not None
                        and declared_binding != actual_binding
                    ):
                        errors.append(
                            f"crc_framework.{name}.{method}: method binding differs"
                        )
            actual_parameters = [
                parameter.__name__ for parameter in getattr(value, "__parameters__", ())
            ]
            if actual_parameters != definition.get("generic_parameters", []):
                errors.append(f"crc_framework.{name}: type parameters differ")
            annotations = {
                field: annotation
                for base in reversed(value.__mro__)
                for field, annotation in vars(base).get("__annotations__", {}).items()
            }
            for field, expected_annotation in definition.get("properties", {}).items():
                member = inspect.getattr_static(value, field, None)
                if isinstance(member, property) and member.fget is not None:
                    annotation = inspect.signature(member.fget).return_annotation
                else:
                    annotation = annotations.get(field, inspect.Signature.empty)
                if _render_annotation(annotation) != _render_annotation(
                    expected_annotation
                ):
                    errors.append(
                        f"crc_framework.{name}.{field}: missing or incorrect property annotation"
                    )
            declared = set(definition.get("methods", {})) | set(
                definition.get("properties", {})
            )
            inherited = (
                set(dir(BaseException)) if issubclass(value, BaseException) else set()
            )
            undeclared_members = sorted(
                member
                for member in dir(value)
                if not member.startswith("_")
                and member not in declared
                and member not in inherited
            )
            if undeclared_members:
                errors.append(
                    f"crc_framework.{name}: undeclared public members {undeclared_members}"
                )
    for qualified in contract["removals"]["symbols"]:
        name = qualified.rsplit(".", 1)[-1]
        if hasattr(module, name):
            errors.append(f"crc_framework.{name}: removed name remains importable")
    for owner, methods in contract["removals"]["methods"].items():
        owners = list(exports) if owner == "*" else [owner]
        for target in owners:
            value = getattr(module, target, None)
            if not inspect.isclass(value):
                continue
            for method in methods:
                if method == "__call__":
                    present = any(
                        method in base.__dict__
                        for base in value.__mro__
                        if base is not type
                    )
                else:
                    present = hasattr(value, method)
                if present:
                    errors.append(
                        f"crc_framework.{target}.{method}: removed operation remains available"
                    )
    return errors


def validate_installed_contract(
    installed: Path, declared: Path = PACKAGE
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract, release = load_contracts(installed)
    validate_contract(contract, release)
    source_contract, source_release = load_contracts(declared)
    _require(
        contract == source_contract,
        "installed API declaration differs from the release checkout",
    )
    _require(
        release == source_release,
        "installed release ledger differs from the release checkout",
    )
    return contract, release


def packaged_errors(installed: Path, declared: Path = PACKAGE) -> list[str]:
    def inventory(package: Path) -> set[Path]:
        return {
            path.relative_to(package)
            for path in package.rglob("*")
            if path.is_file()
            and (path.suffix in {".py", ".pyi", ".json"} or path.name == "py.typed")
        }

    expected = inventory(declared)
    actual = inventory(installed)
    errors = [
        f"{path}: missing from installed package" for path in sorted(expected - actual)
    ]
    errors.extend(
        f"{path}: unexpected file in installed package"
        for path in sorted(actual - expected)
    )
    for path in sorted(expected & actual):
        source = (declared / path).read_text(encoding="utf-8")
        if (installed / path).read_text(encoding="utf-8") != source:
            errors.append(f"{path}: installed resource differs from release checkout")
    return errors


def check_runtime(package_root: Path | None = None) -> None:
    if package_root is not None:
        sys.path.insert(0, str(package_root.resolve()))
    module = importlib.import_module("crc_framework")
    if module.__file__ is None:
        raise ContractError("installed crc_framework has no package location")
    installed = Path(module.__file__).resolve().parent
    if package_root is not None:
        _require(
            installed.is_relative_to(package_root.resolve()),
            f"imported {installed}, expected a package inside {package_root.resolve()}",
        )
    contract, release = validate_installed_contract(installed)
    errors = packaged_errors(installed)
    errors.extend(runtime_errors(contract, module))
    for name in contract["removals"]["modules"]:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                errors.append(
                    f"{name}: import failed because of missing {error.name}, not because the removed module is absent"
                )
        else:
            errors.append(f"{name}: removed import path remains available")
    version = release["packages"]["framework"]
    from crc_framework._version import python_version

    distribution_version = python_version(version)
    if importlib.metadata.version("crc-framework") != distribution_version:
        errors.append(f"installed distribution version must be {distribution_version}")
    native = importlib.import_module("crc_framework._core")
    if getattr(native, "__version__", None) != version:
        errors.append(f"native extension version must be {version}")
    if errors:
        raise ContractError(
            "Python release contract is not implemented:\n"
            + "\n".join(f"- {item}" for item in errors)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="enforce the installed package's own release contract",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        help="site-packages or unpacked wheel directory containing crc_framework",
    )
    args = parser.parse_args()
    try:
        if args.runtime:
            check_runtime(args.package_root)
        else:
            package = (
                args.package_root / "crc_framework" if args.package_root else PACKAGE
            )
            contract, release = load_contracts(package)
            validate_contract(contract, release)
            if args.package_root is None:
                validate_source_versions(ROOT, release)
    except (
        ContractError,
        OSError,
        ValueError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        "Python release contract verified"
        if args.runtime
        else "Python contract declarations and release versions verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
