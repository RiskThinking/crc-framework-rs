"""The declared release surface is coherent and cannot be bypassed by aliases."""

from __future__ import annotations

import importlib.util
import json
import shutil
from copy import deepcopy
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "framework_api_contract_check", ROOT / "tools" / "api_contracts" / "check.py"
)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_declared_contract_and_source_versions_are_coherent() -> None:
    contract, release = CHECK.load_contracts()
    CHECK.validate_contract(contract, release)
    CHECK.validate_source_versions(ROOT, release)
    assert contract["canonical_module"] == "crc_framework"
    assert "SampleFit" in contract["exports"]
    assert "crc_framework.EmpiricalDistribution" in contract["removals"]["symbols"]
    assert (
        contract["exports"]["Impact"]["methods"]["pushforward"]["returns"]
        == "Distribution"
    )


@pytest.mark.parametrize(
    ("owner", "field", "annotation"),
    [
        ("Distribution", "support", "Any"),
        ("FitRecord", "weights", "Sequence[Any]"),
        ("Measurement", "fields", "Mapping[str, UndeclaredMeaning]"),
    ],
)
def test_public_types_cannot_be_opaque_or_undeclared(
    owner: str, field: str, annotation: str
) -> None:
    contract, release = CHECK.load_contracts()
    contract["exports"][owner]["properties"][field] = annotation
    with pytest.raises(CHECK.ContractError, match="Any|unknown types"):
        CHECK.validate_contract(contract, release)


def test_enum_aliases_are_rejected() -> None:
    contract, release = CHECK.load_contracts()
    contract["exports"]["Tail"]["members"]["EXCEEDANCE"] = "upper"
    with pytest.raises(CHECK.ContractError, match="aliases are forbidden"):
        CHECK.validate_contract(contract, release)


def test_removed_names_cannot_be_reintroduced_as_public_exports() -> None:
    contract, release = CHECK.load_contracts()
    contract["exports"]["EmpiricalDistribution"] = {"kind": "class"}
    with pytest.raises(CHECK.ContractError, match="also be removed"):
        CHECK.validate_contract(contract, release)


def test_enum_default_must_match_declared_type() -> None:
    contract, release = CHECK.load_contracts()
    parameters = contract["exports"]["SampleFit"]["constructor"]["parameters"]
    estimator = next(
        parameter for parameter in parameters if parameter["name"] == "estimator"
    )
    estimator["default"] = {"enum": "Tail.UPPER"}
    with pytest.raises(CHECK.ContractError, match="contradicts annotation"):
        CHECK.validate_contract(contract, release)


def test_superseded_draft_versions_cannot_force_another_major_bump() -> None:
    contract, release = CHECK.load_contracts()
    release["formats"]["evaluation"] = "3.0"
    with pytest.raises(CHECK.ContractError, match="single release ledger"):
        CHECK.validate_contract(contract, release)


def _fixture() -> tuple[dict[str, Any], ModuleType]:
    class Choice(str, Enum):
        FIRST = "first"

    class Example:
        def __init__(self, *, enabled: bool = False) -> None:
            self.enabled = enabled

        def quantile(self, probabilities: float) -> float:
            return probabilities

        @property
        def support(self) -> tuple[float, float]:
            return (0.0, 1.0)

    module = ModuleType("crc_framework")
    module.__dict__.update(
        __all__=["Choice", "Example"], Choice=Choice, Example=Example
    )
    contract = {
        "exports": {
            "Choice": {"kind": "enum", "members": {"FIRST": "first"}},
            "Example": {
                "kind": "class",
                "constructor": {
                    "parameters": [
                        {
                            "name": "enabled",
                            "kind": "keyword_only",
                            "annotation": "bool",
                            "default": False,
                        }
                    ],
                    "returns": "Example",
                },
                "methods": {
                    "quantile": {
                        "parameters": [
                            {
                                "name": "probabilities",
                                "kind": "positional_or_keyword",
                                "annotation": "float",
                            }
                        ],
                        "returns": "float",
                    }
                },
                "properties": {"support": "tuple[float, float]"},
            },
        },
        "removals": {
            "symbols": ["crc_framework.OldExample"],
            "methods": {"Example": ["ppf"]},
        },
    }
    return contract, module


def test_runtime_checks_methods_properties_annotations_and_exact_exports() -> None:
    contract, module = _fixture()
    assert CHECK.runtime_errors(contract, module) == []
    changed = deepcopy(contract)
    changed["exports"]["Example"]["methods"]["quantile"]["returns"] = "int"
    assert any(
        "return annotation differs" in error
        for error in CHECK.runtime_errors(changed, module)
    )
    changed = deepcopy(contract)
    changed["exports"]["Example"]["properties"]["support"] = "str"
    assert any(
        "property annotation" in error
        for error in CHECK.runtime_errors(changed, module)
    )
    module.__all__.append("NotImplemented")
    assert any(
        "exact public export" in error
        for error in CHECK.runtime_errors(contract, module)
    )


def test_removing_a_name_from_all_is_not_enough() -> None:
    contract, module = _fixture()
    module.__dict__["OldExample"] = module.Example
    errors = CHECK.runtime_errors(contract, module)
    assert any(
        "OldExample: removed name remains importable" in error for error in errors
    )
    module.__dict__["AnotherName"] = module.Example
    assert any(
        "undeclared public attributes" in error
        for error in CHECK.runtime_errors(contract, module)
    )


def test_removing_an_old_method_from_inventory_is_not_enough() -> None:
    contract, module = _fixture()
    module.Example.ppf = module.Example.quantile
    assert any(
        "ppf: removed operation" in error
        for error in CHECK.runtime_errors(contract, module)
    )


def test_unknown_contract_fields_are_rejected() -> None:
    contract, release = CHECK.load_contracts()
    contract["compatibility_mode"] = "retain_old_names"
    with pytest.raises(CHECK.ContractError, match="unknown fields"):
        CHECK.validate_contract(contract, release)


def test_duplicate_json_keys_cannot_shadow_a_contract(tmp_path: Path) -> None:
    source = tmp_path / "public-api.json"
    source.write_text('{"exports": {}, "exports": {"Unexpected": {}}}')
    with pytest.raises(CHECK.ContractError, match="duplicate JSON key"):
        CHECK._load(source)


def test_runtime_method_bindings_and_undeclared_members_are_checked() -> None:
    contract, module = _fixture()
    contract["exports"]["Example"]["methods"]["quantile"]["binding"] = "class"
    assert any(
        "method binding differs" in error
        for error in CHECK.runtime_errors(contract, module)
    )
    module.Example.unexpected = lambda self: 1
    assert any(
        "undeclared public members" in error
        for error in CHECK.runtime_errors(contract, module)
    )


def test_installed_resources_must_match_the_release_checkout(tmp_path: Path) -> None:
    installed = tmp_path / "crc_framework"
    directory = installed / "contracts" / "api"
    shutil.copytree(ROOT / "python/crc_framework/contracts/api", directory)
    CHECK.validate_installed_contract(installed)
    path = directory / "public-api.json"
    contract = json.loads(path.read_text())
    contract["semantics"]["additional_rule"] = "A different internally valid draft."
    path.write_text(json.dumps(contract))
    with pytest.raises(CHECK.ContractError, match="differs from the release checkout"):
        CHECK.validate_installed_contract(installed)
