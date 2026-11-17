"""Installed Python and native consumers of the shared current wire contracts."""

from __future__ import annotations

import json
import math
from importlib.resources import files

import pytest
from crc_framework import Distribution, QuantileCurve, _core
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource
from tools.numerical_contracts.fixtures import build as numerical_fixtures
from tools.numerical_contracts.generate import build as numerical_schemas

CONTRACTS = files("crc_framework").joinpath("contracts")
NUMERICAL_URI = "https://climateriskcommons.org/schemas/numerical/1.0"
DATASET_URI = "https://climateriskcommons.org/schemas/dataset/2.0"
TAXONOMY_URI = "https://climateriskcommons.org/schemas/taxonomy/1.0"


def _load(path):
    return json.loads(CONTRACTS.joinpath(path).read_text(encoding="utf-8"))


def _offline(uri):
    raise NoSuchResource(ref=uri)


SCHEMAS = {}
for directory in ("numerical", "scientific", "dataset", "taxonomy"):
    for resource in CONTRACTS.joinpath(directory).iterdir():
        if resource.name.endswith(".schema.json"):
            schema = json.loads(resource.read_text(encoding="utf-8"))
            SCHEMAS[schema["$id"]] = schema

REGISTRY = Registry(retrieve=_offline).with_resources(
    (uri, Resource.from_contents(schema)) for uri, schema in SCHEMAS.items()
)

NUMERICAL = _load("numerical/fixtures-1.0.json")
SCIENTIFIC = _load("scientific/fixtures-1.0.json")["cases"]
DATASET = _load("dataset/fixtures.json")
TAXONOMY = _load("taxonomy/fixtures.json")


def _validator(uri):
    return Draft202012Validator(
        SCHEMAS[uri], registry=REGISTRY, format_checker=FormatChecker()
    )


def _text(document):
    return json.dumps(document, ensure_ascii=False, allow_nan=False)


def _numerical_loader(document):
    return (
        QuantileCurve.from_json
        if document.get("object") == "quantile_curve"
        else Distribution.from_json
    )


@pytest.mark.parametrize("uri", SCHEMAS)
def test_packaged_schemas_are_valid_and_resolve_offline(uri):
    Draft202012Validator.check_schema(SCHEMAS[uri])
    assert REGISTRY.get_or_retrieve(uri).value.contents == SCHEMAS[uri]
    with pytest.raises(NoSuchResource):
        REGISTRY.get_or_retrieve("https://unregistered.invalid/schema")


def test_offline_contract_checks_enforce_timestamp_formats():
    # jsonschema omits optional format checkers when their dependencies are absent.
    assert "date-time" in FormatChecker.checkers
    checker = FormatChecker()
    assert checker.conforms("2024-01-02t03:04:05z", "date-time")
    assert not checker.conforms("2024-01-02 03:04:05Z", "date-time")


def test_packaged_numerical_resources_match_their_generators():
    generated = {
        **numerical_schemas(),
        "fixtures-1.0.json": numerical_fixtures(),
    }
    for name, document in generated.items():
        expected = (
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        assert CONTRACTS.joinpath("numerical", name).read_bytes() == expected.encode()


@pytest.mark.parametrize("case", NUMERICAL["accepted"], ids=lambda case: case["name"])
def test_numerical_consumers_admit_and_evaluate_shared_objects(case):
    document = case["document"]
    text = _text(document)
    _validator(NUMERICAL_URI).validate(document)
    _core.validate_document(NUMERICAL_URI, text)
    native = _core.NativeNumericalDocument(text)
    loader = _numerical_loader(document)
    python = loader(text)
    restored = loader(python.to_json())
    _core.validate_document(NUMERICAL_URI, python.to_json())
    for query in case["queries"]:
        probability = query["probability"]
        if query.get("outcome") == "out_of_support":
            for evaluate in (python.quantile, restored.quantile, native.quantile):
                with pytest.raises(ValueError):
                    evaluate(probability)
            continue
        for evaluate in (python.quantile, restored.quantile, native.quantile):
            actual = evaluate(probability)
            if query.get("outcome") == "positive_infinity":
                assert actual == math.inf
            elif query.get("outcome") == "negative_infinity":
                assert actual == -math.inf
            else:
                assert actual == pytest.approx(query["value"], rel=1e-9, abs=1e-11)


@pytest.mark.parametrize("case", NUMERICAL["rejected"], ids=lambda case: case["name"])
def test_numerical_consumers_reject_invalid_current_contracts(case):
    document = case["document"]
    text = _text(document)
    assert _validator(NUMERICAL_URI).is_valid(document) is not case["structural"]
    for validate in (
        lambda: _core.validate_document(NUMERICAL_URI, text),
        lambda: _core.NativeNumericalDocument(text),
        lambda: _numerical_loader(document)(text),
    ):
        with pytest.raises((TypeError, ValueError)):
            validate()


@pytest.mark.parametrize(
    "case", NUMERICAL["rejected_text"], ids=lambda case: case["name"]
)
def test_numerical_consumers_reject_duplicate_json_keys(case):
    for validate in (
        lambda: _core.validate_document(NUMERICAL_URI, case["text"]),
        lambda: _core.NativeNumericalDocument(case["text"]),
        lambda: Distribution.from_json(case["text"]),
        lambda: QuantileCurve.from_json(case["text"]),
    ):
        with pytest.raises(ValueError):
            validate()


@pytest.mark.parametrize("case", SCIENTIFIC, ids=lambda case: case["name"])
def test_native_scientific_consumer_matches_shared_semantics(case):
    document = case["document"]
    assert _validator(case["schema"]).is_valid(document) is case.get(
        "schema_valid", case["valid"]
    )
    if case["valid"]:
        _core.validate_document(case["schema"], _text(document))
    else:
        with pytest.raises(ValueError):
            _core.validate_document(case["schema"], _text(document))


@pytest.mark.parametrize(
    "uri,fixtures",
    [(DATASET_URI, DATASET), (TAXONOMY_URI, TAXONOMY)],
    ids=["dataset", "taxonomy"],
)
def test_native_data_contract_consumers_match_shared_semantics(uri, fixtures):
    validator = _validator(uri)
    for case in fixtures["accepted"]:
        validator.validate(case["document"])
        _core.validate_document(uri, _text(case["document"]))
    for case in fixtures["rejected"]:
        assert validator.is_valid(case["document"]) is not case.get(
            "structural", True
        ), case["name"]
        with pytest.raises(ValueError, match=".+"):
            _core.validate_document(uri, _text(case["document"]))
