"""Scalar, vector and multi-threaded queries agree bit for bit."""

from __future__ import annotations

import json

import numpy as np
import pytest
from crc_framework import (
    Distribution,
    DistributionFamily,
    QuantileCurve,
    TailCompletion,
)

_GEV = Distribution.parametric(
    DistributionFamily.GENERALIZED_EXTREME_VALUE, location=1.0, scale=2.0, shape=-0.1
)
_CURVE = QuantileCurve([0.1, 0.5, 0.9, 0.99], [-3.0, -0.5, 2.0, 9.0])
_LAWS = [
    _GEV,
    Distribution.parametric(DistributionFamily.GUMBEL_LEFT, location=-2.0, scale=0.5),
    Distribution.point_mass(1.5),
    Distribution.hurdle(_GEV, atom_probability=0.3, atom_location=0.0),
    Distribution.empirical([1.0, 2.0, 2.0, 5.0]),
    Distribution.histogram([0.0, 1.0, 4.0], [0.4, 0.6]),
    _CURVE.complete(TailCompletion(lower=-10.0, upper=20.0)),
]


@pytest.mark.parametrize("law", _LAWS, ids=lambda law: law.kind.value)
def test_scalar_queries_equal_vector_queries(law: Distribution) -> None:
    rng = np.random.default_rng(11)
    probabilities = np.concatenate([[0.0, 1.0], rng.random(100)])
    vector = law.quantile(probabilities)
    assert [law.quantile(float(q)) for q in probabilities] == vector.tolist()
    values = np.concatenate([vector, vector + 0.25, [-np.inf, np.inf]])
    for inclusive in (False, True):
        assert [law.cdf(float(x), inclusive=inclusive) for x in values] == law.cdf(
            values, inclusive=inclusive
        ).tolist()
        assert [
            law.exceedance(float(x), inclusive=inclusive) for x in values
        ] == law.exceedance(values, inclusive=inclusive).tolist()
    assert [law.mass(float(x)) for x in values] == law.mass(values).tolist()


def test_scalar_curve_queries_equal_vector_queries() -> None:
    probabilities = np.concatenate(
        [[0.1, 0.5, 0.99], np.random.default_rng(5).uniform(0.1, 0.99, 200)]
    )
    assert [_CURVE.quantile(float(q)) for q in probabilities] == _CURVE.quantile(
        probabilities
    ).tolist()


def test_long_queries_split_across_threads_equal_short_ones() -> None:
    probabilities = np.random.default_rng(2).random(200_003)
    whole = _GEV.quantile(probabilities)
    parts = np.concatenate(
        [_GEV.quantile(probabilities[i : i + 1000]) for i in range(0, 200_003, 1000)]
    )
    assert whole.tobytes() == parts.tobytes()
    assert not whole.flags.writeable
    with pytest.raises(ValueError):
        whole.setflags(write=True)


@pytest.mark.parametrize("query", [float("nan"), -0.5, 1.5, float("inf")])
def test_invalid_scalar_probabilities_keep_their_errors(query: float) -> None:
    with pytest.raises(ValueError) as scalar:
        _GEV.quantile(query)
    with pytest.raises(ValueError) as vector:
        _GEV.quantile([query])
    assert str(scalar.value) == str(vector.value)


def test_from_json_keeps_parse_errors_ahead_of_schema_errors() -> None:
    text = json.loads(_GEV.to_json())
    duplicate = _GEV.to_json().replace('"scale":2.0', '"scale":2.0,"scale":3.0')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        Distribution.from_json(duplicate)
    with pytest.raises(ValueError, match="expected numerical distribution schema"):
        Distribution.from_json(json.dumps({**text, "object": "quantile_curve"}))
    with pytest.raises(ValueError, match="non-finite JSON number"):
        Distribution.from_json(_GEV.to_json().replace("2.0", "NaN"))
    with pytest.raises(ValueError, match="numerical representation must be an object"):
        Distribution.from_json("[]")
