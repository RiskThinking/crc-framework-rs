"""Build the current numerical and fitting interchange schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "python/crc_framework/contracts/numerical"
FAMILIES = [
    "genextreme",
    "weibull_min",
    "weibull_max",
    "skewnorm",
    "gumbel_r",
    "gumbel_l",
    "genpareto",
]
NUMBER = {
    "type": "number",
    "minimum": -1.7976931348623157e308,
    "maximum": 1.7976931348623157e308,
}
PROBABILITY = {**NUMBER, "minimum": 0, "maximum": 1}


def obj(properties, required=None):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties if required is None else required),
        "additionalProperties": False,
    }


def array(items, **kwargs):
    return {"type": "array", "items": items, **kwargs}


def nullable(value):
    return {"anyOf": [value, {"type": "null"}]}


def specification():
    family = {"enum": FAMILIES}
    bounds = obj(
        {
            "minimum_value": nullable(NUMBER),
            "maximum_value": nullable(NUMBER),
            "probability": nullable(PROBABILITY),
            "require_sample_support": {"type": "boolean"},
        },
        [],
    )
    base = {"family": family, "constraints": nullable(bounds)}
    result = {}
    for kind, extra, required in [
        (
            "samples",
            {
                "estimator": {"enum": ["mle", "lmoments"]},
                "minimum_samples": {"type": "integer", "minimum": 3},
                "fallbacks": array({"$ref": "#/$defs/samples_fit_specification"}),
            },
            ["kind", "family"],
        ),
        (
            "quantiles",
            {
                "weights": nullable(array({**NUMBER, "minimum": 0}, minItems=1)),
                "minimum_knots": {"type": "integer", "minimum": 4},
            },
            ["kind", "family"],
        ),
        (
            "censored_quantiles",
            {
                "censor_location": NUMBER,
                "residual_scale_floor_ratio": {**NUMBER, "exclusiveMinimum": 0},
                "minimum_knots": {"type": "integer", "minimum": 4},
            },
            ["kind", "family", "censor_location"],
        ),
        (
            "hurdle_quantiles",
            {
                "atom_probability": {
                    **PROBABILITY,
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
                "atom_location": NUMBER,
                "minimum_knots": {"type": "integer", "minimum": 4},
            },
            ["kind", "family", "atom_probability"],
        ),
    ]:
        result[f"{kind}_fit_specification"] = obj(
            {"kind": {"const": kind}, **base, **extra}, required
        )
    result["concrete_specification"] = {
        "oneOf": [{"$ref": f"#/$defs/{name}"} for name in result]
    }
    concrete = {"$ref": "#/$defs/concrete_specification"}
    result["model_selection"] = obj(
        {
            "kind": {"const": "model_selection"},
            "candidates": array(concrete, minItems=1),
            "criterion": {"enum": ["ks", "normalized_rmse"]},
        },
        ["kind", "candidates"],
    )
    result["specification"] = {"oneOf": [concrete, {"$ref": "#/$defs/model_selection"}]}
    return result


def fitting_schema():
    specs = specification()
    count = {"type": "integer", "minimum": 0}
    nonnegative = {**NUMBER, "minimum": 0}
    positive = {**NUMBER, "exclusiveMinimum": 0, "minimum": 0}
    optimizer = {"iterations": count, "evaluations": count}
    comparison = {"normalized_rmse": nonnegative, "normalization_scale": positive}
    residuals = {
        "rmse": nonnegative,
        "weighted_r_squared": NUMBER,
        "maximum_absolute_residual": nonnegative,
    }
    details = {
        "samples": {
            "ks_statistic": PROBABILITY,
            "ks_pvalue": PROBABILITY,
            "rmse": nonnegative,
            "r_squared": NUMBER,
        },
        "quantiles": {**residuals, **comparison},
        "censored_quantiles": {
            **residuals,
            **comparison,
            "negative_log_likelihood": NUMBER,
            "residual_scale": positive,
            "residual_scale_floor": positive,
            "censored_count": count,
            "uncensored_count": count,
        },
        "hurdle_quantiles": {
            **comparison,
            "tail_rmse": nonnegative,
            "tail_weighted_r_squared": NUMBER,
            "tail_maximum_absolute_residual": nonnegative,
            "atom_count": count,
            "tail_count": count,
        },
    }

    attempts = [
        obj(
            {
                "specification": {"$ref": f"#/$defs/{kind}_fit_specification"},
                "status": {"const": "converged"},
                "diagnostic": obj({**fields, **optimizer}),
            }
        )
        for kind, fields in details.items()
    ]
    attempts.append(
        obj(
            {
                "specification": {"$ref": "#/$defs/concrete_specification"},
                "status": {"enum": ["rejected", "failed"]},
                "diagnostic": obj({"message": {"type": "string", "minLength": 1}}),
            }
        )
    )
    record = obj(
        {
            "specification": {"$ref": "#/$defs/specification"},
            "attempts": array({"oneOf": attempts}, minItems=1),
            "selected_attempt": nullable({"type": "integer", "minimum": 0}),
            "estimator": nullable({"enum": ["mle", "lmoments"]}),
            "converged": {"type": "boolean"},
            "evidence_count": {"type": "integer", "minimum": 0},
            "used_count": {"type": "integer", "minimum": 0},
            "excluded_count": {"type": "integer", "minimum": 0},
            "weights": nullable(array(PROBABILITY)),
            "value_interval": nullable(array(NUMBER, minItems=2, maxItems=2)),
            "probability_interval": nullable(
                array(PROBABILITY, minItems=2, maxItems=2)
            ),
            "engine_version": {"type": "string", "minLength": 1},
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://climateriskcommons.org/schemas/fitting/1.0",
        "$defs": specs,
        **record,
    }


def numerical_schema():
    numbers = array(NUMBER, minItems=1)
    knots = array(PROBABILITY, minItems=2)
    curve = obj(
        {
            "schema_version": {"const": "1.0"},
            "object": {"const": "quantile_curve"},
            "probabilities": knots,
            "values": array(NUMBER, minItems=2),
        }
    )
    source = {"$ref": "#/$defs/quantile_curve"}
    evidence = {
        "oneOf": [
            obj(
                {
                    "source": source,
                    "operation": {"const": "condition"},
                    "q_min": PROBABILITY,
                    "q_max": PROBABILITY,
                }
            ),
            obj(
                {
                    "source": source,
                    "operation": {"const": "complete"},
                    "policy": obj(
                        {
                            "name": {"const": "finite_endpoints"},
                            "lower": NUMBER,
                            "upper": NUMBER,
                        }
                    ),
                }
            ),
        ]
    }
    bodies = [
        obj(
            {
                "kind": {"const": "parametric"},
                "family": {"enum": ["gumbel_r", "gumbel_l"]},
                "location": NUMBER,
                "scale": {**NUMBER, "exclusiveMinimum": 0},
            }
        ),
        obj(
            {
                "kind": {"const": "parametric"},
                "family": {
                    "enum": [f for f in FAMILIES if not f.startswith("gumbel_")]
                },
                "location": NUMBER,
                "scale": {**NUMBER, "exclusiveMinimum": 0},
                "shape": NUMBER,
            }
        ),
        obj(
            {
                "kind": {"const": "discrete"},
                "values": numbers,
                "probabilities": array(PROBABILITY, minItems=1),
                "counts": array(
                    {"type": "integer", "minimum": 1, "maximum": 2**53},
                    minItems=1,
                ),
            },
            ["kind", "values", "probabilities"],
        ),
        obj(
            {
                "kind": {"const": "histogram"},
                "edges": array(NUMBER, minItems=2),
                "probabilities": array(PROBABILITY, minItems=1),
            }
        ),
        obj({"kind": {"const": "point_mass"}, "value": NUMBER}),
        obj(
            {
                "kind": {"const": "hurdle"},
                "base": {"$ref": "#/$defs/data"},
                "atom_probability": PROBABILITY,
                "atom_location": NUMBER,
            }
        ),
        obj(
            {
                "kind": {"const": "quantile"},
                "probabilities": knots,
                "values": array(NUMBER, minItems=2),
                "evidence": evidence,
            },
            ["kind", "probabilities", "values"],
        ),
        obj(
            {
                "kind": {"const": "transformed"},
                "source": {"$ref": "#/$defs/distribution"},
                "impact": {
                    "$ref": "https://climateriskcommons.org/schemas/impact-2.0.schema.json"
                },
            }
        ),
    ]
    distributions = []
    for body in bodies:
        props = body["properties"]
        distributions.append(
            obj(
                {
                    "schema_version": {"const": "1.0"},
                    "object": {"const": "distribution"},
                    **props,
                },
                ["schema_version", "object", *body["required"]],
            )
        )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://climateriskcommons.org/schemas/numerical/1.0",
        "description": "Current numerical objects. Ordered knots, exact masses and modelling records require the numerical semantic validator in addition to structural validation.",
        "$defs": {
            "data": {"oneOf": bodies},
            "quantile_curve": curve,
            "distribution": {"oneOf": distributions},
        },
        "oneOf": [{"$ref": "#/$defs/quantile_curve"}, {"$ref": "#/$defs/distribution"}],
    }


def build():
    return {
        "numerical-1.0.schema.json": numerical_schema(),
        "fitting-1.0.schema.json": fitting_schema(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, value in build().items():
        data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        path = OUTPUT / name
        if args.check:
            if not path.exists() or path.read_bytes() != data.encode("utf-8"):
                raise SystemExit(f"{path} is stale")
        else:
            path.write_text(data, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
