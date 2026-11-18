"""Regenerate common numerical admission and query fixtures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from crc_framework import (
    Distribution,
    DistributionFamily,
    Impact,
    QuantileCurve,
    Ratio,
    TailCompletion,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "python/crc_framework/contracts/numerical/fixtures-1.0.json"


def build():
    accepted = []
    objects = []
    for family, shape in [
        (DistributionFamily.GENERALIZED_EXTREME_VALUE, 0.1),
        (DistributionFamily.WEIBULL_MINIMUM, 2.2),
        (DistributionFamily.WEIBULL_MAXIMUM, 1.7),
        (DistributionFamily.SKEW_NORMAL, 1.5),
        (DistributionFamily.GUMBEL_RIGHT, None),
        (DistributionFamily.GUMBEL_LEFT, None),
        (DistributionFamily.GENERALIZED_PARETO, 0.15),
    ]:
        objects.append(
            (
                family.value,
                Distribution.parametric(family, location=1.2, scale=2, shape=shape),
            )
        )
    objects.extend(
        [
            ("point", Distribution.point_mass(-0.0)),
            ("empirical", Distribution.empirical([0, 0, 2, 10])),
            ("discrete", Distribution.discrete([0, 1e6], [0.999999, 0.000001])),
            ("histogram", Distribution.histogram([0, 1, 2, 3], [0.5, 0, 0.5])),
            ("partial", QuantileCurve([0.9, 0.99], [0.5, 2])),
            ("plateau", QuantileCurve([0, 0.2, 0.5, 0.9, 1], [0, 1, 1, 8, 10])),
        ]
    )
    curve = QuantileCurve([0.1, 0.5, 0.9], [0, 1, 3])
    complete = curve.complete(TailCompletion(lower=-1, upper=4))
    conditioned = QuantileCurve(
        [0.1, 0.4, 0.7, 0.9], [-50, -31, 7, 11]
    ).condition(q_min=0.2, q_max=0.6)
    discrete_base = Distribution.discrete([-1, 2, 3], [0.5, 0.1, 0.4])
    discrete_hurdle = Distribution.hurdle(discrete_base, atom_probability=0.2)
    decreasing = Impact.linear(slope=-2, intercept=1, input=Ratio(), output=Ratio())
    sigmoid = Impact.sigmoid(midpoint=0, steepness=1, input=Ratio(), output=Ratio())
    clamped_affine = Impact.linear(
        slope=0.5, intercept=3, maximum=6, input=Ratio(), output=Ratio()
    )
    objects.extend(
        [
            ("completed", complete),
            ("conditioned", curve.condition(q_min=0.2, q_max=0.8)),
            ("conditioned-interpolated-endpoints", conditioned),
            (
                "hurdle",
                Distribution.hurdle(
                    Distribution.parametric(
                        DistributionFamily.GUMBEL_RIGHT, location=0, scale=1
                    ),
                    atom_probability=0.3,
                ),
            ),
            ("discrete-hurdle", discrete_hurdle),
            (
                "conditioned-discrete-law",
                Distribution.hurdle(discrete_base, atom_probability=0.0),
            ),
        ]
    )
    objects.extend(
        [
            ("hurdle-completed", Distribution.hurdle(complete, atom_probability=0.2)),
            ("decreasing", decreasing.pushforward(objects[0][1])),
            ("transformed-discrete-hurdle", decreasing.pushforward(discrete_hurdle)),
            (
                "composed-clamped-affine",
                decreasing.pushforward(clamped_affine.pushforward(objects[0][1])),
            ),
            (
                "composed-piecewise-affine",
                sigmoid.pushforward(clamped_affine.pushforward(complete)),
            ),
        ]
    )
    for name, value in objects:
        queries = []
        coordinates = [0, 0.1, 0.5, 0.9, 0.99, 1]
        if name == "discrete-hurdle":
            coordinates[2:2] = [0.2, math.nextafter(0.2, 1.0)]
        for q in coordinates:
            try:
                result = float(value.quantile(q))
                query = (
                    {"probability": q, "value": result}
                    if math.isfinite(result)
                    else {
                        "probability": q,
                        "outcome": "positive_infinity"
                        if result > 0
                        else "negative_infinity",
                    }
                )
            except ValueError:
                query = {"probability": q, "outcome": "out_of_support"}
            queries.append(query)
        accepted.append(
            {"name": name, "document": json.loads(value.to_json()), "queries": queries}
        )
    rejected = []

    def bad(name, document, structural=True):
        rejected.append({"name": name, "document": document, "structural": structural})

    bad(
        "old-curve-fields",
        {"curve_kind": "point_mass", "curve_type": "point_mass", "curve_location": 1},
    )
    bad("old-abbreviated-point", {"kind": "point_mass", "value": 1})
    bad(
        "unknown-version",
        {
            "schema_version": "0.6",
            "object": "distribution",
            "kind": "point_mass",
            "value": 1,
        },
    )
    bad(
        "point-dead-fields",
        {
            "schema_version": "1.0",
            "object": "distribution",
            "kind": "point_mass",
            "value": 1,
            "scale": 0,
        },
    )
    bad(
        "gumbel-unused-shape",
        {
            "schema_version": "1.0",
            "object": "distribution",
            "kind": "parametric",
            "family": "gumbel_r",
            "location": 0,
            "scale": 1,
            "shape": None,
        },
    )
    bad(
        "negative-scale",
        {
            "schema_version": "1.0",
            "object": "distribution",
            "kind": "parametric",
            "family": "gumbel_r",
            "location": 0,
            "scale": 0,
        },
    )
    bad(
        "invalid-masses",
        {
            "schema_version": "1.0",
            "object": "distribution",
            "kind": "discrete",
            "values": [0, 1],
            "probabilities": [0.2, 0.3],
        },
        False,
    )
    bad(
        "unordered-knots",
        {
            "schema_version": "1.0",
            "object": "quantile_curve",
            "probabilities": [0.9, 0.1],
            "values": [0, 1],
        },
        False,
    )
    bad(
        "curve-lengths",
        {
            "schema_version": "1.0",
            "object": "quantile_curve",
            "probabilities": [0.1, 0.9],
            "values": [0, 1, 2],
        },
        False,
    )
    bad(
        "decreasing-values",
        {
            "schema_version": "1.0",
            "object": "quantile_curve",
            "probabilities": [0.1, 0.9],
            "values": [1, 0],
        },
        False,
    )
    bad(
        "law-partial-support",
        {
            "schema_version": "1.0",
            "object": "distribution",
            "kind": "quantile",
            "probabilities": [0.1, 0.9],
            "values": [0, 1],
        },
        False,
    )
    broken = json.loads(complete.to_json())
    broken["values"][1] = 0.2
    bad("forged-completion", broken, False)
    broken = json.loads(conditioned.to_json())
    broken["values"][-1] += 1
    bad("forged-conditioning", broken, False)
    broken = json.loads(complete.to_json())
    broken["evidence"] = None
    bad("null-modelling-record", broken)
    return {
        "schema_version": "1.0",
        "accepted": accepted,
        "rejected": rejected,
        "rejected_text": [
            {
                "name": "duplicate-key",
                "text": '{"schema_version":"1.0","object":"distribution","kind":"point_mass","value":1,"value":2}',
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = json.dumps(build(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != data:
            raise SystemExit(f"{OUTPUT} is stale")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(data)


if __name__ == "__main__":
    main()
