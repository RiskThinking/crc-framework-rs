"""Time the release-built dense annual loss simulator on a synthetic book."""

from __future__ import annotations

import argparse
import json
import platform
from time import perf_counter

from crc_framework import (
    AnnualLossSpec,
    InsuredInterest,
    PerilDependence,
    PolicyTerms,
    QuantileCurve,
    simulate_annual_losses,
    simulate_separate_peril_losses,
    summarize_annual_losses,
)


def benchmark(
    count: int,
    trials: int,
    *,
    perils: int,
    correlation: float | None,
    peak_correlation: float,
    seed: int,
    components: bool,
) -> dict[str, object]:
    spec = AnnualLossSpec(trials=trials, peak_correlation=peak_correlation, seed=seed)
    curve = QuantileCurve([0, 0.9, 0.99, 1], [0.0, 0.05, 0.35, 0.85])
    book = [
        InsuredInterest(
            f"asset-{index:06}",
            longitude=-125 + (index % 100) * 0.6,
            latitude=30 + (index // 100) * 0.45,
            total_insured_value=100_000,
            damage_curve=curve,
            terms=PolicyTerms(deductible=2_000, limit=60_000, share=0.8),
        )
        for index in range(count)
    ]
    names = tuple(f"peril-{index:02}" for index in range(perils))
    dependence = PerilDependence.independent()
    if correlation is not None:
        matrix = [
            [1.0 if i == j else correlation for j in range(perils)]
            for i in range(perils)
        ]
        dependence = PerilDependence.from_correlation(
            names, matrix, calibration_source="synthetic benchmark"
        )

    record = {
        "interests": count,
        "trials": trials,
        "perils": perils,
        "cross_peril_correlation": correlation,
        "peak_spatial_correlation": peak_correlation,
        "seed": seed,
        "retain_components": components,
        "platform": platform.machine(),
    }
    started = perf_counter()
    try:
        if perils == 1:
            result = simulate_annual_losses(
                book, spec=spec, include_components=components
            )
        else:
            result = simulate_separate_peril_losses(
                dict.fromkeys(names, book),
                dependence=dependence,
                spec=spec,
                include_components=components,
            )
    except ValueError as error:
        if "exceeds memory_limit" not in str(error) and "exceeds work_limit" not in str(
            error
        ):
            raise
        return {
            **record,
            "elapsed_seconds": perf_counter() - started,
            "rejected": str(error),
        }
    elapsed = perf_counter() - started
    summary = summarize_annual_losses(
        result.totals,
        total_insured_value=result.total_insured_value,
        levels=(),
        return_periods=(250,),
    )
    return {
        **record,
        "elapsed_seconds": elapsed,
        "insured_value": result.total_insured_value,
        "output_bytes": result.totals.nbytes
        + sum(losses.nbytes for losses in (result.components or {}).values()),
        "average_annual_loss": summary.average_annual_loss,
        "aep_250_loss": summary.aep[0].loss,
        "aep_250_nominal_tail_years": summary.aep[0].nominal_tail_years,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interests", type=int)
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--perils", type=int, default=2)
    parser.add_argument("--correlation", type=float)
    parser.add_argument("--peak-correlation", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=20_261_027)
    parser.add_argument("--components", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.interests <= 10_000 or min(args.trials, args.perils) < 1:
        parser.error(
            "provide 1–10,000 interests, positive trials and positive peril count"
        )
    if args.perils == 1 and args.correlation is not None:
        parser.error("cross-peril correlation requires multiple perils")
    print(
        json.dumps(
            benchmark(
                args.interests,
                args.trials,
                perils=args.perils,
                correlation=args.correlation,
                peak_correlation=args.peak_correlation,
                seed=args.seed,
                components=args.components,
            )
        )
    )


if __name__ == "__main__":
    main()
