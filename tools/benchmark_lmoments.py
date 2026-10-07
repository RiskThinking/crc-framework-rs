"""Research audit of the current implementation; does not modify production code.

Requirements: Python 3.12, scipy==1.18.0, mpmath, maturin, threadpoolctl,
an editable installation of this framework, and a Rust toolchain.
Run: python tools/benchmark_lmoments.py --output /tmp/crc-lmoments-audit.json

The temporary release-mode PyO3 probe embeds the actual production source and
extracts its first-three-moment block verbatim. This allows equivalent-work
comparisons against scipy.stats.lmoment(order=[1,2,3], standardize=False).
The sorted option and raw-moment entry point exist in the probe only.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import importlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import timeit
import warnings
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from crc_framework import fit_distribution
from crc_framework.distributions.fitting import _samples
from scipy import stats
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/core/src/distribution/lmoments.rs"


def build_probe(folder: Path):
    source = SOURCE.read_text()
    # Embed our own code, not a reimplementation of either reference library.
    body = source[
        source.index("    let mut sorted = samples.to_vec();") : source.index(
            "    let (shape, location, scale) = match family"
        )
    ]
    body = body.replace(
        "    sorted.sort_unstable_by(f64::total_cmp);",
        "    if !presorted { sorted.sort_unstable_by(f64::total_cmp); }",
    )
    helpers = (
        """
pub(super) fn moments(samples: &[f64], presorted: bool) -> Result<[f64; 3]> {
    if samples.len() < 4 || samples.iter().any(|x| !x.is_finite()) {
        return Err(CrcError::InvalidInput("invalid samples".into()));
    }
"""
        + body
        + """
    Ok([l1 * magnitude + offset, l2 * magnitude, l3 * magnitude])
}
pub(super) fn parameters(l1: f64, l2: f64, skew: f64) -> Result<(Option<f64>, f64, f64)> {
    gev_parameters(l1, l2, skew)
}
pub(super) fn gamma_log(x: f64) -> f64 { log_gamma_one_plus(x) }
"""
    )
    (folder / "src").mkdir()
    (folder / "Cargo.lock").write_bytes((ROOT / "Cargo.lock").read_bytes())
    (folder / "src/kernel.rs").write_text(source + helpers)
    (folder / "Cargo.toml").write_text(f"""
[package]
name = "crc-lmoments-research-probe"
version = "0.0.0"
edition = "2024"
[lib]
name = "_lmoments_probe"
crate-type = ["cdylib"]
[profile.release]
# Avoid LLVM debug-stripping LINKEDIT misalignment on macOS 27
# (rust-lang/rust#157750); optimization remains the release default.
debug = 1
strip = "none"
[dependencies]
crc-framework-core = {{ path = "{ROOT / "crates/core"}" }}
pyo3 = {{ version = "=0.29.0", features = ["extension-module"] }}
""")
    (folder / "src/lib.rs").write_text("""
pub use crc_framework_core::{DistributionFamily, FittedDistribution};
mod error { pub use crc_framework_core::error::*; }
#[allow(dead_code)] mod kernel;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
#[pyfunction]
#[pyo3(signature=(samples, presorted=false))]
fn moments(py: Python<'_>, samples: Vec<f64>, presorted: bool) -> PyResult<Vec<f64>> {
    py.detach(|| kernel::moments(&samples, presorted))
        .map(|x| x.to_vec()).map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
fn moments_batch(py: Python<'_>, samples: Vec<f64>, n: usize) -> PyResult<Vec<f64>> {
    if n < 4 || samples.len() % n != 0 { return Err(PyValueError::new_err("invalid batch")); }
    py.detach(|| {
        let mut output = Vec::with_capacity(samples.len() / n * 3);
        for row in samples.chunks_exact(n) {
            output.extend(kernel::moments(row, false)?);
        }
        Ok::<_, error::CrcError>(output)
    }).map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
fn parameters(l1: f64, l2: f64, skew: f64) -> PyResult<(Option<f64>, f64, f64)> {
    kernel::parameters(l1, l2, skew).map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
fn gamma_log(x: f64) -> f64 { kernel::gamma_log(x) }
#[pymodule]
fn _lmoments_probe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(moments, m)?)?;
    m.add_function(wrap_pyfunction!(moments_batch, m)?)?;
    m.add_function(wrap_pyfunction!(parameters, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_log, m)?)?;
    Ok(())
}
""")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--release",
            "--manifest-path",
            str(folder / "Cargo.toml"),
            "--interpreter",
            sys.executable,
            "--out",
            str(folder / "wheels"),
        ],
        check=True,
    )
    # Load the wheel locally without installing anything into the environment.
    import zipfile

    wheel = next((folder / "wheels").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(folder / "unpacked")
    sys.path.insert(0, str(folder / "unpacked"))
    return importlib.import_module("_lmoments_probe")


def measure(fn, cells=1):
    fn()  # import/lazy setup excluded, all computations kept
    timer = timeit.Timer(fn)
    count = 1
    while timer.timeit(count) < 0.03:
        count *= 2
    measurements = [t / count / cells * 1e6 for t in timer.repeat(7, count)]
    return {
        "median_us": statistics.median(measurements),
        "min_us": min(measurements),
        "max_us": max(measurements),
        "repeats": 7,
        "calls_per_repeat": count,
        "cells_per_call": cells,
    }


def exact_moments(values):
    x = sorted(mp.mpf(float(v)) for v in values)
    n = len(x)
    b0 = mp.fsum(x) / n
    b1 = mp.fsum(mp.mpf(i) / (n - 1) * v for i, v in enumerate(x)) / n
    b2 = (
        mp.fsum(mp.mpf(i * (i - 1)) / ((n - 1) * (n - 2)) * v for i, v in enumerate(x))
        / n
    )
    return [b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0]


def accuracy(probe, rng):
    mp.mp.dps = 90
    rows = []
    worst = {"rust_error_in_L2_units": 0.0, "scipy_error_in_L2_units": 0.0}
    cases = {
        "ordinary_30": rng.lognormal(size=30),
        "large_offset": 1e15 + np.arange(30, dtype=float),
        "tiny_scale": rng.lognormal(size=30) * 1e-280,
        "huge_scale": rng.lognormal(size=30) * 1e280,
        "mixed_near_float_limit": np.array([-1e308, -5e307, 5e307, 1e308]),
        "ties_and_outlier": np.array([0, 0, 1, 3, 8, 15, 30, 100.0]),
    }
    for i in range(200):
        cases[f"random_{i}"] = rng.lognormal(sigma=1.5, size=int(rng.integers(4, 201)))
    for name, values in cases.items():
        reference = exact_moments(values)
        rust = probe.moments(_samples(values))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scipy_value = stats.lmoment(values, order=[1, 2, 3], standardize=False)

        def err(result, reference=reference):
            return max(
                float(abs(mp.mpf(float(a)) - b) / reference[1])
                if np.isfinite(a)
                else float("inf")
                for a, b in zip(result, reference)
            )

        item = {
            "case": name,
            "n": len(values),
            "rust": rust,
            "scipy": scipy_value.tolist(),
            "reference": [str(v) for v in reference],
            "rust_error_in_L2_units": err(rust),
            "scipy_error_in_L2_units": err(scipy_value),
        }
        if name.startswith("random_"):
            for key, previous in worst.items():
                worst[key] = max(previous, item[key])
        else:
            rows.append(item)
    gamma = []
    for c in [
        -0.999999,
        -0.99,
        -0.8,
        -0.1,
        -0.01,
        -0.00999,
        -1e-9,
        0.0,
        1e-9,
        0.00999,
        0.01,
        0.1,
        1,
        5,
        20,
        50,
    ]:
        actual = probe.gamma_log(c)
        expected = mp.loggamma(1 + mp.mpf(float(c)))
        gamma.append({"c": c, "absolute_error": float(abs(mp.mpf(actual) - expected))})
    solvers = []
    for c in [
        -0.999999,
        -0.99,
        -0.8,
        -0.1,
        -1e-5,
        -1e-9,
        0.0,
        1e-9,
        1e-5,
        0.1,
        1,
        5,
        20,
        50,
    ]:
        cm = mp.mpf(float(c))
        tau = (
            2 * mp.expm1(-cm * mp.log(3)) / mp.expm1(-cm * mp.log(2)) - 3
            if c
            else 2 * mp.log(3) / mp.log(2) - 3
        )
        l2 = (
            3 * mp.gamma(1 + cm) * -mp.expm1(-cm * mp.log(2)) / cm
            if c
            else 3 * mp.log(2)
        )
        l1 = 10 + 3 * (1 - mp.gamma(1 + cm)) / cm if c else 10 + 3 * mp.euler
        actual = probe.parameters(float(l1), float(l2), float(tau))
        solvers.append(
            {
                "shape": c,
                "fitted": actual,
                "shape_abs_error": abs(actual[0] - c),
                "location_abs_error": abs(actual[1] - 10),
                "scale_rel_error": abs(actual[2] / 3 - 1),
            }
        )
    return {
        "stress_cases": rows,
        "random_200_worst": worst,
        "log_gamma": gamma,
        "population_parameter_roundtrip": solvers,
    }


def benchmarks(probe, rng):
    rows = []
    for n in [20, 30, 100, 1000, 100000]:
        values = rng.lognormal(size=n)
        for presorted in [False, True]:
            x = np.sort(values) if presorted else values

            def rust(x=x, presorted=presorted):
                return probe.moments(_samples(x), presorted)

            def scipy_fn(x=x, presorted=presorted):
                return stats.lmoment(
                    x,
                    order=[1, 2, 3],
                    standardize=False,
                    sorted=presorted,
                    nan_policy="raise",
                )

            np.testing.assert_allclose(rust(), scipy_fn(), rtol=2e-11, atol=1e-14)
            rows.append(
                {
                    "kind": "moments_1d_numpy",
                    "n": n,
                    "sorted": presorted,
                    "rust": measure(rust),
                    "scipy": measure(scipy_fn),
                }
            )
        rows.append(
            {
                "kind": "proposed_conversion_only",
                "n": n,
                "production_samples": measure(lambda values=values: _samples(values)),
                "numpy_tolist": measure(lambda values=values: values.tolist()),
                "rust_with_tolist": measure(
                    lambda values=values: probe.moments(values.tolist())
                ),
            }
        )
        # Actual fitting includes parameter solving, diagnostics and Python result construction.
        rows.append(
            {
                "kind": "crc_full_gev_fit_context_only",
                "n": n,
                "crc": measure(
                    lambda values=values: fit_distribution(
                        values, "genextreme", method="lmoments"
                    )
                ),
            }
        )
    for count in [100, 10000]:
        x = rng.lognormal(size=(count, 30))

        def rust(x=x):
            return np.asarray([probe.moments(_samples(row)) for row in x]).T

        def scipy_fn(x=x):
            return stats.lmoment(
                x, order=[1, 2, 3], axis=1, standardize=False, nan_policy="raise"
            )

        np.testing.assert_allclose(rust(), scipy_fn(), rtol=2e-11, atol=1e-14)
        rows.append(
            {
                "kind": "moments_batch_numpy",
                "n": 30,
                "cells": count,
                "rust_python_loop": measure(rust, count),
                "scipy_vectorized": measure(scipy_fn, count),
            }
        )

        def native_batch(x=x, count=count):
            return (
                np.asarray(probe.moments_batch(x.ravel().tolist(), 30))
                .reshape(count, 3)
                .T
            )

        np.testing.assert_allclose(native_batch(), scipy_fn(), rtol=2e-11, atol=1e-14)
        rows.append(
            {
                "kind": "proposed_native_batch",
                "n": 30,
                "cells": count,
                "rust_single_native_call": measure(native_batch, count),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20261007)
    with tempfile.TemporaryDirectory(prefix="crc-lmoments-probe-") as directory:
        probe = build_probe(Path(directory))
        with threadpool_limits(limits=1):
            gc.collect()
            result = {
                "environment": {
                    "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "python": sys.version,
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "platform": platform.platform(),
                    "cpu": subprocess.check_output(
                        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                    ).strip()
                    if sys.platform == "darwin"
                    else platform.processor(),
                    "rust": subprocess.check_output(
                        ["rustc", "--version"], text=True
                    ).strip(),
                    "production_source_sha256": hashlib.sha256(
                        SOURCE.read_bytes()
                    ).hexdigest(),
                    "seed": 20261007,
                    "probe_build": "release, opt-level=3, debug=1, strip=none",
                    "thread_limit": 1,
                },
                "accuracy": accuracy(probe, rng),
                "benchmarks": benchmarks(probe, rng),
            }

            # Keep the artifact valid JSON even for intentional overflow cases.
            def json_safe(value):
                if isinstance(value, dict):
                    return {key: json_safe(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [json_safe(item) for item in value]
                if isinstance(value, float) and not math.isfinite(value):
                    return "NaN" if math.isnan(value) else str(value)
                return value

            result = json_safe(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
