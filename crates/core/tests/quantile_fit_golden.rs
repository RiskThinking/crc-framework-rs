//! Bit-exact golden tests for the quantile Nelder-Mead fitting path.
//!
//! `fit_quantiles` drives `quantile_nelder_mead`, the simplex optimizer used to
//! produce production risk-model parameters. These tests pin the exact IEEE-754
//! bit patterns of the fitted parameters together with the simplex iteration and
//! evaluation counts, so any change to the optimizer that is meant to be purely
//! mechanical (allocation strategy, code motion) can be verified as producing
//! identical output rather than merely similar output.
//!
//! The objective calls `exp`, `ln` and `pow` from the platform's libm, so the
//! table is pinned to the libm that produced it (it was generated on macOS
//! 15 / aarch64 and verified identical in both debug and release builds). Treat
//! a mismatch on a different platform as a portability finding to investigate,
//! not as an automatic regression, and regenerate the table only for an
//! intentional numerical change:
//!
//! ```text
//! CRC_GOLDEN_REGEN=1 cargo test -p crc-framework-core --test quantile_fit_golden -- --nocapture
//! ```

use crc_framework_core::{DistributionFamily, fit_hurdle_quantiles, fit_quantiles};

/// Deterministic value generator: SplitMix64, so the cases are identical on
/// every platform and every run without pulling in an RNG dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) built from the top 53 bits, i.e. exactly representable.
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

const PROBABILITIES_WIDE: [f64; 13] = [
    0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
];
const PROBABILITIES_NARROW: [f64; 5] = [0.1, 0.25, 0.5, 0.75, 0.9];

/// A single fit problem: strictly increasing probabilities and non-decreasing
/// values, which is what `fit_quantiles` requires.
struct Case {
    label: &'static str,
    probabilities: Vec<f64>,
    values: Vec<f64>,
    weights: Option<Vec<f64>>,
}

/// Builds a monotone value curve by accumulating positive jitter onto a base
/// curve, so the resulting shapes are irregular enough to exercise multi-start
/// selection, expansion, contraction and shrink branches of the simplex.
fn monotone_values(seed: u64, count: usize, base: f64, slope: f64, jitter: f64) -> Vec<f64> {
    let mut rng = SplitMix64(seed);
    let mut value = base;
    let mut out = Vec::with_capacity(count);
    for step in 0..count {
        value += slope * (1.0 + step as f64 * 0.1) + jitter * rng.next_unit();
        out.push(value);
    }
    out
}

fn cases() -> Vec<Case> {
    let wide = PROBABILITIES_WIDE.to_vec();
    let narrow = PROBABILITIES_NARROW.to_vec();
    vec![
        Case {
            label: "wide-gentle",
            probabilities: wide.clone(),
            values: monotone_values(0x1234_5678, wide.len(), 0.0, 1.0, 0.25),
            weights: None,
        },
        Case {
            label: "wide-steep-tail",
            probabilities: wide.clone(),
            values: monotone_values(0xDEAD_BEEF, wide.len(), 2.0, 4.0, 3.0),
            weights: None,
        },
        Case {
            label: "wide-large-scale",
            probabilities: wide.clone(),
            values: monotone_values(0x0BAD_C0DE, wide.len(), 1000.0, 250.0, 90.0),
            weights: None,
        },
        Case {
            label: "wide-tiny-scale",
            probabilities: wide.clone(),
            values: monotone_values(0x00C0_FFEE, wide.len(), 0.001, 0.0004, 0.0002),
            weights: None,
        },
        Case {
            label: "wide-weighted",
            probabilities: wide.clone(),
            values: monotone_values(0xFEED_FACE, wide.len(), 5.0, 2.0, 1.5),
            weights: Some(vec![
                3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.5, 4.0,
            ]),
        },
        Case {
            label: "wide-flat-then-jump",
            probabilities: wide.clone(),
            values: vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 9.0, 14.0, 30.0,
            ],
            weights: None,
        },
        Case {
            label: "narrow-gentle",
            probabilities: narrow.clone(),
            values: monotone_values(0xA5A5_5A5A, narrow.len(), 10.0, 3.0, 1.0),
            weights: None,
        },
        Case {
            label: "narrow-negative-support",
            probabilities: narrow.clone(),
            values: monotone_values(0x5150_5150, narrow.len(), -40.0, 6.0, 2.0),
            weights: None,
        },
    ]
}

/// One line per (family, case) with the exact bit patterns of everything the
/// optimizer produces. Bits, not decimals: this is the whole point of the test.
fn fingerprint() -> Vec<String> {
    let mut lines = Vec::new();
    for family in DistributionFamily::ALL {
        for case in cases() {
            let outcome = fit_quantiles(
                &case.probabilities,
                &case.values,
                case.weights.as_deref(),
                family,
            );
            let rendered = match outcome {
                Err(error) => format!("err({error})"),
                Ok(result) => {
                    let shape = result.distribution.shape;
                    let loc = result.distribution.location;
                    let scale = result.distribution.scale;
                    let diagnostics = result.diagnostics;
                    format!(
                        "shape={} loc={:016x} scale={:016x} converged={} iterations={} \
                         evaluations={} points={} rmse={:016x} nrmse={:016x} r2={:016x} \
                         maxres={:016x}",
                        shape
                            .map_or_else(|| "none".to_owned(), |v| format!("{:016x}", v.to_bits())),
                        loc.to_bits(),
                        scale.to_bits(),
                        diagnostics.converged,
                        diagnostics.iterations,
                        diagnostics.evaluations,
                        diagnostics.point_count,
                        diagnostics.rmse.to_bits(),
                        diagnostics.normalized_rmse.to_bits(),
                        diagnostics.weighted_r_squared.to_bits(),
                        diagnostics.maximum_absolute_residual.to_bits(),
                    )
                }
            };
            lines.push(format!("{}/{} {rendered}", family.name(), case.label));
        }
    }
    lines
}

#[test]
fn quantile_fits_are_bit_identical_to_golden() {
    let actual = fingerprint();

    if std::env::var_os("CRC_GOLDEN_REGEN").is_some() {
        println!("---BEGIN GOLDEN---");
        for line in &actual {
            println!("{line}");
        }
        println!("---END GOLDEN---");
        return;
    }

    let expected: Vec<&str> = GOLDEN.lines().filter(|line| !line.is_empty()).collect();
    assert_eq!(
        actual.len(),
        expected.len(),
        "golden table is stale: case count changed"
    );
    let mut mismatches = Vec::new();
    for (actual_line, expected_line) in actual.iter().zip(&expected) {
        if actual_line != expected_line {
            mismatches.push(format!(
                "  expected: {expected_line}\n  actual:   {actual_line}"
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} quantile fits changed:\n{}",
        mismatches.len(),
        expected.len(),
        mismatches.join("\n")
    );
}

const GOLDEN: &str = include_str!("quantile_fit_golden.txt");

/// Wall-clock micro-benchmark for the fitting hot path, at the knot widths the
/// curve_fit_cdf pipeline actually uses (999 knots per curve) plus a narrower
/// width for contrast. Both objective paths are measured, because that pipeline
/// overwhelmingly takes the truncated (hurdle) one.
///
/// Ignored by default because it is a timing measurement, not an assertion:
///
/// ```text
/// cargo test --release -p crc-framework-core --test quantile_fit_golden -- \
///     --ignored --nocapture bench_wide_quantile_fits
/// ```
#[test]
#[ignore = "timing benchmark, run explicitly with --ignored"]
fn bench_wide_quantile_fits() {
    const WARMUP_ROUNDS: usize = 2;
    const MEASURED_ROUNDS: usize = 7;
    const PROBLEMS: usize = 4;

    /// A curve with `plateau` leading knots pinned at zero (the hurdle atom)
    /// and a strictly increasing tail, which is the shape the pipeline feeds in.
    fn wide_problem(knots: usize, plateau: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = SplitMix64(seed);
        let probabilities: Vec<f64> = (1..=knots)
            .map(|index| index as f64 / (knots as f64 + 1.0))
            .collect();
        let mut value = 0.0;
        let mut values = Vec::with_capacity(knots);
        for step in 0..knots {
            if step >= plateau {
                value += 0.05 + rng.next_unit() * (1.0 + step as f64 * 0.01);
            }
            values.push(value);
        }
        (probabilities, values)
    }

    for knots in [999usize, 200] {
        let plateau = knots / 4;
        let problems: Vec<(Vec<f64>, Vec<f64>)> = (0..PROBLEMS)
            .map(|seed| wide_problem(knots, plateau, 0x5EED + seed as u64))
            .collect();
        for family in [
            DistributionFamily::GumbelRight,
            DistributionFamily::GenExtreme,
        ] {
            for truncated in [false, true] {
                let mut failures = 0usize;
                let mut round = || {
                    let start = std::time::Instant::now();
                    let mut checksum = 0u64;
                    for (probabilities, values) in &problems {
                        let fitted = if truncated {
                            fit_hurdle_quantiles(
                                probabilities,
                                values,
                                None,
                                family,
                                values[0],
                                probabilities[plateau - 1],
                            )
                            .map(|result| {
                                (
                                    result.distribution.base().location,
                                    result.diagnostics.tail.iterations,
                                )
                            })
                        } else {
                            fit_quantiles(probabilities, values, None, family).map(|result| {
                                (result.distribution.location, result.diagnostics.iterations)
                            })
                        };
                        match fitted {
                            // Consume the result so nothing is optimized away.
                            Ok((location, iterations)) => {
                                checksum = checksum
                                    .wrapping_mul(0x0100_0000_01b3)
                                    .wrapping_add(location.to_bits())
                                    .wrapping_add(iterations as u64);
                            }
                            Err(_) => failures += 1,
                        }
                    }
                    (start.elapsed().as_secs_f64(), checksum)
                };

                for _ in 0..WARMUP_ROUNDS {
                    round();
                }
                let mut timings = Vec::with_capacity(MEASURED_ROUNDS);
                let mut checksum = 0u64;
                for _ in 0..MEASURED_ROUNDS {
                    let (elapsed, round_checksum) = round();
                    checksum = round_checksum;
                    timings.push(elapsed);
                }
                timings.sort_by(f64::total_cmp);
                // A benchmark that measured only error returns would be silently
                // meaningless, so the fit failures are reported alongside.
                assert_eq!(failures, 0, "benchmark inputs must produce real fits");
                println!(
                    "knots={knots:<4} {:<11} {:<7} best {:>10.2} us/fit  median {:>10.2} us/fit  checksum {checksum:016x}",
                    family.name(),
                    if truncated { "hurdle" } else { "plain" },
                    timings[0] / PROBLEMS as f64 * 1.0e6,
                    timings[timings.len() / 2] / PROBLEMS as f64 * 1.0e6,
                );
            }
        }
    }
}
