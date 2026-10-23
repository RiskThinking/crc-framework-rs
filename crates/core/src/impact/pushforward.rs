//! Distribution of `Y = f(X)` for a certified monotone impact.
//!
//! Continuous responses use `f(Q_X(p))` or `f(Q_X(1-p))`. Piecewise-constant
//! responses become finite atoms and use the generalized inverse, so a tie on
//! a jump returns the lower value. Flats of a continuous response are reported
//! as atoms without replacing that quantile rule.

use crate::Interpolation;

use super::definition::{Clamp, Interpolation as TableInterpolation, Model, OutsideDomain};
use super::error::{ContractResult, fail};
use super::kernel;
use super::quadrature::neumaier;
use super::source::{ImpactSource, Tail};

pub const ALGORITHM: &str = "crc-pushforward/1";
const MASS_COVER: f64 = 1e-12;

/// An atom of the pushforward distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpactAtom {
    pub value: f64,
    pub probability: f64,
}

#[derive(Clone, Debug)]
enum Law {
    Discrete,
    Continuous {
        decreasing: bool,
    },
    /// Response is `floor` on `(-inf, threshold)` and the continuous branch after it.
    LowerThreshold {
        cut: f64,
        floor: f64,
        decreasing: bool,
    },
}

/// Native pushforward. It keeps the definition and source instead of tabulating them.
#[derive(Clone, Debug)]
pub struct ImpactDistribution {
    definition_hash: String,
    model: Model,
    clamp: Clamp,
    source: ImpactSource,
    atoms: Vec<ImpactAtom>,
    law: Law,
}

impl ImpactDistribution {
    pub fn definition_hash(&self) -> &str {
        &self.definition_hash
    }

    pub fn algorithm(&self) -> &'static str {
        ALGORITHM
    }

    pub fn atoms(&self) -> &[ImpactAtom] {
        &self.atoms
    }

    pub fn quantile(&self, probability: f64) -> ContractResult<f64> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(fail(
                "invalid_value",
                "",
                "probability must be finite and in [0, 1]",
            ));
        }
        match &self.law {
            Law::Discrete => Ok(discrete_quantile(&self.atoms, probability)),
            Law::Continuous { decreasing } => {
                if probability == 0.0 {
                    return self.essential(true);
                }
                if probability == 1.0 {
                    return self.essential(false);
                }
                let source_probability = if *decreasing {
                    1.0 - probability
                } else {
                    probability
                };
                let exposure = self.source.quantile(source_probability)?;
                kernel::evaluate(&self.model, &self.clamp, exposure)
            }
            Law::LowerThreshold { cut, floor, .. } => {
                if probability == 0.0 {
                    return self.essential(true);
                }
                if probability == 1.0 {
                    return self.essential(false);
                }
                if probability <= *cut {
                    Ok(*floor)
                } else {
                    let exposure = self.source.quantile(probability)?;
                    kernel::evaluate(&self.model, &self.clamp, exposure)
                }
            }
        }
    }

    pub fn quantiles(&self, probabilities: &[f64]) -> ContractResult<Vec<f64>> {
        probabilities
            .iter()
            .map(|probability| self.quantile(*probability))
            .collect()
    }

    /// Explicit approximate quantile table. The interpolation is part of the approximation.
    pub fn tabulate(
        &self,
        probabilities: &[f64],
        interpolation: Interpolation,
    ) -> ContractResult<crate::TabulatedDistribution> {
        if probabilities.len() < 2 {
            return Err(fail(
                "invalid_value",
                "/probabilities",
                "tabulation requires at least two probabilities",
            ));
        }
        if probabilities
            .iter()
            .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
            || probabilities.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(fail(
                "invalid_value",
                "/probabilities",
                "tabulation probabilities must be finite and strictly increasing in [0, 1]",
            ));
        }
        let values = self.quantiles(probabilities)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(fail(
                "unsupported_value",
                "/probabilities",
                "an infinite essential endpoint cannot be stored in a finite quantile table",
            ));
        }
        crate::TabulatedDistribution::new(probabilities.to_vec(), values, interpolation, false)
            .map_err(|error| fail("invalid_value", "/probabilities", error.to_string()))
    }

    fn essential(&self, infimum: bool) -> ContractResult<f64> {
        if let Law::Discrete = self.law {
            return Ok(discrete_quantile(
                &self.atoms,
                if infimum { 0.0 } else { 1.0 },
            ));
        }
        let decreasing = matches!(
            self.law,
            Law::Continuous { decreasing: true }
                | Law::LowerThreshold {
                    decreasing: true,
                    ..
                }
        );
        let at_lower_support = if decreasing { !infimum } else { infimum };
        let (lower, upper) = self.source.support();
        let tail = if at_lower_support { lower } else { upper };
        match tail {
            Tail::Finite(exposure) => kernel::evaluate(&self.model, &self.clamp, exposure),
            Tail::NegativeInfinity => Ok(limit_at_infinity(&self.model, &self.clamp, false)),
            Tail::PositiveInfinity => Ok(limit_at_infinity(&self.model, &self.clamp, true)),
        }
    }
}

pub(crate) fn transform(
    model: &Model,
    clamp: &Clamp,
    definition_hash: &str,
    source: &ImpactSource,
) -> ContractResult<ImpactDistribution> {
    if !source.is_full_probability_support() {
        return Err(fail(
            "missing_probability_support",
            "",
            "distribution transformation requires a full probability distribution",
        ));
    }
    let direction = kernel::monotonicity(model, clamp);
    let decreasing = match direction {
        "constant" => false,
        "non_decreasing" => false,
        "non_increasing" => true,
        "non_monotone" | "unknown" => {
            return Err(fail(
                "monotonicity_required",
                "",
                "distribution transformation requires a certified monotone function",
            ));
        }
        other => {
            return Err(fail(
                "monotonicity_required",
                "",
                format!("unrecognized monotonicity class {other}"),
            ));
        }
    };
    let regions = constant_regions(model, clamp)?;
    let mut atoms = Vec::new();
    for region in &regions {
        let mass = region.probability(source);
        if mass > 1e-15 {
            atoms.push(ImpactAtom {
                value: region.y,
                probability: mass,
            });
        }
    }
    for (location, mass) in source.point_atoms() {
        if mass <= 1e-15 || regions.iter().any(|region| region.contains(location)) {
            continue;
        }
        atoms.push(ImpactAtom {
            value: kernel::evaluate(model, clamp, location)?,
            probability: mass,
        });
    }
    let atoms = merge_atoms(atoms);
    let covered = neumaier(atoms.iter().map(|atom| atom.probability));
    let piecewise_constant = matches!(
        model,
        Model::Binned { .. }
            | Model::Tabulated {
                interpolation: TableInterpolation::StepLeft | TableInterpolation::StepRight,
                ..
            }
    );
    if piecewise_constant && covered < 1.0 - MASS_COVER {
        return Err(fail(
            "out_of_domain",
            "",
            "the source support leaves a piecewise-constant table domain",
        ));
    }
    let law = if covered >= 1.0 - MASS_COVER {
        Law::Discrete
    } else if let Some((cut, floor)) = lower_threshold(model, clamp, source)? {
        if direction == "non_decreasing" {
            Law::LowerThreshold {
                cut,
                floor,
                decreasing,
            }
        } else {
            Law::Continuous { decreasing }
        }
    } else {
        Law::Continuous { decreasing }
    };
    Ok(ImpactDistribution {
        definition_hash: definition_hash.to_string(),
        model: model.clone(),
        clamp: clamp.clone(),
        source: source.clone(),
        atoms,
        law,
    })
}

fn discrete_quantile(atoms: &[ImpactAtom], probability: f64) -> f64 {
    if atoms.is_empty() {
        return f64::NAN;
    }
    if probability <= 0.0 {
        return atoms[0].value;
    }
    let mut cumulative = 0.0;
    for atom in atoms {
        cumulative += atom.probability;
        if probability <= cumulative {
            return atom.value;
        }
    }
    atoms.last().expect("atoms are non-empty").value
}

fn merge_atoms(atoms: Vec<ImpactAtom>) -> Vec<ImpactAtom> {
    let mut ordered = atoms;
    ordered.sort_by(|left, right| left.value.total_cmp(&right.value));
    let mut merged: Vec<ImpactAtom> = Vec::new();
    for atom in ordered {
        if merged.last().is_some_and(|last| last.value == atom.value) {
            merged.last_mut().expect("atom group exists").probability += atom.probability;
        } else {
            merged.push(atom);
        }
    }
    merged
}

#[derive(Clone, Copy)]
struct Region {
    start: f64,
    end: f64,
    start_closed: bool,
    end_closed: bool,
    y: f64,
}

impl Region {
    fn contains(self, x: f64) -> bool {
        let after_start = if self.start.is_infinite() {
            true
        } else if self.start_closed {
            x >= self.start
        } else {
            x > self.start
        };
        let before_end = if self.end.is_infinite() {
            true
        } else if self.end_closed {
            x <= self.end
        } else {
            x < self.end
        };
        after_start && before_end
    }

    fn probability(self, source: &ImpactSource) -> f64 {
        let right = if self.end.is_infinite() && self.end.is_sign_positive() {
            1.0
        } else if self.end_closed {
            source.cdf(self.end)
        } else {
            source.cdf_below(self.end)
        };
        let left = if self.start.is_infinite() && self.start.is_sign_negative() {
            0.0
        } else if self.start_closed {
            source.cdf_below(self.start)
        } else {
            source.cdf(self.start)
        };
        (right - left).max(0.0)
    }
}

fn constant_regions(model: &Model, clamp: &Clamp) -> ContractResult<Vec<Region>> {
    let mut regions = Vec::new();
    match model {
        Model::Linear { slope, intercept } => {
            linear_regions(*slope, *intercept, clamp, &mut regions)?;
        }
        Model::Tabulated {
            points,
            interpolation,
            outside_domain,
        } => match interpolation {
            TableInterpolation::Linear => {
                tabulated_linear_regions(points, clamp, &mut regions)?;
            }
            TableInterpolation::StepLeft => {
                step_regions(points, *outside_domain, clamp, true, &mut regions)?;
            }
            TableInterpolation::StepRight => {
                step_regions(points, *outside_domain, clamp, false, &mut regions)?;
            }
        },
        Model::Binned {
            edges,
            values,
            outside_domain,
        } => binned_regions(edges, values, *outside_domain, clamp, &mut regions)?,
        Model::Sigmoid { zero_below, .. } => {
            if let Some(threshold) = zero_below {
                let floor = kernel::evaluate(model, clamp, threshold - 1.0)?;
                regions.push(Region {
                    start: f64::NEG_INFINITY,
                    end: *threshold,
                    start_closed: false,
                    end_closed: false,
                    y: floor,
                });
            }
        }
        Model::Cooling { .. } => {
            regions.push(Region {
                start: f64::NEG_INFINITY,
                end: 1.0,
                start_closed: false,
                end_closed: true,
                y: kernel::evaluate(model, clamp, 1.0)?,
            });
        }
        Model::Spei { .. } => {
            let level = kernel::evaluate(model, clamp, 0.0)?;
            regions.push(Region {
                start: 0.0,
                end: f64::INFINITY,
                start_closed: true,
                end_closed: false,
                y: level,
            });
        }
        Model::Crop { .. } => {}
    }
    Ok(regions)
}

fn linear_regions(
    slope: f64,
    intercept: f64,
    clamp: &Clamp,
    regions: &mut Vec<Region>,
) -> ContractResult<()> {
    let model = Model::Linear { slope, intercept };
    if slope == 0.0 {
        regions.push(Region {
            start: f64::NEG_INFINITY,
            end: f64::INFINITY,
            start_closed: false,
            end_closed: false,
            y: kernel::evaluate(&model, clamp, 0.0)?,
        });
        return Ok(());
    }
    let mut cuts = Vec::new();
    for limit in [clamp.minimum, clamp.maximum].into_iter().flatten() {
        cuts.push((limit - intercept) / slope);
    }
    cuts.retain(|cut| cut.is_finite());
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|left, right| *left == *right);
    let mut bounds = vec![f64::NEG_INFINITY];
    bounds.extend(cuts);
    bounds.push(f64::INFINITY);
    for pair in bounds.windows(2) {
        let (first, second) = probes(pair[0], pair[1]);
        let y1 = kernel::evaluate(&model, clamp, first)?;
        let y2 = kernel::evaluate(&model, clamp, second)?;
        if y1 == y2 {
            regions.push(Region {
                start: pair[0],
                end: pair[1],
                start_closed: pair[0].is_finite(),
                end_closed: false,
                y: y1,
            });
        }
    }
    Ok(())
}

fn probes(start: f64, end: f64) -> (f64, f64) {
    if start.is_finite() && end.is_finite() {
        let width = end - start;
        (start + width * 0.25, start + width * 0.75)
    } else if start.is_finite() {
        (start, start + 1.0)
    } else if end.is_finite() {
        (end - 1.0, end - 2.0)
    } else {
        (0.0, 1.0)
    }
}

fn tabulated_linear_regions(
    points: &[(f64, f64)],
    clamp: &Clamp,
    regions: &mut Vec<Region>,
) -> ContractResult<()> {
    regions.push(Region {
        start: f64::NEG_INFINITY,
        end: points[0].0,
        start_closed: false,
        end_closed: false,
        y: apply_clamp(clamp, points[0].1),
    });
    for pair in points.windows(2) {
        let (x0, y0) = pair[0];
        let (x1, y1) = pair[1];
        let mut cuts = vec![x0, x1];
        if y0 != y1 {
            for limit in [clamp.minimum, clamp.maximum].into_iter().flatten() {
                let low = y0.min(y1);
                let high = y0.max(y1);
                if limit > low && limit < high {
                    cuts.push(x0 + (x1 - x0) * (limit - y0) / (y1 - y0));
                }
            }
        }
        cuts.sort_by(f64::total_cmp);
        cuts.dedup_by(|left, right| *left == *right);
        for window in cuts.windows(2) {
            let mid = 0.5 * (window[0] + window[1]);
            let y_mid = apply_clamp(clamp, lerp(x0, y0, x1, y1, mid));
            let y_left = apply_clamp(clamp, lerp(x0, y0, x1, y1, window[0]));
            if y_left == y_mid {
                regions.push(Region {
                    start: window[0],
                    end: window[1],
                    start_closed: true,
                    end_closed: window[1] == x1,
                    y: y_left,
                });
            }
        }
    }
    let last = points.last().expect("tabulated curve has points");
    regions.push(Region {
        start: last.0,
        end: f64::INFINITY,
        start_closed: false,
        end_closed: false,
        y: apply_clamp(clamp, last.1),
    });
    Ok(())
}

fn step_regions(
    points: &[(f64, f64)],
    outside: OutsideDomain,
    clamp: &Clamp,
    step_left: bool,
    regions: &mut Vec<Region>,
) -> ContractResult<()> {
    let first = points[0];
    let last = *points.last().expect("step curve has points");
    if outside == OutsideDomain::Clamp {
        regions.push(Region {
            start: f64::NEG_INFINITY,
            end: first.0,
            start_closed: false,
            end_closed: false,
            y: apply_clamp(clamp, first.1),
        });
    }
    for pair in points.windows(2) {
        let y = if step_left { pair[0].1 } else { pair[1].1 };
        regions.push(Region {
            start: pair[0].0,
            end: pair[1].0,
            start_closed: step_left || pair[0].0 == first.0,
            end_closed: !step_left,
            y: apply_clamp(clamp, y),
        });
    }
    if outside == OutsideDomain::Clamp {
        regions.push(Region {
            start: last.0,
            end: f64::INFINITY,
            start_closed: false,
            end_closed: false,
            y: apply_clamp(clamp, last.1),
        });
    } else if step_left {
        regions.push(Region {
            start: last.0,
            end: last.0,
            start_closed: true,
            end_closed: true,
            y: apply_clamp(clamp, last.1),
        });
    }
    Ok(())
}

fn binned_regions(
    edges: &[f64],
    values: &[f64],
    outside: OutsideDomain,
    clamp: &Clamp,
    regions: &mut Vec<Region>,
) -> ContractResult<()> {
    let first = edges[0];
    if values.len() == 1 && outside == OutsideDomain::Clamp {
        regions.push(Region {
            start: f64::NEG_INFINITY,
            end: f64::INFINITY,
            start_closed: false,
            end_closed: false,
            y: apply_clamp(clamp, values[0]),
        });
        return Ok(());
    }
    if outside == OutsideDomain::Clamp {
        regions.push(Region {
            start: f64::NEG_INFINITY,
            end: edges[1],
            start_closed: false,
            end_closed: false,
            y: apply_clamp(clamp, values[0]),
        });
    } else {
        regions.push(Region {
            start: first,
            end: edges[1],
            start_closed: true,
            end_closed: false,
            y: apply_clamp(clamp, values[0]),
        });
    }
    for index in 1..values.len() {
        let end = edges[index + 1];
        let last_bin = index + 1 == values.len();
        regions.push(Region {
            start: edges[index],
            end: if last_bin && outside == OutsideDomain::Clamp {
                f64::INFINITY
            } else {
                end
            },
            start_closed: true,
            end_closed: last_bin && outside == OutsideDomain::Error,
            y: apply_clamp(clamp, values[index]),
        });
    }
    Ok(())
}

fn lower_threshold(
    model: &Model,
    clamp: &Clamp,
    source: &ImpactSource,
) -> ContractResult<Option<(f64, f64)>> {
    let Model::Sigmoid {
        zero_below: Some(threshold),
        ..
    } = model
    else {
        return Ok(None);
    };
    let floor = kernel::evaluate(model, clamp, threshold - 1.0)?;
    let at_threshold = kernel::evaluate(model, clamp, *threshold)?;
    if floor == at_threshold {
        return Ok(None);
    }
    Ok(Some((source.cdf_below(*threshold), floor)))
}

fn lerp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    if x1 == x0 {
        y0
    } else {
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }
}

fn apply_clamp(clamp: &Clamp, mut value: f64) -> f64 {
    if let Some(minimum) = clamp.minimum {
        value = value.max(minimum);
    }
    if let Some(maximum) = clamp.maximum {
        value = value.min(maximum);
    }
    if value == 0.0 { 0.0 } else { value }
}

fn limit_at_infinity(model: &Model, clamp: &Clamp, positive: bool) -> f64 {
    let value = match model {
        Model::Linear { slope, intercept } => {
            if *slope == 0.0 {
                *intercept
            } else if (*slope > 0.0) == positive {
                return clamp.maximum.unwrap_or(f64::INFINITY);
            } else {
                return clamp.minimum.unwrap_or(f64::NEG_INFINITY);
            }
        }
        Model::Sigmoid {
            steepness,
            offset,
            amplitude,
            origin_adjustment,
            zero_below,
            midpoint,
        } => {
            if !positive && zero_below.is_some() {
                0.0
            } else {
                let mut unit = if *steepness == 0.0 {
                    0.5
                } else if (*steepness > 0.0) == positive {
                    1.0
                } else {
                    0.0
                };
                if *origin_adjustment == super::definition::OriginAdjustment::SubtractAtZero {
                    let at_zero = if *steepness == 0.0 {
                        0.5
                    } else {
                        kernel_logistic(*steepness * (0.0 - midpoint))
                    };
                    unit -= at_zero;
                }
                offset + amplitude * unit
            }
        }
        Model::Tabulated { points, .. } => {
            if positive {
                points.last().expect("points").1
            } else {
                points[0].1
            }
        }
        Model::Binned { values, .. } => {
            if positive {
                *values.last().expect("values")
            } else {
                values[0]
            }
        }
        Model::Cooling { cooling_cost, .. } => {
            if !positive || *cooling_cost == 0.0 {
                return apply_clamp(clamp, 0.0);
            }
            return clamp.maximum.unwrap_or(f64::INFINITY);
        }
        Model::Spei {
            water_consumption_asset,
            original_water_unit_cost,
        } => {
            if positive {
                original_water_unit_cost * water_consumption_asset / 10_000_000.0
            } else {
                original_water_unit_cost * 11.0 * water_consumption_asset / 10_000_000.0
            }
        }
        Model::Crop { weights, index, .. } => {
            let mut finite = 0.0;
            let mut negative = false;
            for crop in 0..3 {
                if weights[crop] == 0.0 || index[crop] == 0.0 {
                    continue;
                }
                let grows = if positive {
                    index[crop] > 0.0
                } else {
                    index[crop] < 0.0
                };
                if grows {
                    negative = true;
                } else {
                    finite += weights[crop];
                }
            }
            if negative {
                return clamp.minimum.unwrap_or(f64::NEG_INFINITY);
            }
            finite
        }
    };
    apply_clamp(clamp, value)
}

fn kernel_logistic(value: f64) -> f64 {
    if value >= 0.0 {
        let exponential = (-value).exp();
        1.0 / (1.0 + exponential)
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}
