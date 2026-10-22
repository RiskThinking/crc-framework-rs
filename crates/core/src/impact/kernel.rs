use super::definition::{Clamp, Interpolation, Model, OriginAdjustment, OutsideDomain};
use super::error::{ContractResult, fail};

const CROPS: usize = 3;

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
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

fn table_response(
    xs: &[f64],
    ys: &[f64],
    outside: OutsideDomain,
    binned: bool,
    interpolation: Interpolation,
    value: f64,
) -> ContractResult<f64> {
    if value < xs[0] || value > *xs.last().expect("table has endpoints") {
        if outside == OutsideDomain::Error {
            return Err(fail(
                "out_of_domain",
                "",
                "exposure is outside the table domain",
            ));
        }
        return Ok(if value < xs[0] {
            ys[0]
        } else {
            *ys.last().expect("table has values")
        });
    }
    if binned {
        for (index, response) in ys.iter().enumerate() {
            if value < xs[index + 1] {
                return Ok(*response);
            }
        }
        return Ok(*ys.last().expect("binned model has a final value"));
    }
    if let Some(index) = xs.iter().position(|knot| *knot == value) {
        return Ok(ys[index]);
    }
    let upper = xs.partition_point(|knot| *knot < value);
    let lower = upper - 1;
    Ok(match interpolation {
        Interpolation::StepLeft => ys[lower],
        Interpolation::StepRight => ys[upper],
        Interpolation::Linear => {
            let weight = (value - xs[lower]) / (xs[upper] - xs[lower]);
            ys[lower] + weight * (ys[upper] - ys[lower])
        }
    })
}

pub(crate) fn raw_response(model: &Model, value: f64) -> ContractResult<f64> {
    match model {
        Model::Linear { slope, intercept } => Ok(slope * value + intercept),
        Model::Sigmoid {
            midpoint,
            steepness,
            offset,
            amplitude,
            origin_adjustment,
            zero_below,
        } => {
            if zero_below.is_some_and(|threshold| value < threshold) {
                return Ok(0.0);
            }
            if *amplitude == 0.0 {
                return Ok(*offset);
            }
            let mut unit = if *steepness == 0.0 {
                0.5
            } else {
                logistic(steepness * (value - midpoint))
            };
            if *origin_adjustment == OriginAdjustment::SubtractAtZero {
                unit -= logistic(steepness * (0.0 - midpoint));
            }
            Ok(offset + amplitude * unit)
        }
        Model::Tabulated {
            points,
            interpolation,
            outside_domain,
        } => {
            let xs = points.iter().map(|point| point.0).collect::<Vec<_>>();
            let ys = points.iter().map(|point| point.1).collect::<Vec<_>>();
            table_response(&xs, &ys, *outside_domain, false, *interpolation, value)
        }
        Model::Binned {
            edges,
            values,
            outside_domain,
        } => table_response(
            edges,
            values,
            *outside_domain,
            true,
            Interpolation::Linear,
            value,
        ),
        Model::Spei {
            water_consumption_asset,
            original_water_unit_cost,
        } => {
            let midpoint = -1.0 + 199.0_f64.ln() / -3.0;
            let drought = logistic(-3.0 * (value - midpoint));
            let mut cost = *original_water_unit_cost;
            if value < 0.0 {
                cost *= 1.0 + 10.0 * drought;
            }
            Ok(cost * water_consumption_asset / 10_000_000.0)
        }
        Model::Cooling {
            cooling_cost,
            historic_mean,
        } => {
            if value <= 1.0 {
                Ok(0.0)
            } else {
                Ok(cooling_cost * value / (historic_mean * 10_000_000.0))
            }
        }
        Model::Crop {
            weights,
            country,
            yearly,
            index,
            historic_mean,
        } => {
            let mut total = 0.0;
            for crop in 0..CROPS {
                let base = country[crop] + yearly[crop] * 2010.0;
                let historical = (base + index[crop] * historic_mean).exp();
                let future = (base + index[crop] * value).exp();
                if historical > 0.0 {
                    total += weights[crop] * -(future - historical) / historical;
                }
            }
            Ok(total)
        }
    }
}

pub(crate) fn evaluate(model: &Model, clamp: &Clamp, value: f64) -> ContractResult<f64> {
    if !value.is_finite() {
        return Err(fail(
            "non_finite_exposure",
            "",
            "exposure values must be finite",
        ));
    }
    let result = raw_response(model, value)?;
    if !result.is_finite() {
        return Err(fail(
            "non_finite_result",
            "",
            "impact evaluation produced a non-finite response",
        ));
    }
    Ok(apply_clamp(clamp, result))
}

fn sequence_class(values: &[f64]) -> &'static str {
    let mut non_positive = true;
    let mut non_negative = true;
    let mut zero = true;
    for pair in values.windows(2) {
        let step = pair[1] - pair[0];
        non_positive &= step <= 0.0;
        non_negative &= step >= 0.0;
        zero &= step == 0.0;
    }
    if zero {
        "constant"
    } else if non_negative {
        "non_decreasing"
    } else if non_positive {
        "non_increasing"
    } else {
        "non_monotone"
    }
}

fn jump(base: &str, before: f64, at: f64) -> &'static str {
    let rising = at > before;
    let falling = at < before;
    match base {
        "constant" if at == before => "constant",
        "constant" if rising => "non_decreasing",
        "constant" => "non_increasing",
        "non_decreasing" if falling => "non_monotone",
        "non_decreasing" => "non_decreasing",
        _ if rising => "non_monotone",
        _ => "non_increasing",
    }
}

fn clamped(direction: &'static str, clamp: &Clamp) -> &'static str {
    if clamp
        .minimum
        .is_some_and(|minimum| clamp.maximum == Some(minimum))
    {
        "constant"
    } else {
        direction
    }
}

pub(crate) fn monotonicity(model: &Model, clamp: &Clamp) -> &'static str {
    if clamp
        .minimum
        .is_some_and(|minimum| clamp.maximum == Some(minimum))
    {
        return "constant";
    }
    let direction = match model {
        Model::Linear { slope, .. } => {
            if *slope == 0.0 {
                "constant"
            } else if *slope > 0.0 {
                "non_decreasing"
            } else {
                "non_increasing"
            }
        }
        Model::Sigmoid {
            amplitude,
            steepness,
            zero_below,
            ..
        } => {
            let base = if *amplitude == 0.0 || *steepness == 0.0 {
                "constant"
            } else if *steepness > 0.0 {
                "non_decreasing"
            } else {
                "non_increasing"
            };
            match zero_below {
                None => base,
                Some(threshold) => {
                    let before = apply_clamp(clamp, 0.0);
                    let at = apply_clamp(
                        clamp,
                        raw_response(model, *threshold).expect("threshold is finite"),
                    );
                    jump(base, before, at)
                }
            }
        }
        Model::Tabulated { points, .. } => sequence_class(
            &points
                .iter()
                .map(|point| apply_clamp(clamp, point.1))
                .collect::<Vec<_>>(),
        ),
        Model::Binned { values, .. } => sequence_class(
            &values
                .iter()
                .map(|value| apply_clamp(clamp, *value))
                .collect::<Vec<_>>(),
        ),
        Model::Spei {
            water_consumption_asset,
            original_water_unit_cost,
        } => {
            if *original_water_unit_cost * *water_consumption_asset == 0.0 {
                "constant"
            } else {
                "non_increasing"
            }
        }
        Model::Cooling { cooling_cost, .. } => {
            if *cooling_cost == 0.0 {
                "constant"
            } else {
                "non_decreasing"
            }
        }
        Model::Crop { weights, index, .. } => {
            let mut positive = false;
            let mut negative = false;
            for crop in 0..CROPS {
                if weights[crop] > 0.0 {
                    positive |= index[crop] > 0.0;
                    negative |= index[crop] < 0.0;
                }
            }
            if !positive && !negative {
                "constant"
            } else if positive && !negative {
                "non_increasing"
            } else if negative && !positive {
                "non_decreasing"
            } else {
                "unknown"
            }
        }
    };
    if direction == "non_monotone" || direction == "unknown" {
        direction
    } else {
        clamped(direction, clamp)
    }
}
