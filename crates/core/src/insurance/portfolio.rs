use super::dependence::{CorrelationFactor, factor_perils};
use super::sampling::{Draws, SeededDraws};
use super::simulation::{
    AnnualLossSimulation, AnnualLossSpec, InsuredInterest, SimulationOptions, book_value,
    check_budget, invalid, ordered_interests, reserve_trials, spatial_factor,
};
use crate::Result;
use crate::numerical::CompensatedSum;

struct PerilSimulation<'a> {
    name: &'a str,
    book: Vec<&'a InsuredInterest>,
    draws: SeededDraws,
}

/// Simulate separate peril coverages on the same fixed book of insured interests.
///
/// `cross_peril_correlation` has one row and column per supplied peril, in
/// that order. If omitted, primary streams are independent across perils.
/// The cross-peril and common spatial latent correlations are separable.
/// Deductible, limit, and share apply before summing peril losses per year.
/// The TIV denominator counts each interest once. Optional component losses
/// follow the sorted peril names and reconcile to each annual total.
pub fn simulate_separate_peril_losses(
    perils: &[(String, Vec<InsuredInterest>)],
    cross_peril_correlation: Option<&[Vec<f64>]>,
    spec: &AnnualLossSpec,
    options: &SimulationOptions,
) -> Result<AnnualLossSimulation> {
    spec.validate()?;
    if perils.len() < 2 {
        return Err(invalid(
            "separate-peril simulation requires at least two perils",
        ));
    }
    let book_size = perils[0].1.len();
    if book_size == 0 {
        return Err(invalid(
            "separate-peril simulation requires at least one insured interest",
        ));
    }
    if let Some((name, _)) = perils.iter().find(|(_, book)| book.len() != book_size) {
        return Err(invalid(format!(
            "peril {name:?} must cover the same insured interests"
        )));
    }
    check_budget(
        book_size,
        perils.len(),
        cross_peril_correlation.is_some(),
        spec,
        options,
    )?;
    let mut order = (0..perils.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|&a, &b| perils[a].0.cmp(&perils[b].0));
    if perils
        .iter()
        .any(|(name, _)| name.is_empty() || name.trim() != name)
    {
        return Err(invalid(
            "peril names must be nonempty without boundary whitespace",
        ));
    }
    if let Some(pair) = order
        .windows(2)
        .find(|pair| perils[pair[0]].0 == perils[pair[1]].0)
    {
        return Err(invalid(format!(
            "duplicate peril name {:?}",
            perils[pair[0]].0
        )));
    }
    let mut simulations = Vec::with_capacity(perils.len());
    for &index in &order {
        let (name, interests) = &perils[index];
        let book = ordered_interests(interests)?;
        if let Some(PerilSimulation { book: baseline, .. }) = simulations.first() {
            verify_book(baseline, &book, name)?;
        }
        simulations.push(PerilSimulation {
            name,
            book,
            draws: SeededDraws::for_peril(spec.seed, options.secondary_stream, name),
        });
    }

    let book = &simulations[0].book;
    let total_insured_value = book_value(book)?;
    let (spatial, correlation_repair) = spatial_factor(book, spec)?;
    let across_perils = cross_peril_correlation
        .map_or(Ok(CorrelationFactor::Independent), |matrix| {
            factor_perils(matrix, &order)
        })?;
    let mut totals = reserve_trials(spec.trials)?;
    let mut components = options
        .include_components
        .then(|| {
            simulations
                .iter()
                .map(|peril| Ok((peril.name.to_owned(), reserve_trials(spec.trials)?)))
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;

    let mut independent = vec![vec![0.0; book_size]; simulations.len()];
    let mut peril_innovations = vec![0.0; book_size];
    let mut correlated = vec![0.0; book_size];
    for first in (0..spec.trials).step_by(options.block_size) {
        let block = (spec.trials - first).min(options.block_size);
        for _ in 0..block {
            for (source, draws) in simulations.iter_mut().zip(&mut independent) {
                draws.fill_with(|| source.draws.standard_normal());
            }
            let mut annual = CompensatedSum::default();
            for (row, peril) in simulations.iter_mut().enumerate() {
                let innovations = across_perils.mix_row(&independent, row, &mut peril_innovations);
                spatial.apply(innovations, &mut correlated);
                let mut peril_annual = CompensatedSum::default();
                for (interest, &normal) in peril.book.iter().zip(&correlated) {
                    peril_annual.add(interest.simulate(
                        normal,
                        spec.damage_cv,
                        &mut peril.draws,
                    )?);
                }
                let loss = peril_annual.total();
                if !loss.is_finite() {
                    return Err(invalid(
                        "annual insured peril loss exceeds the representable range",
                    ));
                }
                annual.add(loss);
                if let Some(components) = &mut components {
                    components[row].1.push(loss);
                }
            }
            let loss = annual.total();
            if !loss.is_finite() {
                return Err(invalid(
                    "annual insured loss exceeds the representable range",
                ));
            }
            totals.push(loss);
        }
    }
    Ok(AnnualLossSimulation {
        totals,
        components,
        total_insured_value,
        correlation_repair,
    })
}

fn verify_book(
    baseline: &[&InsuredInterest],
    book: &[&InsuredInterest],
    peril: &str,
) -> Result<()> {
    for (before, after) in baseline.iter().zip(book) {
        if before.id != after.id {
            return Err(invalid(format!(
                "peril {peril:?} must cover the same insured interests: expected {:?}, found {:?}",
                before.id, after.id,
            )));
        }
        if (
            before.longitude,
            before.latitude,
            before.total_insured_value,
        ) != (after.longitude, after.latitude, after.total_insured_value)
        {
            return Err(invalid(format!(
                "peril {peril:?}, interest {:?} must use the same WGS84 location and total insured value",
                before.id,
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::insurance::{DamageCurve, PolicyTerms, simulate_annual_losses};

    fn interest(id: &str, longitude: f64, ratios: [f64; 2]) -> InsuredInterest {
        InsuredInterest {
            id: id.to_owned(),
            longitude,
            latitude: 43.0,
            total_insured_value: 100.0,
            terms: PolicyTerms::new(0.0, None, 1.0).unwrap(),
            damage_curve: DamageCurve::new(vec![0.0, 1.0], ratios.to_vec()).unwrap(),
        }
    }

    fn peril_book() -> Vec<InsuredInterest> {
        vec![
            interest("north", 24.0, [0.0, 1.0]),
            interest("south", 24.2, [0.1, 0.8]),
        ]
    }

    fn correlation(first: &[f64], second: &[f64]) -> f64 {
        let size = first.len() as f64;
        let means = (
            first.iter().sum::<f64>() / size,
            second.iter().sum::<f64>() / size,
        );
        let (cross, first_square, second_square) = first.iter().zip(second).fold(
            (0.0, 0.0, 0.0),
            |(cross, first_sq, second_sq), (x, y)| {
                let (a, b) = (x - means.0, y - means.1);
                (cross + a * b, first_sq + a * a, second_sq + b * b)
            },
        );
        cross / (first_square * second_square).sqrt()
    }

    #[test]
    fn primary_correlation_and_independent_secondary_draws_have_expected_covariance() {
        let options = SimulationOptions {
            include_components: true,
            ..SimulationOptions::default()
        };
        let spec = AnnualLossSpec {
            damage_cv: 0.0,
            ..AnnualLossSpec::default()
        };
        let books = |ratios| {
            ["flood", "wind"].map(|name| (name.to_owned(), vec![interest("site", 24.0, ratios)]))
        };
        let perils = books([0.0, 1.0]);
        for latent in [0.0_f64, 0.8, -0.8] {
            let matrix = [vec![1.0, latent], vec![latent, 1.0]];
            let result = simulate_separate_peril_losses(
                &perils,
                (latent != 0.0).then_some(&matrix[..]),
                &spec,
                &options,
            )
            .unwrap();
            let components = result.components.unwrap();
            let actual = correlation(&components[0].1, &components[1].1);
            // The linear curves return uniform ranks. Their Pearson correlation
            // is Spearman's rho of the bivariate latent Gaussian.
            let expected = 6.0 * (latent / 2.0).asin() / std::f64::consts::PI;
            assert!(
                (actual - expected).abs() < 0.03,
                "sampled={actual}, expected={expected}"
            );
        }
        // Each peril pays at a different interest. These co-located interests
        // have latent covariance equal to the product of the two factors.
        let across_locations = [
            (
                "flood".to_owned(),
                vec![
                    interest("north", 24.0, [0.0, 1.0]),
                    interest("south", 24.0, [0.0, 0.0]),
                ],
            ),
            (
                "wind".to_owned(),
                vec![
                    interest("north", 24.0, [0.0, 0.0]),
                    interest("south", 24.0, [0.0, 1.0]),
                ],
            ),
        ];
        let matrix = [vec![1.0, 0.8], vec![0.8, 1.0]];
        let spatial =
            simulate_separate_peril_losses(&across_locations, Some(&matrix), &spec, &options)
                .unwrap()
                .components
                .unwrap();
        let expected = 6.0 * (0.8 * spec.peak_correlation / 2.0).asin() / std::f64::consts::PI;
        assert!((correlation(&spatial[0].1, &spatial[1].1) - expected).abs() < 0.03);

        let result = simulate_separate_peril_losses(
            &books([0.5, 0.5]),
            Some(&matrix),
            &AnnualLossSpec::default(),
            &options,
        )
        .unwrap();
        let parts = result.components.unwrap();
        assert!(correlation(&parts[0].1, &parts[1].1).abs() < 0.03);
    }

    #[test]
    fn reorders_matrix_axes_and_keeps_independent_peril_streams_stable() {
        let perils = ["flood", "hail", "wind"].map(|name| (name.to_owned(), peril_book()));
        let matrix = [
            vec![1.0, 0.5, 0.1],
            vec![0.5, 1.0, -0.2],
            vec![0.1, -0.2, 1.0],
        ];
        let spec = AnnualLossSpec {
            trials: 150,
            ..AnnualLossSpec::default()
        };
        let options = SimulationOptions {
            include_components: true,
            ..SimulationOptions::default()
        };
        let baseline =
            simulate_separate_peril_losses(&perils, Some(&matrix), &spec, &options).unwrap();
        let mut reversed = perils.clone();
        reversed.reverse();
        for (_, book) in &mut reversed {
            book.reverse();
        }
        let reordered = matrix
            .iter()
            .rev()
            .map(|row| row.iter().rev().copied().collect())
            .collect::<Vec<_>>();
        let replay = simulate_separate_peril_losses(
            &reversed,
            Some(&reordered),
            &spec,
            &SimulationOptions {
                block_size: 3,
                ..options
            },
        )
        .unwrap();
        assert_eq!(baseline, replay);

        let all = simulate_separate_peril_losses(&perils, None, &spec, &options)
            .unwrap()
            .components
            .unwrap();
        let subset = simulate_separate_peril_losses(&perils[1..], None, &spec, &options)
            .unwrap()
            .components
            .unwrap();
        assert_eq!(&all[1..], subset);
        let different_secondary = SimulationOptions {
            secondary_stream: 1,
            ..options
        };
        let changed =
            simulate_separate_peril_losses(&perils, Some(&matrix), &spec, &different_secondary)
                .unwrap();
        assert_ne!(baseline.totals, changed.totals);
        let deterministic = AnnualLossSpec {
            damage_cv: 0.0,
            ..spec
        };
        assert_eq!(
            simulate_separate_peril_losses(&perils, Some(&matrix), &deterministic, &options)
                .unwrap(),
            simulate_separate_peril_losses(
                &perils,
                Some(&matrix),
                &deterministic,
                &different_secondary
            )
            .unwrap(),
        );
    }

    #[test]
    fn rejects_different_books_and_budgets_aggregate_work_and_memory() {
        let mut perils = ["flood", "wind"].map(|name| (name.to_owned(), peril_book()));
        let spec = AnnualLossSpec {
            trials: 100,
            ..AnnualLossSpec::default()
        };
        let defaults = SimulationOptions::default();
        for (options, name) in [
            (
                SimulationOptions {
                    memory_limit: 2500,
                    ..defaults
                },
                "memory_limit",
            ),
            (
                SimulationOptions {
                    work_limit: 1200,
                    ..defaults
                },
                "work_limit",
            ),
        ] {
            assert!(simulate_annual_losses(&perils[0].1, &spec, &options).is_ok());
            assert!(
                simulate_separate_peril_losses(&perils, None, &spec, &options)
                    .unwrap_err()
                    .to_string()
                    .contains(name)
            );
        }
        perils[1].1[0].total_insured_value += 1.0;
        assert!(
            simulate_separate_peril_losses(&perils, None, &spec, &defaults)
                .unwrap_err()
                .to_string()
                .contains("same WGS84 location and total insured value")
        );
        perils[1].1.pop();
        assert!(
            simulate_separate_peril_losses(&perils, None, &spec, &defaults)
                .unwrap_err()
                .to_string()
                .contains("same insured interests")
        );
    }
}
