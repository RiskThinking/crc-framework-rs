use std::collections::BTreeMap;

use crate::{
    distribution::Distribution,
    error::{CrcError, Result},
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScenarioMetadata {
    pub cell: Option<u64>,
    pub factor: Option<String>,
    pub pathway: Option<String>,
    pub horizon: Option<i32>,
    pub country: Option<String>,
    pub continent: Option<String>,
    pub building_type: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct DistributionStatistics {
    pub mean: f64,
    pub standard_deviation: f64,
    pub minimum: f64,
    pub maximum: f64,
    pub cvar_95: f64,
}

#[derive(Debug, Clone)]
pub struct Microscore {
    pub probability: f64,
    pub exposure: f64,
    pub impact: Option<f64>,
    pub probability_below: f64,
    pub probability_above: f64,
    pub impact_probability_below: Option<f64>,
    pub impact_probability_above: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct BinaryOutcome {
    pub factor: String,
    pub downside_probability: f64,
    pub downside_impact: f64,
    pub upside_probability: f64,
    pub upside_impact: f64,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct MicroscoreSuite {
    pub metadata: ScenarioMetadata,
    pub scores: Vec<Microscore>,
    pub exposure_statistics: DistributionStatistics,
    pub impact_statistics: Option<DistributionStatistics>,
}

impl MicroscoreSuite {
    pub fn at(&self, probability: f64) -> Result<BinaryOutcome> {
        let score = self
            .scores
            .iter()
            .find(|score| (score.probability - probability).abs() < 1.0e-12)
            .ok_or_else(|| {
                CrcError::InvalidInput(format!(
                    "probability {probability} was not requested when generating the suite"
                ))
            })?;
        let downside_impact = score.impact.ok_or_else(|| {
            CrcError::MissingParameter(
                "an impact distribution is required for risk outcomes".into(),
            )
        })?;
        let downside_probability = score
            .impact_probability_above
            .unwrap_or(score.probability_above);
        Ok(BinaryOutcome {
            factor: self
                .metadata
                .factor
                .clone()
                .unwrap_or_else(|| "unknown".into()),
            downside_probability,
            downside_impact,
            upside_probability: 1.0 - downside_probability,
            upside_impact: 0.0,
            weight: 1.0,
        })
    }

    pub fn by_probability(&self) -> BTreeMap<String, &Microscore> {
        self.scores
            .iter()
            .map(|score| (format!("{:.12}", score.probability), score))
            .collect()
    }
}

pub fn generate_microscores(
    exposure: &dyn Distribution,
    impact: Option<&dyn Distribution>,
    probabilities: &[f64],
    metadata: ScenarioMetadata,
) -> Result<MicroscoreSuite> {
    validate_probabilities(probabilities)?;
    let mut scores = Vec::with_capacity(probabilities.len());
    for &probability in probabilities {
        let exposure_value = exposure.ppf(probability)?;
        let impact_value = impact
            .map(|distribution| distribution.ppf(probability))
            .transpose()?;
        let impact_probability_below = impact_value.map(|_| probability);
        scores.push(Microscore {
            probability,
            exposure: exposure_value,
            impact: impact_value,
            probability_below: probability,
            probability_above: 1.0 - probability,
            impact_probability_below,
            impact_probability_above: impact_probability_below.map(|value| 1.0 - value),
        });
    }
    Ok(MicroscoreSuite {
        metadata,
        scores,
        exposure_statistics: calculate_statistics(exposure)?,
        impact_statistics: impact.map(calculate_statistics).transpose()?,
    })
}

pub fn calculate_statistics(distribution: &dyn Distribution) -> Result<DistributionStatistics> {
    const COUNT: usize = 1000;
    let (minimum_probability, maximum_probability) = distribution.probability_support();
    let width = maximum_probability - minimum_probability;
    let probabilities: Vec<f64> = (0..COUNT)
        .map(|index| minimum_probability + width * (index as f64 + 0.5) / COUNT as f64)
        .collect();
    let values = distribution.quantiles(&probabilities)?;
    let mean = values.iter().sum::<f64>() / COUNT as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / COUNT as f64;
    let tail_threshold = minimum_probability + 0.95 * width;
    let tail_start = probabilities.partition_point(|probability| *probability < tail_threshold);
    let cvar_95 =
        values[tail_start..].iter().sum::<f64>() / (COUNT.saturating_sub(tail_start)) as f64;
    Ok(DistributionStatistics {
        mean,
        standard_deviation: variance.sqrt(),
        minimum: values[0],
        maximum: *values.last().expect("fixed non-empty grid"),
        cvar_95,
    })
}

fn validate_probabilities(probabilities: &[f64]) -> Result<()> {
    if probabilities.is_empty()
        || probabilities
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value >= 1.0)
    {
        return Err(CrcError::InvalidInput(
            "at least one finite probability strictly between zero and one is required".into(),
        ));
    }
    if probabilities.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(CrcError::InvalidInput(
            "probabilities must be strictly increasing and unique".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::distribution::EmpiricalDistribution;

    use super::*;

    #[test]
    fn metrics_use_only_canonical_probability() {
        let distribution = EmpiricalDistribution::new(vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let suite = generate_microscores(
            &distribution,
            Some(&distribution),
            &[0.5, 0.95],
            ScenarioMetadata {
                factor: Some("test".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(suite.scores.len(), 2);
        assert!((suite.at(0.95).unwrap().downside_probability - 0.05).abs() < 1.0e-12);
    }
}
