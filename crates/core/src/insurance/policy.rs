use crate::{CrcError, Result};

/// Resolved primary terms for one insured interest in its valuation currency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyTerms {
    deductible: f64,
    limit: Option<f64>,
    share: f64,
}

impl PolicyTerms {
    /// Construct terms from monetary amounts and an insurer share in `[0, 1]`.
    ///
    /// An omitted limit resolves to the total insured value of the interest.
    pub fn new(deductible: f64, limit: Option<f64>, share: f64) -> Result<Self> {
        check_amount(deductible, "deductible")?;
        if let Some(amount) = limit {
            check_amount(amount, "limit")?;
        }
        if !(0.0..=1.0).contains(&share) {
            return Err(CrcError::InvalidInput(
                "share must be finite and in [0, 1]".into(),
            ));
        }
        Ok(Self {
            deductible,
            limit,
            share,
        })
    }

    /// Apply these terms to one realized damage ratio.
    pub fn insured_loss(&self, total_insured_value: f64, damage_ratio: f64) -> Result<f64> {
        check_amount(total_insured_value, "total insured value")?;
        self.apply(total_insured_value, damage_ratio)
    }

    /// Apply the same terms to aligned ratios in their original year order.
    pub fn insured_losses(
        &self,
        total_insured_value: f64,
        damage_ratios: &[f64],
    ) -> Result<Vec<f64>> {
        check_amount(total_insured_value, "total insured value")?;
        damage_ratios
            .iter()
            .map(|&ratio| self.apply(total_insured_value, ratio))
            .collect()
    }

    fn apply(&self, total_insured_value: f64, damage_ratio: f64) -> Result<f64> {
        if !(0.0..=1.0).contains(&damage_ratio) {
            return Err(CrcError::InvalidInput(
                "damage ratio must be finite and in [0, 1]".into(),
            ));
        }
        let ground_up_loss = total_insured_value * damage_ratio;
        let limit = self.limit.unwrap_or(total_insured_value);
        let payout = self.share * (ground_up_loss - self.deductible).max(0.0).min(limit);
        Ok(payout.max(0.0))
    }
}

fn check_amount(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(CrcError::InvalidInput(format!(
            "{name} must be finite and nonnegative"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_deductible_limit_and_share_in_order() {
        let policy = PolicyTerms::new(5_000.0, Some(15_000.0), 0.8).unwrap();
        assert_eq!(policy.insured_loss(100_000.0, 0.26).unwrap(), 12_000.0);
        assert_eq!(
            policy
                .insured_losses(100_000.0, &[0.0, 0.05, 0.15, 0.26, 1.0])
                .unwrap(),
            [0.0, 0.0, 8_000.0, 12_000.0, 12_000.0]
        );
    }

    #[test]
    fn distinguishes_an_omitted_limit_from_an_explicit_zero() {
        let full = PolicyTerms::new(0.0, None, 1.0).unwrap();
        let no_limit = PolicyTerms::new(0.0, Some(0.0), 1.0).unwrap();
        let no_share = PolicyTerms::new(0.0, None, 0.0).unwrap();
        assert_eq!(full.insured_loss(100.0, 0.25).unwrap(), 25.0);
        assert_eq!(full.insured_loss(f64::MAX, 1.0).unwrap(), f64::MAX);
        assert_eq!(full.insured_loss(0.0, 1.0).unwrap(), 0.0);
        for terms in [no_limit, no_share] {
            assert_eq!(terms.insured_loss(100.0, 1.0).unwrap(), 0.0);
        }
    }

    #[test]
    fn validates_terms_and_all_ratios_even_when_the_payout_would_be_zero() {
        assert!(PolicyTerms::new(-1.0, None, 1.0).is_err());
        assert!(PolicyTerms::new(0.0, Some(f64::INFINITY), 1.0).is_err());
        assert!(PolicyTerms::new(0.0, None, f64::NAN).is_err());
        let policy = PolicyTerms::new(0.0, Some(0.0), 0.0).unwrap();
        assert!(policy.insured_losses(-1.0, &[]).is_err());
        assert!(policy.insured_losses(100.0, &[0.0, 1.1]).is_err());
    }
}
