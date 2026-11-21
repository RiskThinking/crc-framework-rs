use nalgebra::{
    DMatrix,
    linalg::{Cholesky, SymmetricEigen},
};

use crate::{CrcError, Result};

const EARTH_RADIUS_KM: f64 = 6_371.0;
const EIGENVALUE_FLOOR: f64 = 1e-8;

/// Diagnostics from repairing and renormalizing a latent correlation matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrelationRepair {
    /// Smallest eigenvalue of the original matrix.
    pub minimum_eigenvalue: f64,
    /// Largest absolute entry change after repairing the matrix.
    pub maximum_change: f64,
}

#[derive(Debug)]
pub(super) enum CorrelationFactor {
    Independent,
    Cholesky(DMatrix<f64>),
}

impl CorrelationFactor {
    pub(super) fn apply(&self, independent: &[f64], correlated: &mut [f64]) {
        match self {
            Self::Independent => correlated.copy_from_slice(independent),
            Self::Cholesky(factor) => {
                assert_eq!(correlated.len(), factor.nrows());
                assert_eq!(independent.len(), factor.ncols());
                correlated.fill(0.0);
                // Visit columns in order and accumulate contiguous row ranges.
                // Each output has the same reduction order for every trial.
                for (column, (values, &draw)) in factor.column_iter().zip(independent).enumerate() {
                    let values = &values.as_slice()[column..];
                    for (output, &coefficient) in correlated[column..].iter_mut().zip(values) {
                        *output = coefficient.mul_add(draw, *output);
                    }
                }
            }
        }
    }
}

pub(super) fn validate_dependence(
    peak_correlation: f64,
    decay_length_km: f64,
    repair_tolerance: Option<f64>,
) -> Result<()> {
    if !(0.0..=1.0).contains(&peak_correlation) {
        return Err(CrcError::InvalidInput(
            "peak_correlation must be finite and in [0, 1]".into(),
        ));
    }
    if !decay_length_km.is_finite() || decay_length_km <= 0.0 {
        return Err(CrcError::InvalidInput(
            "decay_length_km must be finite and positive".into(),
        ));
    }
    if repair_tolerance.is_some_and(|tolerance| !tolerance.is_finite() || tolerance < 0.0) {
        return Err(CrcError::InvalidInput(
            "repair_tolerance must be finite and nonnegative".into(),
        ));
    }
    Ok(())
}

/// Factor the exponential great-circle dependence for (longitude, latitude) pairs.
pub(super) fn factor_coordinates(
    coordinates: &[(f64, f64)],
    peak_correlation: f64,
    decay_length_km: f64,
    repair_tolerance: Option<f64>,
) -> Result<(CorrelationFactor, Option<CorrelationRepair>)> {
    validate_dependence(peak_correlation, decay_length_km, repair_tolerance)?;
    if coordinates.is_empty() {
        return Err(CrcError::InvalidInput(
            "dependence requires at least one insured interest".into(),
        ));
    }
    for (position, &(longitude, latitude)) in coordinates.iter().enumerate() {
        if !(-180.0..=180.0).contains(&longitude) || !(-90.0..=90.0).contains(&latitude) {
            return Err(CrcError::InvalidInput(format!(
                "interest {position} requires finite WGS84 longitude in [-180, 180] and latitude in [-90, 90]"
            )));
        }
    }
    let size = coordinates.len();
    if peak_correlation == 0.0 || size == 1 {
        return Ok((CorrelationFactor::Independent, None));
    }

    let mut correlation = DMatrix::identity(size, size);
    let mut requires_repair = false;
    for column in 0..size {
        for row in (column + 1)..size {
            let distance = great_circle_km(coordinates[row], coordinates[column]);
            let value = peak_correlation * (-distance / decay_length_km).exp();
            correlation[(row, column)] = value;
            correlation[(column, row)] = value;
            // A unit off-diagonal makes a principal minor singular, even if
            // its numerical Cholesky step could round to a positive pivot.
            requires_repair |= value == 1.0;
        }
    }
    factor_matrix(correlation, requires_repair, repair_tolerance)
}

fn great_circle_km((lon_a, lat_a): (f64, f64), (lon_b, lat_b): (f64, f64)) -> f64 {
    let lon_gap = (lon_b - lon_a).abs();
    let half_lon = (lon_gap.min(360.0 - lon_gap) / 2.0).to_radians();
    let half_lat = ((lat_b - lat_a) / 2.0).to_radians();
    // The sine of co-latitude is exactly zero at either pole.
    let cos_lat_a = (90.0 - lat_a.abs()).to_radians().sin();
    let cos_lat_b = (90.0 - lat_b.abs()).to_radians().sin();
    let haversine =
        (half_lat.sin().powi(2) + cos_lat_a * cos_lat_b * half_lon.sin().powi(2)).clamp(0.0, 1.0);
    2.0 * EARTH_RADIUS_KM * haversine.sqrt().atan2((1.0 - haversine).sqrt())
}

fn factor_matrix(
    correlation: DMatrix<f64>,
    requires_repair: bool,
    repair_tolerance: Option<f64>,
) -> Result<(CorrelationFactor, Option<CorrelationRepair>)> {
    if !requires_repair {
        if let Some(factor) = Cholesky::new(correlation.clone()) {
            return Ok((CorrelationFactor::Cholesky(factor.unpack_dirty()), None));
        }
    }
    let tolerance = repair_tolerance.ok_or_else(|| {
        CrcError::InvalidInput(
            "correlation matrix must be positive definite. Set repair_tolerance to permit eigenvalue-floor repair"
                .into(),
        )
    })?;

    let size = correlation.nrows();
    let iterations = 100usize.saturating_mul(size);
    let decomposition = SymmetricEigen::try_new(correlation.clone(), f64::EPSILON, iterations)
        .ok_or_else(|| CrcError::ConvergenceFailed {
            family: "correlation eigenvalue repair".into(),
            iterations,
        })?;
    if decomposition
        .eigenvalues
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(CrcError::InvalidInput(
            "correlation eigenvalue repair produced a nonfinite spectrum".into(),
        ));
    }
    let minimum_eigenvalue = decomposition.eigenvalues.min();
    let mut vectors = decomposition.eigenvectors;
    for (mut column, &value) in vectors
        .column_iter_mut()
        .zip(decomposition.eigenvalues.iter())
    {
        column.scale_mut(value.max(EIGENVALUE_FLOOR).sqrt());
    }
    // Reconstruct Q diag(floored eigenvalues) Qᵀ using in-place transpose.
    vectors.transpose_mut();
    let mut repaired = vectors.tr_mul(&vectors);
    let scaling = repaired.diagonal().map(|value| value.sqrt().recip());
    if scaling
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(CrcError::InvalidInput(
            "correlation eigenvalue repair produced an invalid diagonal".into(),
        ));
    }
    let mut maximum_change: f64 = 0.0;
    for column in 0..size {
        repaired[(column, column)] = 1.0;
        for row in (column + 1)..size {
            let value = repaired[(row, column)] * (scaling[row] * scaling[column]);
            maximum_change = maximum_change.max((value - correlation[(row, column)]).abs());
            repaired[(row, column)] = value;
            repaired[(column, row)] = value;
        }
    }
    if maximum_change > tolerance {
        return Err(CrcError::InvalidInput(format!(
            "correlation repair changed a matrix entry by {maximum_change:.3e}, exceeding repair_tolerance {tolerance:.3e}"
        )));
    }
    let factor = Cholesky::new(repaired).ok_or_else(|| {
        CrcError::InvalidInput(
            "correlation matrix remains singular or indefinite after eigenvalue repair".into(),
        )
    })?;
    Ok((
        CorrelationFactor::Cholesky(factor.unpack_dirty()),
        Some(CorrelationRepair {
            minimum_eigenvalue,
            maximum_change,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair_covariance(factor: &CorrelationFactor) -> (f64, f64, f64) {
        let mut left = [0.0; 2];
        let mut right = [0.0; 2];
        factor.apply(&[1.0, 0.0], &mut left);
        factor.apply(&[0.0, 1.0], &mut right);
        (
            left[0].powi(2) + right[0].powi(2),
            left[1].powi(2) + right[1].powi(2),
            left[0] * left[1] + right[0] * right[1],
        )
    }

    #[test]
    fn distance_and_latent_correlation_follow_the_reference_decay() {
        let longitude_pair = [(179.75, 0.0), (-179.75, 0.0)];
        let (factor, repair) = factor_coordinates(&longitude_pair, 0.4, 150.0, None).unwrap();
        let distance_km = EARTH_RADIUS_KM * 0.5_f64.to_radians();
        let expected = 0.4 * (-distance_km / 150.0).exp();
        let (left, right, covariance) = pair_covariance(&factor);
        assert!((covariance - expected).abs() < 1e-14);
        assert_eq!(left, 1.0);
        assert!((right - 1.0).abs() < 1e-14);
        assert!(repair.is_none());
        assert_eq!(great_circle_km((-180.0, 5.0), (180.0, 5.0)), 0.0);
        assert_eq!(great_circle_km((40.0, 90.0), (-60.0, 90.0)), 0.0);
    }

    #[test]
    fn colocation_at_unit_peak_requires_explicit_bounded_repair() {
        let coordinates = [(0.0, 0.0), (0.0, 0.0)];
        assert!(factor_coordinates(&coordinates, 1.0, 150.0, None).is_err());
        assert!(factor_coordinates(&coordinates, 1.0, 150.0, Some(1e-12)).is_err());
        let (factor, diagnostics) =
            factor_coordinates(&coordinates, 1.0, 150.0, Some(1e-6)).unwrap();
        let diagnostics = diagnostics.unwrap();
        let (left, right, covariance) = pair_covariance(&factor);
        assert!(diagnostics.minimum_eigenvalue.abs() < 1e-14);
        assert!((diagnostics.maximum_change - EIGENVALUE_FLOOR).abs() < 1e-14);
        assert!((covariance - (1.0 - EIGENVALUE_FLOOR)).abs() < 1e-14);
        assert_eq!(left, 1.0);
        assert!((right - 1.0).abs() < 1e-14);

        let (factor, repair) = factor_coordinates(&coordinates, 0.4, 150.0, None).unwrap();
        assert!(repair.is_none());
        assert_eq!(pair_covariance(&factor).2, 0.4);
    }

    #[test]
    fn zero_peak_uses_independent_draws_and_still_validates_coordinates() {
        let coordinates = [(0.0, 0.0), (0.0, 0.0)];
        let (factor, repaired) = factor_coordinates(&coordinates, 0.0, 150.0, None).unwrap();
        let mut correlated = [0.0; 2];
        factor.apply(&[0.4, -3.0], &mut correlated);
        assert_eq!(correlated, [0.4, -3.0]);
        assert!(repaired.is_none());
        assert!(factor_coordinates(&[(181.0, 0.0)], 0.0, 150.0, None).is_err());
        assert!(factor_coordinates(&coordinates, 0.0, 0.0, None).is_err());
    }

    #[test]
    fn an_indefinite_matrix_can_require_a_material_correction() {
        let matrix = DMatrix::from_row_slice(3, 3, &[1.0, 0.9, 0.9, 0.9, 1.0, 0.0, 0.9, 0.0, 1.0]);
        let error = factor_matrix(matrix, false, Some(0.01)).unwrap_err();
        assert!(error.to_string().contains("exceeding repair_tolerance"));
    }
}
