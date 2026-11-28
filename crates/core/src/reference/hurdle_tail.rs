// Copyright (C) 2026 Riskthinking.AI
//
// `bounded_minimum` adapts SciPy's bounded Brent minimizer (`fminbound`).
// Upstream portions remain subject to the BSD 3-Clause terms reproduced in
// THIRD_PARTY_NOTICES.md. Modified by Riskthinking.AI in 2026.
//
// Riskthinking.AI's modifications are licensed under AGPL-3.0-or-later.
// SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

//! Least squares of a hurdle's continuous tail by variable projection.
//!
//! Tail knots `(u, v)` above an atom `a` follow `v = Q(F(a) + u S(a))` for the
//! base law `loc + scale * Z`. With `z = (a - loc) / scale` the knots are
//! `v - a = scale * (Q_Z(F_Z(z) + u S_Z(z)) - z)`, linear in the scale for a
//! fixed `(shape, z)`. The search moves in the shape and `s = logit F_Z(z)`,
//! and solves the scale in closed form.
//!
//! Two limits of that surface are searched on their own, because the optimum
//! often lies on one of them:
//!
//! - The atom at or below the start of the base's support (`F(a) -> 0`). The
//!   tail is then the base itself, location and scale are both linear, and the
//!   search is one-dimensional in the shape.
//! - The atom deep in the base's upper tail. The framework stores no law with
//!   `S(a) <= 1e-12`, and a base bounded above stops being representable
//!   before that once its scale exceeds the optimizer's limit. The search
//!   follows that wall as a function of the shape.
//!
//! The interior is seeded from a coarse grid and refined by Levenberg-Marquardt
//! from the two best separated grid points. A final pass re-profiles `s` at the
//! best shape, because the surface flattens towards the upper limit and a
//! valley can bend into the interior there.

use std::cell::RefCell;

use super::curve_fitting::{
    DistributionKind, MAXIMUM_QUANTILE_SCALE, QuantileOptimization, quantile_nelder_mead,
};
use crate::distribution::{
    Distribution as _, DistributionFamily, FittedDistribution, truncated_ppf,
};

const MINIMUM_SCALE: f64 = 1.0e-12;
/// Below this base cdf at the atom the truncation changes no knot.
const NEGLIGIBLE_ATOM_CDF: f64 = 1.0e-17;
/// Just above the smallest base survival at the atom the objective admits.
const SMALLEST_ATOM_SF: f64 = 1.0e-12 * (1.0 + 1.0e-6);
/// The stored law is evaluated as `loc + scale * Q_Z`, so a tail whose spread
/// vanishes next to `|z|` in double precision cannot be represented.
const REPRESENTABLE_SPREAD: f64 = 1.0e-13;
const SHAPE_GRID_POINTS: usize = 9;
const SHAPE_TOLERANCE: f64 = 1.0e-5;
const LOGIT_TOLERANCE: f64 = 1.0e-5;
const GUMBEL_LOGIT_POINTS: usize = 12;
/// A one-parameter search is the whole fit, with no Levenberg-Marquardt step
/// after it, so it resolves its coordinate as finely as the loss allows.
const LINE_TOLERANCE: f64 = 1.0e-10;
const STARTS: usize = 2;
const LM_ITERATIONS: usize = 40;
const LM_RELATIVE_TOLERANCE: f64 = 1.0e-10;

fn logit_floor() -> f64 {
    NEGLIGIBLE_ATOM_CDF.ln() - (-NEGLIGIBLE_ATOM_CDF).ln_1p()
}

fn logit_ceiling() -> f64 {
    (-SMALLEST_ATOM_SF).ln_1p() - SMALLEST_ATOM_SF.ln()
}

fn logit_grid() -> [f64; 9] {
    [
        logit_floor(),
        -12.0,
        -5.0,
        -1.5,
        0.0,
        1.5,
        5.0,
        12.0,
        logit_ceiling(),
    ]
}

/// Search coordinate range for the shape and its map to the family's shape.
/// `(low, high, shape_of)`: the coordinate range and its map to the shape.
type ShapeCoordinates = (f64, f64, fn(f64) -> f64);

fn shape_coordinates(family: DistributionFamily) -> Option<ShapeCoordinates> {
    match family {
        DistributionFamily::GenExtreme | DistributionFamily::GenPareto => {
            Some((-2.0, 2.0, |shape| shape))
        }
        DistributionFamily::WeibullMin | DistributionFamily::WeibullMax => {
            Some((0.05_f64.ln(), 50.0_f64.ln(), f64::exp))
        }
        _ => None,
    }
}

/// Fit the base of a hurdle's tail by least squares on its knots.
///
/// `probabilities` are conditional on the tail and every value lies above
/// `atom`. With `maximum`, the base's support must end at or below it; for a
/// fixed shape and atom position the least-squares scale is then capped in
/// closed form, so the bound costs no extra search. Only laws `accept` admits
/// are returned. Returns `None` for families the search does not cover (skew
/// normal) or when no admissible law is found; the simplex then handles the fit.
pub(crate) fn profile_truncated_quantile_points(
    kind: DistributionKind,
    probabilities: &[f64],
    values: &[f64],
    weights: &[f64],
    atom: f64,
    maximum: Option<f64>,
    accept: Option<&dyn Fn(&FittedDistribution) -> bool>,
) -> Option<QuantileOptimization> {
    let family = DistributionFamily::from_reference_kind(kind);
    let gumbel = matches!(
        family,
        DistributionFamily::GumbelRight | DistributionFamily::GumbelLeft
    );
    if !gumbel && shape_coordinates(family).is_none() {
        return None;
    }
    let mut tail = Tail::new(family, probabilities, values, weights, atom)?;
    // Aim just inside the bound so the stored endpoint survives rounding.
    tail.maximum = maximum.map(|maximum| maximum - 1.0e-12 * maximum.abs().max(1.0));
    tail.accept = accept;
    if gumbel {
        tail.untruncated(0.0);
        tail.grid_then_minimum(
            |tail, s| tail.truncated(0.0, s, None),
            logit_floor(),
            logit_ceiling(),
            GUMBEL_LOGIT_POINTS,
            LINE_TOLERANCE,
        );
    } else if family == DistributionFamily::GenPareto {
        // Threshold stability: the truncated tail of a generalized Pareto law
        // is again one with the same shape, so the support limit is the whole
        // search.
        tail.grid_then_minimum(
            |tail, x| tail.untruncated(x),
            -2.0,
            2.0,
            SHAPE_GRID_POINTS,
            LINE_TOLERANCE,
        );
    } else {
        tail.search_shape_and_logit();
    }
    tail.result()
}

struct Tail<'a> {
    family: DistributionFamily,
    probabilities: &'a [f64],
    values: &'a [f64],
    /// Weights normalized to sum to one, and their square roots.
    weights: Vec<f64>,
    root_weights: Vec<f64>,
    atom: f64,
    maximum: Option<f64>,
    accept: Option<&'a dyn Fn(&FittedDistribution) -> bool>,
    standardized: Vec<f64>,
    evaluations: usize,
    iterations: usize,
    /// Lowest loss seen with its law: `(loss, shape, loc, scale)`.
    best: Option<(f64, Option<f64>, f64, f64)>,
    /// Lowest truncated-branch loss with its coordinates: `(loss, x, s)`.
    best_truncated: Option<(f64, f64, f64)>,
}

impl<'a> Tail<'a> {
    fn new(
        family: DistributionFamily,
        probabilities: &'a [f64],
        values: &'a [f64],
        weights: &[f64],
        atom: f64,
    ) -> Option<Self> {
        let total = weights.iter().sum::<f64>();
        if !total.is_finite() || total <= 0.0 || values.is_empty() {
            return None;
        }
        let weights: Vec<f64> = weights.iter().map(|weight| weight / total).collect();
        Some(Self {
            family,
            probabilities,
            values,
            root_weights: weights.iter().map(|weight| weight.sqrt()).collect(),
            weights,
            atom,
            maximum: None,
            accept: None,
            standardized: vec![0.0; values.len()],
            evaluations: 0,
            iterations: 0,
            best: None,
            best_truncated: None,
        })
    }

    fn shape(&self, x: f64) -> Option<f64> {
        shape_coordinates(self.family).map(|(_, _, shape_of)| shape_of(x))
    }

    fn law(&self, x: f64) -> Option<FittedDistribution> {
        FittedDistribution::from_parameters(self.family, self.shape(x), 0.0, 1.0).ok()
    }

    /// Whether the law stored from these parameters is one the fit may return:
    /// mass above the atom as the objective requires, and admitted by `accept`.
    fn admits(&self, shape: Option<f64>, loc: f64, scale: f64) -> bool {
        let Ok(stored) = FittedDistribution::from_parameters(self.family, shape, loc, scale) else {
            return false;
        };
        // Near the upper limit, rounding `loc` can leave the stored law with
        // less mass above the atom than the search coordinates say.
        stored.cdf(self.atom).is_finite()
            && stored.sf(self.atom) > 1.0e-12
            && self.accept.is_none_or(|accept| accept(&stored))
    }

    /// The scale that puts the support's end on the bound, when the end sits
    /// `extent` standardized units above the atom.
    fn scale_cap(&self, extent: f64) -> Option<f64> {
        match self.maximum {
            None => Some(f64::INFINITY),
            Some(maximum) if extent.is_finite() => Some((maximum - self.atom) / extent),
            Some(_) => None,
        }
    }

    fn note(&mut self, loss: f64, shape: Option<f64>, loc: f64, scale: f64) {
        if self.best.is_none_or(|(current, ..)| loss < current) {
            self.best = Some((loss, shape, loc, scale));
        }
    }

    /// Weighted least-squares scale of `excess ~ scale * basis`, and its loss.
    fn project(&self, basis: &[f64]) -> Option<(f64, f64)> {
        let mut cross = 0.0;
        let mut square = 0.0;
        for ((&h, &value), &weight) in basis.iter().zip(self.values).zip(&self.weights) {
            cross += weight * h * (value - self.atom);
            square += weight * h * h;
        }
        let scale = cross / square;
        (square > 0.0 && scale.is_finite()).then_some((scale, square))
    }

    /// The atom inside the base's support at `F_Z(z) = sigmoid(s)`.
    ///
    /// With `residuals`, writes the weighted residuals for Levenberg-Marquardt.
    fn truncated(&mut self, x: f64, s: f64, residuals: Option<&mut [f64]>) -> f64 {
        self.evaluations += 1;
        if s > logit_ceiling() + 1.0e-9 {
            return f64::INFINITY;
        }
        let Some(law) = self.law(x) else {
            return f64::INFINITY;
        };
        let atom_cdf = 1.0 / (1.0 + (-s).exp());
        let atom_sf = 1.0 / (1.0 + s.exp());
        let z = if atom_cdf < 0.5 {
            law.ppf(atom_cdf)
        } else {
            law.isf(atom_sf)
        };
        let Ok(z) = z else {
            return f64::INFINITY;
        };
        if !z.is_finite() {
            return f64::INFINITY;
        }
        let mut spread: f64 = 0.0;
        for (out, &probability) in self.standardized.iter_mut().zip(self.probabilities) {
            let Ok(q) = truncated_ppf(&law, atom_cdf, atom_sf, probability) else {
                return f64::INFINITY;
            };
            *out = q - z;
            if !out.is_finite() {
                return f64::INFINITY;
            }
            spread = spread.max(out.abs());
        }
        if spread < REPRESENTABLE_SPREAD * z.abs().max(1.0) {
            return f64::INFINITY;
        }
        let Some((mut scale, _)) = self.project(&self.standardized) else {
            return f64::INFINITY;
        };
        if scale <= MINIMUM_SCALE {
            return f64::INFINITY;
        }
        // Past the scale limit the clamped law is still admissible, only worse.
        // The loss is a convex quadratic in the scale, so capping it at the
        // bound is the constrained optimum for these coordinates.
        let top = law.ppf(1.0).unwrap_or(f64::INFINITY);
        let Some(cap) = self.scale_cap(top - z) else {
            return f64::INFINITY;
        };
        scale = scale.min(MAXIMUM_QUANTILE_SCALE).min(cap);
        if scale <= MINIMUM_SCALE {
            return f64::INFINITY;
        }
        let mut loss = 0.0;
        let mut residuals = residuals;
        for index in 0..self.values.len() {
            let residual = self.root_weights[index]
                * (self.values[index] - self.atom - scale * self.standardized[index]);
            loss += residual * residual;
            if let Some(residuals) = residuals.as_deref_mut() {
                residuals[index] = residual;
            }
        }
        let loc = self.atom - scale * z;
        if !loss.is_finite() || !loc.is_finite() {
            return f64::INFINITY;
        }
        if !self.admits(self.shape(x), loc, scale) {
            return f64::INFINITY;
        }
        if self
            .best_truncated
            .is_none_or(|(current, ..)| loss < current)
        {
            self.best_truncated = Some((loss, x, s));
        }
        self.note(loss, self.shape(x), loc, scale);
        loss
    }

    /// The atom at or below the start of the base's support.
    ///
    /// Fits `v = loc + scale * Q_Z(u)` subject to `F(a) <= NEGLIGIBLE_ATOM_CDF`
    /// and, with a maximum, the support ending at or below it. The objective is
    /// a convex quadratic in `(loc, scale)` and both constraints are linear, so
    /// the constrained optimum is the best feasible solution among the free fit
    /// and the fits with each set of constraints held as equalities.
    fn untruncated(&mut self, x: f64) -> f64 {
        self.evaluations += 1;
        let Some(law) = self.law(x) else {
            return f64::INFINITY;
        };
        let Ok(lowest) = law.ppf(NEGLIGIBLE_ATOM_CDF) else {
            return f64::INFINITY;
        };
        let top = law.ppf(1.0).unwrap_or(f64::INFINITY);
        if !lowest.is_finite() || (self.maximum.is_some() && !top.is_finite()) {
            return f64::INFINITY;
        }
        for (out, &probability) in self.standardized.iter_mut().zip(self.probabilities) {
            let Ok(q) = truncated_ppf(&law, 0.0, 1.0, probability) else {
                return f64::INFINITY;
            };
            *out = q;
        }
        if self.standardized.iter().any(|q| !q.is_finite()) {
            return f64::INFINITY;
        }
        let (atom, maximum) = (self.atom, self.maximum);
        let fits_lower = |loc: f64, scale: f64| loc + scale * lowest >= atom;
        let fits_upper = |loc: f64, scale: f64| maximum.is_none_or(|m| loc + scale * top <= m);
        let mut candidates: Vec<(f64, f64)> = Vec::with_capacity(4);

        let (mut z_mean, mut v_mean) = (0.0, 0.0);
        for ((&q, &value), &weight) in self.standardized.iter().zip(self.values).zip(&self.weights)
        {
            z_mean += weight * q;
            v_mean += weight * value;
        }
        let (mut covariance, mut variance) = (0.0, 0.0);
        for ((&q, &value), &weight) in self.standardized.iter().zip(self.values).zip(&self.weights)
        {
            covariance += weight * (q - z_mean) * (value - v_mean);
            variance += weight * (q - z_mean) * (q - z_mean);
        }
        let scale = covariance / variance;
        let loc = v_mean - scale * z_mean;
        if fits_lower(loc, scale) && fits_upper(loc, scale) {
            candidates.push((loc, scale));
        } else {
            // Lower constraint held: v - a = scale * (q - lowest).
            let (mut cross, mut square) = (0.0, 0.0);
            for ((&q, &value), &weight) in
                self.standardized.iter().zip(self.values).zip(&self.weights)
            {
                cross += weight * (q - lowest) * (value - atom);
                square += weight * (q - lowest) * (q - lowest);
            }
            let scale = cross / square;
            let loc = atom - scale * lowest;
            if fits_upper(loc, scale) {
                candidates.push((loc, scale));
            }
            if let Some(maximum) = maximum {
                // Upper constraint held: m - v = scale * (top - q).
                let (mut cross, mut square) = (0.0, 0.0);
                for ((&q, &value), &weight) in
                    self.standardized.iter().zip(self.values).zip(&self.weights)
                {
                    cross += weight * (top - q) * (maximum - value);
                    square += weight * (top - q) * (top - q);
                }
                let scale = cross / square;
                let loc = maximum - scale * top;
                if fits_lower(loc, scale) {
                    candidates.push((loc, scale));
                }
                // Both held: the support spans exactly [atom, maximum].
                let scale = (maximum - atom) / (top - lowest);
                candidates.push((atom - scale * lowest, scale));
            }
        }

        let mut best = (f64::INFINITY, 0.0, 0.0);
        for (loc, scale) in candidates {
            if !(scale > MINIMUM_SCALE && scale < MAXIMUM_QUANTILE_SCALE && loc.is_finite()) {
                continue;
            }
            let loss = self
                .standardized
                .iter()
                .zip(self.values)
                .zip(&self.weights)
                .map(|((&q, &value), &weight)| weight * (loc + scale * q - value).powi(2))
                .sum::<f64>();
            if loss < best.0 {
                best = (loss, loc, scale);
            }
        }
        let (loss, loc, scale) = best;
        if !loss.is_finite() || !self.admits(self.shape(x), loc, scale) {
            return f64::INFINITY;
        }
        self.note(loss, self.shape(x), loc, scale);
        loss
    }

    /// The truncated branch at its upper limit in `s`.
    ///
    /// For a base bounded above (GEV with positive shape, reverse Weibull) the
    /// limit is where the scale reaches the optimizer's limit, located from the
    /// tail's generalized Pareto limit in one pass: there the standardized tail
    /// above the atom is `g * c * h(u)` with `h(u) = -expm1(c ln(1 - u)) / c`
    /// and `g = (-ln F(a))^c`, where `c` is the GEV shape (`c = 1 / k` for the
    /// reverse Weibull, whose `c` factor is 1 for the GEV).
    fn wall(&mut self, x: f64) -> f64 {
        let Some(shape) = self.shape(x) else {
            return f64::INFINITY;
        };
        let (c, factor) = match self.family {
            DistributionFamily::GenExtreme if shape > 0.0 => (shape, 1.0),
            DistributionFamily::WeibullMax => (1.0 / shape, 1.0 / shape),
            _ => return self.truncated(x, logit_ceiling(), None),
        };
        self.evaluations += 1;
        for (out, &probability) in self.standardized.iter_mut().zip(self.probabilities) {
            *out = -(c * (-probability).ln_1p()).exp_m1() / c;
        }
        let Some((scale, _)) = self.project(&self.standardized) else {
            return f64::INFINITY;
        };
        if scale <= 0.0 {
            return f64::INFINITY;
        }
        let neg_log_cdf = (scale / (factor * MAXIMUM_QUANTILE_SCALE)).powf(1.0 / c);
        if neg_log_cdf.is_nan() || neg_log_cdf >= 30.0 {
            return f64::INFINITY;
        }
        let s = if neg_log_cdf > 0.0 {
            -neg_log_cdf.exp_m1().ln()
        } else {
            logit_ceiling()
        };
        self.truncated(x, s.min(logit_ceiling()), None)
    }

    fn search_shape_and_logit(&mut self) {
        let (low, high, _) = shape_coordinates(self.family).expect("shaped family");
        self.grid_then_minimum(
            |tail, x| tail.untruncated(x),
            low,
            high,
            SHAPE_GRID_POINTS,
            SHAPE_TOLERANCE,
        );
        self.grid_then_minimum(
            |tail, x| tail.wall(x),
            low,
            high,
            SHAPE_GRID_POINTS,
            SHAPE_TOLERANCE,
        );

        let shapes: Vec<f64> = if self.family == DistributionFamily::GenExtreme {
            vec![-1.5, -0.6, -0.3, 0.0, 0.3, 0.6, 1.5]
        } else {
            (0..5)
                .map(|i| low + (high - low) * i as f64 / 4.0)
                .collect()
        };
        let logits = logit_grid();
        let mut grid = Vec::with_capacity(shapes.len() * logits.len());
        for (i, &x) in shapes.iter().enumerate() {
            for (j, &s) in logits.iter().enumerate() {
                grid.push((self.truncated(x, s, None), i, j));
            }
        }
        grid.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut starts: Vec<(usize, usize)> = Vec::with_capacity(STARTS);
        for &(loss, i, j) in &grid {
            if !loss.is_finite() || starts.len() == STARTS {
                break;
            }
            if starts
                .iter()
                .all(|&(i2, j2)| i.abs_diff(i2) > 1 || j.abs_diff(j2) > 1)
            {
                starts.push((i, j));
            }
        }
        let lower = [low, logit_floor()];
        let upper = [high, logit_ceiling()];
        let max_step = if self.family == DistributionFamily::GenExtreme {
            [0.2, 4.0]
        } else {
            [1.0, 4.0]
        };
        for (i, j) in starts {
            self.levenberg_marquardt([shapes[i], logits[j]], lower, upper, max_step);
        }

        // Re-profile `s` at the best shape and restart from there if lower.
        let Some((loss, x, _)) = self.best_truncated else {
            return;
        };
        let profiled = logits.map(|s| self.truncated(x, s, None));
        let index = argmin(&profiled);
        if !profiled[index].is_finite() {
            return;
        }
        let (s, refined) = bounded_minimum(
            |s| self.truncated(x, s, None),
            logits[index.saturating_sub(1)],
            logits[(index + 1).min(logits.len() - 1)],
            LOGIT_TOLERANCE,
            40,
        );
        let (s, profiled_loss) = if refined < profiled[index] {
            (s, refined)
        } else {
            (logits[index], profiled[index])
        };
        if profiled_loss < loss * (1.0 - 1.0e-12) {
            self.levenberg_marquardt([x, s], lower, upper, max_step);
        }

        // A maximum caps the scale, which puts a kink in the surface where the
        // cap starts to bind; the optimum often sits on it, in a narrow valley
        // Levenberg-Marquardt can stall against. A simplex needs no gradient.
        if self.maximum.is_some() {
            let Some((_, x, s)) = self.best_truncated else {
                return;
            };
            let tail = RefCell::new(&mut *self);
            quantile_nelder_mead(
                vec![x, s],
                vec![0.1 * x.abs().max(1.0e-2), 0.5],
                |point| tail.borrow_mut().truncated(point[0], point[1], None),
                |point| {
                    point[0] = point[0].clamp(lower[0], upper[0]);
                    point[1] = point[1].clamp(lower[1], upper[1]);
                },
                400,
                1.0e-10,
                1.0e-14,
            );
        }
    }

    /// Evaluate `objective` on a uniform grid, then refine around its best point.
    fn grid_then_minimum(
        &mut self,
        mut objective: impl FnMut(&mut Self, f64) -> f64,
        low: f64,
        high: f64,
        points: usize,
        tolerance: f64,
    ) {
        let grid: Vec<f64> = (0..points)
            .map(|i| low + (high - low) * i as f64 / (points - 1) as f64)
            .collect();
        let losses: Vec<f64> = grid.iter().map(|&x| objective(self, x)).collect();
        let index = argmin(&losses);
        if !losses[index].is_finite() {
            return;
        }
        bounded_minimum(
            |x| objective(self, x),
            grid[index.saturating_sub(1)],
            grid[(index + 1).min(points - 1)],
            tolerance,
            60,
        );
    }

    /// Levenberg-Marquardt on the truncated branch in `(x, s)`, with box
    /// bounds handled by an active set and a per-coordinate step limit.
    fn levenberg_marquardt(
        &mut self,
        start: [f64; 2],
        lower: [f64; 2],
        upper: [f64; 2],
        max_step: [f64; 2],
    ) {
        let n = self.values.len();
        let mut theta = start;
        let mut residuals = vec![0.0; n];
        let mut trial = vec![0.0; n];
        let mut jacobian = [vec![0.0; n], vec![0.0; n]];
        let mut loss = self.truncated(theta[0], theta[1], Some(&mut residuals));
        if !loss.is_finite() {
            return;
        }
        let mut damping = 1.0e-3;
        for _ in 0..LM_ITERATIONS {
            self.iterations += 1;
            for j in 0..2 {
                let mut step = 1.0e-6 * (1.0 + theta[j].abs());
                if theta[j] + step > upper[j] {
                    step = -step;
                }
                let mut probe = theta;
                probe[j] += step;
                if !self
                    .truncated(probe[0], probe[1], Some(&mut trial))
                    .is_finite()
                {
                    step = -step;
                    probe = theta;
                    probe[j] += step;
                    if !self
                        .truncated(probe[0], probe[1], Some(&mut trial))
                        .is_finite()
                    {
                        return;
                    }
                }
                for k in 0..n {
                    jacobian[j][k] = (trial[k] - residuals[k]) / step;
                }
            }
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
            let gradient = [dot(&jacobian[0], &residuals), dot(&jacobian[1], &residuals)];
            let hessian = [
                [
                    dot(&jacobian[0], &jacobian[0]),
                    dot(&jacobian[0], &jacobian[1]),
                ],
                [
                    dot(&jacobian[1], &jacobian[0]),
                    dot(&jacobian[1], &jacobian[1]),
                ],
            ];
            let free = [0, 1].map(|j| {
                !((theta[j] <= lower[j] + 1.0e-12 && gradient[j] > 0.0)
                    || (theta[j] >= upper[j] - 1.0e-12 && gradient[j] < 0.0))
            });
            if !free[0] && !free[1] {
                return;
            }
            let mut improvement = None;
            while damping < 1.0e12 {
                let diagonal =
                    [0, 1].map(|j| hessian[j][j] + damping * hessian[j][j].max(1.0e-300));
                let mut step = [0.0; 2];
                if free[0] && free[1] {
                    let determinant = diagonal[0] * diagonal[1] - hessian[0][1] * hessian[1][0];
                    if determinant == 0.0 || !determinant.is_finite() {
                        damping *= 10.0;
                        continue;
                    }
                    step[0] =
                        (-gradient[0] * diagonal[1] + gradient[1] * hessian[0][1]) / determinant;
                    step[1] =
                        (-gradient[1] * diagonal[0] + gradient[0] * hessian[1][0]) / determinant;
                } else {
                    let j = usize::from(free[1]);
                    step[j] = -gradient[j] / diagonal[j];
                }
                let excess = (step[0].abs() / max_step[0]).max(step[1].abs() / max_step[1]);
                if excess > 1.0 {
                    step = step.map(|value| value / excess);
                }
                let candidate = [0, 1].map(|j| (theta[j] + step[j]).clamp(lower[j], upper[j]));
                let candidate_loss = self.truncated(candidate[0], candidate[1], Some(&mut trial));
                if candidate_loss.is_finite() && candidate_loss < loss {
                    improvement = Some((loss - candidate_loss) / loss.max(1.0e-300));
                    theta = candidate;
                    loss = candidate_loss;
                    std::mem::swap(&mut residuals, &mut trial);
                    damping = (damping / 3.0).max(1.0e-9);
                    break;
                }
                damping *= 4.0;
            }
            if improvement.is_none_or(|relative| relative < LM_RELATIVE_TOLERANCE) {
                return;
            }
        }
    }

    fn result(&self) -> Option<QuantileOptimization> {
        let (_, shape, loc, scale) = self.best?;
        Some(QuantileOptimization {
            shape,
            loc,
            scale,
            converged: true,
            iterations: self.iterations,
            evaluations: self.evaluations,
        })
    }
}

fn argmin(values: &[f64]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value < values[best] {
            best = index;
        }
    }
    best
}

/// Brent's bounded scalar minimization (SciPy's `fminbound`).
///
/// Returns the best point and its value after at most `max_evaluations` calls.
fn bounded_minimum(
    mut objective: impl FnMut(f64) -> f64,
    low: f64,
    high: f64,
    tolerance: f64,
    max_evaluations: usize,
) -> (f64, f64) {
    let sqrt_eps = f64::EPSILON.sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0_f64.sqrt());
    let (mut a, mut b) = (low, high);
    let mut fulc = a + golden_mean * (b - a);
    let mut nfc = fulc;
    let mut xf = fulc;
    let mut rat: f64 = 0.0;
    let mut e: f64 = 0.0;
    let mut fx = objective(xf);
    let mut evaluations = 1;
    let mut ffulc = fx;
    let mut fnfc = fx;
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + tolerance / 3.0;
    let mut tol2 = 2.0 * tol1;
    while (xf - xm).abs() > tol2 - 0.5 * (b - a) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                let x = xf + rat;
                if x - a < tol2 || b - x < tol2 {
                    rat = tol1 * if xm >= xf { 1.0 } else { -1.0 };
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let direction = if rat >= 0.0 { 1.0 } else { -1.0 };
        let x = xf + direction * rat.abs().max(tol1);
        let fu = objective(x);
        evaluations += 1;
        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + tolerance / 3.0;
        tol2 = 2.0 * tol1;
        if evaluations >= max_evaluations {
            break;
        }
    }
    (xf, fx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gev_tail(shape: f64, loc: f64, scale: f64, atom: f64) -> (Vec<f64>, Vec<f64>) {
        let law = FittedDistribution::from_parameters(
            DistributionFamily::GenExtreme,
            Some(shape),
            loc,
            scale,
        )
        .unwrap();
        let (cdf, sf) = (law.cdf(atom), law.sf(atom));
        let probabilities: Vec<f64> = (1..200).map(|i| i as f64 / 200.0).collect();
        let values = probabilities
            .iter()
            .map(|&u| truncated_ppf(&law, cdf, sf, u).unwrap())
            .collect();
        (probabilities, values)
    }

    #[test]
    fn recovers_a_truncated_gev_tail() {
        let (probabilities, values) = gev_tail(-0.2, 3.0, 2.0, 2.0);
        let weights = vec![1.0; values.len()];
        let fit = profile_truncated_quantile_points(
            DistributionKind::GenExtreme,
            &probabilities,
            &values,
            &weights,
            2.0,
            None,
            None,
        )
        .unwrap();
        assert!((fit.shape.unwrap() + 0.2).abs() < 1e-4, "{fit:?}");
        assert!(
            (fit.loc - 3.0).abs() < 1e-3 && (fit.scale - 2.0).abs() < 1e-3,
            "{fit:?}"
        );
    }

    fn truncated_tail(
        family: DistributionFamily,
        shape: Option<f64>,
        atom: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let law = FittedDistribution::from_parameters(family, shape, 0.0, 1.0).unwrap();
        let (cdf, sf) = (law.cdf(atom), law.sf(atom));
        let probabilities: Vec<f64> = (1..20).map(|i| i as f64 / 20.0).collect();
        let values = probabilities
            .iter()
            .map(|&u| truncated_ppf(&law, cdf, sf, u).unwrap())
            .collect();
        (probabilities, values)
    }

    #[test]
    fn one_parameter_searches_recover_exact_tails_to_full_precision() {
        // Gumbel searches only the truncation point and the generalized Pareto
        // only its shape; nothing refines either search afterwards.
        for (kind, family, shape, atom) in [
            (
                DistributionKind::GumbelR,
                DistributionFamily::GumbelRight,
                None,
                -0.5,
            ),
            (
                DistributionKind::GenPareto,
                DistributionFamily::GenPareto,
                Some(0.25),
                0.0,
            ),
        ] {
            let (probabilities, values) = truncated_tail(family, shape, atom);
            let weights = vec![1.0; values.len()];
            let fit = profile_truncated_quantile_points(
                kind,
                &probabilities,
                &values,
                &weights,
                atom,
                None,
                None,
            )
            .unwrap();
            assert!(
                fit.loc.abs() < 1e-8 && (fit.scale - 1.0).abs() < 1e-8,
                "{fit:?}"
            );
        }
    }

    #[test]
    fn an_atom_below_the_support_fits_the_untruncated_base() {
        // Weibull support starts at loc = 5, above the atom at 0.
        let law = FittedDistribution::from_parameters(
            DistributionFamily::WeibullMin,
            Some(1.7),
            5.0,
            3.0,
        )
        .unwrap();
        let probabilities: Vec<f64> = (1..100).map(|i| i as f64 / 100.0).collect();
        let values: Vec<f64> = probabilities.iter().map(|&u| law.ppf(u).unwrap()).collect();
        let fit = profile_truncated_quantile_points(
            DistributionKind::WeibullMin,
            &probabilities,
            &values,
            &vec![1.0; values.len()],
            0.0,
            None,
            None,
        )
        .unwrap();
        assert!((fit.shape.unwrap() - 1.7).abs() < 1e-3, "{fit:?}");
        assert!((fit.loc - 5.0).abs() < 1e-2, "{fit:?}");
    }

    #[test]
    fn a_maximum_caps_the_support_of_the_fitted_tail() {
        // The generating law ends at 3 + 2 / 0.3 = 9.67; the bound is below it.
        let (probabilities, values) = gev_tail(0.3, 3.0, 2.0, 2.0);
        let weights = vec![1.0; values.len()];
        for kind in [DistributionKind::GenExtreme, DistributionKind::WeibullMax] {
            let fit = profile_truncated_quantile_points(
                kind,
                &probabilities,
                &values,
                &weights,
                2.0,
                Some(9.0),
                None,
            )
            .unwrap();
            let law = FittedDistribution::from_parameters(
                DistributionFamily::from_reference_kind(kind),
                fit.shape,
                fit.loc,
                fit.scale,
            )
            .unwrap();
            let end = law.ppf(1.0).unwrap();
            assert!(end <= 9.0 && end > 8.9, "{kind:?} {fit:?} ends at {end}");
        }
        // A law unbounded above cannot meet a maximum.
        assert!(
            profile_truncated_quantile_points(
                DistributionKind::GumbelR,
                &probabilities,
                &values,
                &weights,
                2.0,
                Some(9.0),
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn the_brent_minimizer_finds_an_interior_minimum() {
        let (x, value) = bounded_minimum(|x| (x - 0.3).powi(2) + 1.0, -2.0, 2.0, 1e-8, 100);
        assert!((x - 0.3).abs() < 1e-6 && (value - 1.0).abs() < 1e-12);
    }
}
