//! Native evaluation of the promoted hazard model.
//!
//! `decide.rs` states that inference deliberately does not live in Rust,
//! because reimplementing a model in the hot path introduces a SECOND model
//! that can disagree with the one the promotion gate validated. That
//! reasoning is right, and this module does not contradict it -- it removes
//! the condition that made it true.
//!
//! Nothing here is trained or reimplemented from a description. The fitted
//! parameters are lifted out of the promoted artifact by
//! `tools/export_hazard_model.py` and replayed as arithmetic:
//!
//! ```text
//! p_raw   = sigmoid(intercept + coef . features)
//! p_final = isotonic(p_raw)        piecewise-linear, clipped at both ends
//! ```
//!
//! An exact port is only possible because the promoted heads are
//! LogisticRegression with IsotonicRegression calibrators. A gradient
//! boosted forest would be a far larger surface for two implementations to
//! drift apart on, and the honest answer there would be to keep handing the
//! probability in.
//!
//! Only the hazard heads exist here. The return model's last report reads
//! REJECTED and has no artifact, so there is nothing to port; a native
//! evaluator for a model that never passed validation would be inventing
//! one.
//!
//! Parity is not assumed. `examples/parity.rs` and the Python side compare
//! outputs across the feature space, and the export refuses any artifact
//! that did not pass validation.

/// One trained head: a linear score, optionally isotonically calibrated.
#[derive(Debug, Clone, Default)]
pub struct Head {
    pub intercept: f64,
    pub coef: Vec<f64>,
    /// Calibrator breakpoints, ascending in `x`. Empty means the raw
    /// sigmoid is the answer -- which is a real case, not a missing one.
    pub calibrator_x: Vec<f64>,
    pub calibrator_y: Vec<f64>,
}

/// Why a head could not answer. Distinguished from a probability of zero,
/// because "we cannot say" and "we say it will not happen" are different
/// claims and only one of them is safe to act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceError {
    /// The caller supplied a different number of features than the head was
    /// fitted with. Positional indexing makes this a silent wrong answer
    /// rather than an error unless it is checked.
    FeatureCountMismatch { expected: usize, got: usize },
    /// A feature was NaN or infinite. Propagating it would produce a
    /// confident-looking probability from a meaningless input.
    NonFiniteFeature { index: usize },
}

impl Head {
    /// Calibrated probability for one feature vector.
    pub fn predict(&self, features: &[f64]) -> Result<f64, InferenceError> {
        if features.len() != self.coef.len() {
            return Err(InferenceError::FeatureCountMismatch {
                expected: self.coef.len(),
                got: features.len(),
            });
        }
        let mut score = self.intercept;
        for (index, (value, weight)) in features.iter().zip(self.coef.iter()).enumerate() {
            if !value.is_finite() {
                return Err(InferenceError::NonFiniteFeature { index });
            }
            score += value * weight;
        }
        Ok(self.calibrate(sigmoid(score)))
    }

    /// Piecewise-linear interpolation over the fitted breakpoints, clamped
    /// outside them -- the `out_of_bounds="clip"` the exporter enforces.
    fn calibrate(&self, raw: f64) -> f64 {
        let xs = &self.calibrator_x;
        let ys = &self.calibrator_y;
        if xs.is_empty() || xs.len() != ys.len() {
            return raw;
        }
        if raw <= xs[0] {
            return ys[0];
        }
        if raw >= xs[xs.len() - 1] {
            return ys[ys.len() - 1];
        }
        // Ascending breakpoints, so the first x above `raw` bounds the
        // segment that contains it.
        let upper = match xs.iter().position(|&x| x >= raw) {
            Some(index) if index > 0 => index,
            _ => return ys[0],
        };
        let (x0, x1) = (xs[upper - 1], xs[upper]);
        let (y0, y1) = (ys[upper - 1], ys[upper]);
        let span = x1 - x0;
        if span <= 0.0 {
            // A vertical step in the fitted calibrator. Taking the upper
            // value matches scipy's interpolation at a duplicated knot.
            return y1;
        }
        y0 + (y1 - y0) * ((raw - x0) / span)
    }
}

/// The logistic link. Written out rather than pulled from a crate so the
/// arithmetic on this path is visible and cannot change under a dependency
/// bump.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        // Algebraically identical, but evaluating exp on the negative side
        // avoids overflow for large |z|.
        let e = z.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head() -> Head {
        // The promoted rug_30s head, 2026-08-29.
        Head {
            intercept: -1.913_481_239_140_569_3,
            coef: vec![
                0.120_746_973_885_906_27,
                -3.578_713_757_972_267,
                1.569_651_981_518_294_8,
                6.813_627_0,
                -1.010_759_0,
                0.0,
                0.0,
                0.0,
            ],
            calibrator_x: vec![0.1, 0.5, 0.9],
            calibrator_y: vec![0.0, 0.25, 1.0],
        }
    }

    #[test]
    fn a_wrong_feature_count_is_an_error_not_a_guess() {
        let err = head().predict(&[0.0; 3]).unwrap_err();
        assert_eq!(
            err,
            InferenceError::FeatureCountMismatch { expected: 8, got: 3 }
        );
    }

    #[test]
    fn a_non_finite_feature_is_refused() {
        let mut features = [0.0; 8];
        features[2] = f64::NAN;
        assert_eq!(
            head().predict(&features).unwrap_err(),
            InferenceError::NonFiniteFeature { index: 2 }
        );
    }

    #[test]
    fn calibration_clamps_outside_the_fitted_range() {
        let h = head();
        assert_eq!(h.calibrate(-5.0), 0.0);
        assert_eq!(h.calibrate(5.0), 1.0);
    }

    #[test]
    fn calibration_interpolates_inside_it() {
        // Midway between (0.5, 0.25) and (0.9, 1.0).
        let value = head().calibrate(0.7);
        assert!((value - 0.625).abs() < 1e-12, "got {value}");
    }

    #[test]
    fn a_duplicated_knot_does_not_divide_by_zero() {
        let h = Head {
            intercept: 0.0,
            coef: vec![0.0],
            calibrator_x: vec![0.2, 0.2, 0.8],
            calibrator_y: vec![0.1, 0.4, 0.9],
        };
        assert!(h.calibrate(0.2).is_finite());
    }

    #[test]
    fn sigmoid_does_not_overflow_at_the_extremes() {
        assert!(sigmoid(1000.0).is_finite() && sigmoid(1000.0) <= 1.0);
        assert!(sigmoid(-1000.0).is_finite() && sigmoid(-1000.0) >= 0.0);
    }

    #[test]
    fn an_uncalibrated_head_returns_the_raw_sigmoid() {
        let h = Head { intercept: 0.0, coef: vec![0.0], ..Default::default() };
        assert!((h.predict(&[0.0]).unwrap() - 0.5).abs() < 1e-12);
    }
}
