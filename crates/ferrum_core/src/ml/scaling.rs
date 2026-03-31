//! Feature scaling — StandardScaler (Z-score) and MinMaxScaler.
//!
//! Both store fitted parameters for consistent inference-time transforms.


//! ## File guide: Scaling
//!
//! This file belongs to FerrumChem's **machine-learning pipeline layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct StandardScaler`
//! - `fn fit`
//! - `fn transform`
//! - `fn transform_one`
//! - `fn fit_transform`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Scaling` assumes `ndarray` row-major sample layout (`n_samples x n_features`) throughout fit/predict paths.
//! - `Scaling` prioritizes reproducibility via explicit seeds and deterministic iteration order where applicable.
//! - Model quality metrics for `Scaling` depend on consistent preprocessing choices (scaling, splitting, and leakage control).
//! - Specific scope anchor: `crates/ferrum_core/src/ml/scaling.rs` defines scaling behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Canonical ML literature for `Scaling` algorithm family (statistical learning and validation texts).
//! - scikit-learn/RDKit-style API ergonomics that inform `Scaling` fit/transform/predict contracts.
//! - OECD QSAR validation principles where `Scaling` is used in predictive chemistry contexts.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/ml/scaling.rs`.

use ndarray::{Array1, Array2, Axis, Zip};
use serde::{Deserialize, Serialize};

/// Z-score normalisation: x' = (x − μ) / σ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

impl StandardScaler {
    /// Fit on training data: compute per-feature mean and std.
    pub fn fit(data: &Array2<f64>) -> Self {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let n = data.nrows() as f64;
        let mut sq_sum = Array1::<f64>::zeros(data.ncols());
        for row in data.rows() {
            Zip::from(&mut sq_sum)
                .and(&mean)
                .and(row)
                .for_each(|acc, &m, &x| {
                    let diff = x - m;
                    *acc += diff * diff;
                });
        }
        let mut std = sq_sum;
        std.mapv_inplace(|v| (v / n).sqrt().max(1e-12));
        Self { mean, std }
    }

    /// Transform data using fitted params.
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut result = data.clone();
        Zip::from(result.rows_mut()).for_each(|mut row| {
            Zip::from(&mut row)
                .and(&self.mean)
                .and(&self.std)
                .for_each(|x, &m, &s| *x = (*x - m) / s);
        });
        result
    }

    /// Transform a single sample using fitted params.
    pub fn transform_one(&self, sample: &[f64]) -> Vec<f64> {
        assert_eq!(
            sample.len(),
            self.mean.len(),
            "Sample length ({}) must match scaler feature count ({})",
            sample.len(),
            self.mean.len()
        );
        sample
            .iter()
            .zip(self.mean.iter())
            .zip(self.std.iter())
            .map(|((&x, &m), &s)| (x - m) / s)
            .collect()
    }

    /// Fit and transform in one step.
    pub fn fit_transform(data: &Array2<f64>) -> (Self, Array2<f64>) {
        let scaler = Self::fit(data);
        let transformed = scaler.transform(data);
        (scaler, transformed)
    }

    /// Inverse transform: x = x' * σ + μ
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut result = data.clone();
        Zip::from(result.rows_mut()).for_each(|mut row| {
            Zip::from(&mut row)
                .and(&self.std)
                .and(&self.mean)
                .for_each(|x, &s, &m| *x = *x * s + m);
        });
        result
    }
}

/// Min-max scaling: x' = (x − min) / (max − min) → [0, 1]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxScaler {
    pub min: Array1<f64>,
    pub range: Array1<f64>, // max - min, clamped to avoid div-by-zero
}

impl MinMaxScaler {
    /// Fit on training data.
    pub fn fit(data: &Array2<f64>) -> Self {
        let n_features = data.ncols();
        let mut min = Array1::from_elem(n_features, f64::INFINITY);
        let mut max = Array1::from_elem(n_features, f64::NEG_INFINITY);

        for row in data.rows() {
            for (j, &v) in row.iter().enumerate() {
                if v < min[j] {
                    min[j] = v;
                }
                if v > max[j] {
                    max[j] = v;
                }
            }
        }

        let mut range = &max - &min;
        range.mapv_inplace(|v| if v.abs() < 1e-12 { 1.0 } else { v });

        Self { min, range }
    }

    /// Transform data to [0, 1].
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut result = data.clone();
        Zip::from(result.rows_mut()).for_each(|mut row| {
            Zip::from(&mut row)
                .and(&self.min)
                .and(&self.range)
                .for_each(|x, &min, &range| *x = (*x - min) / range);
        });
        result
    }

    /// Transform a single sample to [0, 1].
    pub fn transform_one(&self, sample: &[f64]) -> Vec<f64> {
        assert_eq!(
            sample.len(),
            self.min.len(),
            "Sample length ({}) must match scaler feature count ({})",
            sample.len(),
            self.min.len()
        );
        sample
            .iter()
            .zip(self.min.iter())
            .zip(self.range.iter())
            .map(|((&x, &min), &range)| (x - min) / range)
            .collect()
    }

    /// Fit and transform in one step.
    pub fn fit_transform(data: &Array2<f64>) -> (Self, Array2<f64>) {
        let scaler = Self::fit(data);
        let transformed = scaler.transform(data);
        (scaler, transformed)
    }

    /// Inverse transform: x = x' * range + min
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut result = data.clone();
        Zip::from(result.rows_mut()).for_each(|mut row| {
            Zip::from(&mut row)
                .and(&self.range)
                .and(&self.min)
                .for_each(|x, &range, &min| *x = *x * range + min);
        });
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    /// Runs regression test for standard scaler zero mean.
    ///
    /// Function context: `crates/ferrum_core/src/ml/scaling.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_standard_scaler_zero_mean() {
        let data = Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
            .unwrap();
        let (scaler, transformed) = StandardScaler::fit_transform(&data);
        // Mean of each column should be ~0
        let col_means = transformed.mean_axis(Axis(0)).unwrap();
        assert!(col_means[0].abs() < 1e-10);
        assert!(col_means[1].abs() < 1e-10);
        // Inverse should recover original
        let recovered = scaler.inverse_transform(&transformed);
        for i in 0..4 {
            for j in 0..2 {
                assert!((recovered[[i, j]] - data[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    /// Runs regression test for standard scaler unit variance.
    ///
    /// Function context: `crates/ferrum_core/src/ml/scaling.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_standard_scaler_unit_variance() {
        let data = Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
            .unwrap();
        let (_, transformed) = StandardScaler::fit_transform(&data);
        // Variance of each column should be ~1
        for j in 0..2 {
            let col = transformed.column(j);
            let mean = col.mean().unwrap();
            let var: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;
            assert!((var - 1.0).abs() < 1e-10, "Variance of col {j} = {var}");
        }
    }

    #[test]
    /// Runs regression test for minmax scaler range.
    ///
    /// Function context: `crates/ferrum_core/src/ml/scaling.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_minmax_scaler_range() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, -5.0, 5.0, 0.0, 3.0, 5.0]).unwrap();
        let (scaler, transformed) = MinMaxScaler::fit_transform(&data);
        // Min should be 0, max should be 1
        for j in 0..2 {
            let col = transformed.column(j);
            let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!((min - 0.0).abs() < 1e-10);
            assert!((max - 1.0).abs() < 1e-10);
        }
        // Inverse
        let recovered = scaler.inverse_transform(&transformed);
        for i in 0..3 {
            for j in 0..2 {
                assert!((recovered[[i, j]] - data[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    /// Runs regression test for constant feature.
    ///
    /// Function context: `crates/ferrum_core/src/ml/scaling.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_constant_feature() {
        // Constant column should not produce NaN
        let data = Array2::from_shape_vec((3, 2), vec![5.0, 1.0, 5.0, 2.0, 5.0, 3.0]).unwrap();
        let (_, t1) = StandardScaler::fit_transform(&data);
        assert!(t1.iter().all(|v| v.is_finite()));
        let (_, t2) = MinMaxScaler::fit_transform(&data);
        assert!(t2.iter().all(|v| v.is_finite()));
    }

    #[test]
    /// Runs regression test for transform one matches batch standard.
    ///
    /// Function context: `crates/ferrum_core/src/ml/scaling.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_transform_one_matches_batch_standard() {
        let data =
            Array2::from_shape_vec((3, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();
        let scaler = StandardScaler::fit(&data);
        let sample = [2.5, 25.0];
        let one = scaler.transform_one(&sample);
        let batch = scaler
            .transform(&Array2::from_shape_vec((1, 2), sample.to_vec()).unwrap())
            .row(0)
            .to_vec();
        for (a, b) in one.iter().zip(batch.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    /// Runs regression test for transform one matches batch minmax.
    ///
    /// Function context: `crates/ferrum_core/src/ml/scaling.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_transform_one_matches_batch_minmax() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, -5.0, 5.0, 0.0, 3.0, 5.0]).unwrap();
        let scaler = MinMaxScaler::fit(&data);
        let sample = [2.5, 1.0];
        let one = scaler.transform_one(&sample);
        let batch = scaler
            .transform(&Array2::from_shape_vec((1, 2), sample.to_vec()).unwrap())
            .row(0)
            .to_vec();
        for (a, b) in one.iter().zip(batch.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
