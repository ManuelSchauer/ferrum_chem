//! Applicability Domain — k-NN distance AD, leverage AD.


//! ## File guide: Ad
//!
//! This file belongs to FerrumChem's **machine-learning pipeline layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `trait ApplicabilityDomain`
//! - `fn is_in_domain`
//! - `fn confidence`
//! - `struct KnnDistanceAD`
//! - `fn fit`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Ad` assumes `ndarray` row-major sample layout (`n_samples x n_features`) throughout fit/predict paths.
//! - `Ad` prioritizes reproducibility via explicit seeds and deterministic iteration order where applicable.
//! - Model quality metrics for `Ad` depend on consistent preprocessing choices (scaling, splitting, and leakage control).
//! - Specific scope anchor: `crates/ferrum_core/src/ml/ad.rs` defines ad behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Canonical ML literature for `Ad` algorithm family (statistical learning and validation texts).
//! - scikit-learn/RDKit-style API ergonomics that inform `Ad` fit/transform/predict contracts.
//! - OECD QSAR validation principles where `Ad` is used in predictive chemistry contexts.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/ml/ad.rs`.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Applicability Domain trait.
pub trait ApplicabilityDomain: Send + Sync {
    /// Check if a sample is within the domain.
    fn is_in_domain(&self, sample: &[f64]) -> bool;

    /// Confidence score (higher = more reliable prediction).
    fn confidence(&self, sample: &[f64]) -> f64;
}

/// k-NN distance-based applicability domain.
///
/// A sample is in domain if its mean distance to k nearest training neighbors
/// is below a threshold (mean + z*std of training distances).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnDistanceAD {
    pub train_features: Array2<f64>,
    pub k: usize,
    pub threshold: f64,
    pub mean_dist: f64,
    pub std_dist: f64,
}

impl KnnDistanceAD {
    /// Fit the AD from training data.
    ///
    /// `z_cutoff`: number of standard deviations for threshold (typically 1.0-3.0).
    pub fn fit(features: &Array2<f64>, k: usize, z_cutoff: f64) -> Self {
        let n = features.nrows();
        let k = k.min(n - 1).max(1);

        // Compute mean k-NN distance for each training sample
        let mut distances = Vec::with_capacity(n);
        for i in 0..n {
            let sample = features.row(i);
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    sample
                        .iter()
                        .zip(features.row(j).iter())
                        .map(|(&a, &b)| {
                            let d = a - b;
                            d * d
                        })
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            dists.sort_by(|a, b| a.total_cmp(b));
            let mean_k_dist: f64 = dists.iter().take(k).sum::<f64>() / k as f64;
            distances.push(mean_k_dist);
        }

        let mean_dist = distances.iter().sum::<f64>() / n as f64;
        let std_dist = (distances
            .iter()
            .map(|&d| (d - mean_dist).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();

        let threshold = mean_dist + z_cutoff * std_dist;

        Self {
            train_features: features.clone(),
            k,
            threshold,
            mean_dist,
            std_dist,
        }
    }

    /// Implements knn distance.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn mean_knn_distance(&self, sample: &[f64]) -> f64 {
        let mut dists: Vec<f64> = (0..self.train_features.nrows())
            .map(|i| {
                sample
                    .iter()
                    .zip(self.train_features.row(i).iter())
                    .map(|(&a, &b)| {
                        let d = a - b;
                        d * d
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        dists.sort_by(|a, b| a.total_cmp(b));
        dists.iter().take(self.k).sum::<f64>() / self.k as f64
    }
}

impl ApplicabilityDomain for KnnDistanceAD {
    /// Checks whether in domain.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn is_in_domain(&self, sample: &[f64]) -> bool {
        self.mean_knn_distance(sample) <= self.threshold
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn confidence(&self, sample: &[f64]) -> f64 {
        let dist = self.mean_knn_distance(sample);
        if dist <= self.mean_dist {
            1.0
        } else if dist >= self.threshold {
            0.0
        } else {
            1.0 - (dist - self.mean_dist) / (self.threshold - self.mean_dist)
        }
    }
}

/// Leverage-based applicability domain (Williams plot).
///
/// h_i = x_i' (X'X)^{-1} x_i
/// Threshold: h* = 3p/n (p = features, n = samples)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeverageAD {
    /// (X'X)^{-1} matrix.
    pub hat_matrix_inv: Array2<f64>,
    /// Training feature means (for centering).
    pub x_mean: Array1<f64>,
    /// Leverage threshold h*.
    pub threshold: f64,
}

impl LeverageAD {
    /// Fit leverage AD from training features.
    pub fn fit(features: &Array2<f64>) -> Self {
        let n = features.nrows();
        let p = features.ncols();

        let x_mean = features.mean_axis(ndarray::Axis(0)).unwrap();

        // Center features
        let x_centered = features - &x_mean;

        // Compute X'X
        let mut xtx = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in i..p {
                let dot: f64 = x_centered.column(i).dot(&x_centered.column(j));
                xtx[[i, j]] = dot;
                xtx[[j, i]] = dot;
            }
        }

        // Ridge regularization improves numerical stability on collinear descriptors.
        let ridge = 1e-8;
        for d in 0..p {
            xtx[[d, d]] += ridge;
        }

        // Invert X'X using a regularized pseudoinverse for improved stability.
        let hat_matrix_inv = pseudoinverse(&xtx);

        let threshold = 3.0 * p as f64 / n as f64;

        Self {
            hat_matrix_inv,
            x_mean,
            threshold,
        }
    }

    /// Compute leverage for a sample.
    pub fn leverage(&self, sample: &[f64]) -> f64 {
        let p = self.x_mean.len();
        let mut centered = vec![0.0; p];
        for j in 0..p {
            centered[j] = sample[j] - self.x_mean[j];
        }

        // h = x' (X'X)^{-1} x
        let mut h = 0.0;
        for i in 0..p {
            let mut inner = 0.0;
            for j in 0..p {
                inner += self.hat_matrix_inv[[i, j]] * centered[j];
            }
            h += centered[i] * inner;
        }
        h
    }
}

impl ApplicabilityDomain for LeverageAD {
    /// Checks whether in domain.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn is_in_domain(&self, sample: &[f64]) -> bool {
        self.leverage(sample) <= self.threshold
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn confidence(&self, sample: &[f64]) -> f64 {
        let h = self.leverage(sample);
        if h <= 0.0 {
            1.0
        } else if h >= self.threshold {
            0.0
        } else {
            1.0 - h / self.threshold
        }
    }
}

/// Simple matrix inversion via Gauss-Jordan elimination.
fn invert_matrix(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut aug = Array2::<f64>::zeros((n, 2 * n));

    // [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            // Additional diagonal stabilization when the pivot is near singular.
            aug[[col, col]] += 1e-8;
            max_val = aug[[col, col]].abs();
            if max_val < 1e-15 {
                continue;
            }
        }

        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Scale pivot row
        let pivot = aug[[col, col]];
        if pivot.abs() < 1e-15 {
            continue;
        }
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }

        // Eliminate
        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..(2 * n) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    inv
}

/// Compute the Moore-Penrose pseudoinverse using a damped normal-equation solve.
/// This implementation is intentionally regularized to remain stable with collinear QSAR features.
pub fn pseudoinverse(a: &Array2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Array2::<f64>::zeros((n, m));
    }

    if m >= n {
        let mut ata = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let dot = a.column(i).dot(&a.column(j));
                ata[[i, j]] = dot;
                ata[[j, i]] = dot;
            }
        }
        let ridge = 1e-8;
        for d in 0..n {
            ata[[d, d]] += ridge;
        }
        let ata_inv = invert_matrix(&ata);
        ata_inv.dot(&a.t())
    } else {
        let mut aat = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in i..m {
                let dot = a.row(i).dot(&a.row(j));
                aat[[i, j]] = dot;
                aat[[j, i]] = dot;
            }
        }
        let ridge = 1e-8;
        for d in 0..m {
            aat[[d, d]] += ridge;
        }
        let aat_inv = invert_matrix(&aat);
        a.t().dot(&aat_inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    /// Runs regression test for knn ad in domain.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_knn_ad_in_domain() {
        let features = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0],
        )
        .unwrap();
        let ad = KnnDistanceAD::fit(&features, 3, 2.0);

        // Point inside cluster → in domain
        assert!(ad.is_in_domain(&[0.5, 0.5]));
        // Point far away → out of domain
        assert!(!ad.is_in_domain(&[100.0, 100.0]));
    }

    #[test]
    /// Runs regression test for knn ad confidence.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_knn_ad_confidence() {
        let features = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let ad = KnnDistanceAD::fit(&features, 2, 2.0);

        let conf_in = ad.confidence(&[1.5]);
        let conf_out = ad.confidence(&[100.0]);
        assert!(conf_in > conf_out, "in={conf_in}, out={conf_out}");
    }

    #[test]
    /// Runs regression test for leverage ad.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_leverage_ad() {
        let features = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0],
        )
        .unwrap();
        let ad = LeverageAD::fit(&features);

        // Interior point should have low leverage
        let h_in = ad.leverage(&[3.0, 4.0]);
        // Extrapolation point should have high leverage
        let h_out = ad.leverage(&[100.0, 200.0]);
        assert!(h_in < h_out, "h_in={h_in}, h_out={h_out}");
    }

    #[test]
    /// Runs regression test for invert identity.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_invert_identity() {
        let eye = Array2::<f64>::eye(3);
        let inv = invert_matrix(&eye);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[[i, j]] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    /// Runs regression test for pseudoinverse shapes.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_pseudoinverse_shapes() {
        let a = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let pinv = pseudoinverse(&a);
        assert_eq!(pinv.nrows(), 2);
        assert_eq!(pinv.ncols(), 3);
    }

    #[test]
    /// Runs regression test for pseudoinverse rank deficient finite.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_pseudoinverse_rank_deficient_finite() {
        // Collinear columns.
        let a = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
        let pinv = pseudoinverse(&a);
        assert!(pinv.iter().all(|v| v.is_finite()));
    }
}
