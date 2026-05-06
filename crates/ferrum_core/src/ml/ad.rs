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

#[cfg(feature = "native-linalg")]
use ndarray_linalg::{Inverse, SVD};

/// Default AD sampling step (exact mode).
fn default_knn_ad_sample_step() -> usize {
    1
}

/// Read positive usize env var or return None.
fn env_positive_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&v| v > 0)
}

/// Resolve AD neighbor sampling stride from environment.
///
/// `FERRUM_AD_MODE=approx` enables sampled neighbor scans for large datasets.
/// `FERRUM_AD_APPROX_MIN_ROWS` controls activation threshold (default: 50000).
/// `FERRUM_AD_APPROX_STEP` controls stride (default: 4, clamped to >=1).
fn resolve_knn_ad_sample_step(n_rows: usize) -> usize {
    let mode = std::env::var("FERRUM_AD_MODE").unwrap_or_else(|_| "exact".to_string());
    let min_rows = env_positive_usize("FERRUM_AD_APPROX_MIN_ROWS").unwrap_or(50_000);
    let step = env_positive_usize("FERRUM_AD_APPROX_STEP").unwrap_or(4);
    resolve_knn_ad_sample_step_with(&mode, min_rows, step, n_rows)
}

/// Pure helper for resolving AD sample-step behavior.
fn resolve_knn_ad_sample_step_with(
    mode: &str,
    min_rows: usize,
    step: usize,
    n_rows: usize,
) -> usize {
    if !mode.eq_ignore_ascii_case("approx") {
        return default_knn_ad_sample_step();
    }
    if n_rows < min_rows {
        return default_knn_ad_sample_step();
    }
    step.max(default_knn_ad_sample_step())
}

/// Return index and value of the largest item among first `count` entries.
fn largest_entry(values: &[f64], count: usize) -> (usize, f64) {
    let mut idx = 0usize;
    let mut val = values[0];
    for i in 1..count {
        if values[i] > val {
            val = values[i];
            idx = i;
        }
    }
    (idx, val)
}

/// Compute squared Euclidean distance between equal-length vectors.
fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean_distance_with_cutoff(a, b, f64::INFINITY)
}

/// Compute squared Euclidean distance with early exit if partial sum exceeds cutoff.
fn squared_euclidean_distance_with_cutoff(a: &[f64], b: &[f64], cutoff: f64) -> f64 {
    let mut acc = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc = d.mul_add(d, acc);
        if acc > cutoff {
            break;
        }
    }
    acc
}

/// Select k nearest distances for a sample.
fn knn_select_k_distances(
    train_features: &Array2<f64>,
    sample: &[f64],
    skip_index: Option<usize>,
    step: usize,
    best_dists: &mut [f64],
) -> usize {
    let k = best_dists.len();
    let mut filled = 0usize;
    let mut worst_idx = 0usize;
    let mut worst_dist = f64::NEG_INFINITY;
    let stride = step.max(1);

    scan_rows(train_features, |i, row| {
        if Some(i) == skip_index {
            return;
        }
        if stride > 1 && i % stride != 0 {
            return;
        }
        let cutoff = if filled < k {
            f64::INFINITY
        } else {
            worst_dist
        };
        let candidate = squared_euclidean_distance_with_cutoff(sample, row, cutoff);
        if filled < k {
            best_dists[filled] = candidate;
            if candidate > worst_dist {
                worst_dist = candidate;
                worst_idx = filled;
            }
            filled += 1;
            if filled == k {
                let (idx, val) = largest_entry(best_dists, k);
                worst_idx = idx;
                worst_dist = val;
            }
            return;
        }
        if candidate < worst_dist {
            best_dists[worst_idx] = candidate;
            let (idx, val) = largest_entry(best_dists, k);
            worst_idx = idx;
            worst_dist = val;
        }
    });

    filled
}

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
    #[serde(default)]
    pub train_features_t: Array2<f64>,
    pub k: usize,
    pub threshold: f64,
    pub mean_dist: f64,
    pub std_dist: f64,
    #[serde(skip, default)]
    pub train_mean_knn_distances: Vec<f64>,
    #[serde(default = "default_knn_ad_sample_step")]
    pub sample_step: usize,
}

impl KnnDistanceAD {
    /// Fit the AD from training data.
    ///
    /// `z_cutoff`: number of standard deviations for threshold (typically 1.0-3.0).
    pub fn fit(features: &Array2<f64>, k: usize, z_cutoff: f64) -> Self {
        let n = features.nrows();
        let k = k.min(n - 1).max(1);
        let train_features_t = features.t().to_owned();
        let sample_step = resolve_knn_ad_sample_step(n);

        // Compute mean k-NN distance for each training sample
        let mut distances = Vec::with_capacity(n);
        let mut best_dists = vec![f64::INFINITY; k];
        scan_rows(features, |i, sample| {
            let filled = if train_features_t.nrows() == features.ncols()
                && train_features_t.ncols() == features.nrows()
            {
                knn_select_k_distances_colmajor(
                    features,
                    &train_features_t,
                    sample,
                    Some(i),
                    sample_step,
                    &mut best_dists,
                )
            } else {
                knn_select_k_distances(features, sample, Some(i), sample_step, &mut best_dists)
            };
            if filled == 0 {
                distances.push(0.0);
                return;
            }
            let mean_k_dist = best_dists
                .iter()
                .take(filled)
                .map(|d| d.sqrt())
                .sum::<f64>()
                / filled as f64;
            distances.push(mean_k_dist);
        });

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
            train_features_t,
            k,
            threshold,
            mean_dist,
            std_dist,
            train_mean_knn_distances: distances,
            sample_step,
        }
    }

    /// Implements knn distance.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn mean_knn_distance(&self, sample: &[f64]) -> f64 {
        let mut best_dists = vec![f64::INFINITY; self.k];
        let filled = if self.train_features_t.nrows() == self.train_features.ncols()
            && self.train_features_t.ncols() == self.train_features.nrows()
        {
            knn_select_k_distances_colmajor(
                &self.train_features,
                &self.train_features_t,
                sample,
                None,
                self.sample_step,
                &mut best_dists,
            )
        } else {
            knn_select_k_distances(
                &self.train_features,
                sample,
                None,
                self.sample_step,
                &mut best_dists,
            )
        };
        if filled == 0 {
            return 0.0;
        }
        best_dists
            .iter()
            .take(filled)
            .map(|d| d.sqrt())
            .sum::<f64>()
            / filled as f64
    }

    /// Returns cached in-domain flags for the training rows used during AD fitting.
    pub fn training_in_domain_flags(&self) -> Vec<bool> {
        self.train_mean_knn_distances
            .iter()
            .map(|dist| *dist <= self.threshold)
            .collect()
    }
}

/// Iterate rows of a feature matrix with a contiguous fast path.
#[inline]
fn scan_rows(features: &Array2<f64>, mut f: impl FnMut(usize, &[f64])) {
    let nrows = features.nrows();
    let ncols = features.ncols();
    if ncols == 0 {
        for i in 0..nrows {
            f(i, &[]);
        }
        return;
    }

    if let Some(buf) = features.as_slice_memory_order() {
        if buf.len() == nrows * ncols {
            for (i, row) in buf.chunks_exact(ncols).enumerate() {
                f(i, row);
            }
            return;
        }
    }

    for i in 0..nrows {
        let row = features.row(i);
        if let Some(slice) = row.as_slice() {
            f(i, slice);
        } else {
            let owned = row.to_vec();
            f(i, &owned);
        }
    }
}

/// Select k nearest distances for a sample using pre-transposed feature storage.
fn knn_select_k_distances_colmajor(
    train_features: &Array2<f64>,
    train_features_t: &Array2<f64>,
    sample: &[f64],
    skip_index: Option<usize>,
    step: usize,
    best_dists: &mut [f64],
) -> usize {
    let k = best_dists.len();
    let mut filled = 0usize;
    let mut worst_idx = 0usize;
    let mut worst_dist = f64::NEG_INFINITY;
    let nrows = train_features.nrows();
    let ncols = train_features.ncols();
    let stride = step.max(1);

    if nrows == 0 || ncols == 0 {
        return 0;
    }

    let Some(features_t) = train_features_t.as_slice_memory_order() else {
        return knn_select_k_distances(train_features, sample, skip_index, stride, best_dists);
    };
    if features_t.len() != nrows * ncols {
        return knn_select_k_distances(train_features, sample, skip_index, stride, best_dists);
    }

    for row_idx in 0..nrows {
        if Some(row_idx) == skip_index {
            continue;
        }
        if stride > 1 && row_idx % stride != 0 {
            continue;
        }
        let cutoff = if filled < k {
            f64::INFINITY
        } else {
            worst_dist
        };

        let mut acc = 0.0f64;
        for col in 0..ncols {
            let x = features_t[col * nrows + row_idx];
            let d = sample[col] - x;
            acc = d.mul_add(d, acc);
            if acc > cutoff {
                break;
            }
        }

        if filled < k {
            best_dists[filled] = acc;
            if acc > worst_dist {
                worst_dist = acc;
                worst_idx = filled;
            }
            filled += 1;
            if filled == k {
                let (idx, val) = largest_entry(best_dists, k);
                worst_idx = idx;
                worst_dist = val;
            }
            continue;
        }

        if acc < worst_dist {
            best_dists[worst_idx] = acc;
            let (idx, val) = largest_entry(best_dists, k);
            worst_idx = idx;
            worst_dist = val;
        }
    }

    filled
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
        let mut xtx = x_centered.t().dot(&x_centered);

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
        let centered_arr = Array1::from_vec(centered);
        let tmp = self.hat_matrix_inv.dot(&centered_arr);
        centered_arr.dot(&tmp)
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

/// Cholesky factorization for symmetric positive-definite matrices.
///
/// Returns lower-triangular `L` such that `a = L * L^T`.
fn cholesky_decompose_spd(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1e-15 {
                    return None;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                let denom = l[[j, j]];
                if denom <= 1e-15 {
                    return None;
                }
                l[[i, j]] = sum / denom;
            }
        }
    }
    Some(l)
}

/// Solve `A * X = B` for SPD `A` using Cholesky factorization.
///
/// Falls back to Gauss-Jordan inverse multiplication only when Cholesky fails.
fn solve_spd_system(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    debug_assert_eq!(a.ncols(), n);
    debug_assert_eq!(b.nrows(), n);
    if n == 0 {
        return Array2::<f64>::zeros((0, b.ncols()));
    }

    if let Some(l) = cholesky_decompose_spd(a) {
        let rhs_cols = b.ncols();
        let mut y = Array2::<f64>::zeros((n, rhs_cols));
        for col in 0..rhs_cols {
            for i in 0..n {
                let mut sum = b[[i, col]];
                for k in 0..i {
                    sum -= l[[i, k]] * y[[k, col]];
                }
                y[[i, col]] = sum / l[[i, i]];
            }
        }

        let mut x = Array2::<f64>::zeros((n, rhs_cols));
        for col in 0..rhs_cols {
            for i_rev in 0..n {
                let i = n - 1 - i_rev;
                let mut sum = y[[i, col]];
                for k in (i + 1)..n {
                    sum -= l[[k, i]] * x[[k, col]];
                }
                x[[i, col]] = sum / l[[i, i]];
            }
        }
        return x;
    }

    // Rare fallback path for numerically pathological input.
    invert_matrix(a).dot(b)
}

/// Compute the Moore-Penrose pseudoinverse using a damped normal-equation solve.
/// This implementation is intentionally regularized to remain stable with collinear QSAR features.
pub fn pseudoinverse(a: &Array2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Array2::<f64>::zeros((n, m));
    }

    #[cfg(feature = "native-linalg")]
    {
        if let Some(pin) = pseudoinverse_native_linalg(a) {
            return pin;
        }
    }

    if m >= n {
        let mut ata = a.t().dot(a);
        let ridge = 1e-8;
        for d in 0..n {
            ata[[d, d]] += ridge;
        }
        let rhs = a.t().to_owned();
        solve_spd_system(&ata, &rhs)
    } else {
        let mut aat = a.dot(&a.t());
        let ridge = 1e-8;
        for d in 0..m {
            aat[[d, d]] += ridge;
        }
        let solved = solve_spd_system(&aat, a);
        solved.t().to_owned()
    }
}

/// Native-linalg pseudoinverse path (SVD preferred, inverse fallback).
#[cfg(feature = "native-linalg")]
fn pseudoinverse_native_linalg(a: &Array2<f64>) -> Option<Array2<f64>> {
    let (m, n) = a.dim();
    let (u_opt, s, vt_opt) = a.clone().svd(true, true).ok()?;
    let u = u_opt?;
    let vt = vt_opt?;

    let mut s_plus = Array2::<f64>::zeros((n, m));
    for (i, &sigma) in s.iter().enumerate() {
        if sigma > 1e-12 {
            s_plus[[i, i]] = 1.0 / sigma;
        }
    }

    // A^+ = V * S^+ * U^T
    let v = vt.t().to_owned();
    let ut = u.t().to_owned();
    let pin = v.dot(&s_plus).dot(&ut);
    if pin.iter().all(|v| v.is_finite()) {
        return Some(pin);
    }

    // Fallback: regularized inverse for square full-rank matrices.
    if m == n
        && let Ok(inv) = a.clone().inv()
        && inv.iter().all(|v| v.is_finite())
    {
        return Some(inv);
    }
    None
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
    /// Runs regression test for knn ad cached training flags.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_knn_ad_training_flags_cover_all_rows() {
        let features = Array2::from_shape_vec((5, 1), vec![0.0, 0.2, 0.4, 0.6, 0.8]).unwrap();
        let ad = KnnDistanceAD::fit(&features, 2, 2.0);
        let flags = ad.training_in_domain_flags();
        assert_eq!(flags.len(), features.nrows());
    }

    #[test]
    /// Runs regression test for knn ad defaults to exact sampling.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_knn_ad_defaults_to_exact_sampling() {
        let features = Array2::from_shape_vec((6, 1), vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).unwrap();
        let mut ad = KnnDistanceAD::fit(&features, 2, 2.0);
        ad.sample_step = resolve_knn_ad_sample_step_with("exact", 1, 7, features.nrows());
        assert_eq!(ad.sample_step, 1);
    }

    #[test]
    /// Runs regression test for knn ad approx mode sampling env.
    ///
    /// Function context: `crates/ferrum_core/src/ml/ad.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_knn_ad_approx_mode_sampling_env() {
        let features = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )
        .unwrap();
        let mut ad = KnnDistanceAD::fit(&features, 2, 2.0);
        ad.sample_step = resolve_knn_ad_sample_step_with("approx", 1, 3, features.nrows());
        assert_eq!(ad.sample_step, 3);
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
    fn test_solve_spd_system_identity_rhs() {
        let a = Array2::<f64>::eye(3);
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        let x = solve_spd_system(&a, &b);
        for i in 0..3 {
            for j in 0..2 {
                assert!((x[[i, j]] - b[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_solve_spd_system_reconstructs_rhs() {
        let a = Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0])
            .unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![2.0, -1.0, 0.5, 3.0, -2.0, 4.0]).unwrap();
        let x = solve_spd_system(&a, &b);
        let recon = a.dot(&x);
        for i in 0..3 {
            for j in 0..2 {
                assert!((recon[[i, j]] - b[[i, j]]).abs() < 1e-8);
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
