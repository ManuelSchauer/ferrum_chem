//! k-Nearest Neighbors — regression and classification.
//!
//! Supports Euclidean distance (for descriptors) and Tanimoto distance (for fingerprints).

//! ## File guide: Knn
//!
//! This file belongs to FerrumChem's **machine-learning pipeline layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `enum DistanceMetric`
//! - `struct KnnRegressor`
//! - `fn fit`
//! - `fn predict_one`
//! - `fn predict`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Knn` assumes `ndarray` row-major sample layout (`n_samples x n_features`) throughout fit/predict paths.
//! - `Knn` prioritizes reproducibility via explicit seeds and deterministic iteration order where applicable.
//! - Model quality metrics for `Knn` depend on consistent preprocessing choices (scaling, splitting, and leakage control).
//! - Specific scope anchor: `crates/ferrum_core/src/ml/knn.rs` defines knn behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Canonical ML literature for `Knn` algorithm family (statistical learning and validation texts).
//! - scikit-learn/RDKit-style API ergonomics that inform `Knn` fit/transform/predict contracts.
//! - OECD QSAR validation principles where `Knn` is used in predictive chemistry contexts.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/ml/knn.rs`.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Distance metric for k-NN.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Tanimoto,
}

/// k-NN model (stores training data for lazy prediction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnRegressor {
    pub k: usize,
    pub metric: DistanceMetric,
    pub train_features: Array2<f32>,
    pub train_targets: Array1<f32>,
}

impl KnnRegressor {
    /// Train (store) the k-NN regressor.
    pub fn fit(
        features: Array2<f64>,
        targets: Array1<f64>,
        k: usize,
        metric: DistanceMetric,
    ) -> Self {
        assert_eq!(features.nrows(), targets.len());
        assert!(
            k > 0 && k <= features.nrows(),
            "k must be in [1, n_samples]"
        );
        Self {
            k,
            metric,
            train_features: features.mapv(|v| v as f32),
            train_targets: targets.mapv(|v| v as f32),
        }
    }

    /// Predict for a single sample.
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        let mut best_dists = vec![f64::INFINITY; self.k];
        let mut best_indices = vec![0usize; self.k];
        knn_mean_target(
            &self.train_features,
            &self.train_targets,
            sample,
            self.metric,
            &mut best_dists,
            &mut best_indices,
        )
    }

    /// Predict for multiple samples.
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        let mut best_dists = vec![f64::INFINITY; self.k];
        let mut best_indices = vec![0usize; self.k];
        let mut preds = Vec::with_capacity(features.nrows());
        scan_rows_f64(features, |_, sample| {
            preds.push(knn_mean_target(
                &self.train_features,
                &self.train_targets,
                sample,
                self.metric,
                &mut best_dists,
                &mut best_indices,
            ));
        });
        Array1::from_vec(preds)
    }

    /// Find k nearest neighbors. Returns Vec<(distance, target)>.
    pub fn find_k_nearest(&self, sample: &[f64]) -> Vec<(f64, f64)> {
        let mut best_dists = vec![f64::INFINITY; self.k];
        let mut best_indices = vec![0usize; self.k];
        let filled = knn_select_k(
            &self.train_features,
            sample,
            self.metric,
            &mut best_dists,
            &mut best_indices,
        );
        let mut neighbors = Vec::with_capacity(filled);
        for i in 0..filled {
            let dist = if matches!(self.metric, DistanceMetric::Euclidean) {
                best_dists[i].sqrt()
            } else {
                best_dists[i]
            };
            neighbors.push((dist, f64::from(self.train_targets[best_indices[i]])));
        }
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0));
        neighbors
    }

    /// Mean distance to k nearest neighbors (useful for applicability domain).
    pub fn mean_knn_distance(&self, sample: &[f64]) -> f64 {
        let mut best_dists = vec![f64::INFINITY; self.k];
        let mut best_indices = vec![0usize; self.k];
        let filled = knn_select_k(
            &self.train_features,
            sample,
            self.metric,
            &mut best_dists,
            &mut best_indices,
        );
        if filled == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        if matches!(self.metric, DistanceMetric::Euclidean) {
            for &dist_sq in &best_dists[..filled] {
                sum += dist_sq.sqrt();
            }
        } else {
            for &dist in &best_dists[..filled] {
                sum += dist;
            }
        }
        sum / filled as f64
    }
}

/// k-NN classifier — majority vote.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnClassifier {
    pub k: usize,
    pub metric: DistanceMetric,
    pub train_features: Array2<f32>,
    pub train_targets: Array1<f32>,
}

impl KnnClassifier {
    /// Fits the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/ml/knn.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    pub fn fit(
        features: Array2<f64>,
        targets: Array1<f64>,
        k: usize,
        metric: DistanceMetric,
    ) -> Self {
        assert_eq!(features.nrows(), targets.len());
        assert!(k > 0 && k <= features.nrows());
        Self {
            k,
            metric,
            train_features: features.mapv(|v| v as f32),
            train_targets: targets.mapv(|v| v as f32),
        }
    }

    /// Predict class (0.0 or 1.0) for a single sample.
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        let prob = self.predict_proba_one(sample);
        if prob >= 0.5 { 1.0 } else { 0.0 }
    }

    /// Predict probability of positive class.
    pub fn predict_proba_one(&self, sample: &[f64]) -> f64 {
        let mut best_dists = vec![f64::INFINITY; self.k];
        let mut best_indices = vec![0usize; self.k];
        knn_positive_fraction(
            &self.train_features,
            &self.train_targets,
            sample,
            self.metric,
            &mut best_dists,
            &mut best_indices,
        )
    }

    /// Predict classes for multiple samples.
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        let mut best_dists = vec![f64::INFINITY; self.k];
        let mut best_indices = vec![0usize; self.k];
        let mut preds = Vec::with_capacity(features.nrows());
        scan_rows_f64(features, |_, sample| {
            let prob = knn_positive_fraction(
                &self.train_features,
                &self.train_targets,
                sample,
                self.metric,
                &mut best_dists,
                &mut best_indices,
            );
            preds.push(if prob >= 0.5 { 1.0 } else { 0.0 });
        });
        Array1::from_vec(preds)
    }
}

/// Select k nearest training rows for a sample.
fn knn_select_k(
    train_features: &Array2<f32>,
    sample: &[f64],
    metric: DistanceMetric,
    best_dists: &mut [f64],
    best_indices: &mut [usize],
) -> usize {
    let k = best_dists.len();
    debug_assert_eq!(best_indices.len(), k);

    let mut filled = 0usize;
    let mut worst_idx = 0usize;
    let mut worst_dist = f64::NEG_INFINITY;

    scan_rows_f32(train_features, |i, row| {
        let candidate = match metric {
            DistanceMetric::Euclidean => {
                let cutoff = if filled < k {
                    f64::INFINITY
                } else {
                    worst_dist
                };
                squared_euclidean_distance_with_cutoff_mixed(sample, row, cutoff)
            }
            DistanceMetric::Tanimoto => tanimoto_distance_mixed(sample, row),
        };

        if filled < k {
            best_dists[filled] = candidate;
            best_indices[filled] = i;
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
            best_indices[worst_idx] = i;
            let (idx, val) = largest_entry(best_dists, k);
            worst_idx = idx;
            worst_dist = val;
        }
    });

    filled
}

/// Compute the mean target value of the k nearest neighbors.
fn knn_mean_target(
    train_features: &Array2<f32>,
    train_targets: &Array1<f32>,
    sample: &[f64],
    metric: DistanceMetric,
    best_dists: &mut [f64],
    best_indices: &mut [usize],
) -> f64 {
    let filled = knn_select_k(train_features, sample, metric, best_dists, best_indices);
    if filled == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for &idx in &best_indices[..filled] {
        sum += f64::from(train_targets[idx]);
    }
    sum / filled as f64
}

/// Compute fraction of positives among k nearest neighbors.
fn knn_positive_fraction(
    train_features: &Array2<f32>,
    train_targets: &Array1<f32>,
    sample: &[f64],
    metric: DistanceMetric,
    best_dists: &mut [f64],
    best_indices: &mut [usize],
) -> f64 {
    let filled = knn_select_k(train_features, sample, metric, best_dists, best_indices);
    if filled == 0 {
        return 0.0;
    }
    let mut n_pos = 0usize;
    for &idx in &best_indices[..filled] {
        if f64::from(train_targets[idx]) >= 0.5 {
            n_pos += 1;
        }
    }
    n_pos as f64 / filled as f64
}

/// Return index and value of the largest entry among the first `count` items.
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

/// Compute squared Euclidean distance.
fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean_distance_with_cutoff(a, b, f64::INFINITY)
}

/// Compute squared Euclidean distance with early exit if partial sum exceeds cutoff.
fn squared_euclidean_distance_with_cutoff(a: &[f64], b: &[f64], cutoff: f64) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc = d.mul_add(d, acc);
        if acc > cutoff {
            break;
        }
    }
    acc
}

/// Compute squared Euclidean distance with mixed precision and early exit.
fn squared_euclidean_distance_with_cutoff_mixed(a: &[f64], b: &[f32], cutoff: f64) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        let d = a[i] - f64::from(b[i]);
        acc = d.mul_add(d, acc);
        if acc > cutoff {
            break;
        }
    }
    acc
}

/// Compute Tanimoto distance = 1 - Tanimoto similarity.
fn tanimoto_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut ab = 0.0;
    let mut aa = 0.0;
    let mut bb = 0.0;
    for i in 0..a.len() {
        let x = a[i];
        let y = b[i];
        ab = x.mul_add(y, ab);
        aa = x.mul_add(x, aa);
        bb = y.mul_add(y, bb);
    }
    let denom = aa + bb - ab;
    if denom < 1e-15 { 0.0 } else { 1.0 - ab / denom }
}

/// Compute Tanimoto distance with mixed precision.
fn tanimoto_distance_mixed(a: &[f64], b: &[f32]) -> f64 {
    let mut ab = 0.0;
    let mut aa = 0.0;
    let mut bb = 0.0;
    for i in 0..a.len() {
        let x = a[i];
        let y = f64::from(b[i]);
        ab = x.mul_add(y, ab);
        aa = x.mul_add(x, aa);
        bb = y.mul_add(y, bb);
    }
    let denom = aa + bb - ab;
    if denom < 1e-15 { 0.0 } else { 1.0 - ab / denom }
}

/// Iterate rows of a feature matrix with a contiguous fast path.
#[inline]
fn scan_rows_f32(features: &Array2<f32>, mut f: impl FnMut(usize, &[f32])) {
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

/// Iterate f64 rows of a feature matrix with a contiguous fast path.
#[inline]
fn scan_rows_f64(features: &Array2<f64>, mut f: impl FnMut(usize, &[f64])) {
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

/// Compute distance between two vectors.
pub fn compute_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => squared_euclidean_distance(a, b).sqrt(),
        DistanceMetric::Tanimoto => tanimoto_distance(a, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    /// Runs regression test for knn regressor simple.
    ///
    /// Function context: `crates/ferrum_core/src/ml/knn.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_knn_regressor_simple() {
        let features =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let targets = array![0.0, 1.0, 1.0, 2.0];
        let model = KnnRegressor::fit(features, targets, 1, DistanceMetric::Euclidean);

        // Query point at (0, 0) → nearest is itself → 0.0
        assert!((model.predict_one(&[0.0, 0.0]) - 0.0).abs() < 1e-6);
        // Query point at (1, 1) → nearest is itself → 2.0
        assert!((model.predict_one(&[1.0, 1.0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    /// Runs regression test for knn regressor k3.
    ///
    /// Function context: `crates/ferrum_core/src/ml/knn.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_knn_regressor_k3() {
        let features = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let targets = array![0.0, 2.0, 4.0, 6.0];
        let model = KnnRegressor::fit(features, targets, 3, DistanceMetric::Euclidean);

        // Query at 1.0 → 3 nearest: 0, 1, 2 → targets: 0, 2, 4 → mean = 2.0
        assert!((model.predict_one(&[1.0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    /// Runs regression test for knn classifier.
    ///
    /// Function context: `crates/ferrum_core/src/ml/knn.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_knn_classifier() {
        let features = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 1.0, 1.0, 1.1, 1.0, 1.0, 1.1],
        )
        .unwrap();
        let targets = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let model = KnnClassifier::fit(features, targets, 3, DistanceMetric::Euclidean);

        assert!((model.predict_one(&[0.05, 0.05]) - 0.0).abs() < 1e-6);
        assert!((model.predict_one(&[1.05, 1.05]) - 1.0).abs() < 1e-6);
    }

    #[test]
    /// Runs regression test for tanimoto distance.
    ///
    /// Function context: `crates/ferrum_core/src/ml/knn.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_tanimoto_distance() {
        let a = vec![1.0, 0.0, 1.0, 1.0];
        let b = vec![1.0, 0.0, 1.0, 1.0];
        assert!(compute_distance(&a, &b, DistanceMetric::Tanimoto).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0, 0.0];
        let d = compute_distance(&a, &c, DistanceMetric::Tanimoto);
        assert!(d > 0.9); // very different fingerprints
    }

    #[test]
    /// Runs regression test for batch predict.
    ///
    /// Function context: `crates/ferrum_core/src/ml/knn.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_batch_predict() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 5.0, 10.0]).unwrap();
        let targets = array![0.0, 5.0, 10.0];
        let model = KnnRegressor::fit(features, targets, 1, DistanceMetric::Euclidean);

        let test = Array2::from_shape_vec((2, 1), vec![0.0, 10.0]).unwrap();
        let preds = model.predict(&test);
        assert!((preds[0] - 0.0).abs() < 1e-6);
        assert!((preds[1] - 10.0).abs() < 1e-6);
    }

    #[test]
    /// Runs regression test for mean knn distance.
    ///
    /// Function context: `crates/ferrum_core/src/ml/knn.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_mean_knn_distance() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let targets = array![0.0, 1.0, 2.0];
        let model = KnnRegressor::fit(features, targets, 2, DistanceMetric::Euclidean);
        // Distance from 0.5 to nearest 2: [0, 1] → distances [0.5, 0.5] → mean = 0.5
        let dist = model.mean_knn_distance(&[0.5]);
        assert!((dist - 0.5).abs() < 1e-6);
    }
}
