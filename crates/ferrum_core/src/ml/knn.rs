#![allow(dead_code)]
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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Distance metric for k-NN.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Tanimoto,
}

/// Node for Vantage-Point Tree (VP-Tree)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpNode {
    pub vp_index: usize,
    pub threshold: f64,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

/// k-NN model (stores training data for lazy prediction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnRegressor {
    pub k: usize,
    pub metric: DistanceMetric,
    pub train_features: Array2<f32>,
    pub train_targets: Array1<f32>,
    pub vp_tree: Vec<VpNode>,
}

impl KnnRegressor {
    /// Train (store) the k-NN regressor.
    pub fn fit(
        features: ArrayView2<f64>,
        targets: ArrayView1<f64>,
        k: usize,
        metric: DistanceMetric,
    ) -> Self {
        assert_eq!(features.nrows(), targets.len());
        assert!(
            k > 0 && k <= features.nrows(),
            "k must be in [1, n_samples]"
        );
        let train_features = features.mapv(|v| v as f32);
        let vp_tree = build_vp_tree(&train_features, metric);
        Self {
            k,
            metric,
            train_features,
            train_targets: targets.mapv(|v| v as f32),
            vp_tree,
        }
    }

    /// Predict for a single sample.
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        let sample32: Vec<f32> = sample.iter().map(|&v| v as f32).collect();
        let mut best_dists = vec![f64::INFINITY; self.k];
        let mut best_indices = vec![0usize; self.k];
        knn_mean_target(
            &self.train_features,
            &self.train_targets,
            &self.vp_tree,
            &sample32,
            self.metric,
            &mut best_dists,
            &mut best_indices,
        )
    }

    /// Predict for multiple samples.
    pub fn predict(&self, features: ArrayView2<f64>) -> Array1<f64> {
        let preds = (0..features.nrows())
            .into_par_iter()
            .map_init(
                || {
                    (
                        vec![f64::INFINITY; self.k],
                        vec![0usize; self.k],
                        vec![0.0f32; features.ncols()],
                    )
                },
                |(best_dists, best_indices, sample32), i| {
                    let sample = features.row(i);
                    for (dst, &src) in sample32.iter_mut().zip(sample.iter()) {
                        *dst = src as f32;
                    }
                    knn_mean_target(
                        &self.train_features,
                        &self.train_targets,
                        &self.vp_tree,
                        sample32,
                        self.metric,
                        best_dists,
                        best_indices,
                    )
                },
            )
            .collect::<Vec<_>>();
        Array1::from_vec(preds)
    }

    /// Find k nearest neighbors. Returns Vec<(distance, target)>.
    pub fn find_k_nearest(&self, sample: &[f64]) -> Vec<(f64, f64)> {
        let sample32: Vec<f32> = sample.iter().map(|&v| v as f32).collect();
        let mut best_dists = vec![f64::INFINITY; self.k];
        let mut best_indices = vec![0usize; self.k];
        let filled = knn_select_k(
            &self.train_features,
            &self.vp_tree,
            &sample32,
            self.metric,
            &mut best_dists,
            &mut best_indices,
        );
        let mut neighbors = Vec::with_capacity(filled);
        for i in 0..filled {
            neighbors.push((best_dists[i], f64::from(self.train_targets[best_indices[i]])));
        }
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0));
        neighbors
    }

    /// Mean distance to k nearest neighbors (useful for applicability domain).
    pub fn mean_knn_distance(&self, sample: &[f64]) -> f64 {
        let sample32: Vec<f32> = sample.iter().map(|&v| v as f32).collect();
        let mut best_dists = vec![f64::INFINITY; self.k];
        let mut best_indices = vec![0usize; self.k];
        let filled = knn_select_k(
            &self.train_features,
            &self.vp_tree,
            &sample32,
            self.metric,
            &mut best_dists,
            &mut best_indices,
        );
        if filled == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for &dist in &best_dists[..filled] {
            sum += dist;
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
    pub vp_tree: Vec<VpNode>,
}

impl KnnClassifier {
    /// Fits the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/ml/knn.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    pub fn fit(
        features: ArrayView2<f64>,
        targets: ArrayView1<f64>,
        k: usize,
        metric: DistanceMetric,
    ) -> Self {
        assert_eq!(features.nrows(), targets.len());
        assert!(k > 0 && k <= features.nrows());
        let train_features = features.mapv(|v| v as f32);
        let vp_tree = build_vp_tree(&train_features, metric);
        Self {
            k,
            metric,
            train_features,
            train_targets: targets.mapv(|v| v as f32),
            vp_tree,
        }
    }

    /// Predict class (0.0 or 1.0) for a single sample.
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        let prob = self.predict_proba_one(sample);
        if prob >= 0.5 { 1.0 } else { 0.0 }
    }

    /// Predict probability of positive class.
    pub fn predict_proba_one(&self, sample: &[f64]) -> f64 {
        let sample32: Vec<f32> = sample.iter().map(|&v| v as f32).collect();
        let mut best_dists = vec![f64::INFINITY; self.k];
        let mut best_indices = vec![0usize; self.k];
        knn_positive_fraction(
            &self.train_features,
            &self.train_targets,
            &self.vp_tree,
            &sample32,
            self.metric,
            &mut best_dists,
            &mut best_indices,
        )
    }

    /// Predict classes for multiple samples.
    pub fn predict(&self, features: ArrayView2<f64>) -> Array1<f64> {
        let preds = (0..features.nrows())
            .into_par_iter()
            .map_init(
                || {
                    (
                        vec![f64::INFINITY; self.k],
                        vec![0usize; self.k],
                        vec![0.0f32; features.ncols()],
                    )
                },
                |(best_dists, best_indices, sample32), i| {
                    let sample = features.row(i);
                    for (dst, &src) in sample32.iter_mut().zip(sample.iter()) {
                        *dst = src as f32;
                    }
                    let prob = knn_positive_fraction(
                        &self.train_features,
                        &self.train_targets,
                        &self.vp_tree,
                        sample32,
                        self.metric,
                        best_dists,
                        best_indices,
                    );
                    if prob >= 0.5 { 1.0 } else { 0.0 }
                },
            )
            .collect::<Vec<_>>();
        Array1::from_vec(preds)
    }
}

/// Select k nearest training rows for a sample.
fn knn_select_k(
    train_features: &Array2<f32>,
    vp_tree: &[VpNode],
    sample: &[f32],
    metric: DistanceMetric,
    best_dists: &mut [f64],
    best_indices: &mut [usize],
) -> usize {
    let k = best_dists.len();
    debug_assert_eq!(best_indices.len(), k);

    let mut filled = 0usize;
    let mut worst_idx = 0usize;
    let mut worst_dist = f64::INFINITY;

    let slice = train_features.as_slice_memory_order().unwrap();
    let ncols = train_features.ncols();

    if !vp_tree.is_empty() {
        vp_tree_search(
            vp_tree, slice, ncols, sample, metric, best_dists, best_indices,
            &mut filled, &mut worst_idx, &mut worst_dist,
        );
    } else {
        // Fallback to linear scan if tree is empty
        scan_rows_f32(train_features, |i, row| {
            let candidate_sq = match metric {
                DistanceMetric::Euclidean => squared_euclidean_distance_f32(sample, row),
                DistanceMetric::Tanimoto => tanimoto_distance_f32(sample, row),
            };
            let candidate = match metric {
                DistanceMetric::Euclidean => candidate_sq.sqrt(),
                DistanceMetric::Tanimoto => candidate_sq,
            };

            if filled < k {
                best_dists[filled] = candidate;
                best_indices[filled] = i;
                if candidate > worst_dist || worst_dist == f64::INFINITY {
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
    }

    filled
}

/// Compute the mean target value of the k nearest neighbors.
fn knn_mean_target(
    train_features: &Array2<f32>,
    train_targets: &Array1<f32>,
    vp_tree: &[VpNode],
    sample: &[f32],
    metric: DistanceMetric,
    best_dists: &mut [f64],
    best_indices: &mut [usize],
) -> f64 {
    let filled = knn_select_k(train_features, vp_tree, sample, metric, best_dists, best_indices);
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
    vp_tree: &[VpNode],
    sample: &[f32],
    metric: DistanceMetric,
    best_dists: &mut [f64],
    best_indices: &mut [usize],
) -> f64 {
    let filled = knn_select_k(train_features, vp_tree, sample, metric, best_dists, best_indices);
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
    let mut acc0 = 0.0;
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;
    let mut acc4 = 0.0;
    let mut acc5 = 0.0;
    let mut acc6 = 0.0;
    let mut acc7 = 0.0;

    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);

    for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        let d0 = chunk_a[0] - chunk_b[0];
        let d1 = chunk_a[1] - chunk_b[1];
        let d2 = chunk_a[2] - chunk_b[2];
        let d3 = chunk_a[3] - chunk_b[3];
        let d4 = chunk_a[4] - chunk_b[4];
        let d5 = chunk_a[5] - chunk_b[5];
        let d6 = chunk_a[6] - chunk_b[6];
        let d7 = chunk_a[7] - chunk_b[7];

        acc0 = d0.mul_add(d0, acc0);
        acc1 = d1.mul_add(d1, acc1);
        acc2 = d2.mul_add(d2, acc2);
        acc3 = d3.mul_add(d3, acc3);
        acc4 = d4.mul_add(d4, acc4);
        acc5 = d5.mul_add(d5, acc5);
        acc6 = d6.mul_add(d6, acc6);
        acc7 = d7.mul_add(d7, acc7);
    }

    let mut acc = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    for (&x, &y) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
        let d = x - y;
        acc = d.mul_add(d, acc);
    }
    acc
}

/// Compute squared Euclidean distance on f32 vectors.
fn squared_euclidean_distance_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;
    let mut acc6 = 0.0f32;
    let mut acc7 = 0.0f32;

    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);

    for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        let d0 = chunk_a[0] - chunk_b[0];
        let d1 = chunk_a[1] - chunk_b[1];
        let d2 = chunk_a[2] - chunk_b[2];
        let d3 = chunk_a[3] - chunk_b[3];
        let d4 = chunk_a[4] - chunk_b[4];
        let d5 = chunk_a[5] - chunk_b[5];
        let d6 = chunk_a[6] - chunk_b[6];
        let d7 = chunk_a[7] - chunk_b[7];

        acc0 = d0.mul_add(d0, acc0);
        acc1 = d1.mul_add(d1, acc1);
        acc2 = d2.mul_add(d2, acc2);
        acc3 = d3.mul_add(d3, acc3);
        acc4 = d4.mul_add(d4, acc4);
        acc5 = d5.mul_add(d5, acc5);
        acc6 = d6.mul_add(d6, acc6);
        acc7 = d7.mul_add(d7, acc7);
    }

    let mut acc = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    for (&x, &y) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
        let d = x - y;
        acc = d.mul_add(d, acc);
    }
    f64::from(acc)
}

/// Compute Tanimoto distance = 1 - Tanimoto similarity.
fn tanimoto_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut ab0 = 0.0; let mut aa0 = 0.0; let mut bb0 = 0.0;
    let mut ab1 = 0.0; let mut aa1 = 0.0; let mut bb1 = 0.0;
    let mut ab2 = 0.0; let mut aa2 = 0.0; let mut bb2 = 0.0;
    let mut ab3 = 0.0; let mut aa3 = 0.0; let mut bb3 = 0.0;
    let mut ab4 = 0.0; let mut aa4 = 0.0; let mut bb4 = 0.0;
    let mut ab5 = 0.0; let mut aa5 = 0.0; let mut bb5 = 0.0;
    let mut ab6 = 0.0; let mut aa6 = 0.0; let mut bb6 = 0.0;
    let mut ab7 = 0.0; let mut aa7 = 0.0; let mut bb7 = 0.0;

    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);

    for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        let x0 = chunk_a[0]; let y0 = chunk_b[0];
        let x1 = chunk_a[1]; let y1 = chunk_b[1];
        let x2 = chunk_a[2]; let y2 = chunk_b[2];
        let x3 = chunk_a[3]; let y3 = chunk_b[3];
        let x4 = chunk_a[4]; let y4 = chunk_b[4];
        let x5 = chunk_a[5]; let y5 = chunk_b[5];
        let x6 = chunk_a[6]; let y6 = chunk_b[6];
        let x7 = chunk_a[7]; let y7 = chunk_b[7];

        ab0 = x0.mul_add(y0, ab0); aa0 = x0.mul_add(x0, aa0); bb0 = y0.mul_add(y0, bb0);
        ab1 = x1.mul_add(y1, ab1); aa1 = x1.mul_add(x1, aa1); bb1 = y1.mul_add(y1, bb1);
        ab2 = x2.mul_add(y2, ab2); aa2 = x2.mul_add(x2, aa2); bb2 = y2.mul_add(y2, bb2);
        ab3 = x3.mul_add(y3, ab3); aa3 = x3.mul_add(x3, aa3); bb3 = y3.mul_add(y3, bb3);
        ab4 = x4.mul_add(y4, ab4); aa4 = x4.mul_add(x4, aa4); bb4 = y4.mul_add(y4, bb4);
        ab5 = x5.mul_add(y5, ab5); aa5 = x5.mul_add(x5, aa5); bb5 = y5.mul_add(y5, bb5);
        ab6 = x6.mul_add(y6, ab6); aa6 = x6.mul_add(x6, aa6); bb6 = y6.mul_add(y6, bb6);
        ab7 = x7.mul_add(y7, ab7); aa7 = x7.mul_add(x7, aa7); bb7 = y7.mul_add(y7, bb7);
    }

    let mut ab = ab0 + ab1 + ab2 + ab3 + ab4 + ab5 + ab6 + ab7;
    let mut aa = aa0 + aa1 + aa2 + aa3 + aa4 + aa5 + aa6 + aa7;
    let mut bb = bb0 + bb1 + bb2 + bb3 + bb4 + bb5 + bb6 + bb7;

    for (&x, &y) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
        ab = x.mul_add(y, ab);
        aa = x.mul_add(x, aa);
        bb = y.mul_add(y, bb);
    }
    let denom = aa + bb - ab;
    if denom < 1e-15 { 0.0 } else { 1.0 - ab / denom }
}

/// Compute Tanimoto distance on f32 vectors.
fn tanimoto_distance_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut ab0 = 0.0f32; let mut aa0 = 0.0f32; let mut bb0 = 0.0f32;
    let mut ab1 = 0.0f32; let mut aa1 = 0.0f32; let mut bb1 = 0.0f32;
    let mut ab2 = 0.0f32; let mut aa2 = 0.0f32; let mut bb2 = 0.0f32;
    let mut ab3 = 0.0f32; let mut aa3 = 0.0f32; let mut bb3 = 0.0f32;
    let mut ab4 = 0.0f32; let mut aa4 = 0.0f32; let mut bb4 = 0.0f32;
    let mut ab5 = 0.0f32; let mut aa5 = 0.0f32; let mut bb5 = 0.0f32;
    let mut ab6 = 0.0f32; let mut aa6 = 0.0f32; let mut bb6 = 0.0f32;
    let mut ab7 = 0.0f32; let mut aa7 = 0.0f32; let mut bb7 = 0.0f32;

    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);

    for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        let x0 = chunk_a[0]; let y0 = chunk_b[0];
        let x1 = chunk_a[1]; let y1 = chunk_b[1];
        let x2 = chunk_a[2]; let y2 = chunk_b[2];
        let x3 = chunk_a[3]; let y3 = chunk_b[3];
        let x4 = chunk_a[4]; let y4 = chunk_b[4];
        let x5 = chunk_a[5]; let y5 = chunk_b[5];
        let x6 = chunk_a[6]; let y6 = chunk_b[6];
        let x7 = chunk_a[7]; let y7 = chunk_b[7];

        ab0 = x0.mul_add(y0, ab0); aa0 = x0.mul_add(x0, aa0); bb0 = y0.mul_add(y0, bb0);
        ab1 = x1.mul_add(y1, ab1); aa1 = x1.mul_add(x1, aa1); bb1 = y1.mul_add(y1, bb1);
        ab2 = x2.mul_add(y2, ab2); aa2 = x2.mul_add(x2, aa2); bb2 = y2.mul_add(y2, bb2);
        ab3 = x3.mul_add(y3, ab3); aa3 = x3.mul_add(x3, aa3); bb3 = y3.mul_add(y3, bb3);
        ab4 = x4.mul_add(y4, ab4); aa4 = x4.mul_add(x4, aa4); bb4 = y4.mul_add(y4, bb4);
        ab5 = x5.mul_add(y5, ab5); aa5 = x5.mul_add(x5, aa5); bb5 = y5.mul_add(y5, bb5);
        ab6 = x6.mul_add(y6, ab6); aa6 = x6.mul_add(x6, aa6); bb6 = y6.mul_add(y6, bb6);
        ab7 = x7.mul_add(y7, ab7); aa7 = x7.mul_add(x7, aa7); bb7 = y7.mul_add(y7, bb7);
    }

    let mut ab = ab0 + ab1 + ab2 + ab3 + ab4 + ab5 + ab6 + ab7;
    let mut aa = aa0 + aa1 + aa2 + aa3 + aa4 + aa5 + aa6 + aa7;
    let mut bb = bb0 + bb1 + bb2 + bb3 + bb4 + bb5 + bb6 + bb7;

    for (&x, &y) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
        ab = x.mul_add(y, ab);
        aa = x.mul_add(x, aa);
        bb = y.mul_add(y, bb);
    }
    let denom = aa + bb - ab;
    if denom < 1e-15f32 {
        0.0
    } else {
        1.0 - f64::from(ab / denom)
    }
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

/// Iterate f64 rows of a feature matrix view with a contiguous fast path.
#[inline]
fn scan_rows_f64_view(features: ArrayView2<f64>, mut f: impl FnMut(usize, &[f64])) {
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


pub(crate) fn vp_tree_search(
    nodes: &[VpNode],
    slice: &[f32],
    ncols: usize,
    sample: &[f32],
    metric: DistanceMetric,
    best_dists: &mut [f64],
    best_indices: &mut [usize],
    filled: &mut usize,
    worst_idx: &mut usize,
    worst_dist: &mut f64,
) {
    if nodes.is_empty() {
        return;
    }

    let k = best_dists.len();
    let mut stack = Vec::with_capacity(64);
    stack.push(0usize);

    while let Some(node_idx) = stack.pop() {
        let node = &nodes[node_idx];
        let vp_start = node.vp_index * ncols;
        let vp_feature = &slice[vp_start..vp_start + ncols];

        let d_sq = match metric {
            DistanceMetric::Euclidean => squared_euclidean_distance_f32(sample, vp_feature),
            DistanceMetric::Tanimoto => tanimoto_distance_f32(sample, vp_feature),
        };
        let d_true = match metric {
            DistanceMetric::Euclidean => d_sq.sqrt(),
            DistanceMetric::Tanimoto => d_sq,
        };

        if *filled < k {
            best_dists[*filled] = d_true;
            best_indices[*filled] = node.vp_index;
            if d_true > *worst_dist || *worst_dist == f64::INFINITY {
                *worst_dist = d_true;
                *worst_idx = *filled;
            }
            *filled += 1;
            if *filled == k {
                let (idx, val) = largest_entry(best_dists, k);
                *worst_idx = idx;
                *worst_dist = val;
            }
        } else if d_true < *worst_dist {
            best_dists[*worst_idx] = d_true;
            best_indices[*worst_idx] = node.vp_index;
            let (idx, val) = largest_entry(best_dists, k);
            *worst_idx = idx;
            *worst_dist = val;
        }

        let worst_dist_true = if *filled < k {
            f64::INFINITY
        } else {
            *worst_dist
        };

        let is_left_first = d_true < node.threshold;
        let (first, second) = if is_left_first {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(sec) = second {
            let intersects = if is_left_first {
                d_true + worst_dist_true >= node.threshold
            } else {
                d_true - worst_dist_true <= node.threshold
            };
            if intersects {
                stack.push(sec);
            }
        }

        if let Some(fst) = first {
            stack.push(fst);
        }
    }
}

pub(crate) fn build_vp_tree(
    train_features: &Array2<f32>,
    metric: DistanceMetric,
) -> Vec<VpNode> {
    let n = train_features.nrows();
    if n == 0 {
        return vec![];
    }
    let mut indices: Vec<usize> = (0..n).collect();
    let mut nodes = Vec::new();
    let slice = train_features.as_slice_memory_order().unwrap();
    let ncols = train_features.ncols();

    /// Implements build recursive.
    ///
    /// Function context: `crates/ferrum_core/src/ml/knn.rs`.
    /// ML-layer note: optimized for batch computation and zero-allocation passes.
    fn build_recursive(
        indices: &mut [usize],
        slice: &[f32],
        ncols: usize,
        metric: DistanceMetric,
        nodes: &mut Vec<VpNode>,
    ) -> Option<usize> {
        if indices.is_empty() {
            return None;
        }

        let node_idx = nodes.len();
        nodes.push(VpNode {
            vp_index: 0,
            threshold: 0.0,
            left: None,
            right: None,
        });

        // Pick vantage point (e.g. last element to avoid RNG overhead)
        let vp_idx = indices[indices.len() - 1];
        let vp_start = vp_idx * ncols;
        let vp_feature = &slice[vp_start..vp_start + ncols];

        if indices.len() == 1 {
            nodes[node_idx].vp_index = vp_idx;
            return Some(node_idx);
        }

        let mut dists = Vec::with_capacity(indices.len() - 1);
        for &i in &indices[..indices.len() - 1] {
            let row_start = i * ncols;
            let row = &slice[row_start..row_start + ncols];
            let d_sq = match metric {
                DistanceMetric::Euclidean => squared_euclidean_distance_f32(vp_feature, row),
                DistanceMetric::Tanimoto => tanimoto_distance_f32(vp_feature, row),
            };
            let d_true = match metric {
                DistanceMetric::Euclidean => d_sq.sqrt(),
                DistanceMetric::Tanimoto => d_sq,
            };
            dists.push((i, d_true));
        }

        let mid = dists.len() / 2;
        dists.select_nth_unstable_by(mid, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let median_dist = dists[mid].1;

        let mut left_indices = Vec::with_capacity(mid);
        let mut right_indices = Vec::with_capacity(dists.len() - mid);
        for (i, d) in dists {
            if d < median_dist {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        nodes[node_idx].vp_index = vp_idx;
        nodes[node_idx].threshold = median_dist;

        nodes[node_idx].left = build_recursive(&mut left_indices, slice, ncols, metric, nodes);
        nodes[node_idx].right = build_recursive(&mut right_indices, slice, ncols, metric, nodes);

        Some(node_idx)
    }

    build_recursive(&mut indices, slice, ncols, metric, &mut nodes);
    nodes
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
        let model = KnnRegressor::fit(features.view(), targets.view(), 1, DistanceMetric::Euclidean);

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
        let model = KnnRegressor::fit(features.view(), targets.view(), 3, DistanceMetric::Euclidean);

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
        let model = KnnClassifier::fit(features.view(), targets.view(), 3, DistanceMetric::Euclidean);

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
        let model = KnnRegressor::fit(features.view(), targets.view(), 1, DistanceMetric::Euclidean);

        let test = Array2::from_shape_vec((2, 1), vec![0.0, 10.0]).unwrap();
        let preds = model.predict(test.view());
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
        let model = KnnRegressor::fit(features.view(), targets.view(), 2, DistanceMetric::Euclidean);
        // Distance from 0.5 to nearest 2: [0, 1] → distances [0.5, 0.5] → mean = 0.5
        let dist = model.mean_knn_distance(&[0.5]);
        assert!((dist - 0.5).abs() < 1e-6);
    }
}
