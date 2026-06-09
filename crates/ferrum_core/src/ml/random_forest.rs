//! Random Forest — decision tree + bagging, from scratch.
//!
//! Supports both regression and classification.
//! Optimized for binary fingerprint features.

//! ## File guide: Random Forest
//!
//! This file belongs to FerrumChem's **machine-learning pipeline layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `enum TreeNode`
//! - `fn predict`
//! - `struct DecisionTreeBuilder`
//! - `fn build`
//! - `fn leaf_value`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Random Forest` assumes `ndarray` row-major sample layout (`n_samples x n_features`) throughout fit/predict paths.
//! - `Random Forest` prioritizes reproducibility via explicit seeds and deterministic iteration order where applicable.
//! - Model quality metrics for `Random Forest` depend on consistent preprocessing choices (scaling, splitting, and leakage control).
//! - Specific scope anchor: `crates/ferrum_core/src/ml/random_forest.rs` defines random forest behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Canonical ML literature for `Random Forest` algorithm family (statistical learning and validation texts).
//! - scikit-learn/RDKit-style API ergonomics that inform `Random Forest` fit/transform/predict contracts.
//! - OECD QSAR validation principles where `Random Forest` is used in predictive chemistry contexts.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/ml/random_forest.rs`.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Default memory budget for optional Random Forest feature pre-sorting.
const DEFAULT_RF_PRESORT_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Cached global row-ordering per feature for root-node split evaluation.
#[derive(Debug, Clone)]
struct FeaturePresort {
    orders: Vec<Vec<usize>>,
}

/// Parse a positive usize from an environment variable.
fn env_positive_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&v| v > 0)
}

/// Resolve the Random Forest pre-sort memory budget.
///
/// `FERRUM_RF_PRESORT_MAX_BYTES` can override the default budget.
fn resolve_rf_presort_max_bytes() -> usize {
    env_positive_usize("FERRUM_RF_PRESORT_MAX_BYTES").unwrap_or(DEFAULT_RF_PRESORT_MAX_BYTES)
}

/// Build feature-wise global row ordering when the memory budget allows it.
fn build_feature_presort_with_budget(
    features: ArrayView2<f64>,
    max_bytes: usize,
) -> Option<FeaturePresort> {
    let n_rows = features.nrows();
    let n_features = features.ncols();
    if n_rows == 0 || n_features == 0 {
        return None;
    }

    let index_count = n_rows.checked_mul(n_features)?;
    let required_bytes = index_count.checked_mul(std::mem::size_of::<usize>())?;
    if required_bytes > max_bytes {
        return None;
    }

    let orders = (0..n_features)
        .into_par_iter()
        .map(|feat_idx| {
            let mut rows: Vec<usize> = (0..n_rows).collect();
            rows.sort_unstable_by(|&a, &b| {
                features[[a, feat_idx]].total_cmp(&features[[b, feat_idx]])
            });
            rows
        })
        .collect::<Vec<_>>();

    Some(FeaturePresort { orders })
}

/// Try to build a shared feature pre-sort cache for Random Forest training.
fn maybe_build_feature_presort(features: ArrayView2<f64>) -> Option<Arc<FeaturePresort>> {
    let max_bytes = resolve_rf_presort_max_bytes();
    build_feature_presort_with_budget(features, max_bytes).map(Arc::new)
}

// ---------------------------------------------------------------------------
// Decision Tree Node
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Implements TreeNode.
///
/// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
/// ML-layer note: optimized for batch computation and zero-allocation passes.
enum TreeNode {
    Leaf {
        value: f32,
    },
    Split {
        feature_idx: usize,
        threshold: f32,
        left: u32,
        right: u32,
    },
}

impl TreeNode {
    /// Predicts the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn predict(&self, sample: &[f64], nodes: &[TreeNode]) -> f64 {
        let mut current = self;
        loop {
            match current {
                TreeNode::Leaf { value } => return f64::from(*value),
                TreeNode::Split {
                    feature_idx,
                    threshold,
                    left,
                    right,
                } => {
                    if sample[*feature_idx] <= f64::from(*threshold) {
                        current = &nodes[*left as usize];
                    } else {
                        current = &nodes[*right as usize];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decision Tree Builder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
/// Implements DecisionTreeBuilder.
///
/// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
/// ML-layer note: optimized for batch computation and zero-allocation passes.
struct DecisionTreeBuilder {
    max_depth: usize,
    min_samples_split: usize,
    max_features: usize, // number of features to consider at each split
    is_classifier: bool,
}

impl DecisionTreeBuilder {
    /// Evaluate root-node split candidates using a global per-feature row ordering cache.
    ///
    /// This path preserves bootstrap multiplicities by weighting each original row with
    /// its count in the bootstrap sample and only applies at depth 0.
    fn best_root_split_from_presort(
        &self,
        features: ArrayView2<f64>,
        targets: &[f64],
        indices: &[usize],
        selected_features: &[usize],
        presort: &FeaturePresort,
    ) -> Option<(f64, usize, f64)> {
        if indices.len() <= 1 {
            return None;
        }
        let n_rows = features.nrows();
        if n_rows == 0 {
            return None;
        }

        let mut row_counts = vec![0u32; n_rows];
        for &row_idx in indices {
            row_counts[row_idx] = row_counts[row_idx].saturating_add(1);
        }

        let mut best_score = f64::INFINITY;
        let mut best_feature = 0usize;
        let mut best_threshold = 0.0f64;
        let mut found_split = false;
        let mut active_rows: Vec<(f64, usize, usize)> = Vec::with_capacity(indices.len());

        for &feat_idx in selected_features {
            let Some(order) = presort.orders.get(feat_idx) else {
                continue;
            };

            active_rows.clear();
            for &row_idx in order {
                let weight = row_counts[row_idx] as usize;
                if weight > 0 {
                    active_rows.push((features[[row_idx, feat_idx]], row_idx, weight));
                }
            }
            if active_rows.len() <= 1 {
                continue;
            }

            if self.is_classifier {
                let mut left_count = 0usize;
                let mut right_count = indices.len();
                let mut left_pos = 0usize;
                let mut right_pos = active_rows
                    .iter()
                    .filter_map(|(_, row_idx, weight)| {
                        if targets[*row_idx] >= 0.5 {
                            Some(*weight)
                        } else {
                            None
                        }
                    })
                    .sum::<usize>();

                for split_idx in 0..(active_rows.len() - 1) {
                    let (current_val, row_idx, weight) = active_rows[split_idx];
                    let next_val = active_rows[split_idx + 1].0;

                    left_count += weight;
                    right_count -= weight;
                    if targets[row_idx] >= 0.5 {
                        left_pos += weight;
                        right_pos -= weight;
                    }

                    if !current_val.is_finite()
                        || !next_val.is_finite()
                        || current_val >= next_val
                        || left_count == 0
                        || right_count == 0
                    {
                        continue;
                    }

                    let left_n = left_count as f64;
                    let right_n = right_count as f64;
                    let n_total = (left_count + right_count) as f64;

                    let p_left = left_pos as f64 / left_n;
                    let p_right = right_pos as f64 / right_n;
                    let gini_left = 2.0 * p_left * (1.0 - p_left);
                    let gini_right = 2.0 * p_right * (1.0 - p_right);
                    let score = (left_n * gini_left + right_n * gini_right) / n_total;

                    if score < best_score {
                        best_score = score;
                        best_feature = feat_idx;
                        best_threshold = current_val + (next_val - current_val) * 0.5;
                        found_split = true;
                    }
                }
            } else {
                let mut left_count = 0usize;
                let mut right_count = indices.len();
                let mut left_sum = 0.0f64;
                let mut left_sum_sq = 0.0f64;
                let mut right_sum = 0.0f64;
                let mut right_sum_sq = 0.0f64;
                for (_, row_idx, weight) in &active_rows {
                    let t = targets[*row_idx];
                    let w = *weight as f64;
                    right_sum += w * t;
                    right_sum_sq += w * t * t;
                }

                for split_idx in 0..(active_rows.len() - 1) {
                    let (current_val, row_idx, weight) = active_rows[split_idx];
                    let next_val = active_rows[split_idx + 1].0;
                    let t = targets[row_idx];
                    let w = weight as f64;

                    left_count += weight;
                    right_count -= weight;
                    left_sum += w * t;
                    left_sum_sq += w * t * t;
                    right_sum -= w * t;
                    right_sum_sq -= w * t * t;

                    if !current_val.is_finite()
                        || !next_val.is_finite()
                        || current_val >= next_val
                        || left_count == 0
                        || right_count == 0
                    {
                        continue;
                    }

                    let left_n = left_count as f64;
                    let right_n = right_count as f64;
                    let n_total = (left_count + right_count) as f64;

                    let left_mean = left_sum / left_n;
                    let right_mean = right_sum / right_n;
                    let left_var = (left_sum_sq / left_n - left_mean * left_mean).max(0.0);
                    let right_var = (right_sum_sq / right_n - right_mean * right_mean).max(0.0);
                    let score = (left_n * left_var + right_n * right_var) / n_total;

                    if score < best_score {
                        best_score = score;
                        best_feature = feat_idx;
                        best_threshold = current_val + (next_val - current_val) * 0.5;
                        found_split = true;
                    }
                }
            }
        }

        if found_split {
            Some((best_score, best_feature, best_threshold))
        } else {
            None
        }
    }

    /// Builds the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn build(
        &self,
        features: ArrayView2<f64>,
        targets: &[f64],
        indices: &mut [usize],
        depth: usize,
        rng: &mut StdRng,
        root_presort: Option<&FeaturePresort>,
        scratch: &mut Vec<(f64, usize)>,
        nodes: &mut Vec<TreeNode>,
    ) -> u32 {
        let node_idx = nodes.len() as u32;
        // Push a dummy node to reserve the slot
        nodes.push(TreeNode::Leaf { value: 0.0 });
        if depth >= self.max_depth || indices.len() < self.min_samples_split || indices.len() <= 1 {
            nodes[node_idx as usize] = TreeNode::Leaf {
                value: self.leaf_value(targets, indices) as f32,
            };
            return node_idx;
        }

        let first_target = targets[indices[0]];
        if indices
            .iter()
            .all(|&i| (targets[i] - first_target).abs() < 1e-15)
        {
            nodes[node_idx as usize] = TreeNode::Leaf {
                value: first_target as f32,
            };
            return node_idx;
        }

        let n_features = features.ncols();
        let n_try = self.max_features.min(n_features);

        // Random feature subset
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        let (selected_features, _) = feature_indices.partial_shuffle(rng, n_try);

        let mut best_score = f64::INFINITY;
        let mut best_feature = 0usize;
        let mut best_threshold = 0.0;
        let mut found_split = false;
        // Using scratch buffer instead of allocating

        if depth == 0
            && let Some(presort) = root_presort
            && let Some((score, feature, threshold)) = self.best_root_split_from_presort(
                features,
                targets,
                indices,
                selected_features,
                presort,
            )
        {
            best_score = score;
            best_feature = feature;
            best_threshold = threshold;
            found_split = true;
        }

        for &feat_idx in selected_features.iter() {
            if depth == 0 && found_split {
                break;
            }
            // Sort candidate rows once by this feature.
            scratch.clear();
            scratch.extend(indices.iter().map(|&idx| (features[[idx, feat_idx]], idx)));
            scratch.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            if scratch.len() <= 1 {
                continue;
            }

            if self.is_classifier {
                let total_count = scratch.len();
                let mut left_count = 0usize;
                let mut right_count = total_count;
                let mut left_pos = 0usize;
                let mut right_pos = scratch
                    .iter()
                    .filter(|(_, idx)| targets[*idx] >= 0.5)
                    .count();

                for split_idx in 0..(total_count - 1) {
                    let (current_val, idx) = scratch[split_idx];
                    let next_val = scratch[split_idx + 1].0;

                    left_count += 1;
                    right_count -= 1;
                    if targets[idx] >= 0.5 {
                        left_pos += 1;
                        right_pos -= 1;
                    }

                    if !current_val.is_finite()
                        || !next_val.is_finite()
                        || current_val >= next_val
                        || left_count == 0
                        || right_count == 0
                    {
                        continue;
                    }

                    let left_n = left_count as f64;
                    let right_n = right_count as f64;
                    let n_total = (left_count + right_count) as f64;

                    let p_left = left_pos as f64 / left_n;
                    let p_right = right_pos as f64 / right_n;
                    let gini_left = 2.0 * p_left * (1.0 - p_left);
                    let gini_right = 2.0 * p_right * (1.0 - p_right);
                    let score = (left_n * gini_left + right_n * gini_right) / n_total;

                    if score < best_score {
                        best_score = score;
                        best_feature = feat_idx;
                        best_threshold = current_val + (next_val - current_val) * 0.5;
                        found_split = true;
                    }
                }
            } else {
                let total_count = scratch.len();
                let mut left_count = 0usize;
                let mut right_count = total_count;
                let mut left_sum = 0.0f64;
                let mut left_sum_sq = 0.0f64;
                let mut right_sum = 0.0f64;
                let mut right_sum_sq = 0.0f64;
                for (_, idx) in &*scratch {
                    let t = targets[*idx];
                    right_sum += t;
                    right_sum_sq += t * t;
                }

                for split_idx in 0..(total_count - 1) {
                    let (current_val, idx) = scratch[split_idx];
                    let next_val = scratch[split_idx + 1].0;
                    let t = targets[idx];

                    left_count += 1;
                    right_count -= 1;
                    left_sum += t;
                    left_sum_sq += t * t;
                    right_sum -= t;
                    right_sum_sq -= t * t;

                    if !current_val.is_finite()
                        || !next_val.is_finite()
                        || current_val >= next_val
                        || left_count == 0
                        || right_count == 0
                    {
                        continue;
                    }

                    let left_n = left_count as f64;
                    let right_n = right_count as f64;
                    let n_total = (left_count + right_count) as f64;

                    let left_mean = left_sum / left_n;
                    let right_mean = right_sum / right_n;
                    let left_var = (left_sum_sq / left_n - left_mean * left_mean).max(0.0);
                    let right_var = (right_sum_sq / right_n - right_mean * right_mean).max(0.0);
                    let score = (left_n * left_var + right_n * right_var) / n_total;

                    if score < best_score {
                        best_score = score;
                        best_feature = feat_idx;
                        best_threshold = current_val + (next_val - current_val) * 0.5;
                        found_split = true;
                    }
                }
            }
        }

        if !found_split {
            nodes[node_idx as usize] = TreeNode::Leaf {
                value: self.leaf_value(targets, indices) as f32,
            };
            return node_idx;
        }

        let mut left = 0usize;
        let mut right = indices.len();
        while left < right {
            let idx = indices[left];
            if features[[idx, best_feature]] <= best_threshold {
                left += 1;
            } else {
                right -= 1;
                indices.swap(left, right);
            }
        }
        let (left_indices, right_indices) = indices.split_at_mut(left);

        if left_indices.is_empty() || right_indices.is_empty() {
            nodes[node_idx as usize] = TreeNode::Leaf {
                value: self.leaf_value(targets, indices) as f32,
            };
            return node_idx;
        }

        let left_idx = self.build(
            features,
            targets,
            left_indices,
            depth + 1,
            rng,
            root_presort,
            scratch,
            nodes,
        );
        let right_idx = self.build(
            features,
            targets,
            right_indices,
            depth + 1,
            rng,
            root_presort,
            scratch,
            nodes,
        );

        nodes[node_idx as usize] = TreeNode::Split {
            feature_idx: best_feature,
            threshold: best_threshold as f32,
            left: left_idx,
            right: right_idx,
        };

        node_idx
    }

    /// Implements value.
    ///
    /// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn leaf_value(&self, targets: &[f64], indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }
        if self.is_classifier {
            // Majority vote
            let n_pos = indices.iter().filter(|&&i| targets[i] >= 0.5).count();
            if n_pos * 2 >= indices.len() { 1.0 } else { 0.0 }
        } else {
            // Mean
            let sum: f64 = indices.iter().map(|&i| targets[i]).sum();
            sum / indices.len() as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Random Forest Regressor
// ---------------------------------------------------------------------------

/// Random Forest Regressor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestRegressor {
    trees: Vec<Vec<TreeNode>>,
    pub n_estimators: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub max_features_fraction: f64,
    pub feature_importances: Vec<f64>,
}

impl RandomForestRegressor {
    /// Train a random forest regressor.
    pub fn fit(
        features: ArrayView2<f64>,
        targets: ArrayView1<f64>,
        n_estimators: usize,
        max_depth: usize,
        min_samples_split: usize,
        max_features_fraction: f64,
        seed: u64,
    ) -> Self {
        let n = features.nrows();
        let n_features = features.ncols();
        let max_features = ((n_features as f64 * max_features_fraction).round() as usize).max(1);
        let target_slice: Vec<f64> = targets.to_vec();

        let builder = DecisionTreeBuilder {
            max_depth,
            min_samples_split,
            max_features,
            is_classifier: false,
        };
        let feature_presort = maybe_build_feature_presort(features);

        let trees = (0..n_estimators)
            .into_par_iter()
            .map_init(
                || (Vec::with_capacity(n), Vec::with_capacity(n)),
                |(indices, scratch), tree_idx| {
                    let mut rng = StdRng::seed_from_u64(derive_tree_seed(seed, tree_idx));
                    indices.clear();
                    indices.extend((0..n).map(|_| rng.gen_range(0..n)));
                    let mut nodes = Vec::new();
                    let _ = builder.build(
                        features,
                        &target_slice,
                        indices,
                        0,
                        &mut rng,
                        feature_presort.as_deref(),
                        scratch,
                        &mut nodes,
                    );
                    nodes
                },
            )
            .collect::<Vec<_>>();
        let mut feature_counts = vec![0u64; n_features];
        for tree in &trees {
            if !tree.is_empty() {
                count_feature_usage(0, tree, &mut feature_counts);
            }
        }

        // Normalize feature importances
        let total: f64 = feature_counts.iter().sum::<u64>() as f64;
        let feature_importances = if total > 0.0 {
            feature_counts.iter().map(|&c| c as f64 / total).collect()
        } else {
            vec![0.0; n_features]
        };

        Self {
            trees,
            n_estimators,
            max_depth,
            min_samples_split,
            max_features_fraction,
            feature_importances,
        }
    }

    /// Predict for a single sample (mean of all trees).
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        let sum: f64 = self.trees.iter().map(|t| if t.is_empty() { 0.0 } else { t[0].predict(sample, t) }).sum();
        sum / self.trees.len() as f64
    }

    /// Predict for multiple samples.
    pub fn predict(&self, features: ArrayView2<f64>) -> Array1<f64> {
        let values = (0..features.nrows())
            .into_par_iter()
            .map(|i| {
                let row = features.row(i);
                if let Some(slice) = row.as_slice() {
                    self.predict_one(slice)
                } else {
                    let owned = row.to_vec();
                    self.predict_one(&owned)
                }
            })
            .collect::<Vec<_>>();
        Array1::from_vec(values)
    }
}

// ---------------------------------------------------------------------------
// Random Forest Classifier
// ---------------------------------------------------------------------------

/// Random Forest Classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestClassifier {
    trees: Vec<Vec<TreeNode>>,
    pub n_estimators: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub max_features_fraction: f64,
    pub feature_importances: Vec<f64>,
}

impl RandomForestClassifier {
    /// Train a random forest classifier.
    pub fn fit(
        features: ArrayView2<f64>,
        targets: ArrayView1<f64>,
        n_estimators: usize,
        max_depth: usize,
        min_samples_split: usize,
        max_features_fraction: f64,
        seed: u64,
    ) -> Self {
        let n = features.nrows();
        let n_features = features.ncols();
        let max_features = ((n_features as f64 * max_features_fraction).round() as usize).max(1);
        let target_slice: Vec<f64> = targets.to_vec();

        let builder = DecisionTreeBuilder {
            max_depth,
            min_samples_split,
            max_features,
            is_classifier: true,
        };
        let feature_presort = maybe_build_feature_presort(features);

        let trees = (0..n_estimators)
            .into_par_iter()
            .map_init(
                || (Vec::with_capacity(n), Vec::with_capacity(n)),
                |(indices, scratch), tree_idx| {
                    let mut rng = StdRng::seed_from_u64(derive_tree_seed(seed, tree_idx));
                    indices.clear();
                    indices.extend((0..n).map(|_| rng.gen_range(0..n)));
                    let mut nodes = Vec::new();
                    let _ = builder.build(
                        features,
                        &target_slice,
                        indices,
                        0,
                        &mut rng,
                        feature_presort.as_deref(),
                        scratch,
                        &mut nodes,
                    );
                    nodes
                },
            )
            .collect::<Vec<_>>();
        let mut feature_counts = vec![0u64; n_features];
        for tree in &trees {
            if !tree.is_empty() {
                count_feature_usage(0, tree, &mut feature_counts);
            }
        }

        let total: f64 = feature_counts.iter().sum::<u64>() as f64;
        let feature_importances = if total > 0.0 {
            feature_counts.iter().map(|&c| c as f64 / total).collect()
        } else {
            vec![0.0; n_features]
        };

        Self {
            trees,
            n_estimators,
            max_depth,
            min_samples_split,
            max_features_fraction,
            feature_importances,
        }
    }

    /// Predict class probability (fraction of trees voting positive).
    pub fn predict_proba_one(&self, sample: &[f64]) -> f64 {
        let votes: f64 = self.trees.iter().map(|t| if t.is_empty() { 0.0 } else { t[0].predict(sample, t) }).sum();
        votes / self.trees.len() as f64
    }

    /// Predict class (0.0 or 1.0) for a single sample.
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        if self.predict_proba_one(sample) >= 0.5 {
            1.0
        } else {
            0.0
        }
    }

    /// Predict for multiple samples.
    pub fn predict(&self, features: ArrayView2<f64>) -> Array1<f64> {
        let values = (0..features.nrows())
            .into_par_iter()
            .map(|i| {
                let row = features.row(i);
                if let Some(slice) = row.as_slice() {
                    self.predict_one(slice)
                } else {
                    let owned = row.to_vec();
                    self.predict_one(&owned)
                }
            })
            .collect::<Vec<_>>();
        Array1::from_vec(values)
    }
}

/// Implements tree seed.
///
/// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
/// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
fn derive_tree_seed(base_seed: u64, tree_idx: usize) -> u64 {
    let mut z = base_seed ^ ((tree_idx as u64).wrapping_add(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn count_feature_usage(node_idx: u32, nodes: &[TreeNode], counts: &mut [u64]) {
    match &nodes[node_idx as usize] {
        TreeNode::Leaf { .. } => {}
        TreeNode::Split {
            feature_idx,
            left,
            right,
            ..
        } => {
            counts[*feature_idx] += 1;
            count_feature_usage(*left, nodes, counts);
            count_feature_usage(*right, nodes, counts);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};
    use rand::SeedableRng;

    #[test]
    /// Runs regression test for rf regressor basic.
    ///
    /// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_rf_regressor_basic() {
        // Simple linear relationship: y = x
        let features =
            Array2::from_shape_vec((20, 1), (0..20).map(|i| i as f64).collect()).unwrap();
        let targets = Array1::from_vec((0..20).map(|i| i as f64).collect());

        let model = RandomForestRegressor::fit(features.view(), targets.view(), 50, 10, 2, 1.0, 42);

        // Should predict reasonably for in-range values
        let pred = model.predict_one(&[5.0]);
        assert!(
            (pred - 5.0).abs() < 3.0,
            "Prediction {pred} too far from 5.0"
        );
    }

    #[test]
    /// Runs regression test for rf classifier basic.
    ///
    /// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_rf_classifier_basic() {
        // Two clusters
        let mut feat_data = Vec::new();
        let mut target_data = Vec::new();
        for i in 0..20 {
            feat_data.push(i as f64);
            feat_data.push(0.0);
            target_data.push(if i < 10 { 0.0 } else { 1.0 });
        }
        let features = Array2::from_shape_vec((20, 2), feat_data).unwrap();
        let targets = Array1::from_vec(target_data);

        let model = RandomForestClassifier::fit(features.view(), targets.view(), 50, 10, 2, 1.0, 42);

        assert!((model.predict_one(&[2.0, 0.0]) - 0.0).abs() < 1e-10);
        assert!((model.predict_one(&[15.0, 0.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    /// Runs regression test for rf feature importances.
    ///
    /// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_rf_feature_importances() {
        // Feature 0 is informative, feature 1 is noise
        let n = 50;
        let mut feat = Vec::new();
        let mut tgt = Vec::new();
        for i in 0..n {
            feat.push(i as f64);
            feat.push(42.0); // constant noise feature
            tgt.push(i as f64);
        }
        let features = Array2::from_shape_vec((n, 2), feat).unwrap();
        let targets = Array1::from_vec(tgt);

        let model = RandomForestRegressor::fit(features.view(), targets.view(), 30, 8, 2, 1.0, 42);

        // Feature 0 should be more important than feature 1
        assert!(
            model.feature_importances[0] > model.feature_importances[1],
            "Feature importances: {:?}",
            model.feature_importances
        );
    }

    #[test]
    /// Runs regression test for rf batch predict.
    ///
    /// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_rf_batch_predict() {
        let features =
            Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let targets = Array1::from_vec((0..10).map(|i| i as f64 * 2.0).collect());

        let model = RandomForestRegressor::fit(features.view(), targets.view(), 20, 5, 2, 1.0, 42);

        let test = Array2::from_shape_vec((3, 1), vec![2.0, 5.0, 8.0]).unwrap();
        let preds = model.predict(test.view());
        assert_eq!(preds.len(), 3);
        // Predictions should be in reasonable range
        for (i, &p) in preds.iter().enumerate() {
            assert!(p.is_finite(), "Prediction {i} is not finite: {p}");
        }
    }

    #[test]
    /// Runs regression test for rf classifier probability.
    ///
    /// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_rf_classifier_probability() {
        let mut feat = Vec::new();
        let mut tgt = Vec::new();
        for i in 0..30 {
            feat.push(i as f64);
            tgt.push(if i < 15 { 0.0 } else { 1.0 });
        }
        let features = Array2::from_shape_vec((30, 1), feat).unwrap();
        let targets = Array1::from_vec(tgt);

        let model = RandomForestClassifier::fit(features.view(), targets.view(), 50, 5, 2, 1.0, 42);

        let prob_low = model.predict_proba_one(&[3.0]);
        let prob_high = model.predict_proba_one(&[25.0]);
        assert!(prob_low < 0.5, "prob_low = {prob_low}");
        assert!(prob_high > 0.5, "prob_high = {prob_high}");
    }

    #[test]
    /// Implements test feature presort budget gate.
    ///
    /// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
    /// ML-layer note: optimized for batch computation and zero-allocation passes.
    fn test_feature_presort_budget_gate() {
        let features =
            Array2::from_shape_vec((4, 2), vec![2.0, 1.0, 0.0, 3.0, 1.0, 2.0, 3.0, 0.0]).unwrap();

        let tiny = build_feature_presort_with_budget(features.view(), 1);
        assert!(tiny.is_none());

        let ok = build_feature_presort_with_budget(features.view(), usize::MAX).unwrap();
        assert_eq!(ok.orders.len(), 2);
        assert_eq!(ok.orders[0], vec![1, 2, 0, 3]);
        assert_eq!(ok.orders[1], vec![3, 0, 2, 1]);
    }

    #[test]
    /// Implements test root presort split matches local sort predictions.
    ///
    /// Function context: `crates/ferrum_core/src/ml/random_forest.rs`.
    /// ML-layer note: optimized for batch computation and zero-allocation passes.
    fn test_root_presort_split_matches_local_sort_predictions() {
        let features = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 1.0, 0.5, 2.0, 1.0, 3.0, 1.5, 4.0, 2.0, 5.0, 2.5, 6.0, 3.0, 7.0, 3.5,
            ],
        )
        .unwrap();
        let targets = vec![0.0, 0.2, 0.5, 0.9, 1.2, 1.7, 2.0, 2.4];
        let mut indices_a = vec![0, 1, 1, 2, 3, 4, 5, 6, 7];
        let mut indices_b = indices_a.clone();
        let builder = DecisionTreeBuilder {
            max_depth: 4,
            min_samples_split: 2,
            max_features: 2,
            is_classifier: false,
        };
        let presort = build_feature_presort_with_budget(features.view(), usize::MAX).unwrap();

        let mut scratch_a = Vec::with_capacity(indices_a.len());
        let mut scratch_b = Vec::with_capacity(indices_b.len());

        let mut rng_a = StdRng::seed_from_u64(42);
        let mut nodes_a = Vec::new();
        let _ = builder.build(
            features.view(),
            &targets,
            &mut indices_a,
            0,
            &mut rng_a,
            Some(&presort),
            &mut scratch_a,
            &mut nodes_a,
        );
        let tree_a = nodes_a;
        let mut rng_b = StdRng::seed_from_u64(42);
        let mut nodes_b = Vec::new();
        let _ = builder.build(features.view(), &targets, &mut indices_b, 0, &mut rng_b, None, &mut scratch_b, &mut nodes_b);
        let tree_b = nodes_b;

        for i in 0..features.nrows() {
            let row = features.row(i).to_vec();
            let pred_a = tree_a[0].predict(features.row(i).to_slice().unwrap(), &tree_a);
            let pred_b = tree_b[0].predict(features.row(i).to_slice().unwrap(), &tree_b);
            assert!((pred_a - pred_b).abs() < 1e-12, "row={i} pa={pred_a} pb={pred_b}");
        }
    }
}
