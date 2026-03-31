//! Random Forest — decision tree + bagging, from scratch.
//!
//! Supports both regression and classification.
//! Optimized for binary fingerprint features.

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Decision Tree Node
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
enum TreeNode {
    Leaf {
        value: f64,
    },
    Split {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    fn predict(&self, sample: &[f64]) -> f64 {
        match self {
            TreeNode::Leaf { value } => *value,
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
            } => {
                if sample[*feature_idx] <= *threshold {
                    left.predict(sample)
                } else {
                    right.predict(sample)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decision Tree Builder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DecisionTreeBuilder {
    max_depth: usize,
    min_samples_split: usize,
    max_features: usize, // number of features to consider at each split
    is_classifier: bool,
}

impl DecisionTreeBuilder {
    fn build(
        &self,
        features: &Array2<f64>,
        targets: &[f64],
        indices: &[usize],
        depth: usize,
        rng: &mut StdRng,
    ) -> TreeNode {
        // Stop conditions
        if depth >= self.max_depth || indices.len() < self.min_samples_split || indices.len() <= 1 {
            return TreeNode::Leaf {
                value: self.leaf_value(targets, indices),
            };
        }

        // Check if all targets are the same
        let first_target = targets[indices[0]];
        if indices
            .iter()
            .all(|&i| (targets[i] - first_target).abs() < 1e-15)
        {
            return TreeNode::Leaf {
                value: first_target,
            };
        }

        let n_features = features.ncols();
        let n_try = self.max_features.min(n_features);

        // Random feature subset
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.shuffle(rng);
        feature_indices.truncate(n_try);

        let mut best_score = f64::INFINITY;
        let mut best_feature = 0usize;
        let mut best_threshold = 0.0;
        let mut found_split = false;

        for &feat_idx in &feature_indices {
            // Sort candidate rows once by this feature.
            let mut sorted_indices = indices.to_vec();
            sorted_indices.sort_by(|&a, &b| features[[a, feat_idx]].total_cmp(&features[[b, feat_idx]]));
            if sorted_indices.len() <= 1 {
                continue;
            }

            if self.is_classifier {
                let total_count = sorted_indices.len();
                let mut left_count = 0usize;
                let mut right_count = total_count;
                let mut left_pos = 0usize;
                let mut right_pos = sorted_indices
                    .iter()
                    .filter(|&&idx| targets[idx] >= 0.5)
                    .count();

                for split_idx in 0..(total_count - 1) {
                    let idx = sorted_indices[split_idx];
                    let current_val = features[[idx, feat_idx]];
                    let next_val = features[[sorted_indices[split_idx + 1], feat_idx]];

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
                let total_count = sorted_indices.len();
                let mut left_count = 0usize;
                let mut right_count = total_count;
                let mut left_sum = 0.0f64;
                let mut left_sum_sq = 0.0f64;
                let mut right_sum = 0.0f64;
                let mut right_sum_sq = 0.0f64;
                for &idx in &sorted_indices {
                    let t = targets[idx];
                    right_sum += t;
                    right_sum_sq += t * t;
                }

                for split_idx in 0..(total_count - 1) {
                    let idx = sorted_indices[split_idx];
                    let current_val = features[[idx, feat_idx]];
                    let next_val = features[[sorted_indices[split_idx + 1], feat_idx]];
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
            return TreeNode::Leaf {
                value: self.leaf_value(targets, indices),
            };
        }

        let mut best_left = Vec::with_capacity(indices.len());
        let mut best_right = Vec::with_capacity(indices.len());
        for &idx in indices {
            if features[[idx, best_feature]] <= best_threshold {
                best_left.push(idx);
            } else {
                best_right.push(idx);
            }
        }

        if best_left.is_empty() || best_right.is_empty() {
            return TreeNode::Leaf {
                value: self.leaf_value(targets, indices),
            };
        }

        TreeNode::Split {
            feature_idx: best_feature,
            threshold: best_threshold,
            left: Box::new(self.build(features, targets, &best_left, depth + 1, rng)),
            right: Box::new(self.build(features, targets, &best_right, depth + 1, rng)),
        }
    }

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
    trees: Vec<TreeNode>,
    pub n_estimators: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub max_features_fraction: f64,
    pub feature_importances: Vec<f64>,
}

impl RandomForestRegressor {
    /// Train a random forest regressor.
    pub fn fit(
        features: &Array2<f64>,
        targets: &Array1<f64>,
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

        let trees = (0..n_estimators)
            .into_par_iter()
            .map(|tree_idx| {
                let mut rng = StdRng::seed_from_u64(derive_tree_seed(seed, tree_idx));
                // Bootstrap sample
                let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
                builder.build(features, &target_slice, &indices, 0, &mut rng)
            })
            .collect::<Vec<_>>();
        let mut feature_counts = vec![0u64; n_features];
        for tree in &trees {
            count_feature_usage(tree, &mut feature_counts);
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
        let sum: f64 = self.trees.iter().map(|t| t.predict(sample)).sum();
        sum / self.trees.len() as f64
    }

    /// Predict for multiple samples.
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
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
    trees: Vec<TreeNode>,
    pub n_estimators: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub max_features_fraction: f64,
    pub feature_importances: Vec<f64>,
}

impl RandomForestClassifier {
    /// Train a random forest classifier.
    pub fn fit(
        features: &Array2<f64>,
        targets: &Array1<f64>,
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

        let trees = (0..n_estimators)
            .into_par_iter()
            .map(|tree_idx| {
                let mut rng = StdRng::seed_from_u64(derive_tree_seed(seed, tree_idx));
                let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
                builder.build(features, &target_slice, &indices, 0, &mut rng)
            })
            .collect::<Vec<_>>();
        let mut feature_counts = vec![0u64; n_features];
        for tree in &trees {
            count_feature_usage(tree, &mut feature_counts);
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
        let votes: f64 = self.trees.iter().map(|t| t.predict(sample)).sum();
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
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
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

fn derive_tree_seed(base_seed: u64, tree_idx: usize) -> u64 {
    let mut z = base_seed ^ ((tree_idx as u64).wrapping_add(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Count how many times each feature is used in splits (proxy for importance).
fn count_feature_usage(node: &TreeNode, counts: &mut [u64]) {
    match node {
        TreeNode::Leaf { .. } => {}
        TreeNode::Split {
            feature_idx,
            left,
            right,
            ..
        } => {
            counts[*feature_idx] += 1;
            count_feature_usage(left, counts);
            count_feature_usage(right, counts);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn test_rf_regressor_basic() {
        // Simple linear relationship: y = x
        let features =
            Array2::from_shape_vec((20, 1), (0..20).map(|i| i as f64).collect()).unwrap();
        let targets = Array1::from_vec((0..20).map(|i| i as f64).collect());

        let model = RandomForestRegressor::fit(&features, &targets, 50, 10, 2, 1.0, 42);

        // Should predict reasonably for in-range values
        let pred = model.predict_one(&[5.0]);
        assert!(
            (pred - 5.0).abs() < 3.0,
            "Prediction {pred} too far from 5.0"
        );
    }

    #[test]
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

        let model = RandomForestClassifier::fit(&features, &targets, 50, 10, 2, 1.0, 42);

        assert!((model.predict_one(&[2.0, 0.0]) - 0.0).abs() < 1e-10);
        assert!((model.predict_one(&[15.0, 0.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
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

        let model = RandomForestRegressor::fit(&features, &targets, 30, 8, 2, 1.0, 42);

        // Feature 0 should be more important than feature 1
        assert!(
            model.feature_importances[0] > model.feature_importances[1],
            "Feature importances: {:?}",
            model.feature_importances
        );
    }

    #[test]
    fn test_rf_batch_predict() {
        let features =
            Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let targets = Array1::from_vec((0..10).map(|i| i as f64 * 2.0).collect());

        let model = RandomForestRegressor::fit(&features, &targets, 20, 5, 2, 1.0, 42);

        let test = Array2::from_shape_vec((3, 1), vec![2.0, 5.0, 8.0]).unwrap();
        let preds = model.predict(&test);
        assert_eq!(preds.len(), 3);
        // Predictions should be in reasonable range
        for (i, &p) in preds.iter().enumerate() {
            assert!(p.is_finite(), "Prediction {i} is not finite: {p}");
        }
    }

    #[test]
    fn test_rf_classifier_probability() {
        let mut feat = Vec::new();
        let mut tgt = Vec::new();
        for i in 0..30 {
            feat.push(i as f64);
            tgt.push(if i < 15 { 0.0 } else { 1.0 });
        }
        let features = Array2::from_shape_vec((30, 1), feat).unwrap();
        let targets = Array1::from_vec(tgt);

        let model = RandomForestClassifier::fit(&features, &targets, 50, 5, 2, 1.0, 42);

        let prob_low = model.predict_proba_one(&[3.0]);
        let prob_high = model.predict_proba_one(&[25.0]);
        assert!(prob_low < 0.5, "prob_low = {prob_low}");
        assert!(prob_high > 0.5, "prob_high = {prob_high}");
    }
}
