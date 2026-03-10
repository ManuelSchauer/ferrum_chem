//! Random Forest — decision tree + bagging, from scratch.
//!
//! Supports both regression and classification.
//! Optimized for binary fingerprint features.

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
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
            TreeNode::Split { feature_idx, threshold, left, right } => {
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
    fn build(&self, features: &Array2<f64>, targets: &[f64], indices: &[usize], depth: usize, rng: &mut StdRng) -> TreeNode {
        // Stop conditions
        if depth >= self.max_depth || indices.len() < self.min_samples_split || indices.len() <= 1 {
            return TreeNode::Leaf { value: self.leaf_value(targets, indices) };
        }

        // Check if all targets are the same
        let first_target = targets[indices[0]];
        if indices.iter().all(|&i| (targets[i] - first_target).abs() < 1e-15) {
            return TreeNode::Leaf { value: first_target };
        }

        let n_features = features.ncols();
        let n_try = self.max_features.min(n_features);

        // Random feature subset
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.shuffle(rng);
        feature_indices.truncate(n_try);

        let mut best_score = f64::INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_left = Vec::new();
        let mut best_right = Vec::new();

        for &feat_idx in &feature_indices {
            // Collect unique values for this feature
            let mut values: Vec<f64> = indices.iter().map(|&i| features[[i, feat_idx]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();

            if values.len() <= 1 {
                continue;
            }

            // Try midpoints as thresholds
            for w in values.windows(2) {
                let threshold = (w[0] + w[1]) / 2.0;
                let (left, right): (Vec<usize>, Vec<usize>) = indices.iter()
                    .partition(|&&i| features[[i, feat_idx]] <= threshold);

                if left.is_empty() || right.is_empty() {
                    continue;
                }

                let score = if self.is_classifier {
                    self.gini_impurity(targets, &left, &right)
                } else {
                    self.mse_impurity(targets, &left, &right)
                };

                if score < best_score {
                    best_score = score;
                    best_feature = feat_idx;
                    best_threshold = threshold;
                    best_left = left;
                    best_right = right;
                }
            }
        }

        if best_left.is_empty() || best_right.is_empty() {
            return TreeNode::Leaf { value: self.leaf_value(targets, indices) };
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

    fn mse_impurity(&self, targets: &[f64], left: &[usize], right: &[usize]) -> f64 {
        let n = (left.len() + right.len()) as f64;
        let mse_left = self.variance(targets, left);
        let mse_right = self.variance(targets, right);
        (left.len() as f64 * mse_left + right.len() as f64 * mse_right) / n
    }

    fn gini_impurity(&self, targets: &[f64], left: &[usize], right: &[usize]) -> f64 {
        let n = (left.len() + right.len()) as f64;
        let gini_left = self.gini(targets, left);
        let gini_right = self.gini(targets, right);
        (left.len() as f64 * gini_left + right.len() as f64 * gini_right) / n
    }

    fn variance(&self, targets: &[f64], indices: &[usize]) -> f64 {
        if indices.is_empty() { return 0.0; }
        let n = indices.len() as f64;
        let mean: f64 = indices.iter().map(|&i| targets[i]).sum::<f64>() / n;
        indices.iter().map(|&i| (targets[i] - mean).powi(2)).sum::<f64>() / n
    }

    fn gini(&self, targets: &[f64], indices: &[usize]) -> f64 {
        if indices.is_empty() { return 0.0; }
        let n = indices.len() as f64;
        let p = indices.iter().filter(|&&i| targets[i] >= 0.5).count() as f64 / n;
        2.0 * p * (1.0 - p)
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

        let mut rng = StdRng::seed_from_u64(seed);
        let mut trees = Vec::with_capacity(n_estimators);
        let mut feature_counts = vec![0u64; n_features];

        let builder = DecisionTreeBuilder {
            max_depth,
            min_samples_split,
            max_features,
            is_classifier: false,
        };

        for _ in 0..n_estimators {
            // Bootstrap sample
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
            let tree = builder.build(features, &target_slice, &indices, 0, &mut rng);
            count_feature_usage(&tree, &mut feature_counts);
            trees.push(tree);
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
        Array1::from_vec(
            (0..features.nrows())
                .map(|i| self.predict_one(features.row(i).as_slice().unwrap()))
                .collect()
        )
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

        let mut rng = StdRng::seed_from_u64(seed);
        let mut trees = Vec::with_capacity(n_estimators);
        let mut feature_counts = vec![0u64; n_features];

        let builder = DecisionTreeBuilder {
            max_depth,
            min_samples_split,
            max_features,
            is_classifier: true,
        };

        for _ in 0..n_estimators {
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
            let tree = builder.build(features, &target_slice, &indices, 0, &mut rng);
            count_feature_usage(&tree, &mut feature_counts);
            trees.push(tree);
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
        if self.predict_proba_one(sample) >= 0.5 { 1.0 } else { 0.0 }
    }

    /// Predict for multiple samples.
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        Array1::from_vec(
            (0..features.nrows())
                .map(|i| self.predict_one(features.row(i).as_slice().unwrap()))
                .collect()
        )
    }
}

/// Count how many times each feature is used in splits (proxy for importance).
fn count_feature_usage(node: &TreeNode, counts: &mut [u64]) {
    match node {
        TreeNode::Leaf { .. } => {}
        TreeNode::Split { feature_idx, left, right, .. } => {
            counts[*feature_idx] += 1;
            count_feature_usage(left, counts);
            count_feature_usage(right, counts);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_rf_regressor_basic() {
        // Simple linear relationship: y = x
        let features = Array2::from_shape_vec((20, 1),
            (0..20).map(|i| i as f64).collect()).unwrap();
        let targets = Array1::from_vec((0..20).map(|i| i as f64).collect());

        let model = RandomForestRegressor::fit(
            &features, &targets, 50, 10, 2, 1.0, 42
        );

        // Should predict reasonably for in-range values
        let pred = model.predict_one(&[5.0]);
        assert!((pred - 5.0).abs() < 3.0, "Prediction {pred} too far from 5.0");
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

        let model = RandomForestClassifier::fit(
            &features, &targets, 50, 10, 2, 1.0, 42
        );

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

        let model = RandomForestRegressor::fit(
            &features, &targets, 30, 8, 2, 1.0, 42
        );

        // Feature 0 should be more important than feature 1
        assert!(
            model.feature_importances[0] > model.feature_importances[1],
            "Feature importances: {:?}", model.feature_importances
        );
    }

    #[test]
    fn test_rf_batch_predict() {
        let features = Array2::from_shape_vec((10, 1),
            (0..10).map(|i| i as f64).collect()).unwrap();
        let targets = Array1::from_vec((0..10).map(|i| i as f64 * 2.0).collect());

        let model = RandomForestRegressor::fit(
            &features, &targets, 20, 5, 2, 1.0, 42
        );

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

        let model = RandomForestClassifier::fit(
            &features, &targets, 50, 5, 2, 1.0, 42
        );

        let prob_low = model.predict_proba_one(&[3.0]);
        let prob_high = model.predict_proba_one(&[25.0]);
        assert!(prob_low < 0.5, "prob_low = {prob_low}");
        assert!(prob_high > 0.5, "prob_high = {prob_high}");
    }
}
