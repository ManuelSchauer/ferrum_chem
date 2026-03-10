//! k-Nearest Neighbors — regression and classification.
//!
//! Supports Euclidean distance (for descriptors) and Tanimoto distance (for fingerprints).

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
    pub train_features: Array2<f64>,
    pub train_targets: Array1<f64>,
}

impl KnnRegressor {
    /// Train (store) the k-NN regressor.
    pub fn fit(features: Array2<f64>, targets: Array1<f64>, k: usize, metric: DistanceMetric) -> Self {
        assert_eq!(features.nrows(), targets.len());
        assert!(k > 0 && k <= features.nrows(), "k must be in [1, n_samples]");
        Self {
            k,
            metric,
            train_features: features,
            train_targets: targets,
        }
    }

    /// Predict for a single sample.
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        let neighbors = self.find_k_nearest(sample);
        neighbors.iter().map(|&(_, t)| t).sum::<f64>() / self.k as f64
    }

    /// Predict for multiple samples.
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        Array1::from_vec(
            (0..features.nrows())
                .map(|i| self.predict_one(features.row(i).as_slice().unwrap()))
                .collect()
        )
    }

    /// Find k nearest neighbors. Returns Vec<(distance, target)>.
    pub fn find_k_nearest(&self, sample: &[f64]) -> Vec<(f64, f64)> {
        let mut distances: Vec<(f64, f64)> = (0..self.train_features.nrows())
            .map(|i| {
                let dist = compute_distance(
                    sample,
                    self.train_features.row(i).as_slice().unwrap(),
                    self.metric,
                );
                (dist, self.train_targets[i])
            })
            .collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.truncate(self.k);
        distances
    }

    /// Mean distance to k nearest neighbors (useful for applicability domain).
    pub fn mean_knn_distance(&self, sample: &[f64]) -> f64 {
        let neighbors = self.find_k_nearest(sample);
        neighbors.iter().map(|&(d, _)| d).sum::<f64>() / self.k as f64
    }
}

/// k-NN classifier — majority vote.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnClassifier {
    pub k: usize,
    pub metric: DistanceMetric,
    pub train_features: Array2<f64>,
    pub train_targets: Array1<f64>,
}

impl KnnClassifier {
    pub fn fit(features: Array2<f64>, targets: Array1<f64>, k: usize, metric: DistanceMetric) -> Self {
        assert_eq!(features.nrows(), targets.len());
        assert!(k > 0 && k <= features.nrows());
        Self {
            k,
            metric,
            train_features: features,
            train_targets: targets,
        }
    }

    /// Predict class (0.0 or 1.0) for a single sample.
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        let prob = self.predict_proba_one(sample);
        if prob >= 0.5 { 1.0 } else { 0.0 }
    }

    /// Predict probability of positive class.
    pub fn predict_proba_one(&self, sample: &[f64]) -> f64 {
        let mut distances: Vec<(f64, f64)> = (0..self.train_features.nrows())
            .map(|i| {
                let dist = compute_distance(
                    sample,
                    self.train_features.row(i).as_slice().unwrap(),
                    self.metric,
                );
                (dist, self.train_targets[i])
            })
            .collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.truncate(self.k);

        let n_pos = distances.iter().filter(|&&(_, t)| t >= 0.5).count();
        n_pos as f64 / self.k as f64
    }

    /// Predict classes for multiple samples.
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        Array1::from_vec(
            (0..features.nrows())
                .map(|i| self.predict_one(features.row(i).as_slice().unwrap()))
                .collect()
        )
    }
}

/// Compute distance between two vectors.
pub fn compute_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => {
            a.iter().zip(b).map(|(&x, &y)| (x - y).powi(2)).sum::<f64>().sqrt()
        }
        DistanceMetric::Tanimoto => {
            // Tanimoto distance = 1 - Tanimoto similarity
            let mut ab = 0.0;
            let mut aa = 0.0;
            let mut bb = 0.0;
            for (&x, &y) in a.iter().zip(b) {
                ab += x * y;
                aa += x * x;
                bb += y * y;
            }
            let denom = aa + bb - ab;
            if denom < 1e-15 {
                0.0
            } else {
                1.0 - ab / denom
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_knn_regressor_simple() {
        let features = Array2::from_shape_vec((4, 2), vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ]).unwrap();
        let targets = array![0.0, 1.0, 1.0, 2.0];
        let model = KnnRegressor::fit(features, targets, 1, DistanceMetric::Euclidean);

        // Query point at (0, 0) → nearest is itself → 0.0
        assert!((model.predict_one(&[0.0, 0.0]) - 0.0).abs() < 1e-10);
        // Query point at (1, 1) → nearest is itself → 2.0
        assert!((model.predict_one(&[1.0, 1.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_knn_regressor_k3() {
        let features = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let targets = array![0.0, 2.0, 4.0, 6.0];
        let model = KnnRegressor::fit(features, targets, 3, DistanceMetric::Euclidean);

        // Query at 1.0 → 3 nearest: 0, 1, 2 → targets: 0, 2, 4 → mean = 2.0
        assert!((model.predict_one(&[1.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_knn_classifier() {
        let features = Array2::from_shape_vec((6, 2), vec![
            0.0, 0.0,
            0.1, 0.0,
            0.0, 0.1,
            1.0, 1.0,
            1.1, 1.0,
            1.0, 1.1,
        ]).unwrap();
        let targets = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let model = KnnClassifier::fit(features, targets, 3, DistanceMetric::Euclidean);

        assert!((model.predict_one(&[0.05, 0.05]) - 0.0).abs() < 1e-10);
        assert!((model.predict_one(&[1.05, 1.05]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tanimoto_distance() {
        let a = vec![1.0, 0.0, 1.0, 1.0];
        let b = vec![1.0, 0.0, 1.0, 1.0];
        assert!(compute_distance(&a, &b, DistanceMetric::Tanimoto).abs() < 1e-10);

        let c = vec![0.0, 1.0, 0.0, 0.0];
        let d = compute_distance(&a, &c, DistanceMetric::Tanimoto);
        assert!(d > 0.9); // very different fingerprints
    }

    #[test]
    fn test_batch_predict() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 5.0, 10.0]).unwrap();
        let targets = array![0.0, 5.0, 10.0];
        let model = KnnRegressor::fit(features, targets, 1, DistanceMetric::Euclidean);

        let test = Array2::from_shape_vec((2, 1), vec![0.0, 10.0]).unwrap();
        let preds = model.predict(&test);
        assert!((preds[0] - 0.0).abs() < 1e-10);
        assert!((preds[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_knn_distance() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let targets = array![0.0, 1.0, 2.0];
        let model = KnnRegressor::fit(features, targets, 2, DistanceMetric::Euclidean);
        // Distance from 0.5 to nearest 2: [0, 1] → distances [0.5, 0.5] → mean = 0.5
        let dist = model.mean_knn_distance(&[0.5]);
        assert!((dist - 0.5).abs() < 1e-10);
    }
}
