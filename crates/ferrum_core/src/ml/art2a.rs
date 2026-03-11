//! ART-2A (Adaptive Resonance Theory 2A) — unsupervised clustering, from scratch.
//!
//! Chemical space clustering and applicability domain definition.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// ART-2A clustering model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Art2a {
    /// Vigilance parameter ρ ∈ (0, 1). Higher → more clusters.
    pub vigilance: f64,
    /// Learning rate η ∈ (0, 1).
    pub learning_rate: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Cluster weight vectors (prototypes), L2-normalized.
    pub prototypes: Vec<Array1<f64>>,
    /// Cluster assignments for training data.
    pub labels: Vec<usize>,
    /// Number of samples per cluster.
    pub cluster_sizes: Vec<usize>,
}

impl Art2a {
    /// Create a new ART-2A model with given hyperparameters.
    pub fn new(vigilance: f64, learning_rate: f64, max_iter: usize) -> Self {
        assert!(
            (0.0..1.0).contains(&vigilance),
            "vigilance must be in (0, 1)"
        );
        assert!(
            (0.0..=1.0).contains(&learning_rate),
            "learning_rate must be in (0, 1]"
        );
        Self {
            vigilance,
            learning_rate,
            max_iter,
            prototypes: Vec::new(),
            labels: Vec::new(),
            cluster_sizes: Vec::new(),
        }
    }

    /// Fit the model on data. Each row is a sample.
    ///
    /// Input vectors are L2-normalized internally.
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        let n = data.len();
        if n == 0 {
            return;
        }

        // L2-normalize all inputs
        let normalized: Vec<Array1<f64>> = data.iter().map(|v| l2_normalize(v)).collect();

        self.prototypes.clear();
        self.labels = vec![0; n];
        self.cluster_sizes.clear();

        for _epoch in 0..self.max_iter {
            let mut changed = false;

            for (i, input) in normalized.iter().enumerate() {
                let (cluster_id, _) = self.find_best_cluster(input);

                if cluster_id != self.labels[i] {
                    changed = true;
                    self.labels[i] = cluster_id;
                }

                // Update prototype with learning
                let proto = &mut self.prototypes[cluster_id];
                for (p, &x) in proto.iter_mut().zip(input.iter()) {
                    *p = (1.0 - self.learning_rate) * *p + self.learning_rate * x;
                }
                // Re-normalize prototype
                let norm = proto.dot(proto).sqrt().max(1e-15);
                proto.mapv_inplace(|v| v / norm);
            }

            if !changed {
                break;
            }
        }

        // Compute cluster sizes
        let n_clusters = self.prototypes.len();
        self.cluster_sizes = vec![0; n_clusters];
        for &label in &self.labels {
            self.cluster_sizes[label] += 1;
        }
    }

    /// Find best matching cluster or create a new one.
    ///
    /// Returns (cluster_id, similarity).
    fn find_best_cluster(&mut self, input: &Array1<f64>) -> (usize, f64) {
        let mut best_id = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (id, proto) in self.prototypes.iter().enumerate() {
            let sim = input.dot(proto); // cosine similarity (both L2-normalized)
            if sim > best_sim {
                best_sim = sim;
                best_id = id;
            }
        }

        // Vigilance check: if best similarity < ρ, create new cluster
        if self.prototypes.is_empty() || best_sim < self.vigilance {
            let new_id = self.prototypes.len();
            self.prototypes.push(input.clone());
            return (new_id, 1.0);
        }

        (best_id, best_sim)
    }

    /// Predict cluster assignment for a new sample.
    pub fn predict_one(&self, sample: &[f64]) -> (usize, f64) {
        let input = l2_normalize(sample);
        let mut best_id = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (id, proto) in self.prototypes.iter().enumerate() {
            let sim = input.dot(proto);
            if sim > best_sim {
                best_sim = sim;
                best_id = id;
            }
        }

        (best_id, best_sim)
    }

    /// Check if a sample is within the applicability domain.
    ///
    /// Returns true if the maximum similarity to any prototype exceeds the vigilance threshold.
    pub fn is_in_domain(&self, sample: &[f64]) -> bool {
        let (_, sim) = self.predict_one(sample);
        sim >= self.vigilance
    }

    /// Number of clusters.
    pub fn n_clusters(&self) -> usize {
        self.prototypes.len()
    }
}

/// L2-normalize a vector.
fn l2_normalize(v: &[f64]) -> Array1<f64> {
    let arr = Array1::from(v.to_vec());
    let norm = arr.dot(&arr).sqrt().max(1e-15);
    arr.mapv(|x| x / norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_art2a_basic_clustering() {
        let data = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.8, 0.2, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.1, 0.9],
            vec![0.1, 0.0, 0.8],
        ];

        let mut model = Art2a::new(0.8, 0.3, 20);
        model.fit(&data);

        // Should find 2 clusters (group near [1,0,0] and group near [0,0,1])
        assert!(
            model.n_clusters() >= 2,
            "Found {} clusters",
            model.n_clusters()
        );

        // Samples in same group should have same label
        assert_eq!(model.labels[0], model.labels[1]);
        assert_eq!(model.labels[3], model.labels[4]);
        // Different groups
        assert_ne!(model.labels[0], model.labels[3]);
    }

    #[test]
    fn test_art2a_vigilance_effect() {
        let data = vec![vec![1.0, 0.0], vec![0.7, 0.3], vec![0.0, 1.0]];

        // Low vigilance → fewer clusters
        let mut model_low = Art2a::new(0.3, 0.3, 20);
        model_low.fit(&data);

        // High vigilance → more clusters
        let mut model_high = Art2a::new(0.95, 0.3, 20);
        model_high.fit(&data);

        assert!(
            model_high.n_clusters() >= model_low.n_clusters(),
            "High vig: {} clusters, low vig: {} clusters",
            model_high.n_clusters(),
            model_low.n_clusters()
        );
    }

    #[test]
    fn test_art2a_applicability_domain() {
        let data = vec![vec![1.0, 0.0, 0.0], vec![0.9, 0.1, 0.0]];

        let mut model = Art2a::new(0.8, 0.3, 10);
        model.fit(&data);

        // Similar to training data → in domain
        assert!(model.is_in_domain(&[1.0, 0.0, 0.0]));
        // Very different → possibly out of domain
        let (_, sim) = model.predict_one(&[0.0, 0.0, 1.0]);
        // The opposite direction should have low similarity
        assert!(sim < 0.5, "Unexpected high similarity: {sim}");
    }

    #[test]
    fn test_art2a_empty_data() {
        let mut model = Art2a::new(0.5, 0.3, 10);
        model.fit(&[]);
        assert_eq!(model.n_clusters(), 0);
    }

    #[test]
    fn test_l2_normalize() {
        let v = l2_normalize(&[3.0, 4.0]);
        let norm = v.dot(&v).sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
        assert!((v[0] - 0.6).abs() < 1e-10);
        assert!((v[1] - 0.8).abs() < 1e-10);
    }
}
