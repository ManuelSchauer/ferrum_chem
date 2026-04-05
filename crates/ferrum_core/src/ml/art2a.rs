//! ART-2A (Adaptive Resonance Theory 2A) — unsupervised clustering, from scratch.
//!
//! Chemical space clustering and applicability domain definition.

//! ## File guide: Art2a
//!
//! This file belongs to FerrumChem's **machine-learning pipeline layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct Art2a`
//! - `fn new`
//! - `fn fit`
//! - `fn find_best_cluster`
//! - `fn predict_one`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Art2A` assumes `ndarray` row-major sample layout (`n_samples x n_features`) throughout fit/predict paths.
//! - `Art2A` prioritizes reproducibility via explicit seeds and deterministic iteration order where applicable.
//! - Model quality metrics for `Art2A` depend on consistent preprocessing choices (scaling, splitting, and leakage control).
//! - Specific scope anchor: `crates/ferrum_core/src/ml/art2a.rs` defines art2a behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Canonical ML literature for `Art2A` algorithm family (statistical learning and validation texts).
//! - scikit-learn/RDKit-style API ergonomics that inform `Art2A` fit/transform/predict contracts.
//! - OECD QSAR validation principles where `Art2A` is used in predictive chemistry contexts.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/ml/art2a.rs`.

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
    pub prototypes: Vec<Array1<f32>>,
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
        let normalized: Vec<Array1<f32>> = data.iter().map(|v| l2_normalize(v)).collect();

        self.prototypes.clear();
        self.labels = vec![0; n];
        self.cluster_sizes.clear();

        for epoch in 0..self.max_iter {
            let allow_new_clusters = epoch == 0;
            let mut changed = false;

            for (i, input) in normalized.iter().enumerate() {
                let (cluster_id, _) = self.find_best_cluster(input, allow_new_clusters);

                if cluster_id != self.labels[i] {
                    changed = true;
                    self.labels[i] = cluster_id;
                }

                // Update prototype with learning
                let proto = &mut self.prototypes[cluster_id];
                for (p, &x) in proto.iter_mut().zip(input.iter()) {
                    let updated = (1.0 - self.learning_rate) * f64::from(*p)
                        + self.learning_rate * f64::from(x);
                    *p = updated as f32;
                }
                // Re-normalize prototype
                let norm = proto
                    .iter()
                    .map(|&v| {
                        let x = f64::from(v);
                        x * x
                    })
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-15);
                proto.mapv_inplace(|v| (f64::from(v) / norm) as f32);
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
    fn find_best_cluster(&mut self, input: &Array1<f32>, allow_new_cluster: bool) -> (usize, f64) {
        if self.prototypes.is_empty() {
            self.prototypes.push(input.clone());
            return (0, 1.0);
        }

        let mut best_id = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (id, proto) in self.prototypes.iter().enumerate() {
            let sim: f64 = input
                .iter()
                .zip(proto.iter())
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            if !sim.is_finite() {
                continue;
            }
            if sim > best_sim {
                best_sim = sim;
                best_id = id;
            }
        }

        // Vigilance check: if best similarity < ρ, create new cluster
        if allow_new_cluster && best_sim < self.vigilance {
            let new_id = self.prototypes.len();
            self.prototypes.push(input.clone());
            return (new_id, 1.0);
        }

        if !best_sim.is_finite() {
            return (0, 0.0);
        }

        (best_id, best_sim)
    }

    /// Predict cluster assignment for a new sample.
    pub fn predict_one(&self, sample: &[f64]) -> (usize, f64) {
        if self.prototypes.is_empty() {
            return (0, 0.0);
        }
        let input = l2_normalize(sample);
        let mut best_id = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (id, proto) in self.prototypes.iter().enumerate() {
            let sim: f64 = input
                .iter()
                .zip(proto.iter())
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            if !sim.is_finite() {
                continue;
            }
            if sim > best_sim {
                best_sim = sim;
                best_id = id;
            }
        }

        if !best_sim.is_finite() {
            return (0, 0.0);
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
fn l2_normalize(v: &[f64]) -> Array1<f32> {
    let finite = v
        .iter()
        .map(|x| if x.is_finite() { *x } else { 0.0 })
        .collect::<Vec<_>>();
    let norm = finite.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
    Array1::from_iter(finite.into_iter().map(|x| (x / norm) as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Runs regression test for art2a basic clustering.
    ///
    /// Function context: `crates/ferrum_core/src/ml/art2a.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
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
    /// Runs regression test for art2a vigilance effect.
    ///
    /// Function context: `crates/ferrum_core/src/ml/art2a.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
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
    /// Runs regression test for art2a applicability domain.
    ///
    /// Function context: `crates/ferrum_core/src/ml/art2a.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
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
    /// Runs regression test for art2a empty data.
    ///
    /// Function context: `crates/ferrum_core/src/ml/art2a.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_art2a_empty_data() {
        let mut model = Art2a::new(0.5, 0.3, 10);
        model.fit(&[]);
        assert_eq!(model.n_clusters(), 0);
    }

    #[test]
    /// Runs regression test for l2 normalize.
    ///
    /// Function context: `crates/ferrum_core/src/ml/art2a.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_l2_normalize() {
        let v = l2_normalize(&[3.0, 4.0]);
        let norm = f64::from(v.dot(&v)).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((f64::from(v[0]) - 0.6).abs() < 1e-6);
        assert!((f64::from(v[1]) - 0.8).abs() < 1e-6);
    }

    #[test]
    /// Runs regression test for art2a cluster count bounded by sample count.
    ///
    /// Function context: `crates/ferrum_core/src/ml/art2a.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_art2a_cluster_count_never_exceeds_sample_count() {
        let data = (0..64)
            .map(|i| vec![i as f64, (i % 7) as f64, (i % 3) as f64])
            .collect::<Vec<_>>();
        let mut model = Art2a::new(0.95, 0.3, 50);
        model.fit(&data);

        assert!(
            model.n_clusters() <= data.len(),
            "Cluster count {} must not exceed sample count {}",
            model.n_clusters(),
            data.len()
        );
        assert_eq!(model.labels.len(), data.len());
        assert_eq!(model.cluster_sizes.iter().sum::<usize>(), data.len());
    }

    #[test]
    /// Runs regression test for art2a handles non finite inputs.
    ///
    /// Function context: `crates/ferrum_core/src/ml/art2a.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_art2a_handles_non_finite_inputs() {
        let data = vec![
            vec![1.0, f64::NAN, 0.2],
            vec![f64::INFINITY, 0.1, 0.3],
            vec![0.0, f64::NEG_INFINITY, 1.0],
            vec![0.9, 0.2, 0.1],
        ];
        let mut model = Art2a::new(0.9, 0.3, 20);
        model.fit(&data);

        assert_eq!(model.labels.len(), data.len());
        assert!(model.n_clusters() >= 1);
        assert!(model.n_clusters() <= data.len());
        assert!(model.cluster_sizes.iter().all(|size| *size > 0));
        assert!(
            model
                .prototypes
                .iter()
                .all(|proto| proto.iter().all(|value| value.is_finite()))
        );
    }
}
