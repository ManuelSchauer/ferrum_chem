//! QsarDataset — typed container for feature matrices and targets.

//! ## File guide: Dataset
//!
//! This file belongs to FerrumChem's **machine-learning pipeline layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct QsarDataset`
//! - `fn new`
//! - `fn from_f32_vectors`
//! - `fn n_samples`
//! - `fn n_features`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Dataset` assumes `ndarray` row-major sample layout (`n_samples x n_features`) throughout fit/predict paths.
//! - `Dataset` prioritizes reproducibility via explicit seeds and deterministic iteration order where applicable.
//! - Model quality metrics for `Dataset` depend on consistent preprocessing choices (scaling, splitting, and leakage control).
//! - Specific scope anchor: `crates/ferrum_core/src/ml/dataset.rs` defines dataset behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Canonical ML literature for `Dataset` algorithm family (statistical learning and validation texts).
//! - scikit-learn/RDKit-style API ergonomics that inform `Dataset` fit/transform/predict contracts.
//! - OECD QSAR validation principles where `Dataset` is used in predictive chemistry contexts.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/ml/dataset.rs`.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// A QSAR dataset holding features (f64), targets, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QsarDataset {
    /// Feature matrix (n_samples × n_features), f64.
    pub features: Array2<f64>,
    /// Target values (n_samples,).
    pub targets: Array1<f64>,
    /// Feature names (length = n_features).
    pub feature_names: Vec<String>,
    /// Optional molecule identifiers (SMILES, names, etc.).
    pub identifiers: Vec<String>,
}

impl QsarDataset {
    /// Create a new dataset from features, targets, and feature names.
    ///
    /// Panics if dimensions are inconsistent.
    pub fn new(features: Array2<f64>, targets: Array1<f64>, feature_names: Vec<String>) -> Self {
        assert_eq!(
            features.nrows(),
            targets.len(),
            "Feature rows ({}) must match target length ({})",
            features.nrows(),
            targets.len()
        );
        assert_eq!(
            features.ncols(),
            feature_names.len(),
            "Feature cols ({}) must match feature_names length ({})",
            features.ncols(),
            feature_names.len()
        );
        Self {
            features,
            targets,
            feature_names,
            identifiers: Vec::new(),
        }
    }

    /// Create from f32 vectors (the output of `build_ml_vectors_parallel`).
    ///
    /// Converts f32 → f64 for numerical stability in ML algorithms.
    pub fn from_f32_vectors(
        vectors: &[Vec<f32>],
        targets: Vec<f64>,
        feature_names: Vec<String>,
    ) -> Self {
        let n_samples = vectors.len();
        assert!(!vectors.is_empty(), "Need at least one sample");
        let n_features = vectors[0].len();

        let mut features = Array2::<f64>::zeros((n_samples, n_features));
        for (i, vec) in vectors.iter().enumerate() {
            assert_eq!(vec.len(), n_features, "All vectors must have same length");
            for (j, &v) in vec.iter().enumerate() {
                features[[i, j]] = v as f64;
            }
        }

        Self::new(features, Array1::from(targets), feature_names)
    }

    /// Number of samples.
    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }

    /// Number of features.
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    /// Set molecule identifiers (SMILES strings, etc.).
    pub fn with_identifiers(mut self, ids: Vec<String>) -> Self {
        assert_eq!(ids.len(), self.n_samples());
        self.identifiers = ids;
        self
    }

    /// Extract a subset of samples by index.
    pub fn subset(&self, indices: &[usize]) -> Self {
        let n = indices.len();
        let n_feat = self.n_features();
        let mut features = Array2::<f64>::zeros((n, n_feat));
        let mut targets = Array1::<f64>::zeros(n);
        let mut identifiers = Vec::with_capacity(n);

        for (new_i, &old_i) in indices.iter().enumerate() {
            features.row_mut(new_i).assign(&self.features.row(old_i));
            targets[new_i] = self.targets[old_i];
            if !self.identifiers.is_empty() {
                identifiers.push(self.identifiers[old_i].clone());
            }
        }

        Self {
            features,
            targets,
            feature_names: self.feature_names.clone(),
            identifiers,
        }
    }

    /// Extract a subset of features by column indices.
    pub fn select_features(&self, col_indices: &[usize]) -> Self {
        let n = self.n_samples();
        let n_new = col_indices.len();
        let mut features = Array2::<f64>::zeros((n, n_new));
        let mut names = Vec::with_capacity(n_new);

        for (new_j, &old_j) in col_indices.iter().enumerate() {
            features
                .column_mut(new_j)
                .assign(&self.features.column(old_j));
            names.push(self.feature_names[old_j].clone());
        }

        Self {
            features,
            targets: self.targets.clone(),
            feature_names: names,
            identifiers: self.identifiers.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    /// Runs regression test for new dataset.
    ///
    /// Function context: `crates/ferrum_core/src/ml/dataset.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_new_dataset() {
        let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let targets = array![0.1, 0.2, 0.3];
        let names = vec!["f1".to_string(), "f2".to_string()];
        let ds = QsarDataset::new(features, targets, names);
        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);
    }

    #[test]
    /// Runs regression test for from f32 vectors.
    ///
    /// Function context: `crates/ferrum_core/src/ml/dataset.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_from_f32_vectors() {
        let vecs = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
        let targets = vec![0.5, 1.5];
        let names = vec!["a".into(), "b".into()];
        let ds = QsarDataset::from_f32_vectors(&vecs, targets, names);
        assert_eq!(ds.n_samples(), 2);
        assert_eq!(ds.n_features(), 2);
        assert!((ds.features[[0, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    /// Runs regression test for subset.
    ///
    /// Function context: `crates/ferrum_core/src/ml/dataset.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_subset() {
        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let targets = array![0.1, 0.2, 0.3, 0.4];
        let names = vec!["f1".into(), "f2".into()];
        let ds = QsarDataset::new(features, targets, names);
        let sub = ds.subset(&[0, 2]);
        assert_eq!(sub.n_samples(), 2);
        assert!((sub.targets[0] - 0.1).abs() < 1e-10);
        assert!((sub.targets[1] - 0.3).abs() < 1e-10);
    }

    #[test]
    /// Runs regression test for select features.
    ///
    /// Function context: `crates/ferrum_core/src/ml/dataset.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_select_features() {
        let features = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let targets = array![0.1, 0.2];
        let names = vec!["a".into(), "b".into(), "c".into()];
        let ds = QsarDataset::new(features, targets, names);
        let sel = ds.select_features(&[0, 2]);
        assert_eq!(sel.n_features(), 2);
        assert_eq!(sel.feature_names, vec!["a", "c"]);
        assert!((sel.features[[0, 1]] - 3.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Feature rows")]
    /// Runs regression test for mismatched dimensions.
    ///
    /// Function context: `crates/ferrum_core/src/ml/dataset.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_mismatched_dimensions() {
        let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let targets = array![0.1, 0.2]; // wrong length
        let names = vec!["f1".into(), "f2".into()];
        QsarDataset::new(features, targets, names);
    }
}
