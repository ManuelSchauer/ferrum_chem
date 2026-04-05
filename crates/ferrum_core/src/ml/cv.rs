//! Cross-validation — K-Fold, Leave-One-Out, Y-randomization.

//! ## File guide: Cv
//!
//! This file belongs to FerrumChem's **machine-learning pipeline layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct KFoldSplits`
//! - `fn k_fold`
//! - `fn leave_one_out`
//! - `fn y_randomization`
//! - `mod tests`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Cv` assumes `ndarray` row-major sample layout (`n_samples x n_features`) throughout fit/predict paths.
//! - `Cv` prioritizes reproducibility via explicit seeds and deterministic iteration order where applicable.
//! - Model quality metrics for `Cv` depend on consistent preprocessing choices (scaling, splitting, and leakage control).
//! - Specific scope anchor: `crates/ferrum_core/src/ml/cv.rs` defines cv behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Canonical ML literature for `Cv` algorithm family (statistical learning and validation texts).
//! - scikit-learn/RDKit-style API ergonomics that inform `Cv` fit/transform/predict contracts.
//! - OECD QSAR validation principles where `Cv` is used in predictive chemistry contexts.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/ml/cv.rs`.

use crate::ml::dataset::QsarDataset;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// K-Fold cross-validation split indices.
pub struct KFoldSplits {
    pub folds: Vec<(Vec<usize>, Vec<usize>)>, // (train_indices, test_indices)
}

/// Generate K-Fold cross-validation splits.
pub fn k_fold(n_samples: usize, k: usize, seed: u64) -> KFoldSplits {
    assert!(k >= 2, "k must be >= 2");
    assert!(n_samples >= k, "n_samples must be >= k");

    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    let fold_size = n_samples / k;
    let remainder = n_samples % k;

    let mut folds = Vec::with_capacity(k);
    let mut start = 0;

    for i in 0..k {
        let end = start + fold_size + if i < remainder { 1 } else { 0 };
        let test_indices: Vec<usize> = indices[start..end].to_vec();
        let train_indices: Vec<usize> = indices[..start]
            .iter()
            .chain(indices[end..].iter())
            .copied()
            .collect();
        folds.push((train_indices, test_indices));
        start = end;
    }

    KFoldSplits { folds }
}

/// Leave-One-Out cross-validation splits.
pub fn leave_one_out(n_samples: usize) -> KFoldSplits {
    let folds: Vec<(Vec<usize>, Vec<usize>)> = (0..n_samples)
        .map(|i| {
            let train: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
            (train, vec![i])
        })
        .collect();
    KFoldSplits { folds }
}

/// Y-randomization validation.
///
/// Shuffles target values and returns shuffled target vectors for each permutation.
/// If the model performs well on shuffled data, the original model is likely overfit.
pub fn y_randomization(
    data: &QsarDataset,
    n_permutations: usize,
    seed: u64,
) -> Vec<ndarray::Array1<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut results = Vec::with_capacity(n_permutations);

    for _ in 0..n_permutations {
        let mut shuffled_targets: Vec<f64> = data.targets.to_vec();
        shuffled_targets.shuffle(&mut rng);
        results.push(ndarray::Array1::from(shuffled_targets));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Builds dataset.
    ///
    /// Function context: `crates/ferrum_core/src/ml/cv.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn make_dataset(n: usize) -> QsarDataset {
        let features = Array2::from_shape_fn((n, 2), |(i, j)| (i + j) as f64);
        let targets = Array1::from_vec((0..n).map(|i| i as f64).collect());
        let names = vec!["a".into(), "b".into()];
        QsarDataset::new(features, targets, names)
    }

    #[test]
    /// Runs regression test for k fold sizes.
    ///
    /// Function context: `crates/ferrum_core/src/ml/cv.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_k_fold_sizes() {
        let splits = k_fold(100, 5, 42);
        assert_eq!(splits.folds.len(), 5);
        for (train, test) in &splits.folds {
            assert_eq!(train.len() + test.len(), 100);
            assert_eq!(test.len(), 20);
        }
    }

    #[test]
    /// Runs regression test for k fold no overlap.
    ///
    /// Function context: `crates/ferrum_core/src/ml/cv.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_k_fold_no_overlap() {
        let splits = k_fold(20, 4, 42);
        let mut all_test: Vec<usize> = Vec::new();
        for (train, test) in &splits.folds {
            // No overlap within a fold
            for &t in test {
                assert!(!train.contains(&t));
            }
            all_test.extend(test);
        }
        // Every sample appears in exactly one test fold
        all_test.sort();
        assert_eq!(all_test, (0..20).collect::<Vec<_>>());
    }

    #[test]
    /// Runs regression test for leave one out.
    ///
    /// Function context: `crates/ferrum_core/src/ml/cv.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_leave_one_out() {
        let splits = leave_one_out(5);
        assert_eq!(splits.folds.len(), 5);
        for (train, test) in &splits.folds {
            assert_eq!(train.len(), 4);
            assert_eq!(test.len(), 1);
        }
    }

    #[test]
    /// Runs regression test for y randomization.
    ///
    /// Function context: `crates/ferrum_core/src/ml/cv.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_y_randomization() {
        let ds = make_dataset(20);
        let shuffled = y_randomization(&ds, 5, 42);
        assert_eq!(shuffled.len(), 5);

        // Each permutation should preserve target length.
        for s in &shuffled {
            assert_eq!(s.len(), ds.targets.len());
        }

        // At least one shuffled dataset should have different target order
        let original: Vec<f64> = ds.targets.to_vec();
        let any_different = shuffled.iter().any(|s| s.to_vec() != original);
        assert!(any_different, "Y-randomization should shuffle targets");
    }
}
