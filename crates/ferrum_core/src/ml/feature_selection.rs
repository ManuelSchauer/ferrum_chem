//! Feature selection — variance threshold and correlation filter.

//! ## File guide: Feature Selection
//!
//! This file belongs to FerrumChem's **machine-learning pipeline layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn variance_threshold`
//! - `fn correlation_filter`
//! - `mod tests`
//! - `fn test_variance_threshold`
//! - `fn test_variance_threshold_higher`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Feature Selection` assumes `ndarray` row-major sample layout (`n_samples x n_features`) throughout fit/predict paths.
//! - `Feature Selection` prioritizes reproducibility via explicit seeds and deterministic iteration order where applicable.
//! - Model quality metrics for `Feature Selection` depend on consistent preprocessing choices (scaling, splitting, and leakage control).
//! - Specific scope anchor: `crates/ferrum_core/src/ml/feature_selection.rs` defines feature selection behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Canonical ML literature for `Feature Selection` algorithm family (statistical learning and validation texts).
//! - scikit-learn/RDKit-style API ergonomics that inform `Feature Selection` fit/transform/predict contracts.
//! - OECD QSAR validation principles where `Feature Selection` is used in predictive chemistry contexts.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/ml/feature_selection.rs`.

use ndarray::{Array2, Axis};

/// Remove features with variance below a threshold.
///
/// Returns indices of features to keep.
pub fn variance_threshold(data: &Array2<f64>, threshold: f64) -> Vec<usize> {
    let mean = data.mean_axis(Axis(0)).unwrap();
    let n = data.nrows() as f64;
    let mut keep = Vec::new();

    for j in 0..data.ncols() {
        let var: f64 = data
            .column(j)
            .iter()
            .map(|&x| (x - mean[j]).powi(2))
            .sum::<f64>()
            / n;
        if var > threshold {
            keep.push(j);
        }
    }
    keep
}

/// Remove highly correlated features (Pearson |r| > threshold).
///
/// Greedily removes the feature with higher mean absolute correlation.
/// Returns indices of features to keep.
pub fn correlation_filter(data: &Array2<f64>, threshold: f64) -> Vec<usize> {
    let n_feat = data.ncols();
    if n_feat == 0 {
        return Vec::new();
    }
    let n = data.nrows() as f64;

    // Compute means and centered matrix once.
    let mean = data.mean_axis(Axis(0)).unwrap();
    let centered_data = data - &mean;

    // Compute stds from centered matrix.
    let mut std_devs = vec![0.0f64; n_feat];
    for j in 0..n_feat {
        let var: f64 = centered_data.column(j).iter().map(|&x| x * x).sum::<f64>() / n;
        std_devs[j] = var.sqrt().max(1e-12);
    }

    // Compute packed upper-triangle correlation matrix.
    let upper_len = n_feat * (n_feat + 1) / 2;
    let mut corr_upper = vec![0.0f64; upper_len];
    let mut row_start = vec![0usize; n_feat];
    for (i, slot) in row_start.iter_mut().enumerate() {
        *slot = i * n_feat - (i * i.saturating_sub(1)) / 2;
    }
    for i in 0..n_feat {
        let row_start_i = row_start[i];
        corr_upper[row_start_i] = 1.0;
        let col_i = centered_data.column(i);
        for j in (i + 1)..n_feat {
            let cov = col_i.dot(&centered_data.column(j)) / n;
            let r = cov / (std_devs[i] * std_devs[j]);
            corr_upper[row_start_i + (j - i)] = r;
        }
    }

    // Greedy removal: find pairs above threshold, remove the one with higher mean |corr|
    let mut removed = vec![false; n_feat];
    loop {
        let mut worst_pair: Option<(usize, usize, f64)> = None;
        for i in 0..n_feat {
            if removed[i] {
                continue;
            }
            let row_start_i = row_start[i];
            for j in (i + 1)..n_feat {
                if removed[j] {
                    continue;
                }
                let r_abs = corr_upper[row_start_i + (j - i)].abs();
                if r_abs > threshold {
                    match &worst_pair {
                        Some((_, _, best_r)) if r_abs <= *best_r => {}
                        _ => worst_pair = Some((i, j, r_abs)),
                    }
                }
            }
        }
        match worst_pair {
            Some((i, j, _)) => {
                // Remove the feature with higher mean absolute correlation to others
                let mean_corr_i: f64 = (0..n_feat)
                    .filter(|&k| !removed[k] && k != i)
                    .map(|k| {
                        if i <= k {
                            corr_upper[row_start[i] + (k - i)].abs()
                        } else {
                            corr_upper[row_start[k] + (i - k)].abs()
                        }
                    })
                    .sum::<f64>();
                let mean_corr_j: f64 = (0..n_feat)
                    .filter(|&k| !removed[k] && k != j)
                    .map(|k| {
                        if j <= k {
                            corr_upper[row_start[j] + (k - j)].abs()
                        } else {
                            corr_upper[row_start[k] + (j - k)].abs()
                        }
                    })
                    .sum::<f64>();
                if mean_corr_i >= mean_corr_j {
                    removed[i] = true;
                } else {
                    removed[j] = true;
                }
            }
            None => break,
        }
    }

    (0..n_feat).filter(|&j| !removed[j]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    /// Runs regression test for variance threshold.
    ///
    /// Function context: `crates/ferrum_core/src/ml/feature_selection.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_variance_threshold() {
        // col 0: constant (var=0), col 1: varies, col 2: varies
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                5.0, 1.0, 10.0, 5.0, 2.0, 20.0, 5.0, 3.0, 30.0, 5.0, 4.0, 40.0,
            ],
        )
        .unwrap();
        let keep = variance_threshold(&data, 0.0);
        assert_eq!(keep, vec![1, 2]); // col 0 removed (zero variance)
    }

    #[test]
    /// Runs regression test for variance threshold higher.
    ///
    /// Function context: `crates/ferrum_core/src/ml/feature_selection.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_variance_threshold_higher() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 100.0, 1.1, 200.0, 0.9, 300.0, 1.0, 400.0])
                .unwrap();
        let keep = variance_threshold(&data, 1.0);
        // col 0 has low variance, col 1 has high variance
        assert!(keep.contains(&1));
        assert!(!keep.contains(&0));
    }

    #[test]
    /// Runs regression test for correlation filter.
    ///
    /// Function context: `crates/ferrum_core/src/ml/feature_selection.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_correlation_filter() {
        // col 0 and col 1 are perfectly correlated, col 2 is independent
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 10.0, 2.0, 4.0, 20.0, 3.0, 6.0, 5.0, 4.0, 8.0, 15.0,
            ],
        )
        .unwrap();
        let keep = correlation_filter(&data, 0.95);
        assert_eq!(keep.len(), 2); // one of the correlated pair removed
        assert!(keep.contains(&2)); // independent feature kept
    }

    #[test]
    /// Runs regression test for no removal below threshold.
    ///
    /// Function context: `crates/ferrum_core/src/ml/feature_selection.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_no_removal_below_threshold() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 5.0, 3.0, 20.0, 4.0, 1.0]).unwrap();
        let keep = correlation_filter(&data, 0.95);
        assert_eq!(keep.len(), 2);
    }
}
