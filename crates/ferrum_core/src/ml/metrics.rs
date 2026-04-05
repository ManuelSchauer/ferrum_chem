//! Evaluation metrics — regression and classification.

//! ## File guide: Metrics
//!
//! This file belongs to FerrumChem's **machine-learning pipeline layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn r_squared`
//! - `fn rmse`
//! - `fn mae`
//! - `fn q_squared`
//! - `fn accuracy`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Metrics` assumes `ndarray` row-major sample layout (`n_samples x n_features`) throughout fit/predict paths.
//! - `Metrics` prioritizes reproducibility via explicit seeds and deterministic iteration order where applicable.
//! - Model quality metrics for `Metrics` depend on consistent preprocessing choices (scaling, splitting, and leakage control).
//! - Specific scope anchor: `crates/ferrum_core/src/ml/metrics.rs` defines metrics behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Canonical ML literature for `Metrics` algorithm family (statistical learning and validation texts).
//! - scikit-learn/RDKit-style API ergonomics that inform `Metrics` fit/transform/predict contracts.
//! - OECD QSAR validation principles where `Metrics` is used in predictive chemistry contexts.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/ml/metrics.rs`.

/// R² (coefficient of determination).
pub fn r_squared(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len();
    assert_eq!(n, predicted.len());
    if n == 0 {
        return 0.0;
    }
    let mean: f64 = actual.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = actual.iter().map(|&y| (y - mean).powi(2)).sum();
    let ss_res: f64 = actual
        .iter()
        .zip(predicted)
        .map(|(&a, p)| (a - p).powi(2))
        .sum();
    if ss_tot < 1e-15 {
        return if ss_res < 1e-15 { 1.0 } else { 0.0 };
    }
    1.0 - ss_res / ss_tot
}

/// Root Mean Squared Error.
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len();
    assert_eq!(n, predicted.len());
    let mse: f64 = actual
        .iter()
        .zip(predicted)
        .map(|(&a, p)| (a - p).powi(2))
        .sum::<f64>()
        / n as f64;
    mse.sqrt()
}

/// Mean Absolute Error.
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len();
    assert_eq!(n, predicted.len());
    actual
        .iter()
        .zip(predicted)
        .map(|(&a, p)| (a - p).abs())
        .sum::<f64>()
        / n as f64
}

/// Q² (cross-validated R²) — same formula as R² but on CV predictions.
pub fn q_squared(actual: &[f64], cv_predicted: &[f64]) -> f64 {
    r_squared(actual, cv_predicted)
}

/// Accuracy for binary classification (threshold = 0.5).
pub fn accuracy(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len();
    assert_eq!(n, predicted.len());
    let correct = actual
        .iter()
        .zip(predicted)
        .filter(|&(&a, &p)| {
            let a_class = if a >= 0.5 { 1 } else { 0 };
            let p_class = if p >= 0.5 { 1 } else { 0 };
            a_class == p_class
        })
        .count();
    correct as f64 / n as f64
}

/// Matthews Correlation Coefficient for binary classification.
pub fn mcc(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len();
    assert_eq!(n, predicted.len());

    let (mut tp, mut tn, mut fp, mut fn_) = (0u64, 0u64, 0u64, 0u64);
    for (&a, &p) in actual.iter().zip(predicted) {
        let a_pos = a >= 0.5;
        let p_pos = p >= 0.5;
        match (a_pos, p_pos) {
            (true, true) => tp += 1,
            (false, false) => tn += 1,
            (false, true) => fp += 1,
            (true, false) => fn_ += 1,
        }
    }

    let denom =
        ((tp + fp) as f64 * (tp + fn_) as f64 * (tn + fp) as f64 * (tn + fn_) as f64).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }
    (tp as f64 * tn as f64 - fp as f64 * fn_ as f64) / denom
}

/// Sensitivity (true positive rate / recall).
pub fn sensitivity(actual: &[f64], predicted: &[f64]) -> f64 {
    let (mut tp, mut fn_) = (0u64, 0u64);
    for (&a, &p) in actual.iter().zip(predicted) {
        if a >= 0.5 {
            if p >= 0.5 {
                tp += 1;
            } else {
                fn_ += 1;
            }
        }
    }
    if tp + fn_ == 0 {
        return 0.0;
    }
    tp as f64 / (tp + fn_) as f64
}

/// Specificity (true negative rate).
pub fn specificity(actual: &[f64], predicted: &[f64]) -> f64 {
    let (mut tn, mut fp) = (0u64, 0u64);
    for (&a, &p) in actual.iter().zip(predicted) {
        if a < 0.5 {
            if p < 0.5 {
                tn += 1;
            } else {
                fp += 1;
            }
        }
    }
    if tn + fp == 0 {
        return 0.0;
    }
    tn as f64 / (tn + fp) as f64
}

/// AUC-ROC via trapezoidal integration of the ROC curve.
///
/// `predicted` should be continuous scores (probabilities).
pub fn auc_roc(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len();
    assert_eq!(n, predicted.len());

    // Sort by predicted score descending
    let mut pairs: Vec<(f64, f64)> = predicted
        .iter()
        .zip(actual)
        .map(|(&p, a)| (p, *a))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let n_pos = actual.iter().filter(|&&a| a >= 0.5).count() as f64;
    let n_neg = actual.iter().filter(|&&a| a < 0.5).count() as f64;

    if n_pos < 1.0 || n_neg < 1.0 {
        return 0.5; // undefined
    }

    let mut auc = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_tp = 0.0;
    let mut prev_fp = 0.0;

    let mut prev_score = f64::NEG_INFINITY;
    for (score, label) in &pairs {
        if (*score - prev_score).abs() > 1e-15 {
            // Trapezoidal rule
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
            prev_tp = tp;
            prev_fp = fp;
            prev_score = *score;
        }
        if *label >= 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
    }
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;

    auc / (n_pos * n_neg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Runs regression test for r squared perfect.
    ///
    /// Function context: `crates/ferrum_core/src/ml/metrics.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_r_squared_perfect() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((r_squared(&actual, &predicted) - 1.0).abs() < 1e-10);
    }

    #[test]
    /// Runs regression test for r squared bad.
    ///
    /// Function context: `crates/ferrum_core/src/ml/metrics.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_r_squared_bad() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![3.0, 3.0, 3.0, 3.0, 3.0]; // mean prediction
        assert!(r_squared(&actual, &predicted).abs() < 1e-10);
    }

    #[test]
    /// Runs regression test for rmse.
    ///
    /// Function context: `crates/ferrum_core/src/ml/metrics.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_rmse() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.0, 2.0, 3.0];
        assert!(rmse(&actual, &predicted).abs() < 1e-10);
    }

    #[test]
    /// Runs regression test for rmse nonzero.
    ///
    /// Function context: `crates/ferrum_core/src/ml/metrics.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_rmse_nonzero() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.1, 2.1, 3.1];
        assert!((rmse(&actual, &predicted) - 0.1).abs() < 1e-10);
    }

    #[test]
    /// Runs regression test for mae.
    ///
    /// Function context: `crates/ferrum_core/src/ml/metrics.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_mae() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.5, 2.5, 3.5];
        assert!((mae(&actual, &predicted) - 0.5).abs() < 1e-10);
    }

    #[test]
    /// Runs regression test for accuracy perfect.
    ///
    /// Function context: `crates/ferrum_core/src/ml/metrics.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_accuracy_perfect() {
        let actual = vec![0.0, 1.0, 0.0, 1.0];
        let predicted = vec![0.1, 0.9, 0.2, 0.8];
        assert!((accuracy(&actual, &predicted) - 1.0).abs() < 1e-10);
    }

    #[test]
    /// Runs regression test for mcc perfect.
    ///
    /// Function context: `crates/ferrum_core/src/ml/metrics.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_mcc_perfect() {
        let actual = vec![0.0, 0.0, 1.0, 1.0];
        let predicted = vec![0.0, 0.0, 1.0, 1.0];
        assert!((mcc(&actual, &predicted) - 1.0).abs() < 1e-10);
    }

    #[test]
    /// Runs regression test for mcc random.
    ///
    /// Function context: `crates/ferrum_core/src/ml/metrics.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_mcc_random() {
        let actual = vec![0.0, 1.0, 0.0, 1.0];
        let predicted = vec![0.0, 0.0, 1.0, 1.0]; // 50% right
        assert!(mcc(&actual, &predicted).abs() < 1e-10);
    }

    #[test]
    /// Runs regression test for sensitivity specificity.
    ///
    /// Function context: `crates/ferrum_core/src/ml/metrics.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_sensitivity_specificity() {
        let actual = vec![1.0, 1.0, 0.0, 0.0];
        let predicted = vec![1.0, 0.0, 0.0, 0.0]; // TP=1, FN=1, TN=2, FP=0
        assert!((sensitivity(&actual, &predicted) - 0.5).abs() < 1e-10);
        assert!((specificity(&actual, &predicted) - 1.0).abs() < 1e-10);
    }

    #[test]
    /// Runs regression test for auc perfect.
    ///
    /// Function context: `crates/ferrum_core/src/ml/metrics.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_auc_perfect() {
        let actual = vec![0.0, 0.0, 1.0, 1.0];
        let predicted = vec![0.1, 0.2, 0.8, 0.9];
        let auc = auc_roc(&actual, &predicted);
        assert!((auc - 1.0).abs() < 1e-10, "AUC = {auc}");
    }

    #[test]
    /// Runs regression test for auc random.
    ///
    /// Function context: `crates/ferrum_core/src/ml/metrics.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_auc_random() {
        let actual = vec![0.0, 1.0, 0.0, 1.0];
        let predicted = vec![0.5, 0.5, 0.5, 0.5]; // all same score
        let auc = auc_roc(&actual, &predicted);
        assert!((auc - 0.5).abs() < 0.01, "AUC = {auc}");
    }
}
