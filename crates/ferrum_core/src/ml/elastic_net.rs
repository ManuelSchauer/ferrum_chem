//! Elastic Net / LASSO regression — coordinate descent, from scratch.
//!
//! Automatic feature selection via L1 regularization.
//! α controls overall regularization strength, l1_ratio controls L1/L2 mix.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Elastic Net regression model.
///
/// Minimizes: (1/2n) ||y - Xβ||² + α * l1_ratio * ||β||₁ + α * (1-l1_ratio)/2 * ||β||²
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticNet {
    /// Regression coefficients (n_features,).
    pub coefficients: Array1<f64>,
    /// Intercept term.
    pub intercept: f64,
    /// Regularization strength.
    pub alpha: f64,
    /// L1/L2 mixing parameter (1.0 = LASSO, 0.0 = Ridge).
    pub l1_ratio: f64,
    /// X mean (for centering).
    pub x_mean: Array1<f64>,
    /// X std (for scaling).
    pub x_std: Array1<f64>,
    /// Y mean.
    pub y_mean: f64,
    /// Number of iterations until convergence.
    pub n_iter: usize,
    /// Feature names with non-zero coefficients.
    pub selected_features: Vec<usize>,
}

impl ElasticNet {
    /// Fit Elastic Net via coordinate descent.
    ///
    /// - `alpha`: regularization strength (0 = OLS)
    /// - `l1_ratio`: 1.0 = LASSO, 0.0 = Ridge, 0.5 = balanced
    /// - `max_iter`: maximum iterations
    /// - `tol`: convergence tolerance
    pub fn fit(
        features: &Array2<f64>,
        targets: &Array1<f64>,
        alpha: f64,
        l1_ratio: f64,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        let n = features.nrows() as f64;
        let p = features.ncols();

        // Center and scale
        let x_mean = features.mean_axis(Axis(0)).unwrap();
        let mut x_std = Array1::<f64>::zeros(p);
        for j in 0..p {
            let var: f64 = features.column(j).iter()
                .map(|&x| (x - x_mean[j]).powi(2))
                .sum::<f64>() / n;
            x_std[j] = var.sqrt().max(1e-12);
        }

        let mut x = features.clone();
        for mut row in x.rows_mut() {
            let centered = (&row.to_owned() - &x_mean) / &x_std;
            row.assign(&centered);
        }

        let y_mean = targets.mean().unwrap();
        let y: Array1<f64> = targets.mapv(|v| v - y_mean);

        // Precompute X'X diagonal and X'y
        let mut x_sq = Array1::<f64>::zeros(p);
        for j in 0..p {
            x_sq[j] = x.column(j).dot(&x.column(j));
        }

        let l1_penalty = alpha * l1_ratio;
        let l2_penalty = alpha * (1.0 - l1_ratio);

        let mut beta = Array1::<f64>::zeros(p);
        let mut residual = y.clone();
        let mut n_iter = 0;

        for iter in 0..max_iter {
            let mut max_change = 0.0f64;

            for j in 0..p {
                let old_beta = beta[j];

                // Partial residual (add back contribution of feature j)
                if old_beta != 0.0 {
                    for i in 0..features.nrows() {
                        residual[i] += x[[i, j]] * old_beta;
                    }
                }

                // Compute the simple least squares update
                let rho = x.column(j).dot(&residual);

                // Soft thresholding (proximal operator for L1)
                let new_beta = soft_threshold(rho, l1_penalty * n)
                    / (x_sq[j] + l2_penalty * n);

                beta[j] = new_beta;

                // Update residual
                if new_beta != 0.0 {
                    for i in 0..features.nrows() {
                        residual[i] -= x[[i, j]] * new_beta;
                    }
                }

                max_change = max_change.max((new_beta - old_beta).abs());
            }

            n_iter = iter + 1;
            if max_change < tol {
                break;
            }
        }

        // Scale coefficients back to original feature scale
        let mut coefficients = Array1::<f64>::zeros(p);
        for j in 0..p {
            coefficients[j] = beta[j] / x_std[j];
        }

        let selected_features: Vec<usize> = (0..p)
            .filter(|&j| coefficients[j].abs() > 1e-12)
            .collect();

        Self {
            coefficients,
            intercept: y_mean,
            alpha,
            l1_ratio,
            x_mean,
            x_std,
            y_mean,
            n_iter,
            selected_features,
        }
    }

    /// Convenience constructor for pure LASSO (l1_ratio = 1.0).
    pub fn lasso(features: &Array2<f64>, targets: &Array1<f64>, alpha: f64) -> Self {
        Self::fit(features, targets, alpha, 1.0, 1000, 1e-6)
    }

    /// Convenience constructor for pure Ridge (l1_ratio = 0.0).
    pub fn ridge(features: &Array2<f64>, targets: &Array1<f64>, alpha: f64) -> Self {
        Self::fit(features, targets, alpha, 0.0, 1000, 1e-6)
    }

    /// Predict for a single sample.
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        let mut pred = self.intercept;
        for (j, &x) in sample.iter().enumerate() {
            pred += (x - self.x_mean[j]) * self.coefficients[j];
        }
        pred
    }

    /// Predict for multiple samples.
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        Array1::from_vec(
            (0..features.nrows())
                .map(|i| self.predict_one(features.row(i).as_slice().unwrap()))
                .collect()
        )
    }

    /// Number of non-zero coefficients (selected features).
    pub fn n_selected(&self) -> usize {
        self.selected_features.len()
    }
}

/// Soft thresholding operator: sign(x) * max(|x| - λ, 0)
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn test_elastic_net_linear() {
        // y = 2*x1 + 3*x2
        let features = Array2::from_shape_vec((8, 2), vec![
            1.0, 1.0,
            2.0, 1.0,
            3.0, 1.0,
            1.0, 2.0,
            2.0, 2.0,
            3.0, 2.0,
            1.0, 3.0,
            2.0, 3.0,
        ]).unwrap();
        let targets: Array1<f64> = features.rows().into_iter()
            .map(|r| 2.0 * r[0] + 3.0 * r[1])
            .collect();

        let model = ElasticNet::fit(&features, &targets, 0.001, 0.5, 1000, 1e-8);
        let preds = model.predict(&features);

        for (&actual, &pred) in targets.iter().zip(preds.iter()) {
            assert!(
                (actual - pred).abs() < 1.0,
                "EN pred {pred} too far from {actual}"
            );
        }
    }

    #[test]
    fn test_lasso_sparsity() {
        // 3 features, only first is relevant: y = x1
        let features = Array2::from_shape_vec((10, 3), vec![
            1.0, 5.0, 3.0,
            2.0, 3.0, 7.0,
            3.0, 8.0, 1.0,
            4.0, 2.0, 9.0,
            5.0, 6.0, 4.0,
            6.0, 1.0, 8.0,
            7.0, 4.0, 2.0,
            8.0, 7.0, 6.0,
            9.0, 9.0, 5.0,
            10.0, 0.0, 0.0,
        ]).unwrap();
        let targets = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let model = ElasticNet::lasso(&features, &targets, 0.1);

        // Feature 0 should have the largest coefficient
        assert!(
            model.coefficients[0].abs() > model.coefficients[1].abs(),
            "Coefficients: {:?}", model.coefficients
        );
        assert!(
            model.coefficients[0].abs() > model.coefficients[2].abs(),
            "Coefficients: {:?}", model.coefficients
        );
    }

    #[test]
    fn test_ridge_no_sparsity() {
        let features = Array2::from_shape_vec((5, 2), vec![
            1.0, 2.0,
            2.0, 3.0,
            3.0, 4.0,
            4.0, 5.0,
            5.0, 6.0,
        ]).unwrap();
        let targets = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = ElasticNet::ridge(&features, &targets, 0.01);
        // Ridge should keep all features (non-zero)
        assert!(model.coefficients[0].abs() > 1e-6);
        assert!(model.coefficients[1].abs() > 1e-6);
    }

    #[test]
    fn test_soft_threshold() {
        assert!((soft_threshold(5.0, 2.0) - 3.0).abs() < 1e-10);
        assert!((soft_threshold(-5.0, 2.0) - (-3.0)).abs() < 1e-10);
        assert!((soft_threshold(1.0, 2.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_selected_features() {
        let features = Array2::from_shape_vec((6, 3), vec![
            1.0, 0.0, 0.5,
            2.0, 0.0, 1.0,
            3.0, 0.0, 1.5,
            4.0, 0.0, 2.0,
            5.0, 0.0, 2.5,
            6.0, 0.0, 3.0,
        ]).unwrap();
        let targets = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let model = ElasticNet::lasso(&features, &targets, 0.5);
        // Constant feature (col 1) should be dropped
        assert!(!model.selected_features.contains(&1),
            "Selected: {:?}, coeffs: {:?}", model.selected_features, model.coefficients);
    }
}
