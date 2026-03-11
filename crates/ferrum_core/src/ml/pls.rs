//! PLS (Partial Least Squares) Regression — NIPALS algorithm, from scratch.
//!
//! Handles collinearity in the mega-vector of descriptors.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// PLS Regression model (PLS1 — single y variable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlsRegression {
    /// Number of latent components.
    pub n_components: usize,
    /// X loadings (n_components × n_features).
    pub x_loadings: Array2<f64>,
    /// X weights (n_components × n_features).
    pub x_weights: Array2<f64>,
    /// Y loadings (n_components,).
    pub y_loadings: Array1<f64>,
    /// Regression coefficients (n_features,).
    pub coefficients: Array1<f64>,
    /// Intercept.
    pub intercept: f64,
    /// X mean (for centering).
    pub x_mean: Array1<f64>,
    /// X std (for scaling).
    pub x_std: Array1<f64>,
    /// Y mean.
    pub y_mean: f64,
}

impl PlsRegression {
    /// Fit PLS regression using NIPALS algorithm.
    pub fn fit(features: &Array2<f64>, targets: &Array1<f64>, n_components: usize) -> Self {
        let n = features.nrows();
        let p = features.ncols();
        let nc = n_components.min(p).min(n);

        // Center and scale X
        let x_mean = features.mean_axis(Axis(0)).unwrap();
        let mut x_std = Array1::<f64>::zeros(p);
        for j in 0..p {
            let var: f64 = features
                .column(j)
                .iter()
                .map(|&x| (x - x_mean[j]).powi(2))
                .sum::<f64>()
                / n as f64;
            x_std[j] = var.sqrt().max(1e-12);
        }

        let mut x = features.clone();
        for mut row in x.rows_mut() {
            let centered = (&row.to_owned() - &x_mean) / &x_std;
            row.assign(&centered);
        }

        // Center y
        let y_mean = targets.mean().unwrap();
        let mut y: Array1<f64> = targets.mapv(|v| v - y_mean);

        let mut x_weights = Array2::<f64>::zeros((nc, p));
        let mut x_loadings = Array2::<f64>::zeros((nc, p));
        let mut y_loadings = Array1::<f64>::zeros(nc);
        let mut t_scores = Array2::<f64>::zeros((n, nc)); // score matrix

        for a in 0..nc {
            // NIPALS iteration
            // w = X'y / ||X'y||
            let mut w = Array1::<f64>::zeros(p);
            for j in 0..p {
                w[j] = x.column(j).dot(&y);
            }
            let w_norm = w.dot(&w).sqrt().max(1e-15);
            w.mapv_inplace(|v| v / w_norm);

            // t = Xw
            let mut t = Array1::<f64>::zeros(n);
            for i in 0..n {
                t[i] = x.row(i).dot(&w);
            }

            // p_loading = X't / (t't)
            let tt = t.dot(&t).max(1e-15);
            let mut p_loading = Array1::<f64>::zeros(p);
            for j in 0..p {
                p_loading[j] = x.column(j).dot(&t) / tt;
            }

            // q = y't / (t't)
            let q = y.dot(&t) / tt;

            // Store
            x_weights.row_mut(a).assign(&w);
            x_loadings.row_mut(a).assign(&p_loading);
            y_loadings[a] = q;
            t_scores.column_mut(a).assign(&t);

            // Deflate X and y
            for i in 0..n {
                for j in 0..p {
                    x[[i, j]] -= t[i] * p_loading[j];
                }
                y[i] -= t[i] * q;
            }
        }

        // Compute regression coefficients: B = W(P'W)^{-1} q
        // For PLS1: simplified via iterative approach
        let coefficients = Self::compute_coefficients(&x_weights, &x_loadings, &y_loadings, &x_std);

        Self {
            n_components: nc,
            x_loadings,
            x_weights,
            y_loadings,
            coefficients,
            intercept: y_mean,
            x_mean,
            x_std,
            y_mean,
        }
    }

    fn compute_coefficients(
        w: &Array2<f64>,
        p: &Array2<f64>,
        q: &Array1<f64>,
        x_std: &Array1<f64>,
    ) -> Array1<f64> {
        let nc = w.nrows();
        let n_feat = w.ncols();

        // W_star = W (P'W)^{-1}
        // For sequential deflation, W_star can be computed iteratively
        let mut w_star = Array2::<f64>::zeros((nc, n_feat));
        for a in 0..nc {
            let mut ws = w.row(a).to_owned();
            for b in 0..a {
                let pw = p.row(b).dot(&w.row(a));
                let ww = w_star.row(b).to_owned();
                ws = ws - &(ww * pw);
            }
            w_star.row_mut(a).assign(&ws);
        }

        // B = sum_a (w_star_a * q_a) / x_std
        let mut coeff = Array1::<f64>::zeros(n_feat);
        for a in 0..nc {
            for j in 0..n_feat {
                coeff[j] += w_star[[a, j]] * q[a];
            }
        }
        // Scale back
        for j in 0..n_feat {
            coeff[j] /= x_std[j];
        }
        coeff
    }

    /// Predict for multiple samples.
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        Array1::from_vec(
            (0..features.nrows())
                .map(|i| self.predict_one(features.row(i).as_slice().unwrap()))
                .collect(),
        )
    }

    /// Predict for a single sample.
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        let mut pred = self.intercept;
        for (j, &x) in sample.iter().enumerate() {
            pred += (x - self.x_mean[j]) * self.coefficients[j];
        }
        pred
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn test_pls_linear() {
        // y = 2*x1 + 3*x2
        let features = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 3.0,
                3.0, 3.0, 4.0, 4.0,
            ],
        )
        .unwrap();
        let targets: Array1<f64> = features
            .rows()
            .into_iter()
            .map(|r| 2.0 * r[0] + 3.0 * r[1])
            .collect();

        let model = PlsRegression::fit(&features, &targets, 2);
        let preds = model.predict(&features);

        for (&actual, &pred) in targets.iter().zip(preds.iter()) {
            assert!(
                (actual - pred).abs() < 0.5,
                "PLS prediction {pred} too far from {actual}"
            );
        }
    }

    #[test]
    fn test_pls_collinear() {
        // x2 = 2*x1, y = x1 (collinear features)
        let features = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0],
        )
        .unwrap();
        let targets = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let model = PlsRegression::fit(&features, &targets, 1);
        let pred = model.predict_one(&[3.0, 6.0]);
        assert!((pred - 3.0).abs() < 0.5, "Collinear PLS pred = {pred}");
    }

    #[test]
    fn test_pls_n_components() {
        let features = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                0.0, 1.0,
            ],
        )
        .unwrap();
        let targets = array![1.0, 2.0, 3.0, 3.0, 5.0, 4.0];

        let model = PlsRegression::fit(&features, &targets, 2);
        assert_eq!(model.n_components, 2);
        let preds = model.predict(&features);
        assert_eq!(preds.len(), 6);
    }
}
