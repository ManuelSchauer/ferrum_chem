//! Applicability Domain — k-NN distance AD, leverage AD.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Applicability Domain trait.
pub trait ApplicabilityDomain: Send + Sync {
    /// Check if a sample is within the domain.
    fn is_in_domain(&self, sample: &[f64]) -> bool;

    /// Confidence score (higher = more reliable prediction).
    fn confidence(&self, sample: &[f64]) -> f64;
}

/// k-NN distance-based applicability domain.
///
/// A sample is in domain if its mean distance to k nearest training neighbors
/// is below a threshold (mean + z*std of training distances).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnDistanceAD {
    pub train_features: Array2<f64>,
    pub k: usize,
    pub threshold: f64,
    pub mean_dist: f64,
    pub std_dist: f64,
}

impl KnnDistanceAD {
    /// Fit the AD from training data.
    ///
    /// `z_cutoff`: number of standard deviations for threshold (typically 1.0-3.0).
    pub fn fit(features: &Array2<f64>, k: usize, z_cutoff: f64) -> Self {
        let n = features.nrows();
        let k = k.min(n - 1).max(1);

        // Compute mean k-NN distance for each training sample
        let mut distances = Vec::with_capacity(n);
        for i in 0..n {
            let sample = features.row(i);
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let diff = &sample.to_owned() - &features.row(j).to_owned();
                    diff.dot(&diff).sqrt()
                })
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean_k_dist: f64 = dists.iter().take(k).sum::<f64>() / k as f64;
            distances.push(mean_k_dist);
        }

        let mean_dist = distances.iter().sum::<f64>() / n as f64;
        let std_dist = (distances.iter()
            .map(|&d| (d - mean_dist).powi(2))
            .sum::<f64>() / n as f64).sqrt();

        let threshold = mean_dist + z_cutoff * std_dist;

        Self {
            train_features: features.clone(),
            k,
            threshold,
            mean_dist,
            std_dist,
        }
    }

    fn mean_knn_distance(&self, sample: &[f64]) -> f64 {
        let sample_arr = Array1::from(sample.to_vec());
        let mut dists: Vec<f64> = (0..self.train_features.nrows())
            .map(|i| {
                let diff = &sample_arr - &self.train_features.row(i).to_owned();
                diff.dot(&diff).sqrt()
            })
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        dists.iter().take(self.k).sum::<f64>() / self.k as f64
    }
}

impl ApplicabilityDomain for KnnDistanceAD {
    fn is_in_domain(&self, sample: &[f64]) -> bool {
        self.mean_knn_distance(sample) <= self.threshold
    }

    fn confidence(&self, sample: &[f64]) -> f64 {
        let dist = self.mean_knn_distance(sample);
        if dist <= self.mean_dist {
            1.0
        } else if dist >= self.threshold {
            0.0
        } else {
            1.0 - (dist - self.mean_dist) / (self.threshold - self.mean_dist)
        }
    }
}

/// Leverage-based applicability domain (Williams plot).
///
/// h_i = x_i' (X'X)^{-1} x_i
/// Threshold: h* = 3p/n (p = features, n = samples)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeverageAD {
    /// (X'X)^{-1} matrix.
    pub hat_matrix_inv: Array2<f64>,
    /// Training feature means (for centering).
    pub x_mean: Array1<f64>,
    /// Leverage threshold h*.
    pub threshold: f64,
}

impl LeverageAD {
    /// Fit leverage AD from training features.
    pub fn fit(features: &Array2<f64>) -> Self {
        let n = features.nrows();
        let p = features.ncols();

        let x_mean = features.mean_axis(ndarray::Axis(0)).unwrap();

        // Center features
        let mut x_centered = features.clone();
        for mut row in x_centered.rows_mut() {
            let c = &row.to_owned() - &x_mean;
            row.assign(&c);
        }

        // Compute X'X
        let mut xtx = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in i..p {
                let dot: f64 = x_centered.column(i).dot(&x_centered.column(j));
                xtx[[i, j]] = dot;
                xtx[[j, i]] = dot;
            }
        }

        // Invert X'X (simple Gauss-Jordan for small matrices)
        let hat_matrix_inv = invert_matrix(&xtx);

        let threshold = 3.0 * p as f64 / n as f64;

        Self {
            hat_matrix_inv,
            x_mean,
            threshold,
        }
    }

    /// Compute leverage for a sample.
    pub fn leverage(&self, sample: &[f64]) -> f64 {
        let p = self.x_mean.len();
        let mut centered = vec![0.0; p];
        for j in 0..p {
            centered[j] = sample[j] - self.x_mean[j];
        }

        // h = x' (X'X)^{-1} x
        let mut h = 0.0;
        for i in 0..p {
            let mut inner = 0.0;
            for j in 0..p {
                inner += self.hat_matrix_inv[[i, j]] * centered[j];
            }
            h += centered[i] * inner;
        }
        h
    }
}

impl ApplicabilityDomain for LeverageAD {
    fn is_in_domain(&self, sample: &[f64]) -> bool {
        self.leverage(sample) <= self.threshold
    }

    fn confidence(&self, sample: &[f64]) -> f64 {
        let h = self.leverage(sample);
        if h <= 0.0 {
            1.0
        } else if h >= self.threshold {
            0.0
        } else {
            1.0 - h / self.threshold
        }
    }
}

/// Simple matrix inversion via Gauss-Jordan elimination.
fn invert_matrix(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut aug = Array2::<f64>::zeros((n, 2 * n));

    // [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            // Singular matrix — add small regularization
            aug[[col, col]] += 1e-10;
        }

        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Scale pivot row
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }

        // Eliminate
        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..(2 * n) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_knn_ad_in_domain() {
        let features = Array2::from_shape_vec((6, 2), vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            0.5, 0.5,
            0.5, 0.0,
        ]).unwrap();
        let ad = KnnDistanceAD::fit(&features, 3, 2.0);

        // Point inside cluster → in domain
        assert!(ad.is_in_domain(&[0.5, 0.5]));
        // Point far away → out of domain
        assert!(!ad.is_in_domain(&[100.0, 100.0]));
    }

    #[test]
    fn test_knn_ad_confidence() {
        let features = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let ad = KnnDistanceAD::fit(&features, 2, 2.0);

        let conf_in = ad.confidence(&[1.5]);
        let conf_out = ad.confidence(&[100.0]);
        assert!(conf_in > conf_out, "in={conf_in}, out={conf_out}");
    }

    #[test]
    fn test_leverage_ad() {
        let features = Array2::from_shape_vec((5, 2), vec![
            1.0, 2.0,
            2.0, 3.0,
            3.0, 4.0,
            4.0, 5.0,
            5.0, 6.0,
        ]).unwrap();
        let ad = LeverageAD::fit(&features);

        // Interior point should have low leverage
        let h_in = ad.leverage(&[3.0, 4.0]);
        // Extrapolation point should have high leverage
        let h_out = ad.leverage(&[100.0, 200.0]);
        assert!(h_in < h_out, "h_in={h_in}, h_out={h_out}");
    }

    #[test]
    fn test_invert_identity() {
        let eye = Array2::<f64>::eye(3);
        let inv = invert_matrix(&eye);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[[i, j]] - expected).abs() < 1e-10);
            }
        }
    }
}
