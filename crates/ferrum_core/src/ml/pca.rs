//! PCA dimensionality reduction — Jacobi eigendecomposition (from scratch).

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Principal Component Analysis via covariance matrix eigendecomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pca {
    /// Mean of training data (for centering).
    pub mean: Array1<f64>,
    /// Principal components (n_components × n_features), row-major.
    pub components: Array2<f64>,
    /// Explained variance for each component.
    pub explained_variance: Array1<f64>,
    /// Fraction of total variance explained.
    pub explained_variance_ratio: Array1<f64>,
}

impl Pca {
    /// Fit PCA, keeping `n_components` principal components.
    pub fn fit(data: &Array2<f64>, n_components: usize) -> Self {
        let n = data.nrows() as f64;
        let p = data.ncols();
        let nc = n_components.min(p);

        // Center data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut centered = data.clone();
        for mut row in centered.rows_mut() {
            let c = &row.to_owned() - &mean;
            row.assign(&c);
        }

        // Covariance matrix (p × p)
        let mut cov = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in i..p {
                let mut s = 0.0;
                for k in 0..data.nrows() {
                    s += centered[[k, i]] * centered[[k, j]];
                }
                s /= n;
                cov[[i, j]] = s;
                cov[[j, i]] = s;
            }
        }

        // Jacobi eigendecomposition
        let (eigenvalues, eigenvectors) = jacobi_eigen(&cov);

        // Sort eigenvalues descending
        let mut indices: Vec<usize> = (0..p).collect();
        indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

        let total_var: f64 = eigenvalues.iter().sum();
        let total_var = if total_var.abs() < 1e-15 { 1.0 } else { total_var };

        let mut components = Array2::<f64>::zeros((nc, p));
        let mut explained_variance = Array1::<f64>::zeros(nc);
        let mut explained_variance_ratio = Array1::<f64>::zeros(nc);

        for (i, &idx) in indices.iter().take(nc).enumerate() {
            for j in 0..p {
                components[[i, j]] = eigenvectors[[j, idx]];
            }
            explained_variance[i] = eigenvalues[idx].max(0.0);
            explained_variance_ratio[i] = eigenvalues[idx].max(0.0) / total_var;
        }

        Self {
            mean,
            components,
            explained_variance,
            explained_variance_ratio,
        }
    }

    /// Transform data into PCA space.
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let n = data.nrows();
        let nc = self.components.nrows();
        let mut result = Array2::<f64>::zeros((n, nc));

        for i in 0..n {
            let centered = &data.row(i).to_owned() - &self.mean;
            for j in 0..nc {
                let dot: f64 = centered.iter()
                    .zip(self.components.row(j).iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                result[[i, j]] = dot;
            }
        }
        result
    }

    /// Fit and transform in one step.
    pub fn fit_transform(data: &Array2<f64>, n_components: usize) -> (Self, Array2<f64>) {
        let pca = Self::fit(data, n_components);
        let transformed = pca.transform(data);
        (pca, transformed)
    }

    /// Inverse transform: project back to original space.
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let n = data.nrows();
        let p = self.mean.len();
        let nc = self.components.nrows();
        let mut result = Array2::<f64>::zeros((n, p));

        for i in 0..n {
            for j in 0..p {
                let mut v = self.mean[j];
                for k in 0..nc {
                    v += data[[i, k]] * self.components[[k, j]];
                }
                result[[i, j]] = v;
            }
        }
        result
    }

    /// Number of components.
    pub fn n_components(&self) -> usize {
        self.components.nrows()
    }
}

/// Jacobi eigenvalue algorithm for symmetric matrices.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors are column-wise.
fn jacobi_eigen(a: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut d = a.clone(); // working matrix
    let mut v = Array2::<f64>::eye(n); // eigenvector accumulator

    let max_iter = 100 * n * n;
    let eps = 1e-12;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = d[[i, j]].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < eps {
            break;
        }

        // Compute rotation angle
        let dpp = d[[p, p]];
        let dqq = d[[q, q]];
        let dpq = d[[p, q]];

        let theta = if (dpp - dqq).abs() < eps {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * dpq / (dpp - dqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Givens rotation
        let mut new_d = d.clone();
        new_d[[p, p]] = c * c * dpp + 2.0 * s * c * dpq + s * s * dqq;
        new_d[[q, q]] = s * s * dpp - 2.0 * s * c * dpq + c * c * dqq;
        new_d[[p, q]] = 0.0;
        new_d[[q, p]] = 0.0;

        for i in 0..n {
            if i != p && i != q {
                let dip = d[[i, p]];
                let diq = d[[i, q]];
                new_d[[i, p]] = c * dip + s * diq;
                new_d[[p, i]] = new_d[[i, p]];
                new_d[[i, q]] = -s * dip + c * diq;
                new_d[[q, i]] = new_d[[i, q]];
            }
        }

        d = new_d;

        // Update eigenvectors
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip + s * viq;
            v[[i, q]] = -s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| d[[i, i]]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_pca_basic() {
        // 2D data with clear principal axis along x=y
        let data = Array2::from_shape_vec((5, 2), vec![
            1.0, 1.0,
            2.0, 2.0,
            3.0, 3.0,
            4.0, 4.0,
            5.0, 5.0,
        ]).unwrap();
        let pca = Pca::fit(&data, 2);
        assert_eq!(pca.n_components(), 2);
        // First component should explain nearly all variance
        assert!(pca.explained_variance_ratio[0] > 0.99);
    }

    #[test]
    fn test_pca_dimensionality_reduction() {
        let data = Array2::from_shape_vec((4, 3), vec![
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0,
            4.0, 8.0, 12.0,
        ]).unwrap();
        let (pca, transformed) = Pca::fit_transform(&data, 1);
        assert_eq!(transformed.ncols(), 1);
        // Reconstruction should be close to original for rank-1 data
        let reconstructed = pca.inverse_transform(&transformed);
        for i in 0..4 {
            for j in 0..3 {
                assert!(
                    (reconstructed[[i, j]] - data[[i, j]]).abs() < 1e-6,
                    "Reconstruction error at [{i},{j}]: {} vs {}",
                    reconstructed[[i, j]], data[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_pca_transform_dimensions() {
        let data = Array2::from_shape_vec((6, 4), vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
        ]).unwrap();
        let (_, transformed) = Pca::fit_transform(&data, 2);
        assert_eq!(transformed.nrows(), 6);
        assert_eq!(transformed.ncols(), 2);
    }

    #[test]
    fn test_jacobi_identity() {
        // Eigenvalues of identity should be all 1
        let eye = Array2::<f64>::eye(3);
        let (vals, _) = jacobi_eigen(&eye);
        for v in &vals {
            assert!((*v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_jacobi_diagonal() {
        let mut diag = Array2::<f64>::zeros((3, 3));
        diag[[0, 0]] = 3.0;
        diag[[1, 1]] = 1.0;
        diag[[2, 2]] = 2.0;
        let (vals, _) = jacobi_eigen(&diag);
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-10);
        assert!((sorted[1] - 2.0).abs() < 1e-10);
        assert!((sorted[2] - 3.0).abs() < 1e-10);
    }
}
