//! PCA dimensionality reduction — Jacobi eigendecomposition (from scratch).

//! ## File guide: Pca
//!
//! This file belongs to FerrumChem's **machine-learning pipeline layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct Pca`
//! - `fn fit`
//! - `fn transform`
//! - `fn transform_one`
//! - `fn fit_transform`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - Chooses covariance eigendecomposition for p<=n and dual Gram eigendecomposition for p>n to control cost.
//! - Centered data and finite eigenvalue sanitation are required before component sorting and ratio computation.
//! - Jacobi solver tolerance/iteration settings trade tiny precision loss for predictable runtime on large descriptor matrices.
//! ### Chemistry/ML references
//! - Jolliffe & Cadima (2016). Principal Component Analysis review.
//! - Golub & Van Loan. Matrix Computations (eigendecomposition and numerical stability).
//! - Kernel/dual PCA derivations for p>>n regimes used to justify Gram-matrix path.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

#[cfg(feature = "native-linalg")]
use ndarray_linalg::{Eigh, UPLO};

/// Attempt symmetric eigendecomposition using native LAPACK when enabled.
#[cfg(feature = "native-linalg")]
fn symmetric_eigendecompose(a: &Array2<f64>) -> Option<(Vec<f64>, Array2<f64>)> {
    let (vals, vecs) = a.clone().eigh(UPLO::Lower).ok()?;
    Some((vals.to_vec(), vecs))
}

/// Fallback stub when native linear algebra is not enabled.
#[cfg(not(feature = "native-linalg"))]
fn symmetric_eigendecompose(_a: &Array2<f64>) -> Option<(Vec<f64>, Array2<f64>)> {
    None
}

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
    pub fn fit(data: &Array2<f64>, n_components: usize) -> Result<Self, String> {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let nc = n_components.min(n_features);
        let mean = data
            .mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::zeros(n_features));

        if n_samples == 0 || n_features == 0 || nc == 0 {
            return Err("Cannot fit PCA on empty dataset".to_string());
        }

        // Center data once and reuse in both covariance paths.
        let centered = data - &mean;
        let n = n_samples as f64;

        // When p >> n, eigendecompose the n×n Gram matrix and map eigenvectors
        // back to feature space. This avoids the enormous p×p covariance + Jacobi cost.
        let use_dual = n_features > n_samples;
        let (mut eigenvalues, eigenvectors) = if use_dual {
            let gram = centered.dot(&centered.t()) / n;
            symmetric_eigendecompose(&gram).unwrap_or_else(|| jacobi_eigen(&gram))
        } else {
            let cov = centered.t().dot(&centered) / n;
            symmetric_eigendecompose(&cov).unwrap_or_else(|| jacobi_eigen(&cov))
        };

        for value in &mut eigenvalues {
            if !value.is_finite() {
                *value = 0.0;
            }
        }

        // Sort eigenvalues descending.
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&a, &b| eigenvalues[b].total_cmp(&eigenvalues[a]));

        let total_var: f64 = eigenvalues.iter().map(|&v| v.max(0.0)).sum();
        let total_var = if total_var.abs() < 1e-15 {
            1.0
        } else {
            total_var
        };

        let mut components = Array2::<f64>::zeros((nc, n_features));
        let mut explained_variance = Array1::<f64>::zeros(nc);
        let mut explained_variance_ratio = Array1::<f64>::zeros(nc);

        for i in 0..nc {
            let Some(&idx) = indices.get(i) else {
                break;
            };
            let lambda = eigenvalues[idx].max(0.0);
            explained_variance[i] = lambda;
            explained_variance_ratio[i] = lambda / total_var;

            if lambda <= 1e-15 {
                continue;
            }

            if use_dual {
                // v = Xᵀu / sqrt(n * λ)
                let u = eigenvectors.column(idx).to_owned();
                let mut component = centered.t().dot(&u);
                let denom = (n * lambda).sqrt();
                if denom > 1e-15 {
                    component.mapv_inplace(|x| x / denom);
                }

                // Normalize to unit norm for numerical stability.
                let norm = component.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-15 {
                    component.mapv_inplace(|x| x / norm);
                }
                components.row_mut(i).assign(&component.view());
            } else {
                for j in 0..n_features {
                    components[[i, j]] = eigenvectors[[j, idx]];
                }
            }
        }

        Ok(Self {
            mean,
            components,
            explained_variance,
            explained_variance_ratio,
        })
    }

    /// Transform data into PCA space.
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let centered = data - &self.mean;
        centered.dot(&self.components.t())
    }

    /// Transform a single sample into PCA space.
    pub fn transform_one(&self, sample: &[f64]) -> Vec<f64> {
        assert_eq!(
            sample.len(),
            self.mean.len(),
            "Sample length ({}) must match PCA feature count ({})",
            sample.len(),
            self.mean.len()
        );
        let nc = self.components.nrows();
        let mut out = vec![0.0; nc];
        for (j, out_j) in out.iter_mut().enumerate() {
            let dot: f64 = sample
                .iter()
                .zip(self.mean.iter())
                .zip(self.components.row(j).iter())
                .map(|((&x, &m), &c)| (x - m) * c)
                .sum();
            *out_j = dot;
        }
        out
    }

    /// Fit and transform in one step.
    pub fn fit_transform(data: &Array2<f64>, n_components: usize) -> Result<(Self, Array2<f64>), String> {
        let pca = Self::fit(data, n_components)?;
        let transformed = pca.transform(data);
        Ok((pca, transformed))
    }

    /// Inverse transform: project back to original space.
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut result = data.dot(&self.components);
        for mut row in result.axis_iter_mut(Axis(0)) {
            row += &self.mean;
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
    if n == 0 {
        return (Vec::new(), Array2::<f64>::zeros((0, 0)));
    }
    if n == 1 {
        return (vec![a[[0, 0]]], Array2::<f64>::eye(1));
    }
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

        // Apply Givens rotation in-place.
        for i in 0..n {
            if i != p && i != q {
                let dip = d[[i, p]];
                let diq = d[[i, q]];
                let new_ip = c * dip + s * diq;
                let new_iq = -s * dip + c * diq;
                d[[i, p]] = new_ip;
                d[[p, i]] = new_ip;
                d[[i, q]] = new_iq;
                d[[q, i]] = new_iq;
            }
        }
        d[[p, p]] = c * c * dpp + 2.0 * s * c * dpq + s * s * dqq;
        d[[q, q]] = s * s * dpp - 2.0 * s * c * dpq + c * c * dqq;
        d[[p, q]] = 0.0;
        d[[q, p]] = 0.0;

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
    /// Runs regression test for pca basic.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pca.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_pca_basic() {
        // 2D data with clear principal axis along x=y
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        )
        .unwrap();
        let pca = Pca::fit(&data, 2).unwrap();
        assert_eq!(pca.n_components(), 2);
        // First component should explain nearly all variance
        assert!(pca.explained_variance_ratio[0] > 0.99);
    }

    #[test]
    /// Runs regression test for pca dimensionality reduction.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pca.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_pca_dimensionality_reduction() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0],
        )
        .unwrap();
        let (pca, transformed) = Pca::fit_transform(&data, 1).unwrap();
        assert_eq!(transformed.ncols(), 1);
        // Reconstruction should be close to original for rank-1 data
        let reconstructed = pca.inverse_transform(&transformed);
        for i in 0..4 {
            for j in 0..3 {
                assert!(
                    (reconstructed[[i, j]] - data[[i, j]]).abs() < 1e-6,
                    "Reconstruction error at [{i},{j}]: {} vs {}",
                    reconstructed[[i, j]],
                    data[[i, j]]
                );
            }
        }
    }

    #[test]
    /// Runs regression test for pca transform dimensions.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pca.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_pca_transform_dimensions() {
        let data = Array2::from_shape_vec(
            (6, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            ],
        )
        .unwrap();
        let (_, transformed) = Pca::fit_transform(&data, 2).unwrap();
        assert_eq!(transformed.nrows(), 6);
        assert_eq!(transformed.ncols(), 2);
    }

    #[test]
    /// Runs regression test for jacobi identity.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pca.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_jacobi_identity() {
        // Eigenvalues of identity should be all 1
        let eye = Array2::<f64>::eye(3);
        let (vals, _) = jacobi_eigen(&eye);
        for v in &vals {
            assert!((*v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    /// Runs regression test for jacobi diagonal.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pca.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
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

    #[test]
    /// Runs regression test for pca high dimensional small sample.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pca.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_pca_high_dimensional_small_sample() {
        let rows = 30usize;
        let cols = 5395usize;
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                let v = ((i as f64 + 1.0) * (j as f64 + 1.0)).sin()
                    + ((i * 17 + j * 7) as f64).cos() * 0.1;
                data.push(v);
            }
        }

        let data = Array2::from_shape_vec((rows, cols), data).unwrap();
        let (pca, transformed) = Pca::fit_transform(&data, 2).unwrap();

        assert_eq!(pca.n_components(), 2);
        assert_eq!(pca.components.ncols(), cols);
        assert_eq!(transformed.nrows(), rows);
        assert_eq!(transformed.ncols(), 2);
        assert!(pca.explained_variance_ratio.iter().all(|v| v.is_finite()));
    }

    #[test]
    /// Runs regression test for transform one matches batch.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pca.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_transform_one_matches_batch() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 5.0, 3.0, 5.0, 8.0, 4.0, 6.0, 10.0],
        )
        .unwrap();
        let pca = Pca::fit(&data, 2).unwrap();
        let sample = vec![2.5, 4.0, 6.5];
        let one = pca.transform_one(&sample);
        let batch = pca
            .transform(&Array2::from_shape_vec((1, 3), sample).unwrap())
            .row(0)
            .to_vec();
        for (a, b) in one.iter().zip(batch.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
