//! Feature scaling — StandardScaler (Z-score) and MinMaxScaler.
//!
//! Both store fitted parameters for consistent inference-time transforms.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Z-score normalisation: x' = (x − μ) / σ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

impl StandardScaler {
    /// Fit on training data: compute per-feature mean and std.
    pub fn fit(data: &Array2<f64>) -> Self {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let n = data.nrows() as f64;
        let mut std = Array1::<f64>::zeros(data.ncols());
        for i in 0..data.nrows() {
            let diff = &data.row(i).to_owned() - &mean;
            std = std + &(&diff * &diff);
        }
        std.mapv_inplace(|v| (v / n).sqrt().max(1e-12));
        Self { mean, std }
    }

    /// Transform data using fitted params.
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut result = data.clone();
        for mut row in result.rows_mut() {
            let scaled = (&row.to_owned() - &self.mean) / &self.std;
            row.assign(&scaled);
        }
        result
    }

    /// Fit and transform in one step.
    pub fn fit_transform(data: &Array2<f64>) -> (Self, Array2<f64>) {
        let scaler = Self::fit(data);
        let transformed = scaler.transform(data);
        (scaler, transformed)
    }

    /// Inverse transform: x = x' * σ + μ
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut result = data.clone();
        for mut row in result.rows_mut() {
            let orig = &row.to_owned() * &self.std + &self.mean;
            row.assign(&orig);
        }
        result
    }
}

/// Min-max scaling: x' = (x − min) / (max − min) → [0, 1]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxScaler {
    pub min: Array1<f64>,
    pub range: Array1<f64>, // max - min, clamped to avoid div-by-zero
}

impl MinMaxScaler {
    /// Fit on training data.
    pub fn fit(data: &Array2<f64>) -> Self {
        let n_features = data.ncols();
        let mut min = Array1::from_elem(n_features, f64::INFINITY);
        let mut max = Array1::from_elem(n_features, f64::NEG_INFINITY);

        for row in data.rows() {
            for (j, &v) in row.iter().enumerate() {
                if v < min[j] {
                    min[j] = v;
                }
                if v > max[j] {
                    max[j] = v;
                }
            }
        }

        let mut range = &max - &min;
        range.mapv_inplace(|v| if v.abs() < 1e-12 { 1.0 } else { v });

        Self { min, range }
    }

    /// Transform data to [0, 1].
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut result = data.clone();
        for mut row in result.rows_mut() {
            let scaled = (&row.to_owned() - &self.min) / &self.range;
            row.assign(&scaled);
        }
        result
    }

    /// Fit and transform in one step.
    pub fn fit_transform(data: &Array2<f64>) -> (Self, Array2<f64>) {
        let scaler = Self::fit(data);
        let transformed = scaler.transform(data);
        (scaler, transformed)
    }

    /// Inverse transform: x = x' * range + min
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut result = data.clone();
        for mut row in result.rows_mut() {
            let orig = &row.to_owned() * &self.range + &self.min;
            row.assign(&orig);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_standard_scaler_zero_mean() {
        let data = Array2::from_shape_vec((4, 2), vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
            4.0, 40.0,
        ]).unwrap();
        let (scaler, transformed) = StandardScaler::fit_transform(&data);
        // Mean of each column should be ~0
        let col_means = transformed.mean_axis(Axis(0)).unwrap();
        assert!(col_means[0].abs() < 1e-10);
        assert!(col_means[1].abs() < 1e-10);
        // Inverse should recover original
        let recovered = scaler.inverse_transform(&transformed);
        for i in 0..4 {
            for j in 0..2 {
                assert!((recovered[[i, j]] - data[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_standard_scaler_unit_variance() {
        let data = Array2::from_shape_vec((4, 2), vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
            4.0, 40.0,
        ]).unwrap();
        let (_, transformed) = StandardScaler::fit_transform(&data);
        // Variance of each column should be ~1
        for j in 0..2 {
            let col = transformed.column(j);
            let mean = col.mean().unwrap();
            let var: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;
            assert!((var - 1.0).abs() < 1e-10, "Variance of col {j} = {var}");
        }
    }

    #[test]
    fn test_minmax_scaler_range() {
        let data = Array2::from_shape_vec((3, 2), vec![
            1.0, -5.0,
            5.0,  0.0,
            3.0,  5.0,
        ]).unwrap();
        let (scaler, transformed) = MinMaxScaler::fit_transform(&data);
        // Min should be 0, max should be 1
        for j in 0..2 {
            let col = transformed.column(j);
            let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!((min - 0.0).abs() < 1e-10);
            assert!((max - 1.0).abs() < 1e-10);
        }
        // Inverse
        let recovered = scaler.inverse_transform(&transformed);
        for i in 0..3 {
            for j in 0..2 {
                assert!((recovered[[i, j]] - data[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_constant_feature() {
        // Constant column should not produce NaN
        let data = Array2::from_shape_vec((3, 2), vec![
            5.0, 1.0,
            5.0, 2.0,
            5.0, 3.0,
        ]).unwrap();
        let (_, t1) = StandardScaler::fit_transform(&data);
        assert!(t1.iter().all(|v| v.is_finite()));
        let (_, t2) = MinMaxScaler::fit_transform(&data);
        assert!(t2.iter().all(|v| v.is_finite()));
    }
}
