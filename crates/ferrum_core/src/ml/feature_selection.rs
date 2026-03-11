//! Feature selection — variance threshold and correlation filter.

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

    // Compute means and stds
    let mean = data.mean_axis(Axis(0)).unwrap();
    let mut std_devs = vec![0.0f64; n_feat];
    for j in 0..n_feat {
        let var: f64 = data
            .column(j)
            .iter()
            .map(|&x| (x - mean[j]).powi(2))
            .sum::<f64>()
            / n;
        std_devs[j] = var.sqrt().max(1e-12);
    }

    // Compute correlation matrix (upper triangle)
    let mut corr = vec![vec![0.0f64; n_feat]; n_feat];
    for i in 0..n_feat {
        corr[i][i] = 1.0;
        for j in (i + 1)..n_feat {
            let mut cov = 0.0;
            for k in 0..data.nrows() {
                cov += (data[[k, i]] - mean[i]) * (data[[k, j]] - mean[j]);
            }
            cov /= n;
            let r = cov / (std_devs[i] * std_devs[j]);
            corr[i][j] = r;
            corr[j][i] = r;
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
            for j in (i + 1)..n_feat {
                if removed[j] {
                    continue;
                }
                let r_abs = corr[i][j].abs();
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
                    .map(|k| corr[i][k].abs())
                    .sum::<f64>();
                let mean_corr_j: f64 = (0..n_feat)
                    .filter(|&k| !removed[k] && k != j)
                    .map(|k| corr[j][k].abs())
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
    fn test_no_removal_below_threshold() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 5.0, 3.0, 20.0, 4.0, 1.0]).unwrap();
        let keep = correlation_filter(&data, 0.95);
        assert_eq!(keep.len(), 2);
    }
}
