//! Train/test splitting — random, stratified, scaffold-based.

use crate::ml::dataset::QsarDataset;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

/// Split result.
pub struct TrainTestSplit {
    pub train: QsarDataset,
    pub test: QsarDataset,
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
}

/// Random train/test split.
pub fn random_split(data: &QsarDataset, test_fraction: f64, seed: u64) -> TrainTestSplit {
    let n = data.n_samples();
    let n_test = (n as f64 * test_fraction).round() as usize;
    let n_test = n_test.max(1).min(n - 1);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let test_indices = indices[..n_test].to_vec();
    let train_indices = indices[n_test..].to_vec();

    TrainTestSplit {
        train: data.subset(&train_indices),
        test: data.subset(&test_indices),
        train_indices,
        test_indices,
    }
}

/// Stratified split for classification (preserves class distribution).
///
/// Targets are rounded to integers for binning.
pub fn stratified_split(data: &QsarDataset, test_fraction: f64, seed: u64) -> TrainTestSplit {
    let mut rng = StdRng::seed_from_u64(seed);

    // Bin by rounded target
    let mut bins: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &t) in data.targets.iter().enumerate() {
        let key = t.round() as i64;
        bins.entry(key).or_default().push(i);
    }

    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    for (_, indices) in &mut bins {
        indices.shuffle(&mut rng);
        let n_test = (indices.len() as f64 * test_fraction).round() as usize;
        let n_test = n_test.max(1).min(indices.len().saturating_sub(1));
        test_indices.extend_from_slice(&indices[..n_test]);
        train_indices.extend_from_slice(&indices[n_test..]);
    }

    train_indices.sort_unstable();
    test_indices.sort_unstable();

    TrainTestSplit {
        train: data.subset(&train_indices),
        test: data.subset(&test_indices),
        train_indices,
        test_indices,
    }
}

/// Scaffold-based split — groups molecules by Murcko-like scaffold hash.
///
/// Uses a simple scaffold approximation: ring system atoms from the SMILES.
/// Pass `scaffold_keys` (one per molecule), molecules with the same key share a scaffold.
pub fn scaffold_split(
    data: &QsarDataset,
    scaffold_keys: &[String],
    test_fraction: f64,
    seed: u64,
) -> TrainTestSplit {
    assert_eq!(scaffold_keys.len(), data.n_samples());

    let mut rng = StdRng::seed_from_u64(seed);

    // Group indices by scaffold
    let mut scaffold_groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, key) in scaffold_keys.iter().enumerate() {
        scaffold_groups.entry(key.as_str()).or_default().push(i);
    }

    // Sort scaffolds by size (largest first) for more balanced splits
    let mut scaffolds: Vec<(&str, Vec<usize>)> = scaffold_groups.into_iter().collect();
    scaffolds.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    // Shuffle within each scaffold for randomness
    for (_, indices) in &mut scaffolds {
        indices.shuffle(&mut rng);
    }

    let n_total = data.n_samples();
    let n_test_target = (n_total as f64 * test_fraction).round() as usize;

    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    // Assign entire scaffolds to test until we reach the target size
    for (_, indices) in scaffolds {
        if test_indices.len() < n_test_target {
            test_indices.extend(indices);
        } else {
            train_indices.extend(indices);
        }
    }

    // Ensure at least 1 in each set
    if train_indices.is_empty() && !test_indices.is_empty() {
        train_indices.push(test_indices.pop().unwrap());
    }
    if test_indices.is_empty() && !train_indices.is_empty() {
        test_indices.push(train_indices.pop().unwrap());
    }

    train_indices.sort_unstable();
    test_indices.sort_unstable();

    TrainTestSplit {
        train: data.subset(&train_indices),
        test: data.subset(&test_indices),
        train_indices,
        test_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn make_dataset(n: usize) -> QsarDataset {
        let features = Array2::from_shape_fn((n, 3), |(i, j)| (i * 3 + j) as f64);
        let targets = Array1::from_vec((0..n).map(|i| i as f64).collect());
        let names = vec!["a".into(), "b".into(), "c".into()];
        QsarDataset::new(features, targets, names)
    }

    #[test]
    fn test_random_split_sizes() {
        let ds = make_dataset(100);
        let split = random_split(&ds, 0.2, 42);
        assert_eq!(split.train.n_samples() + split.test.n_samples(), 100);
        assert_eq!(split.test.n_samples(), 20);
    }

    #[test]
    fn test_random_split_no_overlap() {
        let ds = make_dataset(50);
        let split = random_split(&ds, 0.3, 42);
        for &ti in &split.test_indices {
            assert!(!split.train_indices.contains(&ti));
        }
    }

    #[test]
    fn test_stratified_split() {
        // Binary classification: 60 class-0, 40 class-1
        let n = 100;
        let features = Array2::from_shape_fn((n, 2), |(i, j)| (i + j) as f64);
        let targets = Array1::from_vec(
            (0..n).map(|i| if i < 60 { 0.0 } else { 1.0 }).collect()
        );
        let names = vec!["x".into(), "y".into()];
        let ds = QsarDataset::new(features, targets, names);

        let split = stratified_split(&ds, 0.2, 42);
        assert_eq!(split.train.n_samples() + split.test.n_samples(), 100);

        // Both classes should be represented in test set
        let test_class0 = split.test.targets.iter().filter(|&&t| t < 0.5).count();
        let test_class1 = split.test.targets.iter().filter(|&&t| t > 0.5).count();
        assert!(test_class0 > 0);
        assert!(test_class1 > 0);
    }

    #[test]
    fn test_scaffold_split() {
        let ds = make_dataset(10);
        let scaffolds: Vec<String> = (0..10).map(|i| {
            if i < 3 { "ring_A".into() }
            else if i < 7 { "ring_B".into() }
            else { "ring_C".into() }
        }).collect();

        let split = scaffold_split(&ds, &scaffolds, 0.3, 42);
        assert_eq!(split.train.n_samples() + split.test.n_samples(), 10);

        // Scaffolds should not be split across train/test
        let train_scaffolds: Vec<&str> = split.train_indices.iter()
            .map(|&i| scaffolds[i].as_str()).collect();
        let test_scaffolds: Vec<&str> = split.test_indices.iter()
            .map(|&i| scaffolds[i].as_str()).collect();
        for ts in &test_scaffolds {
            assert!(!train_scaffolds.contains(ts),
                "Scaffold {ts} found in both train and test");
        }
    }
}
