//! QsarPipeline — serializable end-to-end QSAR pipeline.
//!
//! Combines: scaler → optional PCA → model → applicability domain → prediction.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::ml::ad::{ApplicabilityDomain, KnnDistanceAD};
use crate::ml::cv;
use crate::ml::dataset::QsarDataset;
use crate::ml::elastic_net::ElasticNet;
use crate::ml::knn::{DistanceMetric, KnnClassifier, KnnRegressor};
use crate::ml::metrics;
use crate::ml::pca::Pca;
use crate::ml::pls::PlsRegression;
use crate::ml::random_forest::{RandomForestClassifier, RandomForestRegressor};
use crate::ml::scaling::{MinMaxScaler, StandardScaler};

/// Scaler type for pipeline serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalerType {
    Standard(StandardScaler),
    MinMax(MinMaxScaler),
    None,
}

/// Model type for pipeline serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    KnnRegressor(KnnRegressor),
    KnnClassifier(KnnClassifier),
    RandomForestRegressor(RandomForestRegressor),
    RandomForestClassifier(RandomForestClassifier),
    PlsRegression(PlsRegression),
    ElasticNet(ElasticNet),
}

/// Task type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TaskType {
    Regression,
    Classification,
}

/// Complete QSAR pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QsarPipeline {
    pub name: String,
    pub task: TaskType,
    pub scaler: ScalerType,
    pub pca: Option<Pca>,
    pub model: ModelType,
    pub ad: Option<KnnDistanceAD>,
    pub feature_names: Vec<String>,
    pub train_r2: f64,
    pub cv_q2: f64,
}

impl QsarPipeline {
    /// Predict for a single sample (raw features, before scaling).
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        let scaled = self.preprocess_one(sample);
        self.model_predict_one(&scaled)
    }

    /// Predict for multiple samples.
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        let processed = self.preprocess(features);
        self.model_predict(&processed)
    }

    /// Check if a sample is within the applicability domain.
    pub fn is_in_domain(&self, sample: &[f64]) -> bool {
        match &self.ad {
            Some(ad) => {
                let scaled = self.preprocess_one(sample);
                ad.is_in_domain(&scaled)
            }
            None => true,
        }
    }

    /// Preprocess a single sample through scaler and optional PCA.
    fn preprocess_one(&self, sample: &[f64]) -> Vec<f64> {
        let n = sample.len();
        let row = Array2::from_shape_vec((1, n), sample.to_vec()).unwrap();
        let processed = self.preprocess(&row);
        processed.row(0).to_vec()
    }

    /// Preprocess a matrix through scaler and optional PCA.
    fn preprocess(&self, features: &Array2<f64>) -> Array2<f64> {
        let scaled = match &self.scaler {
            ScalerType::Standard(s) => s.transform(features),
            ScalerType::MinMax(s) => s.transform(features),
            ScalerType::None => features.clone(),
        };

        match &self.pca {
            Some(pca) => pca.transform(&scaled),
            None => scaled,
        }
    }

    fn model_predict_one(&self, sample: &[f64]) -> f64 {
        match &self.model {
            ModelType::KnnRegressor(m) => m.predict_one(sample),
            ModelType::KnnClassifier(m) => m.predict_one(sample),
            ModelType::RandomForestRegressor(m) => m.predict_one(sample),
            ModelType::RandomForestClassifier(m) => m.predict_one(sample),
            ModelType::PlsRegression(m) => m.predict_one(sample),
            ModelType::ElasticNet(m) => m.predict_one(sample),
        }
    }

    fn model_predict(&self, features: &Array2<f64>) -> Array1<f64> {
        match &self.model {
            ModelType::KnnRegressor(m) => m.predict(features),
            ModelType::KnnClassifier(m) => m.predict(features),
            ModelType::RandomForestRegressor(m) => m.predict(features),
            ModelType::RandomForestClassifier(m) => m.predict(features),
            ModelType::PlsRegression(m) => m.predict(features),
            ModelType::ElasticNet(m) => m.predict(features),
        }
    }

    /// Serialize pipeline to bytes (bincode).
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Failed to serialize pipeline")
    }

    /// Deserialize pipeline from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        bincode::deserialize(bytes).expect("Failed to deserialize pipeline")
    }
}

/// AutoQsar — train multiple models, compare via CV, report best.
pub struct AutoQsar;

impl AutoQsar {
    /// Train all regression models on the dataset and return the best pipeline by CV Q².
    pub fn regression(data: &QsarDataset, n_cv_folds: usize, seed: u64) -> QsarPipeline {
        let scaler = StandardScaler::fit(&data.features);
        let scaled = scaler.transform(&data.features);

        let mut best_pipeline: Option<QsarPipeline> = None;
        let mut best_q2 = f64::NEG_INFINITY;

        // --- k-NN with various k ---
        for k in [3, 5, 7] {
            let q2 = cv_regression_q2(
                &scaled,
                &data.targets,
                n_cv_folds,
                seed,
                |train_x, train_y| {
                    let model = KnnRegressor::fit(
                        train_x.clone(),
                        train_y.clone(),
                        k,
                        DistanceMetric::Euclidean,
                    );
                    ModelType::KnnRegressor(model)
                },
            );

            if q2 > best_q2 {
                best_q2 = q2;
                let model = KnnRegressor::fit(
                    scaled.clone(),
                    data.targets.clone(),
                    k,
                    DistanceMetric::Euclidean,
                );
                let train_preds = model.predict(&scaled);
                let train_r2 = metrics::r_squared(
                    data.targets.as_slice().unwrap(),
                    train_preds.as_slice().unwrap(),
                );
                best_pipeline = Some(QsarPipeline {
                    name: format!("kNN-{k}"),
                    task: TaskType::Regression,
                    scaler: ScalerType::Standard(scaler.clone()),
                    pca: None,
                    model: ModelType::KnnRegressor(model),
                    ad: None,
                    feature_names: data.feature_names.clone(),
                    train_r2,
                    cv_q2: q2,
                });
            }
        }

        // --- Random Forest ---
        for n_trees in [50, 100] {
            let q2 = cv_regression_q2(
                &scaled,
                &data.targets,
                n_cv_folds,
                seed,
                |train_x, train_y| {
                    let model =
                        RandomForestRegressor::fit(train_x, train_y, n_trees, 10, 2, 0.33, seed);
                    ModelType::RandomForestRegressor(model)
                },
            );

            if q2 > best_q2 {
                best_q2 = q2;
                let model =
                    RandomForestRegressor::fit(&scaled, &data.targets, n_trees, 10, 2, 0.33, seed);
                let train_preds = model.predict(&scaled);
                let train_r2 = metrics::r_squared(
                    data.targets.as_slice().unwrap(),
                    train_preds.as_slice().unwrap(),
                );
                best_pipeline = Some(QsarPipeline {
                    name: format!("RF-{n_trees}"),
                    task: TaskType::Regression,
                    scaler: ScalerType::Standard(scaler.clone()),
                    pca: None,
                    model: ModelType::RandomForestRegressor(model),
                    ad: None,
                    feature_names: data.feature_names.clone(),
                    train_r2,
                    cv_q2: q2,
                });
            }
        }

        // --- PLS ---
        for nc in [2, 3, 5] {
            if nc >= data.n_features() {
                continue;
            }
            let q2 = cv_regression_q2(
                &scaled,
                &data.targets,
                n_cv_folds,
                seed,
                |train_x, train_y| {
                    let model = PlsRegression::fit(train_x, train_y, nc);
                    ModelType::PlsRegression(model)
                },
            );

            if q2 > best_q2 {
                best_q2 = q2;
                let model = PlsRegression::fit(&scaled, &data.targets, nc);
                let train_preds = model.predict(&scaled);
                let train_r2 = metrics::r_squared(
                    data.targets.as_slice().unwrap(),
                    train_preds.as_slice().unwrap(),
                );
                best_pipeline = Some(QsarPipeline {
                    name: format!("PLS-{nc}"),
                    task: TaskType::Regression,
                    scaler: ScalerType::Standard(scaler.clone()),
                    pca: None,
                    model: ModelType::PlsRegression(model),
                    ad: None,
                    feature_names: data.feature_names.clone(),
                    train_r2,
                    cv_q2: q2,
                });
            }
        }

        // --- Elastic Net ---
        for alpha in [0.01, 0.1, 1.0] {
            let q2 = cv_regression_q2(
                &scaled,
                &data.targets,
                n_cv_folds,
                seed,
                |train_x, train_y| {
                    let model = ElasticNet::fit(train_x, train_y, alpha, 0.5, 1000, 1e-6);
                    ModelType::ElasticNet(model)
                },
            );

            if q2 > best_q2 {
                best_q2 = q2;
                let model = ElasticNet::fit(&scaled, &data.targets, alpha, 0.5, 1000, 1e-6);
                let train_preds = model.predict(&scaled);
                let train_r2 = metrics::r_squared(
                    data.targets.as_slice().unwrap(),
                    train_preds.as_slice().unwrap(),
                );
                best_pipeline = Some(QsarPipeline {
                    name: format!("EN-a{alpha}"),
                    task: TaskType::Regression,
                    scaler: ScalerType::Standard(scaler.clone()),
                    pca: None,
                    model: ModelType::ElasticNet(model),
                    ad: None,
                    feature_names: data.feature_names.clone(),
                    train_r2,
                    cv_q2: q2,
                });
            }
        }

        // Add AD to best pipeline
        let mut pipeline = best_pipeline.expect("No model trained");
        let ad = KnnDistanceAD::fit(&scaled, 5, 2.0);
        pipeline.ad = Some(ad);
        pipeline
    }
}

/// Helper: compute cross-validated Q² for a regression model builder.
fn cv_regression_q2<F>(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    n_folds: usize,
    seed: u64,
    build_model: F,
) -> f64
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> ModelType,
{
    let n = features.nrows();
    let splits = cv::k_fold(n, n_folds, seed);
    let mut all_preds = vec![0.0f64; n];

    for (train_idx, test_idx) in &splits.folds {
        let train_x = select_rows(features, train_idx);
        let train_y = select_elements(targets, train_idx);
        let test_x = select_rows(features, test_idx);

        let model = build_model(&train_x, &train_y);
        let preds = match &model {
            ModelType::KnnRegressor(m) => m.predict(&test_x),
            ModelType::RandomForestRegressor(m) => m.predict(&test_x),
            ModelType::PlsRegression(m) => m.predict(&test_x),
            ModelType::ElasticNet(m) => m.predict(&test_x),
            _ => Array1::zeros(test_idx.len()),
        };

        for (i, &idx) in test_idx.iter().enumerate() {
            all_preds[idx] = preds[i];
        }
    }

    metrics::r_squared(targets.as_slice().unwrap(), &all_preds)
}

fn select_rows(data: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let ncols = data.ncols();
    let mut result = Array2::<f64>::zeros((indices.len(), ncols));
    for (new_i, &old_i) in indices.iter().enumerate() {
        result.row_mut(new_i).assign(&data.row(old_i));
    }
    result
}

fn select_elements(data: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
    Array1::from_vec(indices.iter().map(|&i| data[i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn make_regression_dataset() -> QsarDataset {
        // y ≈ 2*x1 + noise
        let n = 30;
        let mut feat = Vec::new();
        let mut tgt = Vec::new();
        for i in 0..n {
            let x = i as f64;
            feat.push(x);
            feat.push(x * 0.5 + 1.0); // correlated feature
            tgt.push(2.0 * x + 1.0);
        }
        let features = Array2::from_shape_vec((n, 2), feat).unwrap();
        let targets = Array1::from_vec(tgt);
        QsarDataset::new(features, targets, vec!["x1".into(), "x2".into()])
    }

    #[test]
    fn test_pipeline_predict() {
        let ds = make_regression_dataset();
        let scaler = StandardScaler::fit(&ds.features);
        let scaled = scaler.transform(&ds.features);
        let model = KnnRegressor::fit(
            scaled.clone(),
            ds.targets.clone(),
            3,
            DistanceMetric::Euclidean,
        );

        let pipeline = QsarPipeline {
            name: "test".into(),
            task: TaskType::Regression,
            scaler: ScalerType::Standard(scaler),
            pca: None,
            model: ModelType::KnnRegressor(model),
            ad: None,
            feature_names: ds.feature_names.clone(),
            train_r2: 0.99,
            cv_q2: 0.95,
        };

        let pred = pipeline.predict_one(&[5.0, 3.5]);
        assert!(pred.is_finite());
    }

    #[test]
    fn test_pipeline_serialization() {
        let ds = make_regression_dataset();
        let scaler = StandardScaler::fit(&ds.features);
        let scaled = scaler.transform(&ds.features);
        let model = KnnRegressor::fit(scaled, ds.targets.clone(), 3, DistanceMetric::Euclidean);

        let pipeline = QsarPipeline {
            name: "test-serial".into(),
            task: TaskType::Regression,
            scaler: ScalerType::Standard(scaler),
            pca: None,
            model: ModelType::KnnRegressor(model),
            ad: None,
            feature_names: ds.feature_names.clone(),
            train_r2: 0.99,
            cv_q2: 0.95,
        };

        let bytes = pipeline.to_bytes();
        let restored = QsarPipeline::from_bytes(&bytes);
        assert_eq!(restored.name, "test-serial");
        assert_eq!(restored.task, TaskType::Regression);
    }

    #[test]
    fn test_auto_qsar_regression() {
        let ds = make_regression_dataset();
        let pipeline = AutoQsar::regression(&ds, 3, 42);

        assert_eq!(pipeline.task, TaskType::Regression);
        assert!(!pipeline.name.is_empty());
        assert!(pipeline.ad.is_some());

        // Should have reasonable performance on linear data
        let pred = pipeline.predict(&ds.features);
        let r2 = metrics::r_squared(ds.targets.as_slice().unwrap(), pred.as_slice().unwrap());
        assert!(r2 > 0.5, "AutoQsar R² = {r2} on linear data");
    }
}
