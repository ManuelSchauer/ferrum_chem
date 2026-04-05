//! QsarPipeline — serializable end-to-end QSAR pipeline.
//!
//! Combines: scaler → optional PCA → model → applicability domain → prediction.

//! ## File guide: Pipeline
//!
//! This file belongs to FerrumChem's **machine-learning pipeline layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `enum ScalerType`
//! - `enum ModelType`
//! - `enum TaskType`
//! - `struct QsarPipeline`
//! - `fn predict_one`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - Training flow is strictly ordered: scaling -> optional PCA -> model fit -> optional AD fit.
//! - Serialized pipeline structs assume version-compatible ndarray/serde layouts for reproducible reloads.
//! - Task-type branching (regression/classification) must remain consistent across CV, metrics, and prediction APIs.
//! ### Chemistry/ML references
//! - Kuhn & Johnson. Applied predictive modeling workflows (preprocess-fit-validate).
//! - Breiman (2001), Zou & Hastie (2005), and KNN/PLS literature reflected by selectable model backends.
//! - OECD QSAR validation principles motivating explicit applicability-domain handling.

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::mpsc;

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
        let scaled = match &self.scaler {
            ScalerType::Standard(s) => s.transform_one(sample),
            ScalerType::MinMax(s) => s.transform_one(sample),
            ScalerType::None => sample.to_vec(),
        };

        match &self.pca {
            Some(pca) => pca.transform_one(&scaled),
            None => scaled,
        }
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

    /// Implements predict one.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pipeline.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
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

    /// Implements predict.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pipeline.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoQsarParallelMode {
    Candidate,
    Fold,
}

impl AutoQsarParallelMode {
    fn as_str(self) -> &'static str {
        match self {
            AutoQsarParallelMode::Candidate => "candidate",
            AutoQsarParallelMode::Fold => "fold",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct AutoQsarRuntimeConfig {
    parallel_mode: AutoQsarParallelMode,
    max_threads: Option<usize>,
    prune_keep: Option<usize>,
    prune_folds: usize,
}

fn resolve_auto_qsar_runtime_config(
    parallel_mode_raw: Option<&str>,
    max_threads_raw: Option<&str>,
    prune_keep_raw: Option<&str>,
    prune_folds_raw: Option<&str>,
    total_candidates: usize,
    n_cv_folds: usize,
) -> AutoQsarRuntimeConfig {
    let parallel_mode = parallel_mode_raw
        .map(parse_parallel_mode)
        .unwrap_or(AutoQsarParallelMode::Candidate);
    let max_threads = max_threads_raw
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0);
    let prune_keep = prune_keep_raw
        .and_then(|raw| raw.parse::<usize>().ok())
        .map(|keep| keep.min(total_candidates))
        .filter(|keep| *keep > 0 && *keep < total_candidates);
    let prune_folds = prune_folds_raw
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(2, n_cv_folds.max(2));
    AutoQsarRuntimeConfig {
        parallel_mode,
        max_threads,
        prune_keep,
        prune_folds,
    }
}

#[cfg(test)]
fn make_runtime_config_for_tests(
    mode: AutoQsarParallelMode,
    max_threads: Option<usize>,
    prune_keep: Option<usize>,
    prune_folds: usize,
) -> AutoQsarRuntimeConfig {
    AutoQsarRuntimeConfig {
        parallel_mode: mode,
        max_threads,
        prune_keep,
        prune_folds,
    }
}

fn parse_parallel_mode(raw: &str) -> AutoQsarParallelMode {
    match raw.to_ascii_lowercase().as_str() {
        "fold" => AutoQsarParallelMode::Fold,
        _ => AutoQsarParallelMode::Candidate,
    }
}

fn parse_runtime_config_from_env(
    total_candidates: usize,
    n_cv_folds: usize,
) -> AutoQsarRuntimeConfig {
    resolve_auto_qsar_runtime_config(
        env::var("FERRUM_AUTOQSAR_PARALLEL_MODE").ok().as_deref(),
        env::var("FERRUM_AUTOQSAR_MAX_THREADS").ok().as_deref(),
        env::var("FERRUM_AUTOQSAR_PRUNE_KEEP").ok().as_deref(),
        env::var("FERRUM_AUTOQSAR_PRUNE_FOLDS").ok().as_deref(),
        total_candidates,
        n_cv_folds,
    )
}

fn build_optional_thread_pool(max_threads: Option<usize>) -> Option<rayon::ThreadPool> {
    max_threads.map(|threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("failed to build AutoQsar thread pool")
    })
}

fn install_in_pool<T, F>(pool: Option<&rayon::ThreadPool>, operation: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    match pool {
        Some(pool) => pool.install(operation),
        None => operation(),
    }
}

#[derive(Debug, Clone, Copy)]
enum RegressionCandidate {
    Knn {
        k: usize,
        label: &'static str,
    },
    RandomForest {
        n_trees: usize,
        label: &'static str,
    },
    Pls {
        n_components: usize,
        label: &'static str,
    },
    ElasticNet {
        alpha: f64,
        label: &'static str,
    },
}

impl RegressionCandidate {
    fn label(self) -> &'static str {
        match self {
            Self::Knn { label, .. }
            | Self::RandomForest { label, .. }
            | Self::Pls { label, .. }
            | Self::ElasticNet { label, .. } => label,
        }
    }

    fn pipeline_name(self) -> String {
        match self {
            Self::Knn { k, .. } => format!("kNN-{k}"),
            Self::RandomForest { n_trees, .. } => format!("RF-{n_trees}"),
            Self::Pls { n_components, .. } => format!("PLS-{n_components}"),
            Self::ElasticNet { alpha, .. } => format!("EN-a{alpha}"),
        }
    }

    fn evaluate_q2(
        self,
        scaled: &Array2<f64>,
        targets: &Array1<f64>,
        n_cv_folds: usize,
        seed: u64,
        parallel_folds: bool,
    ) -> f64 {
        match self {
            Self::Knn { k, .. } => cv_regression_q2_with_mode(
                scaled,
                targets,
                n_cv_folds,
                seed,
                parallel_folds,
                |train_x, train_y| {
                    let model = KnnRegressor::fit(
                        train_x.clone(),
                        train_y.clone(),
                        k,
                        DistanceMetric::Euclidean,
                    );
                    ModelType::KnnRegressor(model)
                },
            ),
            Self::RandomForest { n_trees, .. } => cv_regression_q2_with_mode(
                scaled,
                targets,
                n_cv_folds,
                seed,
                parallel_folds,
                |train_x, train_y| {
                    let model =
                        RandomForestRegressor::fit(train_x, train_y, n_trees, 10, 2, 0.33, seed);
                    ModelType::RandomForestRegressor(model)
                },
            ),
            Self::Pls { n_components, .. } => cv_regression_q2_with_mode(
                scaled,
                targets,
                n_cv_folds,
                seed,
                parallel_folds,
                |train_x, train_y| {
                    let model = PlsRegression::fit(train_x, train_y, n_components);
                    ModelType::PlsRegression(model)
                },
            ),
            Self::ElasticNet { alpha, .. } => cv_regression_q2_with_mode(
                scaled,
                targets,
                n_cv_folds,
                seed,
                parallel_folds,
                |train_x, train_y| {
                    let model = ElasticNet::fit(train_x, train_y, alpha, 0.5, 1000, 1e-6);
                    ModelType::ElasticNet(model)
                },
            ),
        }
    }

    fn fit_model(self, scaled: &Array2<f64>, targets: &Array1<f64>, seed: u64) -> ModelType {
        match self {
            Self::Knn { k, .. } => ModelType::KnnRegressor(KnnRegressor::fit(
                scaled.clone(),
                targets.clone(),
                k,
                DistanceMetric::Euclidean,
            )),
            Self::RandomForest { n_trees, .. } => ModelType::RandomForestRegressor(
                RandomForestRegressor::fit(scaled, targets, n_trees, 10, 2, 0.33, seed),
            ),
            Self::Pls { n_components, .. } => {
                ModelType::PlsRegression(PlsRegression::fit(scaled, targets, n_components))
            }
            Self::ElasticNet { alpha, .. } => {
                ModelType::ElasticNet(ElasticNet::fit(scaled, targets, alpha, 0.5, 1000, 1e-6))
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CandidateEvaluation {
    order: usize,
    candidate: RegressionCandidate,
    q2: f64,
}

#[derive(Debug, Clone, Copy)]
enum CandidateProgressEvent {
    Started { candidate: RegressionCandidate },
    Done(CandidateEvaluation),
}

fn build_regression_candidates(n_features: usize) -> Vec<RegressionCandidate> {
    let mut candidates = vec![
        RegressionCandidate::Knn {
            k: 3,
            label: "kNN-3",
        },
        RegressionCandidate::Knn {
            k: 5,
            label: "kNN-5",
        },
        RegressionCandidate::Knn {
            k: 7,
            label: "kNN-7",
        },
        RegressionCandidate::RandomForest {
            n_trees: 50,
            label: "RF-50",
        },
        RegressionCandidate::RandomForest {
            n_trees: 100,
            label: "RF-100",
        },
    ];

    for nc in [2usize, 3usize, 5usize] {
        if nc < n_features {
            candidates.push(RegressionCandidate::Pls {
                n_components: nc,
                label: match nc {
                    2 => "PLS-2",
                    3 => "PLS-3",
                    _ => "PLS-5",
                },
            });
        }
    }

    candidates.extend([
        RegressionCandidate::ElasticNet {
            alpha: 0.01,
            label: "EN-0.01",
        },
        RegressionCandidate::ElasticNet {
            alpha: 0.1,
            label: "EN-0.1",
        },
        RegressionCandidate::ElasticNet {
            alpha: 1.0,
            label: "EN-1.0",
        },
    ]);

    candidates
}

fn top_k_candidates(
    mut evaluations: Vec<CandidateEvaluation>,
    keep: usize,
) -> Vec<CandidateEvaluation> {
    evaluations.sort_by(|a, b| match b.q2.total_cmp(&a.q2) {
        std::cmp::Ordering::Equal => a.order.cmp(&b.order),
        ord => ord,
    });
    evaluations.truncate(keep.min(evaluations.len()));
    evaluations
}

/// Progress stage emitted by AutoQsar model selection.
#[derive(Debug, Clone, Copy)]
pub enum AutoQsarProgressStage {
    Scaling,
    CandidateStart {
        label: &'static str,
    },
    CandidateDone {
        label: &'static str,
        q2: f64,
        best_q2: f64,
    },
    AdFit,
    Complete,
}

/// Progress payload emitted by AutoQsar model selection.
#[derive(Debug, Clone, Copy)]
pub struct AutoQsarProgress {
    pub stage: AutoQsarProgressStage,
    pub completed_candidates: usize,
    pub total_candidates: usize,
}

impl AutoQsarProgress {
    /// Normalized progress in [0, 1], reserving tail for AD fitting/finalization.
    pub fn fraction(&self) -> f32 {
        let total = self.total_candidates.max(1) as f32;
        let completed = (self.completed_candidates as f32 / total).clamp(0.0, 1.0);
        match self.stage {
            AutoQsarProgressStage::Scaling => 0.0,
            AutoQsarProgressStage::CandidateStart { .. }
            | AutoQsarProgressStage::CandidateDone { .. } => 0.02 + 0.90 * completed,
            AutoQsarProgressStage::AdFit => 0.95,
            AutoQsarProgressStage::Complete => 1.0,
        }
    }
}

impl AutoQsar {
    /// Train all regression models on the dataset and return the best pipeline by CV Q².
    pub fn regression(data: &QsarDataset, n_cv_folds: usize, seed: u64) -> QsarPipeline {
        let mut noop = |_progress: AutoQsarProgress| {};
        Self::regression_with_progress(data, n_cv_folds, seed, &mut noop)
    }

    /// Train all regression models with progress callbacks.
    pub fn regression_with_progress(
        data: &QsarDataset,
        n_cv_folds: usize,
        seed: u64,
        on_progress: &mut dyn FnMut(AutoQsarProgress),
    ) -> QsarPipeline {
        let total_candidates = build_regression_candidates(data.n_features()).len();
        let runtime_cfg = parse_runtime_config_from_env(total_candidates, n_cv_folds);
        Self::regression_with_runtime_config(data, n_cv_folds, seed, on_progress, runtime_cfg)
    }

    /// Read the currently active runtime configuration for AutoQsar training.
    pub fn current_runtime_config(
        n_features: usize,
        n_cv_folds: usize,
    ) -> (String, Option<usize>, Option<usize>, usize) {
        let total_candidates = build_regression_candidates(n_features).len();
        let cfg = parse_runtime_config_from_env(total_candidates, n_cv_folds);
        (
            cfg.parallel_mode.as_str().to_string(),
            cfg.max_threads,
            cfg.prune_keep,
            cfg.prune_folds,
        )
    }

    fn regression_with_runtime_config(
        data: &QsarDataset,
        n_cv_folds: usize,
        seed: u64,
        on_progress: &mut dyn FnMut(AutoQsarProgress),
        runtime_cfg: AutoQsarRuntimeConfig,
    ) -> QsarPipeline {
        let scaler = StandardScaler::fit(&data.features);
        let scaled = scaler.transform(&data.features);
        let candidates = build_regression_candidates(data.n_features());
        let total_candidates = candidates.len();
        let thread_pool = build_optional_thread_pool(runtime_cfg.max_threads);
        let progress_total_candidates = runtime_cfg.prune_keep.unwrap_or(total_candidates);
        let mut completed_candidates = 0usize;
        on_progress(AutoQsarProgress {
            stage: AutoQsarProgressStage::Scaling,
            completed_candidates,
            total_candidates: progress_total_candidates,
        });

        let mut evaluate_candidates = |active: &[(usize, RegressionCandidate)],
                                       folds: usize,
                                       emit_progress: bool|
         -> Vec<CandidateEvaluation> {
            if runtime_cfg.parallel_mode == AutoQsarParallelMode::Candidate {
                let (tx, rx) = mpsc::channel::<CandidateProgressEvent>();
                install_in_pool(thread_pool.as_ref(), || {
                    rayon::scope(|scope| {
                        for (order, candidate) in active.iter().copied() {
                            let tx = tx.clone();
                            let scaled_ref = &scaled;
                            let targets_ref = &data.targets;
                            scope.spawn(move |_| {
                                let _ = tx.send(CandidateProgressEvent::Started { candidate });
                                let q2_raw = candidate.evaluate_q2(
                                    scaled_ref,
                                    targets_ref,
                                    folds,
                                    seed,
                                    false,
                                );
                                let q2 = if q2_raw.is_finite() {
                                    q2_raw
                                } else {
                                    f64::NEG_INFINITY
                                };
                                let _ =
                                    tx.send(CandidateProgressEvent::Done(CandidateEvaluation {
                                        order,
                                        candidate,
                                        q2,
                                    }));
                            });
                        }
                    });
                });
                drop(tx);
                let mut out = Vec::with_capacity(active.len());
                let mut running_best = f64::NEG_INFINITY;
                for event in rx {
                    match event {
                        CandidateProgressEvent::Started { candidate } if emit_progress => {
                            on_progress(AutoQsarProgress {
                                stage: AutoQsarProgressStage::CandidateStart {
                                    label: candidate.label(),
                                },
                                completed_candidates,
                                total_candidates: progress_total_candidates,
                            });
                        }
                        CandidateProgressEvent::Started { .. } => {}
                        CandidateProgressEvent::Done(eval) => {
                            if emit_progress {
                                completed_candidates += 1;
                                running_best = running_best.max(eval.q2);
                                on_progress(AutoQsarProgress {
                                    stage: AutoQsarProgressStage::CandidateDone {
                                        label: eval.candidate.label(),
                                        q2: eval.q2,
                                        best_q2: running_best,
                                    },
                                    completed_candidates,
                                    total_candidates: progress_total_candidates,
                                });
                            }
                            out.push(eval);
                        }
                    }
                }
                out
            } else {
                let mut out = Vec::with_capacity(active.len());
                let mut running_best = f64::NEG_INFINITY;
                for (order, candidate) in active.iter().copied() {
                    if emit_progress {
                        on_progress(AutoQsarProgress {
                            stage: AutoQsarProgressStage::CandidateStart {
                                label: candidate.label(),
                            },
                            completed_candidates,
                            total_candidates: progress_total_candidates,
                        });
                    }
                    let q2_raw = install_in_pool(thread_pool.as_ref(), || {
                        candidate.evaluate_q2(&scaled, &data.targets, folds, seed, true)
                    });
                    let q2 = if q2_raw.is_finite() {
                        q2_raw
                    } else {
                        f64::NEG_INFINITY
                    };
                    if emit_progress {
                        completed_candidates += 1;
                        running_best = running_best.max(q2);
                        on_progress(AutoQsarProgress {
                            stage: AutoQsarProgressStage::CandidateDone {
                                label: candidate.label(),
                                q2,
                                best_q2: running_best,
                            },
                            completed_candidates,
                            total_candidates: progress_total_candidates,
                        });
                    }
                    out.push(CandidateEvaluation {
                        order,
                        candidate,
                        q2,
                    });
                }
                out
            }
        };

        let all_candidates = candidates
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<(usize, RegressionCandidate)>>();

        let candidates_for_full = if let Some(keep) = runtime_cfg.prune_keep {
            let quick_pass = evaluate_candidates(&all_candidates, runtime_cfg.prune_folds, false);
            let mut survivors = top_k_candidates(quick_pass, keep);
            survivors.sort_by_key(|eval| eval.order);
            survivors
                .into_iter()
                .map(|eval| (eval.order, eval.candidate))
                .collect::<Vec<_>>()
        } else {
            all_candidates.clone()
        };

        let full_pass = evaluate_candidates(&candidates_for_full, n_cv_folds, true);

        let mut best_eval: Option<CandidateEvaluation> = None;
        for eval in full_pass {
            let is_better = match best_eval {
                Some(current) => {
                    eval.q2.total_cmp(&current.q2).is_gt()
                        || (eval.q2.total_cmp(&current.q2).is_eq() && eval.order < current.order)
                }
                None => true,
            };
            if is_better {
                best_eval = Some(eval);
            }
        }

        // Add AD to best pipeline
        let best_eval = best_eval.expect("No model trained");
        let best_candidate = best_eval.candidate;
        let model = best_candidate.fit_model(&scaled, &data.targets, seed);
        let train_preds = match &model {
            ModelType::KnnRegressor(m) => m.predict(&scaled),
            ModelType::RandomForestRegressor(m) => m.predict(&scaled),
            ModelType::PlsRegression(m) => m.predict(&scaled),
            ModelType::ElasticNet(m) => m.predict(&scaled),
            _ => unreachable!("AutoQsar regression candidates are regression-only"),
        };
        let train_r2 = metrics::r_squared(
            data.targets.as_slice().unwrap(),
            train_preds.as_slice().unwrap(),
        );
        let mut pipeline = QsarPipeline {
            name: best_candidate.pipeline_name(),
            task: TaskType::Regression,
            scaler: ScalerType::Standard(scaler.clone()),
            pca: None,
            model,
            ad: None,
            feature_names: data.feature_names.clone(),
            train_r2,
            cv_q2: best_eval.q2,
        };
        on_progress(AutoQsarProgress {
            stage: AutoQsarProgressStage::AdFit,
            completed_candidates,
            total_candidates: progress_total_candidates,
        });
        let ad = KnnDistanceAD::fit(&scaled, 5, 2.0);
        pipeline.ad = Some(ad);
        on_progress(AutoQsarProgress {
            stage: AutoQsarProgressStage::Complete,
            completed_candidates: progress_total_candidates,
            total_candidates: progress_total_candidates,
        });
        pipeline
    }
}

/// Helper: compute cross-validated Q² for a regression model builder, optionally
/// parallelizing folds.
fn cv_regression_q2_with_mode<F>(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    n_folds: usize,
    seed: u64,
    parallel_folds: bool,
    build_model: F,
) -> f64
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> ModelType + Sync + Send,
{
    let n = features.nrows();
    let splits = cv::k_fold(n, n_folds, seed);
    let eval_fold = |(train_idx, test_idx): &(Vec<usize>, Vec<usize>)| {
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
        (test_idx.clone(), preds.to_vec())
    };
    let fold_preds = if parallel_folds {
        splits.folds.par_iter().map(eval_fold).collect::<Vec<_>>()
    } else {
        splits.folds.iter().map(eval_fold).collect::<Vec<_>>()
    };

    let mut all_preds = vec![0.0f64; n];
    for (test_idx, preds) in fold_preds {
        for (i, idx) in test_idx.into_iter().enumerate() {
            all_preds[idx] = preds[i];
        }
    }

    metrics::r_squared(targets.as_slice().unwrap(), &all_preds)
}

/// Implements rows.
///
/// Function context: `crates/ferrum_core/src/ml/pipeline.rs`.
/// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
fn select_rows(data: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let ncols = data.ncols();
    if indices.is_empty() {
        return Array2::<f64>::zeros((0, ncols));
    }
    if ncols == 0 {
        return Array2::<f64>::zeros((indices.len(), 0));
    }
    if let Some(buf) = data.as_slice_memory_order() {
        if buf.len() == data.nrows() * ncols {
            let mut packed = Vec::with_capacity(indices.len() * ncols);
            for &old_i in indices {
                let start = old_i * ncols;
                packed.extend_from_slice(&buf[start..start + ncols]);
            }
            return Array2::from_shape_vec((indices.len(), ncols), packed)
                .expect("packed row slice must match target shape");
        }
    }
    let mut result = Array2::<f64>::zeros((indices.len(), ncols));
    for (new_i, &old_i) in indices.iter().enumerate() {
        result.row_mut(new_i).assign(&data.row(old_i));
    }
    result
}

/// Implements elements.
///
/// Function context: `crates/ferrum_core/src/ml/pipeline.rs`.
/// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
fn select_elements(data: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        out.push(data[idx]);
    }
    Array1::from_vec(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Builds regression dataset.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pipeline.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
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
    /// Runs regression test for pipeline predict.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pipeline.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
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
    /// Runs regression test for pipeline serialization.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pipeline.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
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
    /// Runs regression test for auto qsar regression.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pipeline.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
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

    #[test]
    /// Runs regression test for auto qsar progress stages.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pipeline.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_auto_qsar_progress_stages() {
        let ds = make_regression_dataset();
        let mut saw_scaling = false;
        let mut saw_ad_fit = false;
        let mut saw_complete = false;
        let mut candidate_starts = 0usize;
        let mut candidate_done = 0usize;
        let mut prev_completed = 0usize;
        let mut prev_best = f64::NEG_INFINITY;

        let mut cb = |progress: AutoQsarProgress| match progress.stage {
            AutoQsarProgressStage::Scaling => {
                saw_scaling = true;
                assert_eq!(progress.completed_candidates, 0);
            }
            AutoQsarProgressStage::CandidateStart { .. } => {
                candidate_starts += 1;
                assert!(progress.completed_candidates <= progress.total_candidates);
            }
            AutoQsarProgressStage::CandidateDone { q2: _, best_q2, .. } => {
                candidate_done += 1;
                assert!(progress.completed_candidates >= prev_completed);
                assert!(best_q2 >= prev_best || (best_q2.is_infinite() && prev_best.is_infinite()));
                prev_completed = progress.completed_candidates;
                prev_best = best_q2;
            }
            AutoQsarProgressStage::AdFit => {
                saw_ad_fit = true;
                assert_eq!(progress.completed_candidates, progress.total_candidates);
            }
            AutoQsarProgressStage::Complete => {
                saw_complete = true;
                assert_eq!(progress.completed_candidates, progress.total_candidates);
                assert!((progress.fraction() - 1.0).abs() < f32::EPSILON);
            }
        };

        let pipeline = AutoQsar::regression_with_progress(&ds, 3, 42, &mut cb);
        assert_eq!(pipeline.task, TaskType::Regression);
        assert!(saw_scaling);
        assert!(saw_ad_fit);
        assert!(saw_complete);
        assert_eq!(candidate_starts, candidate_done);
        assert!(candidate_done > 0);
    }

    #[test]
    /// Runs regression test for auto qsar fold mode parity.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pipeline.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_auto_qsar_fold_mode_parity() {
        let ds = make_regression_dataset();
        let mut noop = |_progress: AutoQsarProgress| {};
        let candidate_mode = AutoQsar::regression_with_runtime_config(
            &ds,
            3,
            42,
            &mut noop,
            make_runtime_config_for_tests(AutoQsarParallelMode::Candidate, None, None, 2),
        );
        let fold_mode = AutoQsar::regression_with_runtime_config(
            &ds,
            3,
            42,
            &mut noop,
            make_runtime_config_for_tests(AutoQsarParallelMode::Fold, Some(2), None, 2),
        );

        assert!((candidate_mode.cv_q2 - fold_mode.cv_q2).abs() < 1e-9);
        assert_eq!(candidate_mode.name, fold_mode.name);
    }

    #[test]
    /// Runs regression test for runtime config parsing.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pipeline.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_auto_qsar_runtime_config_parsing() {
        let cfg =
            resolve_auto_qsar_runtime_config(Some("fold"), Some("4"), Some("3"), Some("2"), 11, 5);
        assert_eq!(cfg.parallel_mode, AutoQsarParallelMode::Fold);
        assert_eq!(cfg.max_threads, Some(4));
        assert_eq!(cfg.prune_keep, Some(3));
        assert_eq!(cfg.prune_folds, 2);

        let default_cfg = resolve_auto_qsar_runtime_config(None, None, None, None, 11, 5);
        assert_eq!(default_cfg.parallel_mode, AutoQsarParallelMode::Candidate);
        assert!(default_cfg.max_threads.is_none());
        assert!(default_cfg.prune_keep.is_none());
        assert_eq!(default_cfg.prune_folds, 2);
    }

    #[test]
    /// Runs regression test for preprocess one matches batch with pca.
    ///
    /// Function context: `crates/ferrum_core/src/ml/pipeline.rs`.
    /// ML-layer note: numerical stability and reproducibility are important for comparable model evaluation.
    fn test_preprocess_one_matches_batch_with_pca() {
        let ds = make_regression_dataset();
        let scaler = StandardScaler::fit(&ds.features);
        let scaled = scaler.transform(&ds.features);
        let pca = Pca::fit(&scaled, 1);
        let projected = pca.transform(&scaled);
        let model = KnnRegressor::fit(projected, ds.targets.clone(), 3, DistanceMetric::Euclidean);

        let pipeline = QsarPipeline {
            name: "test-pca-one".into(),
            task: TaskType::Regression,
            scaler: ScalerType::Standard(scaler),
            pca: Some(pca),
            model: ModelType::KnnRegressor(model),
            ad: None,
            feature_names: ds.feature_names.clone(),
            train_r2: 0.0,
            cv_q2: 0.0,
        };

        let sample = ds.features.row(0).to_vec();
        let one = pipeline.preprocess_one(&sample);
        let batch = pipeline
            .preprocess(&Array2::from_shape_vec((1, sample.len()), sample).unwrap())
            .row(0)
            .to_vec();

        assert_eq!(one.len(), batch.len());
        for (a, b) in one.iter().zip(batch.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
