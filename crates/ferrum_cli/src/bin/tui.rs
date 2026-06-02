//! ## File guide: Tui
//!
//! This file belongs to FerrumChem's **CLI/demo/benchmark executable layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `enum AppState`
//! - `enum BrowserPurpose`
//! - `enum MlAlgorithm`
//! - `fn all`
//! - `fn label`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - TUI requires an interactive TTY and crossterm raw-mode lifecycle to be initialized successfully.
//! - Main-thread rendering and worker-thread compute channels assume message ordering without shared mutable UI state.
//! - Long-running operations must report progress/cancellation cooperatively to avoid perceived stalls in tab workflows.
//! ### Chemistry/ML references
//! - ratatui and crossterm terminal UI architecture/documentation.
//! - Rust std::sync::mpsc channel patterns for single-producer/single-consumer UI worker designs.
//! - FerrumChem usability learnings from descriptor-parity and 75% progress stall investigations.

#![allow(clippy::module_name_repetitions)]

//! FerrumChem interactive TUI.
//!
//! This binary implements a non-blocking, tab-based interface for five workflows:
//! 1. Import
//! 2. Descriptor
//! 3. Model Select
//! 4. Parameters
//! 5. Analyze
//!
//! Architectural design:
//! - Main thread: rendering and key handling at ~60 FPS
//! - Worker thread: CPU-heavy tasks via std::sync::mpsc channels
//! - Text input fields: tui-input crate for native editing behavior

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs;
use std::io;
use std::io::Write as _;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ndarray::{Array1, Array2};
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, Clear, Gauge, List, ListItem, ListState, Paragraph, Row, Table, Tabs, Wrap,
    canvas::{Canvas, Line as CanvasLine, Points},
};
use tui_input::Input;
use tui_input::backend::crossterm::EventHandler;

use ferrum_core::algo::rings::find_symmetrized_sssr_bonds;
use ferrum_core::desc::batch::{
    BatchDescriptor, FlatDescriptorMatrix, calculate_batch_flat_static_cancellable, write_csv_flat,
};
use ferrum_core::io::csv_reader::CsvReader;
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::ml::ad::KnnDistanceAD;
use ferrum_core::ml::art2a::Art2a;
use ferrum_core::ml::cv;
use ferrum_core::ml::dataset::QsarDataset;
use ferrum_core::ml::elastic_net::ElasticNet;
use ferrum_core::ml::feature_selection::{correlation_filter, variance_threshold};
use ferrum_core::ml::knn::{DistanceMetric, KnnRegressor};
use ferrum_core::ml::metrics;
use ferrum_core::ml::pca::Pca;
use ferrum_core::ml::pipeline::{
    AutoQsar, AutoQsarProgress, AutoQsarProgressStage, ModelType, QsarPipeline, ScalerType,
    TaskType,
};
use ferrum_core::ml::pls::PlsRegression;
use ferrum_core::ml::random_forest::RandomForestRegressor;
use ferrum_core::ml::scaling::{MinMaxScaler, StandardScaler};
use ferrum_core::ml::splitting::{TrainTestSplit, random_split, scaffold_split, stratified_split};
use ferrum_core::mol::molecule::Molecule;
use serde::{Deserialize, Serialize};

const FRAME_INTERVAL: Duration = Duration::from_millis(16);
const PREVIEW_ROWS: usize = 8;
const PREVIEW_COLS: usize = 5;
const DEFAULT_EXPORT_PATH: &str = "/tmp/ferrum_tui_predictions.csv";
const DEFAULT_FEATURE_EXPORT_PATH: &str = "/tmp/ferrum_tui_features.csv";
const DEFAULT_MODEL_EXPORT_PATH: &str = "/tmp/ferrum_tui_model_bundle.fcbundle";
const DEFAULT_CLUSTER_EXPORT_PATH: &str = "/tmp/ferrum_tui_clusters.csv";
const DEFAULT_VARIANCE_THRESHOLD: f64 = 1e-12;
const DEFAULT_CORRELATION_THRESHOLD: f64 = 0.95;
const DEFAULT_TEST_FRACTION: f64 = 0.20;
const DEFAULT_ART2A_VIGILANCE: f64 = 0.80;
const DEFAULT_ART2A_LEARNING_RATE: f64 = 0.30;
const DEFAULT_ART2A_MAX_ITER: usize = 50;
const DESCRIPTOR_COMPUTING_MESSAGE: &str = "Calculating descriptors...";
const DESCRIPTOR_ERROR_LOG_PATH: &str = "/tmp/ferrum_tui_descriptor_errors.log";
const SPINNER_FRAMES: [char; 4] = ['|', '/', '-', '\\'];

const TAB_TITLES: [&str; 6] = [
    "1: Import",
    "2: Descriptor",
    "3: Model Select",
    "4: Parameters",
    "5: Visuals",
    "6: Diagnostics",
];

/// Top-level UI state.
#[derive(Debug, Clone)]
enum AppState {
    Idle,
    Computing { message: String, job: WorkerJobKind },
    ErrorPopup(String),
    FileBrowserOpen(PathBuf),
    TargetSelectionOpen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Implements WorkerJobKind.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
enum WorkerJobKind {
    DatasetLoad,
    DescriptorCalculation,
    ModelTraining,
    Art2aClustering,
    Prediction,
    PredictionExport,
    ClusterExport,
    ModelBundleExport,
    ModelBundleImport,
    VarianceFilter,
    CorrelationFilter,
    FeatureMatrixExport,
}

impl WorkerJobKind {
    /// Returns a human-readable label for worker status and cancellation messages.
    fn label(self) -> &'static str {
        match self {
            Self::DatasetLoad => "dataset loading",
            Self::DescriptorCalculation => "descriptor calculation",
            Self::ModelTraining => "model training",
            Self::Art2aClustering => "ART2-A clustering",
            Self::Prediction => "prediction",
            Self::PredictionExport => "prediction export",
            Self::ClusterExport => "cluster export",
            Self::ModelBundleExport => "model bundle export",
            Self::ModelBundleImport => "model bundle import",
            Self::VarianceFilter => "variance filtering",
            Self::CorrelationFilter => "correlation filtering",
            Self::FeatureMatrixExport => "feature matrix export",
        }
    }

    /// Indicates whether this job kind supports cooperative cancellation.
    fn is_cancelable(self) -> bool {
        matches!(
            self,
            Self::DescriptorCalculation
                | Self::ModelTraining
                | Self::Art2aClustering
                | Self::Prediction
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Implements BrowserPurpose.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
enum BrowserPurpose {
    Dataset,
    Prediction,
    ModelImport,
}

const MODEL_BUNDLE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Implements PersistedModelBundle.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
struct PersistedModelBundle {
    schema_version: u32,
    pipeline_bundle_bytes: Option<Vec<u8>>,
    art2a_model_bytes: Option<Vec<u8>>,
    descriptors: Vec<String>,
    selected_feature_indices: Vec<usize>,
    expected_feature_names: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Implements MlAlgorithm.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
enum MlAlgorithm {
    AutoQsar,
    RandomForest,
    Knn,
    Pls,
    ElasticNet,
}

impl MlAlgorithm {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn all() -> [Self; 5] {
        [
            Self::AutoQsar,
            Self::RandomForest,
            Self::Knn,
            Self::Pls,
            Self::ElasticNet,
        ]
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn label(self) -> &'static str {
        match self {
            Self::AutoQsar => "AutoQsar",
            Self::RandomForest => "Random Forest",
            Self::Knn => "k-NN",
            Self::Pls => "PLS",
            Self::ElasticNet => "Elastic Net",
        }
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn cycle(self, delta: isize) -> Self {
        let all = Self::all();
        let idx = all
            .iter()
            .position(|candidate| *candidate == self)
            .unwrap_or(0);
        let next = (idx as isize + delta).rem_euclid(all.len() as isize) as usize;
        all[next]
    }

    /// Implements all.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn manual_all() -> [Self; 4] {
        [Self::RandomForest, Self::Knn, Self::Pls, Self::ElasticNet]
    }

    /// Checks whether manual.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn is_manual(self) -> bool {
        self != Self::AutoQsar
    }

    /// Implements cycle.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn manual_cycle(self, delta: isize) -> Self {
        let all = Self::manual_all();
        let idx = all
            .iter()
            .position(|candidate| *candidate == self)
            .unwrap_or(0);
        let next = (idx as isize + delta).rem_euclid(all.len() as isize) as usize;
        all[next]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Implements MlMode.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
enum MlMode {
    AutoQsar,
    Art2a,
    Manual,
}

impl MlMode {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn toggle(self) -> Self {
        match self {
            Self::AutoQsar => Self::Art2a,
            Self::Art2a => Self::Manual,
            Self::Manual => Self::AutoQsar,
        }
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn label(self) -> &'static str {
        match self {
            Self::AutoQsar => "AutoQsar",
            Self::Art2a => "ART2-A",
            Self::Manual => "Manual",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Implements MlScalerChoice.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
enum MlScalerChoice {
    Standard,
    MinMax,
}

impl MlScalerChoice {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn label(self) -> &'static str {
        match self {
            Self::Standard => "StandardScaler",
            Self::MinMax => "MinMaxScaler",
        }
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn toggle(self) -> Self {
        match self {
            Self::Standard => Self::MinMax,
            Self::MinMax => Self::Standard,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Implements SplitStrategy.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
enum SplitStrategy {
    Random,
    Stratified,
    Scaffold,
}

impl SplitStrategy {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn cycle(self) -> Self {
        match self {
            Self::Random => Self::Stratified,
            Self::Stratified => Self::Scaffold,
            Self::Scaffold => Self::Random,
        }
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn label(self) -> &'static str {
        match self {
            Self::Random => "Random",
            Self::Stratified => "Stratified",
            Self::Scaffold => "Scaffold",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Implements VisualizationColorMode.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
enum VisualizationColorMode {
    Uniform,
    TargetOverlay,
    ClusterIds,
    ApplicabilityDomain,
}

impl VisualizationColorMode {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn cycle(self) -> Self {
        match self {
            Self::Uniform => Self::TargetOverlay,
            Self::TargetOverlay => Self::ClusterIds,
            Self::ClusterIds => Self::ApplicabilityDomain,
            Self::ApplicabilityDomain => Self::Uniform,
        }
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn label(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::TargetOverlay => "target overlay",
            Self::ClusterIds => "cluster IDs",
            Self::ApplicabilityDomain => "applicability domain",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Implements TargetColumnSource.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
enum TargetColumnSource {
    Csv,
    Descriptor,
}

impl TargetColumnSource {
    /// Returns a short human-readable source label.
    fn label(self) -> &'static str {
        match self {
            Self::Csv => "CSV",
            Self::Descriptor => "DESC",
        }
    }

    /// Returns a stable key prefix used for internal target identifiers.
    fn key_prefix(self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Descriptor => "desc",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Implements TargetColumn.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
struct TargetColumn {
    key: String,
    name: String,
    source: TargetColumnSource,
}

impl TargetColumn {
    /// Creates a new typed target column entry.
    fn new(name: String, source: TargetColumnSource) -> Self {
        let key = format!("{}:{}", source.key_prefix(), name);
        Self { key, name, source }
    }

    /// Formats this target for UI selection lists.
    fn display_label(&self) -> String {
        format!("[{}] {}", self.source.label(), self.name)
    }

    /// Formats this target for model/prediction output names.
    fn output_name(&self) -> String {
        format!(
            "{}__{}",
            self.name,
            self.source.label().to_ascii_lowercase()
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Implements MlParamKey.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
enum MlParamKey {
    CvFolds,
    TestFraction,
    Seed,
    Art2aVigilance,
    Art2aLearningRate,
    Art2aMaxIter,
    KnnK,
    RfTrees,
    RfMaxDepth,
    RfMinSamplesSplit,
    RfMaxFeaturesFraction,
    PlsComponents,
    EnAlpha,
    EnL1Ratio,
    EnMaxIter,
    EnTol,
}

#[derive(Debug, Clone)]
/// Implements MlTrainingConfig.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
struct MlTrainingConfig {
    algorithm: MlAlgorithm,
    scaler: MlScalerChoice,
    split_strategy: SplitStrategy,
    test_fraction: f64,
    cv_folds: usize,
    seed: u64,
    knn_k: usize,
    rf_n_estimators: usize,
    rf_max_depth: usize,
    rf_min_samples_split: usize,
    rf_max_features_fraction: f64,
    pls_components: usize,
    en_alpha: f64,
    en_l1_ratio: f64,
    en_max_iter: usize,
    en_tol: f64,
}

impl Default for MlTrainingConfig {
    /// Builds the default value for this type.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn default() -> Self {
        Self {
            algorithm: MlAlgorithm::AutoQsar,
            scaler: MlScalerChoice::Standard,
            split_strategy: SplitStrategy::Random,
            test_fraction: DEFAULT_TEST_FRACTION,
            cv_folds: 3,
            seed: 42,
            knn_k: 5,
            rf_n_estimators: 100,
            rf_max_depth: 10,
            rf_min_samples_split: 2,
            rf_max_features_fraction: 0.33,
            pls_components: 3,
            en_alpha: 0.1,
            en_l1_ratio: 0.5,
            en_max_iter: 1000,
            en_tol: 1e-6,
        }
    }
}

#[derive(Debug, Clone)]
/// Implements MlParamEntry.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
struct MlParamEntry {
    hotkey: char,
    key: MlParamKey,
    label: &'static str,
    value: String,
}

/// Dense row-major feature matrix used by UI surfaces and worker payloads.
#[derive(Debug, Clone)]
struct FeatureMatrix {
    rows: usize,
    cols: usize,
    feature_names: Vec<String>,
    data: Vec<f64>,
    descriptors: Vec<BatchDescriptor>,
    selected_feature_indices: Vec<usize>,
}

/// Persisted/active feature-layout contract used for model inference/export.
#[derive(Debug, Clone)]
struct ModelFeatureLayout {
    descriptors: Vec<BatchDescriptor>,
    selected_feature_indices: Vec<usize>,
    expected_feature_names: Vec<String>,
}

#[derive(Debug, Clone)]
/// Implements ActiveModelSelection.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
struct ActiveModelSelection {
    layout: ModelFeatureLayout,
    pipeline_bundle_bytes: Option<Vec<u8>>,
    art2a_model_bytes: Option<Vec<u8>>,
    warning: Option<String>,
}

impl ModelFeatureLayout {
    /// Builds a feature-layout snapshot from an in-memory feature matrix.
    fn from_matrix(matrix: &FeatureMatrix) -> Self {
        Self {
            descriptors: matrix.descriptors.clone(),
            selected_feature_indices: matrix.selected_feature_indices.clone(),
            expected_feature_names: matrix.feature_names.clone(),
        }
    }

    /// Builds a feature-layout snapshot from persisted/model response components.
    fn from_parts(
        descriptors: Vec<BatchDescriptor>,
        selected_feature_indices: Vec<usize>,
        expected_feature_names: Vec<String>,
    ) -> Self {
        Self {
            descriptors,
            selected_feature_indices,
            expected_feature_names,
        }
    }
}

impl FeatureMatrix {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn row(&self, idx: usize) -> &[f64] {
        debug_assert!(idx < self.rows);
        let start = idx * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Converts payload between representations.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn to_payload(&self) -> FeatureMatrixPayload {
        FeatureMatrixPayload {
            rows: self.rows,
            cols: self.cols,
            feature_names: self.feature_names.clone(),
            data: self.data.clone(),
            descriptors: self.descriptors.clone(),
            selected_feature_indices: self.selected_feature_indices.clone(),
        }
    }

    /// Constructs payload between representations.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn from_payload(payload: FeatureMatrixPayload) -> Self {
        Self {
            rows: payload.rows,
            cols: payload.cols,
            feature_names: payload.feature_names,
            data: payload.data,
            descriptors: payload.descriptors,
            selected_feature_indices: payload.selected_feature_indices,
        }
    }
}

/// Descriptor profile row shown in analytics table.
#[derive(Debug, Clone)]
struct DescriptorProfileRow {
    name: String,
    mean: f64,
    min: f64,
    max: f64,
    stddev: f64,
    zero_variance: bool,
}

/// 2D PCA projection cache used for visualization tab.
#[derive(Debug, Clone)]
struct ProjectionData {
    points: Vec<(f64, f64)>,
    x_bounds: [f64; 2],
    y_bounds: [f64; 2],
    explained_variance: [f64; 2],
}

#[derive(Debug, Clone)]
/// Implements Art2aConfig.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
struct Art2aConfig {
    vigilance: f64,
    learning_rate: f64,
    max_iter: usize,
}

#[derive(Debug, Clone)]
/// Implements FeatureMatrixPayload.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
struct FeatureMatrixPayload {
    rows: usize,
    cols: usize,
    feature_names: Vec<String>,
    data: Vec<f64>,
    descriptors: Vec<BatchDescriptor>,
    selected_feature_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
/// Implements TargetTrainingSummary.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
struct TargetTrainingSummary {
    target: String,
    target_display_label: String,
    pipeline_name: String,
    cv_q2: f64,
    train_r2: f64,
    rmse: f64,
    mae: f64,
    pearson: f64,
    holdout_r2: f64,
    holdout_rmse: f64,
    holdout_mae: f64,
    holdout_pearson: f64,
    split_label: String,
    split_train_size: usize,
    split_test_size: usize,
    top_feature_importances: Vec<(String, f64)>,
}

#[derive(Debug, Clone)]
/// Implements PredictionOutputRow.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
struct PredictionOutputRow {
    smiles: String,
    predictions: Vec<f64>,
}

#[derive(Debug, Clone)]
/// Implements ClusterExportRow.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
struct ClusterExportRow {
    smiles: String,
    cluster_id: usize,
    similarity: Option<f64>,
    in_domain: Option<bool>,
}

#[derive(Debug, Clone)]
/// Implements Art2aClusterSummary.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
struct Art2aClusterSummary {
    cluster_id: usize,
    size: usize,
    fraction: f64,
    mean_similarity: f64,
    min_similarity: f64,
    max_similarity: f64,
}

#[derive(Debug)]
/// Implements WorkerRequest.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
enum WorkerRequest {
    LoadDataset(PathBuf),
    CalculateDescriptors {
        molecules: Vec<Molecule>,
        descriptors: Vec<BatchDescriptor>,
    },
    TrainModel {
        matrix: FeatureMatrixPayload,
        target_series: Vec<(String, Vec<f64>)>,
        smiles: Vec<String>,
        config: MlTrainingConfig,
        target_columns: Vec<TargetColumn>,
    },
    RunArt2aClustering {
        matrix: FeatureMatrixPayload,
        config: Art2aConfig,
        scaler: MlScalerChoice,
    },
    PredictFile {
        path: PathBuf,
        pipeline_bundle_bytes: Option<Vec<u8>>,
        art2a_model_bytes: Option<Vec<u8>>,
        descriptors: Vec<BatchDescriptor>,
        selected_feature_indices: Vec<usize>,
        expected_feature_names: Vec<String>,
    },
    ExportPredictions {
        path: PathBuf,
        target_names: Vec<String>,
        rows: Vec<PredictionOutputRow>,
    },
    ExportClusters {
        path: PathBuf,
        rows: Vec<ClusterExportRow>,
    },
    ApplyVarianceFilter {
        matrix: FeatureMatrixPayload,
        threshold: f64,
    },
    ApplyCorrelationFilter {
        matrix: FeatureMatrixPayload,
        threshold: f64,
    },
    ExportFeatureMatrix {
        path: PathBuf,
        smiles: Vec<String>,
        matrix: FeatureMatrixPayload,
    },
    ExportModelBundle {
        path: PathBuf,
        bundle: PersistedModelBundle,
    },
    ImportModelBundle {
        path: PathBuf,
    },
    Shutdown,
}

#[derive(Debug)]
/// Implements WorkerResponse.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
enum WorkerResponse {
    Progress(f32),
    DatasetLoaded {
        path: PathBuf,
        molecules: Vec<Molecule>,
        smiles: Vec<String>,
        properties: Vec<Vec<(Arc<str>, Box<str>)>>,
        failed_rows: usize,
    },
    DescriptorsCalculated {
        matrix: FeatureMatrixPayload,
        profiles: Vec<DescriptorProfileRow>,
        projection: Option<ProjectionData>,
    },
    ModelTrained {
        primary_target: String,
        primary_target_display_label: String,
        pipeline_name: String,
        cv_q2: f64,
        train_r2: f64,
        rmse: f64,
        mae: f64,
        pearson: f64,
        train_actuals: Vec<f64>,
        ad_in_domain_flags: Vec<bool>,
        top_feature_importances: Vec<(String, f64)>,
        holdout_r2: f64,
        holdout_rmse: f64,
        holdout_mae: f64,
        holdout_pearson: f64,
        split_label: String,
        split_train_size: usize,
        split_test_size: usize,
        target_summaries: Vec<TargetTrainingSummary>,
        pipeline_bundle_bytes: Vec<u8>,
        descriptors: Vec<BatchDescriptor>,
        selected_feature_indices: Vec<usize>,
        expected_feature_names: Vec<String>,
        preview_predictions: Vec<f64>,
        train_predictions: Vec<f64>,
    },
    ClusteringCompleted {
        labels: Vec<usize>,
        cluster_sizes: Vec<usize>,
        summaries: Vec<Art2aClusterSummary>,
        art2a_model_bytes: Vec<u8>,
        projection: Option<ProjectionData>,
        descriptors: Vec<BatchDescriptor>,
        selected_feature_indices: Vec<usize>,
        expected_feature_names: Vec<String>,
    },
    PredictionsReady {
        path: PathBuf,
        target_names: Vec<String>,
        rows: Vec<PredictionOutputRow>,
        skipped_rows: usize,
        projection: Option<ProjectionData>,
    },
    PredictionsExported {
        path: PathBuf,
        count: usize,
    },
    ClustersExported {
        path: PathBuf,
        count: usize,
    },
    FeatureFilterApplied {
        matrix: FeatureMatrixPayload,
        profiles: Vec<DescriptorProfileRow>,
        projection: Option<ProjectionData>,
        removed_features: usize,
        filter_label: String,
    },
    JobCanceled {
        job: WorkerJobKind,
    },
    FeatureMatrixExported {
        path: PathBuf,
        rows: usize,
        cols: usize,
    },
    ModelBundleExported {
        path: PathBuf,
    },
    ModelBundleImported {
        path: PathBuf,
        bundle: PersistedModelBundle,
    },
    Error(String),
}

/// Main in-memory application model.
struct App {
    // Required plan fields
    active_tab: usize,
    state: AppState,
    available_descriptors: Vec<(BatchDescriptor, bool)>,
    descriptor_list_state: ListState,
    loaded_molecules: Vec<Molecule>,
    feature_matrix: Option<FeatureMatrix>,
    trained_pipeline: Option<QsarPipeline>,
    trained_target_summaries: Vec<TargetTrainingSummary>,
    training_config: MlTrainingConfig,
    selected_ml_mode: MlMode,
    manual_algo_selection: MlAlgorithm,
    ml_param_page: usize,
    editing_ml_param: Option<MlParamKey>,
    ml_param_input: Input,
    last_train_rmse: Option<f64>,
    last_train_mae: Option<f64>,
    last_train_pearson: Option<f64>,
    last_holdout_r2: Option<f64>,
    last_holdout_rmse: Option<f64>,
    last_holdout_mae: Option<f64>,
    last_holdout_pearson: Option<f64>,
    last_split_label: Option<String>,
    last_split_train_size: Option<usize>,
    last_split_test_size: Option<usize>,
    art2a_config: Art2aConfig,
    cluster_labels: Vec<usize>,
    cluster_sizes: Vec<usize>,
    cluster_summaries: Vec<Art2aClusterSummary>,
    art2a_model_bytes: Option<Vec<u8>>,
    art2a_feature_layout: Option<ModelFeatureLayout>,
    color_mode: VisualizationColorMode,

    // Additional state needed for complete workflows
    loaded_smiles: Vec<String>,
    loaded_properties: Vec<Vec<(Arc<str>, Box<str>)>>,
    last_dataset_failed_rows: usize,
    loaded_from_path: Option<PathBuf>,
    target_columns: Vec<TargetColumn>,
    selected_target_columns: BTreeSet<String>,
    target_column_list_state: ListState,
    target_overlay_values: Option<Vec<f64>>,
    target_overlay_error: Option<String>,

    descriptor_profiles: Vec<DescriptorProfileRow>,
    projection: Option<ProjectionData>,
    train_actuals: Vec<f64>,
    training_predictions: Vec<f64>,
    ad_in_domain_flags: Vec<bool>,
    top_feature_importances: Vec<(String, f64)>,
    show_feature_importance: bool,

    prediction_rows: Vec<PredictionOutputRow>,
    prediction_target_names: Vec<String>,
    prediction_source: Option<PathBuf>,
    prediction_export_path: PathBuf,
    feature_export_path: PathBuf,
    model_bundle_path: PathBuf,
    cluster_export_path: PathBuf,

    pipeline_bundle_bytes: Option<Vec<u8>>,
    pipeline_feature_layout: Option<ModelFeatureLayout>,
    include_csv_numeric_in_clustering: bool,
    trained_primary_target: Option<String>,
    importance_target_index: usize,

    status_message: String,
    progress_percent: f32,
    profile_scroll: usize,
    spinner_frame: usize,

    browser_purpose: BrowserPurpose,
    browser_entries: Vec<PathBuf>,
    browser_list_state: ListState,

    worker_tx: Sender<WorkerRequest>,
    worker_rx: Receiver<WorkerResponse>,
    worker_cancel_flag: Arc<AtomicBool>,
    cancel_requested: bool,
    active_job: Option<WorkerJobKind>,
}

impl App {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn new(
        worker_tx: Sender<WorkerRequest>,
        worker_rx: Receiver<WorkerResponse>,
        worker_cancel_flag: Arc<AtomicBool>,
    ) -> Self {
        let mut descriptor_list_state = ListState::default();
        descriptor_list_state.select(Some(0));

        Self {
            active_tab: 0,
            state: AppState::Idle,
            available_descriptors: BatchDescriptor::all()
                .iter()
                .copied()
                .map(|d| (d, true))
                .collect(),
            descriptor_list_state,
            loaded_molecules: Vec::new(),
            feature_matrix: None,
            trained_pipeline: None,
            trained_target_summaries: Vec::new(),
            training_config: MlTrainingConfig::default(),
            selected_ml_mode: MlMode::AutoQsar,
            manual_algo_selection: MlAlgorithm::RandomForest,
            ml_param_page: 0,
            editing_ml_param: None,
            ml_param_input: Input::default(),
            last_train_rmse: None,
            last_train_mae: None,
            last_train_pearson: None,
            last_holdout_r2: None,
            last_holdout_rmse: None,
            last_holdout_mae: None,
            last_holdout_pearson: None,
            last_split_label: None,
            last_split_train_size: None,
            last_split_test_size: None,
            art2a_config: Art2aConfig {
                vigilance: DEFAULT_ART2A_VIGILANCE,
                learning_rate: DEFAULT_ART2A_LEARNING_RATE,
                max_iter: DEFAULT_ART2A_MAX_ITER,
            },
            cluster_labels: Vec::new(),
            cluster_sizes: Vec::new(),
            cluster_summaries: Vec::new(),
            art2a_model_bytes: None,
            art2a_feature_layout: None,
            color_mode: VisualizationColorMode::Uniform,

            loaded_smiles: Vec::new(),
            loaded_properties: Vec::new(),
            last_dataset_failed_rows: 0,
            loaded_from_path: None,
            target_columns: Vec::new(),
            selected_target_columns: BTreeSet::new(),
            target_column_list_state: ListState::default(),
            target_overlay_values: None,
            target_overlay_error: None,

            descriptor_profiles: Vec::new(),
            projection: None,
            train_actuals: Vec::new(),
            training_predictions: Vec::new(),
            ad_in_domain_flags: Vec::new(),
            top_feature_importances: Vec::new(),
            show_feature_importance: true,

            prediction_rows: Vec::new(),
            prediction_target_names: Vec::new(),
            prediction_source: None,
            prediction_export_path: PathBuf::from(DEFAULT_EXPORT_PATH),
            feature_export_path: PathBuf::from(DEFAULT_FEATURE_EXPORT_PATH),
            model_bundle_path: PathBuf::from(DEFAULT_MODEL_EXPORT_PATH),
            cluster_export_path: PathBuf::from(DEFAULT_CLUSTER_EXPORT_PATH),

            pipeline_bundle_bytes: None,
            pipeline_feature_layout: None,
            include_csv_numeric_in_clustering: true,
            trained_primary_target: None,
            importance_target_index: 0,

            status_message: "Ready. Open Tab 1 (Import) and press 'f' to load molecules."
                .to_string(),
            progress_percent: 0.0,
            profile_scroll: 0,
            spinner_frame: 0,

            browser_purpose: BrowserPurpose::Dataset,
            browser_entries: Vec::new(),
            browser_list_state: ListState::default(),

            worker_tx,
            worker_rx,
            worker_cancel_flag,
            cancel_requested: false,
            active_job: None,
        }
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn tick(&mut self) {
        if matches!(self.state, AppState::Computing { .. }) {
            self.spinner_frame = (self.spinner_frame + 1) % SPINNER_FRAMES.len();
        }
    }

    /// Implements descriptors.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn selected_descriptors(&self) -> Vec<BatchDescriptor> {
        self.available_descriptors
            .iter()
            .filter_map(|(descriptor, enabled)| (*enabled).then_some(*descriptor))
            .collect()
    }

    /// Returns the total number of feature values produced by the current descriptor selection.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn selected_descriptor_value_count(&self) -> usize {
        self.available_descriptors
            .iter()
            .filter(|(_, enabled)| *enabled)
            .map(|(descriptor, _)| descriptor.feature_len())
            .sum()
    }

    /// Resolves the active model/layout for prediction/export, allowing single-model fallback.
    fn resolve_active_model_selection(&self) -> Result<ActiveModelSelection, String> {
        let pipeline_layout = self
            .pipeline_bundle_bytes
            .as_ref()
            .and(self.pipeline_feature_layout.as_ref());
        let art2a_layout = self
            .art2a_model_bytes
            .as_ref()
            .and(self.art2a_feature_layout.as_ref());

        match (pipeline_layout, art2a_layout) {
            (Some(pipeline_layout), Some(art2a_layout)) => {
                if pipeline_layout.expected_feature_names == art2a_layout.expected_feature_names
                    && pipeline_layout.selected_feature_indices
                        == art2a_layout.selected_feature_indices
                    && pipeline_layout.descriptors == art2a_layout.descriptors
                {
                    return Ok(ActiveModelSelection {
                        layout: pipeline_layout.clone(),
                        pipeline_bundle_bytes: self.pipeline_bundle_bytes.clone(),
                        art2a_model_bytes: self.art2a_model_bytes.clone(),
                        warning: None,
                    });
                }

                return Ok(ActiveModelSelection {
                    layout: pipeline_layout.clone(),
                    pipeline_bundle_bytes: self.pipeline_bundle_bytes.clone(),
                    art2a_model_bytes: None,
                    warning: Some(
                        "Supervised and ART2-A layouts differ; prediction/export will use supervised model only."
                            .to_string(),
                    ),
                });
            }
            (Some(pipeline_layout), None) => {
                return Ok(ActiveModelSelection {
                    layout: pipeline_layout.clone(),
                    pipeline_bundle_bytes: self.pipeline_bundle_bytes.clone(),
                    art2a_model_bytes: None,
                    warning: None,
                });
            }
            (None, Some(art2a_layout)) => {
                return Ok(ActiveModelSelection {
                    layout: art2a_layout.clone(),
                    pipeline_bundle_bytes: None,
                    art2a_model_bytes: self.art2a_model_bytes.clone(),
                    warning: None,
                });
            }
            (None, None) => {}
        }

        if let Some(matrix) = &self.feature_matrix {
            if !matrix.descriptors.is_empty() && !matrix.feature_names.is_empty() {
                return Ok(ActiveModelSelection {
                    layout: ModelFeatureLayout::from_matrix(matrix),
                    pipeline_bundle_bytes: self.pipeline_bundle_bytes.clone(),
                    art2a_model_bytes: self.art2a_model_bytes.clone(),
                    warning: None,
                });
            }
        }

        Err("No descriptor/model feature layout available. Calculate descriptors or import a model bundle first.".to_string())
    }

    /// Returns CSV numeric column vectors aligned to current matrix rows, keyed by column name.
    fn csv_numeric_targets_aligned_to_matrix_rows(&self) -> Vec<(String, Vec<f64>)> {
        let Some(matrix) = &self.feature_matrix else {
            return Vec::new();
        };
        if self.loaded_properties.len() != matrix.rows || matrix.rows == 0 {
            return Vec::new();
        }

        let mut per_column: HashMap<String, Vec<f64>> = HashMap::new();
        let mut valid_counts: HashMap<String, usize> = HashMap::new();

        for row in &self.loaded_properties {
            for (name, raw) in row {
                if let Ok(parsed) = raw.trim().parse::<f64>() {
                    per_column.entry(name.to_string()).or_default().push(parsed);
                    *valid_counts.entry(name.to_string()).or_insert(0) += 1;
                }
            }
        }

        let mut columns = per_column
            .into_iter()
            .filter(|(name, values)| {
                values.len() == matrix.rows
                    && valid_counts.get(name).copied().unwrap_or(0) == matrix.rows
                    && !matrix.feature_names.iter().any(|feature| feature == name)
            })
            .collect::<Vec<_>>();
        columns.sort_by(|left, right| left.0.cmp(&right.0));
        columns
    }

    /// Builds clustering matrix from descriptor matrix, optionally appending CSV numeric columns.
    fn clustering_payload_with_optional_csv_targets(
        &self,
    ) -> Result<(FeatureMatrixPayload, usize), String> {
        let Some(matrix) = &self.feature_matrix else {
            return Err("No feature matrix available. Calculate descriptors first.".to_string());
        };
        if matrix.rows < 2 || matrix.cols == 0 {
            return Err("ART2-A clustering needs at least 2 rows and 1 feature.".to_string());
        }

        let mut payload = matrix.to_payload();
        if !self.include_csv_numeric_in_clustering {
            return Ok((payload, 0));
        }

        let csv_columns = self.csv_numeric_targets_aligned_to_matrix_rows();
        if csv_columns.is_empty() {
            return Ok((payload, 0));
        }

        let base_cols = payload.cols;
        let total_cols = base_cols + csv_columns.len();
        let mut merged_data = Vec::with_capacity(payload.rows * total_cols);
        for row in 0..payload.rows {
            let base = row * base_cols;
            merged_data.extend_from_slice(&payload.data[base..base + base_cols]);
            for (_, values) in &csv_columns {
                merged_data.push(values[row]);
            }
        }
        payload.data = merged_data;

        for (name, _) in &csv_columns {
            payload.feature_names.push(format!("CSV::{}", name));
            payload
                .selected_feature_indices
                .push(payload.selected_feature_indices.len());
        }
        payload.cols = total_cols;

        Ok((payload, csv_columns.len()))
    }

    /// Implements selected target.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn primary_selected_target(&self) -> Option<&TargetColumn> {
        self.target_columns
            .iter()
            .find(|target| self.selected_target_columns.contains(target.key.as_str()))
    }

    /// Implements targets in order.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn selected_targets_in_order(&self) -> Vec<TargetColumn> {
        self.target_columns
            .iter()
            .filter(|target| self.selected_target_columns.contains(target.key.as_str()))
            .cloned()
            .collect()
    }

    /// Returns the currently displayed target summary for feature-importance view.
    fn current_importance_target_summary(&self) -> Option<&TargetTrainingSummary> {
        if self.trained_target_summaries.is_empty() {
            return None;
        }
        let idx = self
            .importance_target_index
            .min(self.trained_target_summaries.len().saturating_sub(1));
        self.trained_target_summaries.get(idx)
    }

    /// Syncs the active feature-importance panel with the selected importance target index.
    fn sync_importance_view_from_index(&mut self) {
        if let Some(summary) = self.current_importance_target_summary() {
            self.top_feature_importances = summary.top_feature_importances.clone();
        } else {
            self.top_feature_importances.clear();
        }
    }

    /// Cycles which trained target drives the feature-importance panel.
    fn cycle_importance_target(&mut self, delta: isize) {
        if self.trained_target_summaries.is_empty() {
            self.status_message =
                "No trained target summaries available. Train model(s) first.".to_string();
            return;
        }
        let len = self.trained_target_summaries.len() as isize;
        let current = self.importance_target_index as isize;
        let next = (current + delta).rem_euclid(len) as usize;
        self.importance_target_index = next;
        self.sync_importance_view_from_index();
        if let Some(summary) = self.current_importance_target_summary() {
            self.status_message = format!(
                "Feature importance target: {} ({})",
                summary.target_display_label, summary.pipeline_name
            );
        }
    }

    /// Implements target overlay cache.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn refresh_target_overlay_cache(&mut self) {
        let Some(target_column) = self.primary_selected_target().cloned() else {
            self.target_overlay_values = None;
            self.target_overlay_error = None;
            return;
        };
        match self.resolve_target_values(&target_column) {
            Ok(values) => {
                self.target_overlay_values = Some(values);
                self.target_overlay_error = None;
            }
            Err(error) => {
                self.target_overlay_values = None;
                self.target_overlay_error = Some(error);
            }
        }
    }

    /// Implements mode summary.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn ml_mode_summary(&self) -> String {
        let selected = self.selected_targets_in_order();
        if selected.is_empty() {
            return "Mode: Unsupervised (select target(s) to enable supervised training)"
                .to_string();
        }
        let primary = &selected[0].name;
        if self.feature_matrix.is_none() {
            if self.target_overlay_values.is_some() {
                return format!(
                    "Mode: Targets selected [{}], compute descriptors to enable supervised training",
                    truncate_label(
                        &selected
                            .iter()
                            .map(|target| target.display_label())
                            .collect::<Vec<_>>()
                            .join(", "),
                        48
                    )
                );
            }
            if let Some(error) = &self.target_overlay_error {
                return format!(
                    "Mode: Primary target invalid ({})",
                    truncate_label(error, 52)
                );
            }
            return format!("Mode: Target pending validation ('{}')", primary);
        }
        if let (Some(values), Some(matrix)) = (&self.target_overlay_values, &self.feature_matrix) {
            if values.len() == matrix.rows {
                return format!(
                    "Mode: Supervised ready ({} target(s), primary='{}')",
                    selected.len(),
                    primary
                );
            }
            return format!(
                "Mode: Target row mismatch (target rows={}, feature rows={})",
                values.len(),
                matrix.rows
            );
        }
        if let Some(error) = &self.target_overlay_error {
            return format!(
                "Mode: Primary target invalid ({})",
                truncate_label(error, 52)
            );
        }
        format!("Mode: Target pending validation ('{}')", primary)
    }

    /// Implements target columns.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn rebuild_target_columns(&mut self) {
        let mut stats: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for row in &self.loaded_properties {
            for (name, value) in row {
                let entry = stats.entry(name.to_string()).or_insert((0usize, 0usize));
                entry.0 += 1;
                if value.trim().parse::<f64>().is_ok() {
                    entry.1 += 1;
                }
            }
        }

        let row_count = self.loaded_properties.len();
        let mut property_targets = stats
            .into_iter()
            .filter_map(|(name, (present_count, numeric_count))| {
                (present_count == row_count && row_count > 0 && numeric_count == row_count)
                    .then_some(TargetColumn::new(name, TargetColumnSource::Csv))
            })
            .collect::<Vec<_>>();
        property_targets.sort_by(|left, right| {
            left.name
                .cmp(&right.name)
                .then_with(|| left.source.cmp(&right.source))
        });

        let mut merged = property_targets.clone();
        if let Some(matrix) = &self.feature_matrix {
            for feature_name in &matrix.feature_names {
                merged.push(TargetColumn::new(
                    feature_name.clone(),
                    TargetColumnSource::Descriptor,
                ));
            }
        }
        merged.sort_by(|left, right| {
            left.name
                .cmp(&right.name)
                .then_with(|| left.source.cmp(&right.source))
        });
        self.target_columns = merged;

        self.selected_target_columns.retain(|key| {
            self.target_columns
                .iter()
                .any(|candidate| candidate.key == *key)
        });
        if self.selected_target_columns.is_empty() {
            for target in &property_targets {
                self.selected_target_columns.insert(target.key.clone());
            }
        }
        if self.selected_target_columns.is_empty() {
            for target in &self.target_columns {
                self.selected_target_columns.insert(target.key.clone());
            }
        }

        let selected_idx = self
            .target_columns
            .iter()
            .position(|target| self.selected_target_columns.contains(target.key.as_str()))
            .or_else(|| (!self.target_columns.is_empty()).then_some(0));
        self.target_column_list_state.select(selected_idx);
        self.refresh_target_overlay_cache();
    }

    /// Resolves descriptor-backed target column values from the current matrix.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn extract_descriptor_target_values(
        &self,
        target_column: &TargetColumn,
    ) -> Option<(usize, Vec<f64>)> {
        if target_column.source != TargetColumnSource::Descriptor {
            return None;
        }
        let matrix = self.feature_matrix.as_ref()?;
        let column_idx = matrix
            .feature_names
            .iter()
            .position(|name| name == &target_column.name)?;
        let mut values = Vec::with_capacity(matrix.rows);
        for row in 0..matrix.rows {
            values.push(matrix.data[row * matrix.cols + column_idx]);
        }
        Some((column_idx, values))
    }

    /// Resolves any target column values (CSV numeric property or descriptor feature).
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn resolve_target_values(&self, target_column: &TargetColumn) -> Result<Vec<f64>, String> {
        if let Some((_, values)) = self.extract_descriptor_target_values(target_column) {
            return Ok(values);
        }
        extract_targets(&self.loaded_properties, &target_column.name)
    }

    /// Resolves target values and optionally the source descriptor column index to drop.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn resolve_training_target_values(
        &self,
        target_column: &TargetColumn,
    ) -> Result<(Vec<f64>, Option<usize>), String> {
        if let Some((column_idx, values)) = self.extract_descriptor_target_values(target_column) {
            return Ok((values, Some(column_idx)));
        }
        let values = extract_targets(&self.loaded_properties, &target_column.name)?;
        let drop_idx = self
            .feature_matrix
            .as_ref()
            .and_then(|matrix| {
                matrix
                    .feature_names
                    .iter()
                    .position(|name| name == &target_column.name)
            });
        Ok((values, drop_idx))
    }

    /// Implements target selection popup.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn open_target_selection_popup(&mut self) {
        if self.target_columns.is_empty() {
            self.status_message = "No target columns available yet.".to_string();
            return;
        }
        let selected_idx = self
            .target_columns
            .iter()
            .position(|target| self.selected_target_columns.contains(target.key.as_str()))
            .or_else(|| Some(0));
        self.target_column_list_state.select(selected_idx);
        self.state = AppState::TargetSelectionOpen;
    }

    /// Implements selected target column.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn toggle_selected_target_column(&mut self) {
        let Some(selected) = self.target_column_list_state.selected() else {
            return;
        };
        let Some(target) = self.target_columns.get(selected) else {
            return;
        };
        if self.selected_target_columns.contains(target.key.as_str()) {
            self.selected_target_columns.remove(target.key.as_str());
        } else {
            self.selected_target_columns.insert(target.key.clone());
        }
        self.refresh_target_overlay_cache();
    }

    /// Sets all target columns.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn set_all_target_columns(&mut self, enabled: bool) {
        self.selected_target_columns.clear();
        if enabled {
            for target in &self.target_columns {
                self.selected_target_columns.insert(target.key.clone());
            }
        }
        self.refresh_target_overlay_cache();
    }

    /// Implements target selection key.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn handle_target_selection_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Enter => {
                self.refresh_target_overlay_cache();
                if self.fallback_stratified_when_target_unavailable() {
                    self.status_message =
                        "Target selection updated. Stratified split requires valid target, so split reverted to Random.".to_string();
                } else {
                    self.status_message = format!(
                        "Target selection updated ({} selected).",
                        self.selected_target_columns.len()
                    );
                }
                self.state = AppState::Idle;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                let len = self.target_columns.len();
                if len == 0 {
                    return;
                }
                let current = self.target_column_list_state.selected().unwrap_or(0);
                let next = if current == 0 { len - 1 } else { current - 1 };
                self.target_column_list_state.select(Some(next));
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let len = self.target_columns.len();
                if len == 0 {
                    return;
                }
                let current = self.target_column_list_state.selected().unwrap_or(0);
                self.target_column_list_state
                    .select(Some((current + 1) % len));
            }
            KeyCode::Char(' ') => {
                self.toggle_selected_target_column();
            }
            KeyCode::Char('a') => {
                self.set_all_target_columns(true);
            }
            KeyCode::Char('n') => {
                self.set_all_target_columns(false);
            }
            _ => {}
        }
    }

    /// Checks whether valid target for supervised.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn has_valid_target_for_supervised(&self) -> bool {
        if self.selected_target_columns.is_empty() {
            return false;
        }
        let Some(values) = &self.target_overlay_values else {
            return false;
        };
        let Some(matrix) = &self.feature_matrix else {
            return false;
        };
        values.len() == matrix.rows
    }

    /// Checks whether target for stratified split.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn has_target_for_stratified_split(&self) -> bool {
        if self.selected_target_columns.is_empty() {
            return false;
        }
        if self.target_overlay_error.is_some() {
            return false;
        }
        let Some(values) = &self.target_overlay_values else {
            return false;
        };
        if let Some(matrix) = &self.feature_matrix {
            return values.len() == matrix.rows;
        }
        true
    }

    /// Implements stratified when target unavailable.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn fallback_stratified_when_target_unavailable(&mut self) -> bool {
        if self.training_config.split_strategy == SplitStrategy::Stratified
            && !self.has_target_for_stratified_split()
        {
            self.training_config.split_strategy = SplitStrategy::Random;
            return true;
        }
        false
    }

    /// Implements algorithm.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn active_algorithm(&self) -> MlAlgorithm {
        match self.selected_ml_mode {
            MlMode::AutoQsar => MlAlgorithm::AutoQsar,
            MlMode::Art2a => MlAlgorithm::AutoQsar,
            MlMode::Manual => self.manual_algo_selection,
        }
    }

    /// Implements param page count.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn ml_param_page_count(&self) -> usize {
        match self.selected_ml_mode {
            MlMode::AutoQsar | MlMode::Art2a => 1,
            MlMode::Manual => match self.active_algorithm() {
                MlAlgorithm::RandomForest | MlAlgorithm::ElasticNet => 2,
                _ => 1,
            },
        }
    }

    /// Implements ml param page.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn cycle_ml_param_page(&mut self, delta: isize) {
        let page_count = self.ml_param_page_count();
        if page_count <= 1 {
            self.ml_param_page = 0;
            self.status_message = "Current algorithm exposes a single parameter page.".to_string();
            return;
        }
        let next = (self.ml_param_page as isize + delta).rem_euclid(page_count as isize) as usize;
        self.ml_param_page = next;
        self.status_message = format!("Hyperparameter page {}/{}.", next + 1, page_count);
    }

    /// Implements param entries.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn ml_param_entries(&self) -> Vec<MlParamEntry> {
        let page_count = self.ml_param_page_count();
        let page = self.ml_param_page.min(page_count.saturating_sub(1));

        let mut entries = vec![
            MlParamEntry {
                hotkey: '1',
                key: MlParamKey::CvFolds,
                label: "CV folds",
                value: self.training_config.cv_folds.to_string(),
            },
            MlParamEntry {
                hotkey: '2',
                key: MlParamKey::TestFraction,
                label: "Test fraction",
                value: format!("{:.2}", self.training_config.test_fraction),
            },
            MlParamEntry {
                hotkey: '3',
                key: MlParamKey::Seed,
                label: "Seed",
                value: self.training_config.seed.to_string(),
            },
        ];

        if self.selected_ml_mode == MlMode::Art2a {
            entries.push(MlParamEntry {
                hotkey: '4',
                key: MlParamKey::Art2aVigilance,
                label: "ART2-A rho",
                value: format!("{:.2}", self.art2a_config.vigilance),
            });
            entries.push(MlParamEntry {
                hotkey: '5',
                key: MlParamKey::Art2aLearningRate,
                label: "ART2-A eta",
                value: format!("{:.2}", self.art2a_config.learning_rate),
            });
            entries.push(MlParamEntry {
                hotkey: '6',
                key: MlParamKey::Art2aMaxIter,
                label: "ART2-A max iter",
                value: self.art2a_config.max_iter.to_string(),
            });
            return entries;
        }

        if self.selected_ml_mode != MlMode::Manual {
            return entries;
        }

        match self.active_algorithm() {
            MlAlgorithm::AutoQsar => {}
            MlAlgorithm::RandomForest => {
                if page == 0 {
                    entries.push(MlParamEntry {
                        hotkey: '4',
                        key: MlParamKey::RfTrees,
                        label: "RF trees",
                        value: self.training_config.rf_n_estimators.to_string(),
                    });
                    entries.push(MlParamEntry {
                        hotkey: '5',
                        key: MlParamKey::RfMaxDepth,
                        label: "RF max depth",
                        value: self.training_config.rf_max_depth.to_string(),
                    });
                    entries.push(MlParamEntry {
                        hotkey: '6',
                        key: MlParamKey::RfMinSamplesSplit,
                        label: "RF min split",
                        value: self.training_config.rf_min_samples_split.to_string(),
                    });
                } else {
                    entries.push(MlParamEntry {
                        hotkey: '4',
                        key: MlParamKey::RfMaxFeaturesFraction,
                        label: "RF max feat frac",
                        value: format!("{:.3}", self.training_config.rf_max_features_fraction),
                    });
                }
            }
            MlAlgorithm::Knn => {
                entries.push(MlParamEntry {
                    hotkey: '4',
                    key: MlParamKey::KnnK,
                    label: "k-NN k",
                    value: self.training_config.knn_k.to_string(),
                });
            }
            MlAlgorithm::Pls => {
                entries.push(MlParamEntry {
                    hotkey: '4',
                    key: MlParamKey::PlsComponents,
                    label: "PLS components",
                    value: self.training_config.pls_components.to_string(),
                });
            }
            MlAlgorithm::ElasticNet => {
                if page == 0 {
                    entries.push(MlParamEntry {
                        hotkey: '4',
                        key: MlParamKey::EnAlpha,
                        label: "EN alpha",
                        value: format!("{:.6}", self.training_config.en_alpha),
                    });
                    entries.push(MlParamEntry {
                        hotkey: '5',
                        key: MlParamKey::EnL1Ratio,
                        label: "EN l1_ratio",
                        value: format!("{:.4}", self.training_config.en_l1_ratio),
                    });
                    entries.push(MlParamEntry {
                        hotkey: '6',
                        key: MlParamKey::EnMaxIter,
                        label: "EN max iter",
                        value: self.training_config.en_max_iter.to_string(),
                    });
                } else {
                    entries.push(MlParamEntry {
                        hotkey: '4',
                        key: MlParamKey::EnTol,
                        label: "EN tol",
                        value: format!("{:.8}", self.training_config.en_tol),
                    });
                }
            }
        }

        entries
    }

    /// Implements editing ml param.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn begin_editing_ml_param(&mut self, key: MlParamKey) {
        let value = match key {
            MlParamKey::CvFolds => self.training_config.cv_folds.to_string(),
            MlParamKey::TestFraction => format!("{:.2}", self.training_config.test_fraction),
            MlParamKey::Seed => self.training_config.seed.to_string(),
            MlParamKey::Art2aVigilance => format!("{:.2}", self.art2a_config.vigilance),
            MlParamKey::Art2aLearningRate => format!("{:.2}", self.art2a_config.learning_rate),
            MlParamKey::Art2aMaxIter => self.art2a_config.max_iter.to_string(),
            MlParamKey::KnnK => self.training_config.knn_k.to_string(),
            MlParamKey::RfTrees => self.training_config.rf_n_estimators.to_string(),
            MlParamKey::RfMaxDepth => self.training_config.rf_max_depth.to_string(),
            MlParamKey::RfMinSamplesSplit => self.training_config.rf_min_samples_split.to_string(),
            MlParamKey::RfMaxFeaturesFraction => {
                format!("{:.3}", self.training_config.rf_max_features_fraction)
            }
            MlParamKey::PlsComponents => self.training_config.pls_components.to_string(),
            MlParamKey::EnAlpha => format!("{:.6}", self.training_config.en_alpha),
            MlParamKey::EnL1Ratio => format!("{:.4}", self.training_config.en_l1_ratio),
            MlParamKey::EnMaxIter => self.training_config.en_max_iter.to_string(),
            MlParamKey::EnTol => format!("{:.8}", self.training_config.en_tol),
        };
        self.editing_ml_param = Some(key);
        self.ml_param_input = Input::default().with_value(value);
    }

    /// Implements editing ml param.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn commit_editing_ml_param(&mut self) {
        let Some(param) = self.editing_ml_param else {
            return;
        };
        let raw = self.ml_param_input.value().trim();
        if raw.is_empty() {
            self.set_error("Hyperparameter value cannot be empty.".to_string());
            return;
        }
        let parse_result = match param {
            MlParamKey::CvFolds => raw.parse::<usize>().ok().filter(|v| *v >= 2).map(|v| {
                self.training_config.cv_folds = v;
            }),
            MlParamKey::TestFraction => raw
                .parse::<f64>()
                .ok()
                .filter(|v| *v > 0.0 && *v < 1.0)
                .map(|v| {
                    self.training_config.test_fraction = v;
                }),
            MlParamKey::Seed => raw.parse::<u64>().ok().map(|v| {
                self.training_config.seed = v;
            }),
            MlParamKey::Art2aVigilance => raw
                .parse::<f64>()
                .ok()
                .filter(|v| *v > 0.0 && *v < 1.0)
                .map(|v| {
                    self.art2a_config.vigilance = v;
                }),
            MlParamKey::Art2aLearningRate => raw
                .parse::<f64>()
                .ok()
                .filter(|v| *v > 0.0 && *v <= 1.0)
                .map(|v| {
                    self.art2a_config.learning_rate = v;
                }),
            MlParamKey::Art2aMaxIter => raw.parse::<usize>().ok().filter(|v| *v >= 1).map(|v| {
                self.art2a_config.max_iter = v;
            }),
            MlParamKey::KnnK => raw.parse::<usize>().ok().filter(|v| *v >= 1).map(|v| {
                self.training_config.knn_k = v;
            }),
            MlParamKey::RfTrees => raw.parse::<usize>().ok().filter(|v| *v >= 1).map(|v| {
                self.training_config.rf_n_estimators = v;
            }),
            MlParamKey::RfMaxDepth => raw.parse::<usize>().ok().filter(|v| *v >= 1).map(|v| {
                self.training_config.rf_max_depth = v;
            }),
            MlParamKey::RfMinSamplesSplit => {
                raw.parse::<usize>().ok().filter(|v| *v >= 2).map(|v| {
                    self.training_config.rf_min_samples_split = v;
                })
            }
            MlParamKey::RfMaxFeaturesFraction => raw
                .parse::<f64>()
                .ok()
                .filter(|v| *v > 0.0 && *v <= 1.0)
                .map(|v| {
                    self.training_config.rf_max_features_fraction = v;
                }),
            MlParamKey::PlsComponents => raw.parse::<usize>().ok().filter(|v| *v >= 1).map(|v| {
                self.training_config.pls_components = v;
            }),
            MlParamKey::EnAlpha => raw.parse::<f64>().ok().filter(|v| *v > 0.0).map(|v| {
                self.training_config.en_alpha = v;
            }),
            MlParamKey::EnL1Ratio => raw
                .parse::<f64>()
                .ok()
                .filter(|v| *v >= 0.0 && *v <= 1.0)
                .map(|v| {
                    self.training_config.en_l1_ratio = v;
                }),
            MlParamKey::EnMaxIter => raw.parse::<usize>().ok().filter(|v| *v >= 1).map(|v| {
                self.training_config.en_max_iter = v;
            }),
            MlParamKey::EnTol => raw.parse::<f64>().ok().filter(|v| *v > 0.0).map(|v| {
                self.training_config.en_tol = v;
            }),
        };

        if parse_result.is_none() {
            self.set_error(format!(
                "Invalid value '{}' for parameter '{}'.",
                raw,
                ml_param_label(param)
            ));
            return;
        }

        self.editing_ml_param = None;
        self.ml_param_input = Input::default();
        self.status_message = "Hyperparameter updated.".to_string();
    }

    /// Sets error.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn set_error(&mut self, message: String) {
        self.progress_percent = 0.0;
        self.state = AppState::ErrorPopup(message);
    }

    /// Implements computing.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn begin_computing(&mut self, message: String, job: WorkerJobKind) {
        self.progress_percent = -1.0;
        self.cancel_requested = false;
        self.worker_cancel_flag
            .store(false, AtomicOrdering::Relaxed);
        self.active_job = Some(job);
        self.state = AppState::Computing { message, job };
    }

    /// Implements worker.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn send_worker(&mut self, request: WorkerRequest) {
        if let Err(error) = self.worker_tx.send(request) {
            self.set_error(format!("Worker channel error: {}", error));
        }
    }

    /// Implements descriptor error and continue.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn log_descriptor_error_and_continue(&mut self, message: &str) {
        if let Err(log_error) = append_descriptor_error_log(message) {
            self.status_message = format!(
                "Descriptor calculation failed; logging also failed ({}).",
                log_error
            );
            self.progress_percent = 0.0;
            self.state = AppState::Idle;
            return;
        }
        self.status_message = format!(
            "Descriptor calculation failed; details logged to {}.",
            DESCRIPTOR_ERROR_LOG_PATH
        );
        self.progress_percent = 0.0;
        self.cancel_requested = false;
        self.active_job = None;
        self.state = AppState::Idle;
    }

    /// Implements tab.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn next_tab(&mut self) {
        self.active_tab = (self.active_tab + 1) % TAB_TITLES.len();
    }

    /// Implements tab.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn prev_tab(&mut self) {
        if self.active_tab == 0 {
            self.active_tab = TAB_TITLES.len() - 1;
        } else {
            self.active_tab -= 1;
        }
    }

    /// Updates descriptor selection.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn update_descriptor_selection(&mut self, delta: isize) {
        let len = self.available_descriptors.len();
        if len == 0 {
            return;
        }
        let current = self.descriptor_list_state.selected().unwrap_or(0) as isize;
        let next = (current + delta).rem_euclid(len as isize) as usize;
        self.descriptor_list_state.select(Some(next));
    }

    /// Implements selected descriptor.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn toggle_selected_descriptor(&mut self) {
        let Some(index) = self.descriptor_list_state.selected() else {
            return;
        };
        if let Some((_, enabled)) = self.available_descriptors.get_mut(index) {
            *enabled = !*enabled;
        }
    }

    /// Sets all descriptors.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn set_all_descriptors(&mut self, enabled: bool) {
        for (_, flag) in &mut self.available_descriptors {
            *flag = enabled;
        }
    }

    /// Applies the curated fast-ML descriptor preset.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn apply_fast_ml_descriptor_preset(&mut self) {
        let preset = BatchDescriptor::fast_ml_preset();
        for (descriptor, enabled) in &mut self.available_descriptors {
            *enabled = preset.contains(descriptor);
        }
        self.status_message = format!(
            "Applied Fast ML descriptor preset ({} descriptors, {} values/mol).",
            preset.len(),
            self.selected_descriptor_value_count()
        );
    }

    /// Implements file browser.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn open_file_browser(&mut self, purpose: BrowserPurpose) {
        let fallback_dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let start_dir = match purpose {
            BrowserPurpose::Dataset => self
                .loaded_from_path
                .as_ref()
                .and_then(|p| p.parent().map(Path::to_path_buf))
                .unwrap_or(fallback_dir),
            BrowserPurpose::Prediction => self
                .prediction_source
                .as_ref()
                .and_then(|p| p.parent().map(Path::to_path_buf))
                .unwrap_or(fallback_dir),
            BrowserPurpose::ModelImport => self
                .model_bundle_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or(fallback_dir),
        };

        match read_browser_entries(&start_dir, purpose) {
            Ok(entries) => {
                self.browser_purpose = purpose;
                self.browser_entries = entries;
                let selected = (!self.browser_entries.is_empty()).then_some(0);
                self.browser_list_state.select(selected);
                self.state = AppState::FileBrowserOpen(start_dir);
            }
            Err(error) => self.set_error(format!("Failed to open file browser: {}", error)),
        }
    }

    /// Implements browser to.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn navigate_browser_to(&mut self, dir: PathBuf) {
        match read_browser_entries(&dir, self.browser_purpose) {
            Ok(entries) => {
                self.browser_entries = entries;
                let selected = (!self.browser_entries.is_empty()).then_some(0);
                self.browser_list_state.select(selected);
                self.state = AppState::FileBrowserOpen(dir);
            }
            Err(error) => self.set_error(format!("Failed to read '{}': {}", dir.display(), error)),
        }
    }

    /// Implements dataset load.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_dataset_load(&mut self, path: PathBuf) {
        self.begin_computing(
            format!("Loading molecules from '{}'", path.display()),
            WorkerJobKind::DatasetLoad,
        );
        self.send_worker(WorkerRequest::LoadDataset(path));
    }

    /// Implements descriptor calculation.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_descriptor_calculation(&mut self) {
        if self.loaded_molecules.is_empty() {
            self.set_error(
                "No molecules loaded. Press 'f' to load a .csv or .smi file first.".to_string(),
            );
            return;
        }

        let descriptors = self.selected_descriptors();
        if descriptors.is_empty() {
            self.set_error(
                "No descriptors selected. Toggle at least one descriptor on.".to_string(),
            );
            return;
        }

        self.begin_computing(
            DESCRIPTOR_COMPUTING_MESSAGE.to_string(),
            WorkerJobKind::DescriptorCalculation,
        );
        self.send_worker(WorkerRequest::CalculateDescriptors {
            molecules: self.loaded_molecules.clone(),
            descriptors,
        });
    }

    /// Implements descriptor cancellation.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_job_cancellation(&mut self) {
        let Some(job) = self.active_job else {
            self.status_message = "No active worker job to cancel.".to_string();
            return;
        };
        if !job.is_cancelable() {
            self.status_message = format!(
                "Active {} is not cancelable; waiting for completion.",
                job.label()
            );
            return;
        }
        if self.cancel_requested {
            self.status_message = format!(
                "{} cancellation already requested; waiting for worker to stop.",
                job.label()
            );
            return;
        }
        self.cancel_requested = true;
        self.worker_cancel_flag.store(true, AtomicOrdering::Relaxed);
        self.status_message = format!("Cancelling {}...", job.label());
    }

    /// Implements variance filter.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_variance_filter(&mut self) {
        let payload = {
            let Some(matrix) = &self.feature_matrix else {
                self.set_error(
                    "No feature matrix available. Calculate descriptors first.".to_string(),
                );
                return;
            };
            if matrix.cols == 0 {
                self.set_error("Feature matrix has zero columns.".to_string());
                return;
            }
            matrix.to_payload()
        };
        self.begin_computing(
            format!(
                "Applying variance threshold ({:.2e})...",
                DEFAULT_VARIANCE_THRESHOLD
            ),
            WorkerJobKind::VarianceFilter,
        );
        self.send_worker(WorkerRequest::ApplyVarianceFilter {
            matrix: payload,
            threshold: DEFAULT_VARIANCE_THRESHOLD,
        });
    }

    /// Implements correlation filter.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_correlation_filter(&mut self) {
        let payload = {
            let Some(matrix) = &self.feature_matrix else {
                self.set_error(
                    "No feature matrix available. Calculate descriptors first.".to_string(),
                );
                return;
            };
            if matrix.cols <= 1 {
                self.set_error("Correlation filter needs at least 2 features.".to_string());
                return;
            }
            matrix.to_payload()
        };
        self.begin_computing(
            format!(
                "Applying correlation filter (|r| > {:.2})...",
                DEFAULT_CORRELATION_THRESHOLD
            ),
            WorkerJobKind::CorrelationFilter,
        );
        self.send_worker(WorkerRequest::ApplyCorrelationFilter {
            matrix: payload,
            threshold: DEFAULT_CORRELATION_THRESHOLD,
        });
    }

    /// Implements export feature matrix.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_export_feature_matrix(&mut self) {
        let payload = {
            let Some(matrix) = &self.feature_matrix else {
                self.set_error("No feature matrix available to export.".to_string());
                return;
            };
            if matrix.rows == 0 {
                self.set_error("Feature matrix has zero rows.".to_string());
                return;
            }
            if self.loaded_smiles.len() < matrix.rows {
                self.set_error(format!(
                    "Cannot export features: {} SMILES entries for {} matrix rows.",
                    self.loaded_smiles.len(),
                    matrix.rows
                ));
                return;
            }
            matrix.to_payload()
        };
        let path = if let Some(source) = &self.loaded_from_path {
            let stem = source
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "features".to_string());
            source.with_file_name(format!("{}_features.csv", stem))
        } else {
            self.feature_export_path.clone()
        };
        self.begin_computing(
            format!("Exporting feature matrix to '{}'...", path.display()),
            WorkerJobKind::FeatureMatrixExport,
        );
        self.send_worker(WorkerRequest::ExportFeatureMatrix {
            path,
            smiles: self.loaded_smiles.clone(),
            matrix: payload,
        });
    }

    /// Implements training.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_training(&mut self) {
        self.refresh_target_overlay_cache();
        let Some(feature_matrix) = self.feature_matrix.as_ref() else {
            self.set_error("Calculate descriptors first in Tab 2 (Descriptor).".to_string());
            return;
        };
        let matrix_rows = feature_matrix.rows;
        let feature_cols = feature_matrix.cols;

        let selected_targets = self.selected_targets_in_order();
        if selected_targets.is_empty() {
            self.set_error(
                "No target columns selected. Press 't' to choose one or more targets.".to_string(),
            );
            return;
        }

        if self.training_config.split_strategy == SplitStrategy::Stratified
            && !self.has_valid_target_for_supervised()
        {
            self.set_error(
                "Stratified split requires a valid target aligned with feature rows.".to_string(),
            );
            return;
        }

        let mut target_series = Vec::with_capacity(selected_targets.len());
        let mut target_drop_indices = BTreeSet::new();
        for target in selected_targets {
            let (values, drop_idx) = match self.resolve_training_target_values(&target) {
                Ok(values) => values,
                Err(error) => {
                    self.set_error(format!(
                        "Target '{}' validation failed: {}",
                        target.name, error
                    ));
                    return;
                }
            };
            if let Some(idx) = drop_idx {
                target_drop_indices.insert(idx);
            }
            if values.len() != matrix_rows {
                self.set_error(format!(
                    "Target '{}' length mismatch: {} targets for {} feature rows.",
                    target.name,
                    values.len(),
                    matrix_rows
                ));
                return;
            }
            target_series.push((target.output_name(), values));
        }
        if self.loaded_smiles.len() != matrix_rows {
            self.set_error(format!(
                "SMILES row mismatch: {} smiles for {} feature rows.",
                self.loaded_smiles.len(),
                matrix_rows
            ));
            return;
        }
        let matrix_payload = if target_drop_indices.is_empty() {
            feature_matrix.to_payload()
        } else {
            let keep = (0..feature_cols)
                .filter(|idx| !target_drop_indices.contains(idx))
                .collect::<Vec<_>>();
            if keep.is_empty() {
                self.set_error(
                    "Cannot train model: selected targets remove all descriptor features."
                        .to_string(),
                );
                return;
            }
            match select_feature_columns(feature_matrix.to_payload(), &keep) {
                Ok(payload) => payload,
                Err(error) => {
                    self.set_error(format!(
                        "Failed to prepare training matrix after dropping target descriptor columns: {}",
                        error
                    ));
                    return;
                }
            }
        };

        self.training_config.algorithm = self.active_algorithm();
        self.begin_computing(
            format!(
                "Training {} on {} target(s) with {}...",
                self.training_config.algorithm.label(),
                target_series.len(),
                self.training_config.scaler.label()
            ),
            WorkerJobKind::ModelTraining,
        );
        self.send_worker(WorkerRequest::TrainModel {
            matrix: matrix_payload,
            target_series,
            smiles: self.loaded_smiles.clone(),
            config: self.training_config.clone(),
            target_columns: self.target_columns.clone(),
        });
    }

    /// Implements art2a clustering.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_art2a_clustering(&mut self) {
        let (matrix_payload, added_csv_cols) =
            match self.clustering_payload_with_optional_csv_targets() {
                Ok(result) => result,
                Err(error) => {
                    self.set_error(error);
                    return;
                }
            };
        self.begin_computing(
            format!(
                "Running ART2-A clustering (rho={:.2}, eta={:.2}, scaler={}, csv extras={})...",
                self.art2a_config.vigilance,
                self.art2a_config.learning_rate,
                self.training_config.scaler.label(),
                if added_csv_cols > 0 {
                    format!("+{} numeric columns", added_csv_cols)
                } else if self.include_csv_numeric_in_clustering {
                    "enabled (none available)".to_string()
                } else {
                    "off".to_string()
                }
            ),
            WorkerJobKind::Art2aClustering,
        );
        self.send_worker(WorkerRequest::RunArt2aClustering {
            matrix: matrix_payload,
            config: self.art2a_config.clone(),
            scaler: self.training_config.scaler,
        });
    }

    /// Implements prediction.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_prediction(&mut self, path: PathBuf) {
        if self.pipeline_bundle_bytes.is_none() && self.art2a_model_bytes.is_none() {
            self.set_error(
                "No supervised model or ART2-A model available. Train model (Enter) or run ART2-A (g) first.".to_string(),
            );
            return;
        }
        let selection = match self.resolve_active_model_selection() {
            Ok(selection) => selection,
            Err(error) => {
                self.set_error(error);
                return;
            }
        };
        if let Some(warning) = &selection.warning {
            self.status_message = warning.clone();
        }
        let layout = selection.layout;

        self.begin_computing(
            format!("Running prediction for '{}'...", path.display()),
            WorkerJobKind::Prediction,
        );
        self.send_worker(WorkerRequest::PredictFile {
            path,
            pipeline_bundle_bytes: selection.pipeline_bundle_bytes,
            art2a_model_bytes: selection.art2a_model_bytes,
            descriptors: layout.descriptors,
            selected_feature_indices: layout.selected_feature_indices,
            expected_feature_names: layout.expected_feature_names,
        });
    }

    /// Implements export predictions.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_export_predictions(&mut self) {
        if self.prediction_rows.is_empty() {
            self.set_error("No predictions available to export.".to_string());
            return;
        }

        let export_path = if let Some(source) = &self.prediction_source {
            let stem = source
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "predictions".to_string());
            source.with_file_name(format!("{}_predictions.csv", stem))
        } else {
            self.prediction_export_path.clone()
        };

        self.begin_computing(
            format!("Exporting predictions to '{}'...", export_path.display()),
            WorkerJobKind::PredictionExport,
        );
        self.send_worker(WorkerRequest::ExportPredictions {
            path: export_path,
            target_names: self.prediction_target_names.clone(),
            rows: self.prediction_rows.clone(),
        });
    }

    /// Exports ART2-A cluster assignments with source SMILES.
    fn request_export_clusters(&mut self) {
        let rows = match build_cluster_export_rows(
            &self.prediction_rows,
            &self.prediction_target_names,
            &self.cluster_labels,
            &self.loaded_smiles,
        ) {
            Ok(rows) => rows,
            Err(error) => {
                self.set_error(error);
                return;
            }
        };

        let export_path = if let Some(source) = &self.prediction_source {
            let stem = source
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "clusters".to_string());
            source.with_file_name(format!("{}_clusters.csv", stem))
        } else if let Some(source) = &self.loaded_from_path {
            let stem = source
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "clusters".to_string());
            source.with_file_name(format!("{}_clusters.csv", stem))
        } else {
            self.cluster_export_path.clone()
        };

        self.begin_computing(
            format!(
                "Exporting ART2-A clusters to '{}'...",
                export_path.display()
            ),
            WorkerJobKind::ClusterExport,
        );
        self.send_worker(WorkerRequest::ExportClusters {
            path: export_path,
            rows,
        });
    }

    /// Exports the currently trained model bundle to disk.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_export_model_bundle(&mut self) {
        if self.pipeline_bundle_bytes.is_none() && self.art2a_model_bytes.is_none() {
            self.set_error(
                "No trained model available to export. Train model (Enter) or run ART2-A (g) first."
                    .to_string(),
            );
            return;
        }
        let selection = match self.resolve_active_model_selection() {
            Ok(selection) => selection,
            Err(error) => {
                self.set_error(error);
                return;
            }
        };
        if let Some(warning) = &selection.warning {
            self.status_message = warning.clone();
        }
        let layout = selection.layout;

        let export_path = if let Some(source) = &self.loaded_from_path {
            let stem = source
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "ferrum_model".to_string());
            source.with_file_name(format!("{}_model_bundle.fcbundle", stem))
        } else {
            self.model_bundle_path.clone()
        };

        let bundle = PersistedModelBundle {
            schema_version: MODEL_BUNDLE_SCHEMA_VERSION,
            pipeline_bundle_bytes: selection.pipeline_bundle_bytes,
            art2a_model_bytes: selection.art2a_model_bytes,
            descriptors: layout
                .descriptors
                .iter()
                .map(|descriptor| descriptor.name().to_string())
                .collect(),
            selected_feature_indices: layout.selected_feature_indices,
            expected_feature_names: layout.expected_feature_names,
        };

        self.begin_computing(
            format!("Exporting model bundle to '{}'...", export_path.display()),
            WorkerJobKind::ModelBundleExport,
        );
        self.send_worker(WorkerRequest::ExportModelBundle {
            path: export_path,
            bundle,
        });
    }

    /// Opens the file browser for model bundle import.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_import_model_bundle(&mut self) {
        self.open_file_browser(BrowserPurpose::ModelImport);
    }

    /// Imports a persisted model bundle from disk.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn request_import_model_bundle_from_path(&mut self, path: PathBuf) {
        self.begin_computing(
            format!("Importing model bundle from '{}'...", path.display()),
            WorkerJobKind::ModelBundleImport,
        );
        self.send_worker(WorkerRequest::ImportModelBundle { path });
    }

    /// Implements file browser key.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn handle_file_browser_key(&mut self, key: KeyEvent) {
        let current_dir = match &self.state {
            AppState::FileBrowserOpen(path) => path.clone(),
            _ => return,
        };

        match key.code {
            KeyCode::Esc => {
                self.state = AppState::Idle;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                let len = self.browser_entries.len();
                if len == 0 {
                    return;
                }
                let current = self.browser_list_state.selected().unwrap_or(0);
                let next = if current == 0 { len - 1 } else { current - 1 };
                self.browser_list_state.select(Some(next));
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let len = self.browser_entries.len();
                if len == 0 {
                    return;
                }
                let current = self.browser_list_state.selected().unwrap_or(0);
                self.browser_list_state.select(Some((current + 1) % len));
            }
            KeyCode::Backspace => {
                if let Some(parent) = current_dir.parent() {
                    self.navigate_browser_to(parent.to_path_buf());
                }
            }
            KeyCode::Enter => {
                let Some(selected) = self.browser_list_state.selected() else {
                    return;
                };
                let Some(path) = self.browser_entries.get(selected).cloned() else {
                    return;
                };

                if path.is_dir() {
                    self.navigate_browser_to(path);
                    return;
                }

                // Close modal and dispatch corresponding background work.
                self.state = AppState::Idle;
                match self.browser_purpose {
                    BrowserPurpose::Dataset => self.request_dataset_load(path),
                    BrowserPurpose::Prediction => self.request_prediction(path),
                    BrowserPurpose::ModelImport => self.request_import_model_bundle_from_path(path),
                }
            }
            _ => {}
        }
    }

    /// Implements import key.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn handle_import_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('f') => self.open_file_browser(BrowserPurpose::Dataset),
            _ => {}
        }
    }

    /// Implements descriptor key.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn handle_descriptor_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.update_descriptor_selection(-1),
            KeyCode::Down | KeyCode::Char('j') => self.update_descriptor_selection(1),
            KeyCode::Char(' ') => self.toggle_selected_descriptor(),
            KeyCode::Char('a') => self.set_all_descriptors(true),
            KeyCode::Char('n') => self.set_all_descriptors(false),
            KeyCode::Char('r') => self.apply_fast_ml_descriptor_preset(),
            KeyCode::Char('f') => self.open_file_browser(BrowserPurpose::Dataset),
            KeyCode::Char('v') => self.request_variance_filter(),
            KeyCode::Char('c') => self.request_correlation_filter(),
            KeyCode::Char('e') => self.request_export_feature_matrix(),
            KeyCode::Enter => self.request_descriptor_calculation(),
            _ => {}
        }
    }

    /// Implements model select key.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn handle_model_select_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('t') => {
                self.open_target_selection_popup();
            }
            KeyCode::Char('m') => {
                self.selected_ml_mode = self.selected_ml_mode.toggle();
                self.ml_param_page = 0;
                self.status_message = format!("ML mode set to {}.", self.selected_ml_mode.label());
            }
            KeyCode::Char('a') => {
                if self.selected_ml_mode != MlMode::Manual {
                    self.status_message =
                        "Switch to Manual mode ('m') to choose a manual model.".to_string();
                    return;
                }
                self.manual_algo_selection = self.manual_algo_selection.manual_cycle(1);
                self.status_message = format!(
                    "Manual algorithm set to {}.",
                    self.manual_algo_selection.label()
                );
            }
            KeyCode::Char('A') => {
                if self.selected_ml_mode != MlMode::Manual {
                    self.status_message =
                        "Switch to Manual mode ('m') to choose a manual model.".to_string();
                    return;
                }
                self.manual_algo_selection = self.manual_algo_selection.manual_cycle(-1);
                self.status_message = format!(
                    "Manual algorithm set to {}.",
                    self.manual_algo_selection.label()
                );
            }
            KeyCode::Char('z') => {
                self.training_config.scaler = self.training_config.scaler.toggle();
                self.status_message =
                    format!("Scaler set to {}.", self.training_config.scaler.label());
            }
            KeyCode::Char('x') => {
                self.training_config.split_strategy = self.training_config.split_strategy.cycle();
                if self.training_config.split_strategy == SplitStrategy::Stratified
                    && !self.has_target_for_stratified_split()
                {
                    self.training_config.split_strategy = SplitStrategy::Scaffold;
                    self.status_message =
                        "Stratified split requires valid target; switched to Scaffold.".to_string();
                    return;
                }
                self.status_message = format!(
                    "Split strategy set to {}.",
                    self.training_config.split_strategy.label()
                );
            }
            KeyCode::Char('u') => {
                self.request_import_model_bundle();
            }
            KeyCode::Char('b') => {
                self.request_export_model_bundle();
            }
            _ => {}
        }
    }

    /// Implements param key.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn handle_param_key(&mut self, key: KeyEvent) {
        if self.editing_ml_param.is_some() {
            match key.code {
                KeyCode::Esc => {
                    self.editing_ml_param = None;
                    self.ml_param_input = Input::default();
                }
                KeyCode::Enter => self.commit_editing_ml_param(),
                _ => {
                    let _ = self.ml_param_input.handle_event(&Event::Key(key));
                }
            }
            return;
        }

        match key.code {
            KeyCode::Char('[') => {
                self.art2a_config.vigilance = (self.art2a_config.vigilance - 0.05).max(0.05);
                self.status_message = format!(
                    "ART2-A vigilance set to {:.2}.",
                    self.art2a_config.vigilance
                );
            }
            KeyCode::Char(']') => {
                self.art2a_config.vigilance = (self.art2a_config.vigilance + 0.05).min(0.99);
                self.status_message = format!(
                    "ART2-A vigilance set to {:.2}.",
                    self.art2a_config.vigilance
                );
            }
            KeyCode::Char(';') => {
                self.art2a_config.learning_rate =
                    (self.art2a_config.learning_rate - 0.05).max(0.05);
                self.status_message = format!(
                    "ART2-A learning rate set to {:.2}.",
                    self.art2a_config.learning_rate
                );
            }
            KeyCode::Char('\'') => {
                self.art2a_config.learning_rate = (self.art2a_config.learning_rate + 0.05).min(1.0);
                self.status_message = format!(
                    "ART2-A learning rate set to {:.2}.",
                    self.art2a_config.learning_rate
                );
            }
            KeyCode::Char('-') => {
                self.art2a_config.max_iter = self.art2a_config.max_iter.saturating_sub(5).max(1);
                self.status_message =
                    format!("ART2-A max_iter set to {}.", self.art2a_config.max_iter);
            }
            KeyCode::Char('=') => {
                self.art2a_config.max_iter = self.art2a_config.max_iter.saturating_add(5);
                self.status_message =
                    format!("ART2-A max_iter set to {}.", self.art2a_config.max_iter);
            }
            KeyCode::Char(',') => {
                self.cycle_ml_param_page(-1);
            }
            KeyCode::Char('.') => {
                self.cycle_ml_param_page(1);
            }
            KeyCode::Char('1')
            | KeyCode::Char('2')
            | KeyCode::Char('3')
            | KeyCode::Char('4')
            | KeyCode::Char('5')
            | KeyCode::Char('6') => {
                let ch = match key.code {
                    KeyCode::Char(ch) => ch,
                    _ => unreachable!(),
                };
                if let Some(entry) = self
                    .ml_param_entries()
                    .into_iter()
                    .find(|entry| entry.hotkey == ch)
                {
                    self.begin_editing_ml_param(entry.key);
                }
            }
            KeyCode::Enter => {
                self.request_training();
            }
            KeyCode::Char('p') => {
                self.open_file_browser(BrowserPurpose::Prediction);
            }
            KeyCode::Char('s') => {
                self.request_export_predictions();
            }
            KeyCode::Char('K') => {
                self.request_export_clusters();
            }
            KeyCode::Char('b') => {
                self.request_export_model_bundle();
            }
            KeyCode::Char('u') => {
                self.request_import_model_bundle();
            }
            _ => {}
        }
    }

    /// Implements analyze key.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn handle_analyze_key(&mut self, key: KeyEvent) {
        let max_offset = self
            .descriptor_profiles
            .len()
            .saturating_sub(PREVIEW_ROWS.max(1));
        match key.code {
            KeyCode::Enter => match self.selected_ml_mode {
                MlMode::AutoQsar | MlMode::Manual => self.request_training(),
                MlMode::Art2a => {
                    self.status_message =
                        "Mode is ART2-A: press 'g' to run clustering.".to_string();
                }
            },
            KeyCode::Char('g') => {
                self.request_art2a_clustering();
            }
            KeyCode::Char('y') => {
                self.include_csv_numeric_in_clustering = !self.include_csv_numeric_in_clustering;
                self.status_message = format!(
                    "ART2-A CSV numeric features: {}.",
                    if self.include_csv_numeric_in_clustering {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
            }
            KeyCode::Char('p') => {
                self.open_file_browser(BrowserPurpose::Prediction);
            }
            KeyCode::Char('s') => {
                self.request_export_predictions();
            }
            KeyCode::Char('K') => {
                self.request_export_clusters();
            }
            KeyCode::Char('b') => {
                self.request_export_model_bundle();
            }
            KeyCode::Char('u') => {
                self.request_import_model_bundle();
            }
            KeyCode::Char('o') => {
                self.color_mode = self.color_mode.cycle();
                self.status_message =
                    format!("Visualization color mode: {}.", self.color_mode.label());
            }
            KeyCode::Char('[') => {
                self.cycle_importance_target(-1);
            }
            KeyCode::Char(']') => {
                self.cycle_importance_target(1);
            }
            KeyCode::Char(',') => {
                self.cycle_importance_target(-1);
            }
            KeyCode::Char('.') => {
                self.cycle_importance_target(1);
            }
            KeyCode::Char('i') => {
                self.show_feature_importance = !self.show_feature_importance;
                self.status_message = format!(
                    "Feature-importance panel {}.",
                    if self.show_feature_importance {
                        "shown"
                    } else {
                        "hidden"
                    }
                );
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.profile_scroll > 0 {
                    self.profile_scroll -= 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.profile_scroll = (self.profile_scroll + 1).min(max_offset);
            }
            _ => {}
        }
    }

    /// Implements key.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if let KeyCode::Char('q') = key.code {
            return true;
        }

        match self.state.clone() {
            AppState::ErrorPopup(_) => {
                if matches!(key.code, KeyCode::Esc | KeyCode::Enter) {
                    self.state = AppState::Idle;
                }
                return false;
            }
            AppState::TargetSelectionOpen => {
                self.handle_target_selection_key(key);
                return false;
            }
            AppState::FileBrowserOpen(_) => {
                self.handle_file_browser_key(key);
                return false;
            }
            AppState::Computing { .. } => {
                if matches!(key.code, KeyCode::Esc) {
                    self.request_job_cancellation();
                }
                // Keep UI responsive, but suppress workflow actions while worker is busy.
                return false;
            }
            AppState::Idle => {}
        }

        if self.active_tab == 3 && self.editing_ml_param.is_some() {
            self.handle_param_key(key);
            return false;
        }

        match key.code {
            KeyCode::Left | KeyCode::Char('h') => self.prev_tab(),
            KeyCode::Right | KeyCode::Char('l') => self.next_tab(),
            KeyCode::Esc => {
                self.editing_ml_param = None;
                if matches!(self.active_tab, 2 | 3) {
                    self.refresh_target_overlay_cache();
                    if self.fallback_stratified_when_target_unavailable() {
                        self.status_message =
                            "Stratified split requires valid target; split reverted to Random."
                                .to_string();
                    }
                }
            }
            _ => match self.active_tab {
                0 => self.handle_import_key(key),
                1 => self.handle_descriptor_key(key),
                2 => self.handle_model_select_key(key),
                3 => self.handle_param_key(key),
                4 => self.handle_analyze_key(key),
                _ => {}
            },
        }

        false
    }

    /// Implements worker messages.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn drain_worker_messages(&mut self) {
        loop {
            match self.worker_rx.try_recv() {
                Ok(response) => self.apply_worker_response(response),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.set_error("Worker thread disconnected unexpectedly. Restarting...".to_string());
                    self.active_job = None;
                    self.cancel_requested = false;
                    self.state = AppState::Idle;
                    
                    let (tx, rx, flag, _handle) = spawn_worker();
                    self.worker_tx = tx;
                    self.worker_rx = rx;
                    self.worker_cancel_flag = flag;
                    break;
                }
            }
        }
    }

    /// Implements worker response.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
    fn apply_worker_response(&mut self, response: WorkerResponse) {
        match response {
            WorkerResponse::Progress(progress) => {
                self.progress_percent = progress.clamp(0.0, 1.0);
            }
            WorkerResponse::DatasetLoaded {
                path,
                molecules,
                smiles,
                properties,
                failed_rows,
            } => {
                self.progress_percent = 0.0;
                self.cancel_requested = false;
                self.active_job = None;
                self.last_dataset_failed_rows = failed_rows;
                self.loaded_molecules = molecules;
                self.loaded_smiles = smiles;
                self.loaded_properties = properties;
                self.loaded_from_path = Some(path.clone());

                // Reset downstream state because source data changed.
                self.feature_matrix = None;
                self.rebuild_target_columns();
                self.descriptor_profiles.clear();
                self.projection = None;
                self.train_actuals.clear();
                self.training_predictions.clear();
                self.ad_in_domain_flags.clear();
                self.top_feature_importances.clear();
                self.trained_target_summaries.clear();
                self.trained_primary_target = None;
                self.importance_target_index = 0;
                self.trained_pipeline = None;
                self.pipeline_bundle_bytes = None;
                self.pipeline_feature_layout = None;
                self.prediction_rows.clear();
                self.prediction_target_names.clear();
                self.prediction_source = None;
                self.profile_scroll = 0;
                self.last_train_rmse = None;
                self.last_train_mae = None;
                self.last_train_pearson = None;
                self.last_holdout_r2 = None;
                self.last_holdout_rmse = None;
                self.last_holdout_mae = None;
                self.last_holdout_pearson = None;
                self.last_split_label = None;
                self.last_split_train_size = None;
                self.last_split_test_size = None;
                self.cluster_labels.clear();
                self.cluster_sizes.clear();
                self.cluster_summaries.clear();
                self.art2a_model_bytes = None;
                self.art2a_feature_layout = None;
                self.color_mode = VisualizationColorMode::Uniform;
                let split_fallback = self.fallback_stratified_when_target_unavailable();

                self.status_message = format!(
                    "Loaded {} molecules from '{}' ({} failed rows skipped).",
                    self.loaded_molecules.len(),
                    path.display(),
                    failed_rows
                );
                if split_fallback {
                    self.status_message.push_str(
                        " Stratified split requires valid target; split reverted to Random.",
                    );
                }
                self.state = AppState::Idle;
            }
            WorkerResponse::DescriptorsCalculated {
                matrix,
                profiles,
                projection,
            } => {
                self.cancel_requested = false;
                self.active_job = None;
                self.progress_percent = 0.0;
                let rows = matrix.rows;
                let cols = matrix.cols;
                self.feature_matrix = Some(FeatureMatrix::from_payload(matrix));
                self.rebuild_target_columns();
                self.descriptor_profiles = profiles;
                self.projection = projection;
                let split_fallback = self.fallback_stratified_when_target_unavailable();
                self.profile_scroll = 0;

                // New feature set invalidates existing trained model and predictions.
                self.train_actuals.clear();
                self.training_predictions.clear();
                self.ad_in_domain_flags.clear();
                self.top_feature_importances.clear();
                self.trained_target_summaries.clear();
                self.trained_primary_target = None;
                self.importance_target_index = 0;
                self.trained_pipeline = None;
                self.pipeline_bundle_bytes = None;
                self.pipeline_feature_layout = None;
                self.prediction_rows.clear();
                self.prediction_target_names.clear();
                self.prediction_source = None;
                self.last_train_rmse = None;
                self.last_train_mae = None;
                self.last_train_pearson = None;
                self.last_holdout_r2 = None;
                self.last_holdout_rmse = None;
                self.last_holdout_mae = None;
                self.last_holdout_pearson = None;
                self.last_split_label = None;
                self.last_split_train_size = None;
                self.last_split_test_size = None;
                self.cluster_labels.clear();
                self.cluster_sizes.clear();
                self.cluster_summaries.clear();
                self.art2a_model_bytes = None;
                self.art2a_feature_layout = None;
                self.color_mode = VisualizationColorMode::Uniform;

                self.status_message = format!(
                    "Descriptor calculation complete: {} molecules × {} features.",
                    rows, cols
                );
                if split_fallback {
                    self.status_message.push_str(
                        " Stratified split requires valid target; split reverted to Random.",
                    );
                }
                self.state = AppState::Idle;
            }
            WorkerResponse::ModelTrained {
                pipeline_name,
                cv_q2,
                train_r2,
                rmse,
                mae,
                pearson,
                train_actuals,
                ad_in_domain_flags,
                top_feature_importances,
                holdout_r2,
                holdout_rmse,
                holdout_mae,
                holdout_pearson,
                split_label,
                split_train_size,
                split_test_size,
                primary_target,
                primary_target_display_label,
                target_summaries,
                pipeline_bundle_bytes,
                descriptors,
                selected_feature_indices,
                expected_feature_names,
                preview_predictions,
                train_predictions,
            } => match decode_pipeline_bundle(&pipeline_bundle_bytes, &primary_target) {
                Ok(pipeline) => {
                    self.progress_percent = 0.0;
                    self.cancel_requested = false;
                    self.active_job = None;
                    self.trained_pipeline = Some(pipeline);
                    self.trained_target_summaries = target_summaries;
                    self.trained_primary_target = Some(primary_target_display_label.clone());
                    self.importance_target_index = 0;
                    self.pipeline_bundle_bytes = Some(pipeline_bundle_bytes);
                    self.pipeline_feature_layout = Some(ModelFeatureLayout::from_parts(
                        descriptors.clone(),
                        selected_feature_indices.clone(),
                        expected_feature_names.clone(),
                    ));
                    self.train_actuals = train_actuals;
                    self.training_predictions = train_predictions;
                    self.ad_in_domain_flags = ad_in_domain_flags;
                    self.top_feature_importances = top_feature_importances;
                    self.sync_importance_view_from_index();
                    self.last_train_rmse = Some(rmse);
                    self.last_train_mae = Some(mae);
                    self.last_train_pearson = Some(pearson);
                    self.last_holdout_r2 = Some(holdout_r2);
                    self.last_holdout_rmse = Some(holdout_rmse);
                    self.last_holdout_mae = Some(holdout_mae);
                    self.last_holdout_pearson = Some(holdout_pearson);
                    self.last_split_label = Some(split_label.clone());
                    self.last_split_train_size = Some(split_train_size);
                    self.last_split_test_size = Some(split_test_size);
                    if self.feature_matrix.is_none() {
                        self.feature_matrix = Some(FeatureMatrix {
                            rows: 0,
                            cols: expected_feature_names.len(),
                            feature_names: expected_feature_names.clone(),
                            data: Vec::new(),
                            descriptors,
                            selected_feature_indices,
                        });
                        self.rebuild_target_columns();
                    }

                    let preview = preview_predictions
                        .iter()
                        .map(|v| format!("{:.3}", v))
                        .collect::<Vec<_>>()
                        .join(", ");

                    self.status_message = format!(
                        "Training complete for {} target(s). Primary {}: {} (Q2={:.4}, train R2={:.4}, holdout R2={:.4}) [{} split {}:{}]. Preview: [{}]",
                        self.trained_target_summaries.len(),
                        primary_target_display_label,
                        pipeline_name,
                        cv_q2,
                        train_r2,
                        holdout_r2,
                        split_label,
                        split_train_size,
                        split_test_size,
                        preview
                    );
                    self.refresh_target_overlay_cache();
                    self.state = AppState::Idle;
                }
                Err(error) => {
                    self.set_error(format!(
                        "Model trained but pipeline deserialization failed: {}",
                        error
                    ));
                }
            },
            WorkerResponse::PredictionsReady {
                path,
                target_names,
                rows,
                skipped_rows,
                projection,
            } => {
                self.progress_percent = 0.0;
                self.cancel_requested = false;
                self.active_job = None;
                self.prediction_target_names = target_names;
                self.prediction_rows = rows;
                self.prediction_source = Some(path.clone());
                self.projection = projection;
                if let Some(cluster_idx) = self
                    .prediction_target_names
                    .iter()
                    .position(|name| name == "ART2A_ClusterId")
                {
                    let mut labels = Vec::with_capacity(self.prediction_rows.len());
                    let mut valid_labels = true;
                    for row in &self.prediction_rows {
                        match row.predictions.get(cluster_idx).copied() {
                            Some(value) if value.is_finite() && value >= 0.0 => {
                                labels.push(value.round() as usize);
                            }
                            _ => {
                                valid_labels = false;
                                break;
                            }
                        }
                    }

                    if valid_labels {
                        let similarity_idx = self
                            .prediction_target_names
                            .iter()
                            .position(|name| name == "ART2A_Similarity");
                        self.cluster_labels = labels;
                        self.cluster_sizes = cluster_size_histogram(&self.cluster_labels);
                        self.cluster_summaries = cluster_summaries_from_predictions(
                            &self.prediction_rows,
                            cluster_idx,
                            similarity_idx,
                        );
                        self.color_mode = VisualizationColorMode::ClusterIds;
                    } else {
                        self.cluster_labels.clear();
                        self.cluster_sizes.clear();
                        self.cluster_summaries.clear();
                    }
                } else {
                    self.cluster_labels.clear();
                    self.cluster_sizes.clear();
                    self.cluster_summaries.clear();
                }
                self.status_message = format!(
                    "Prediction complete: {} rows × {} output(s) from '{}'. Skipped rows: {}. Press 's' to export.",
                    self.prediction_rows.len(),
                    self.prediction_target_names.len(),
                    path.display(),
                    skipped_rows
                );
                self.state = AppState::Idle;
            }
            WorkerResponse::PredictionsExported { path, count } => {
                self.progress_percent = 0.0;
                self.cancel_requested = false;
                self.active_job = None;
                self.status_message =
                    format!("Exported {} predictions to '{}'.", count, path.display());
                self.state = AppState::Idle;
            }
            WorkerResponse::ClustersExported { path, count } => {
                self.progress_percent = 0.0;
                self.cancel_requested = false;
                self.active_job = None;
                self.cluster_export_path = path.clone();
                self.status_message = format!(
                    "Exported {} ART2-A cluster rows to '{}'.",
                    count,
                    path.display()
                );
                self.state = AppState::Idle;
            }
            WorkerResponse::FeatureFilterApplied {
                matrix,
                profiles,
                projection,
                removed_features,
                filter_label,
            } => {
                self.progress_percent = 0.0;
                self.cancel_requested = false;
                self.active_job = None;
                let rows = matrix.rows;
                let cols = matrix.cols;
                self.feature_matrix = Some(FeatureMatrix::from_payload(matrix));
                self.rebuild_target_columns();
                self.descriptor_profiles = profiles;
                self.projection = projection;
                let split_fallback = self.fallback_stratified_when_target_unavailable();
                self.profile_scroll = 0;
                self.train_actuals.clear();
                self.training_predictions.clear();
                self.ad_in_domain_flags.clear();
                self.top_feature_importances.clear();
                self.trained_target_summaries.clear();
                self.trained_primary_target = None;
                self.importance_target_index = 0;
                self.trained_pipeline = None;
                self.pipeline_bundle_bytes = None;
                self.pipeline_feature_layout = None;
                self.prediction_rows.clear();
                self.prediction_target_names.clear();
                self.prediction_source = None;
                self.last_train_rmse = None;
                self.last_train_mae = None;
                self.last_train_pearson = None;
                self.last_holdout_r2 = None;
                self.last_holdout_rmse = None;
                self.last_holdout_mae = None;
                self.last_holdout_pearson = None;
                self.last_split_label = None;
                self.last_split_train_size = None;
                self.last_split_test_size = None;
                self.cluster_labels.clear();
                self.cluster_sizes.clear();
                self.cluster_summaries.clear();
                self.art2a_model_bytes = None;
                self.art2a_feature_layout = None;
                self.color_mode = VisualizationColorMode::Uniform;
                self.status_message = format!(
                    "{} applied: removed {} feature(s), matrix now {}x{}.",
                    filter_label, removed_features, rows, cols
                );
                if split_fallback {
                    self.status_message.push_str(
                        " Stratified split requires valid target; split reverted to Random.",
                    );
                }
                self.state = AppState::Idle;
            }
            WorkerResponse::ClusteringCompleted {
                labels,
                cluster_sizes,
                summaries,
                art2a_model_bytes,
                projection,
                descriptors,
                selected_feature_indices,
                expected_feature_names,
            } => {
                self.progress_percent = 0.0;
                self.cancel_requested = false;
                self.active_job = None;
                self.cluster_labels = labels;
                self.cluster_sizes = cluster_sizes.clone();
                self.cluster_summaries = summaries;
                self.art2a_model_bytes = Some(art2a_model_bytes);
                self.projection = projection;
                self.art2a_feature_layout = Some(ModelFeatureLayout::from_parts(
                    descriptors.clone(),
                    selected_feature_indices.clone(),
                    expected_feature_names.clone(),
                ));
                if self.feature_matrix.is_none() {
                    self.feature_matrix = Some(FeatureMatrix {
                        rows: 0,
                        cols: expected_feature_names.len(),
                        feature_names: expected_feature_names,
                        data: Vec::new(),
                        descriptors,
                        selected_feature_indices,
                    });
                    self.rebuild_target_columns();
                }
                if self.color_mode == VisualizationColorMode::Uniform {
                    self.color_mode = VisualizationColorMode::ClusterIds;
                }
                let top_summary = self
                    .cluster_summaries
                    .iter()
                    .max_by_key(|summary| summary.size)
                    .map(|summary| {
                        format!(
                            "largest C{} size={} ({:.1}%), mean_sim={:.3}",
                            summary.cluster_id,
                            summary.size,
                            summary.fraction * 100.0,
                            summary.mean_similarity
                        )
                    })
                    .unwrap_or_else(|| "no summary".to_string());
                self.status_message = format!(
                    "ART2-A clustering complete: {} clusters ({})",
                    cluster_sizes.len(),
                    top_summary
                );
                if cluster_sizes.len() == 1 {
                    self.status_message.push_str(
                        " Single cluster found: try higher rho (]) and/or switch scaler (z).",
                    );
                }
                self.state = AppState::Idle;
            }
            WorkerResponse::FeatureMatrixExported { path, rows, cols } => {
                self.progress_percent = 0.0;
                self.cancel_requested = false;
                self.active_job = None;
                self.status_message = format!(
                    "Exported feature matrix ({}x{}) to '{}'.",
                    rows,
                    cols,
                    path.display()
                );
                self.state = AppState::Idle;
            }
            WorkerResponse::ModelBundleExported { path } => {
                self.progress_percent = 0.0;
                self.cancel_requested = false;
                self.active_job = None;
                self.model_bundle_path = path.clone();
                self.status_message = format!("Exported model bundle to '{}'.", path.display());
                self.state = AppState::Idle;
            }
            WorkerResponse::ModelBundleImported { path, bundle } => {
                self.progress_percent = 0.0;
                self.cancel_requested = false;
                self.active_job = None;
                self.model_bundle_path = path.clone();
                self.trained_target_summaries.clear();
                self.trained_primary_target = None;
                self.importance_target_index = 0;
                self.top_feature_importances.clear();
                self.pipeline_bundle_bytes = bundle.pipeline_bundle_bytes;
                self.art2a_model_bytes = bundle.art2a_model_bytes;
                let descriptors = bundle
                    .descriptors
                    .iter()
                    .map(|name| {
                        descriptor_from_name(name).ok_or_else(|| {
                            format!(
                                "Imported model bundle references unknown descriptor '{}'.",
                                name
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>();
                match descriptors {
                    Ok(descriptors) => {
                        let layout = ModelFeatureLayout::from_parts(
                            descriptors.clone(),
                            bundle.selected_feature_indices.clone(),
                            bundle.expected_feature_names.clone(),
                        );
                        self.pipeline_feature_layout =
                            self.pipeline_bundle_bytes.as_ref().map(|_| layout.clone());
                        self.art2a_feature_layout =
                            self.art2a_model_bytes.as_ref().map(|_| layout.clone());
                        self.feature_matrix = Some(FeatureMatrix {
                            rows: 0,
                            cols: bundle.expected_feature_names.len(),
                            feature_names: bundle.expected_feature_names,
                            data: Vec::new(),
                            descriptors,
                            selected_feature_indices: bundle.selected_feature_indices,
                        });
                        self.rebuild_target_columns();
                        self.status_message = format!(
                            "Imported model bundle from '{}'. You can now run prediction ('p').",
                            path.display()
                        );
                        self.state = AppState::Idle;
                    }
                    Err(error) => {
                        self.set_error(error);
                    }
                }
            }
            WorkerResponse::JobCanceled { job } => {
                self.cancel_requested = false;
                self.active_job = None;
                self.progress_percent = 0.0;
                self.status_message = format!("{} canceled.", job.label());
                self.state = AppState::Idle;
            }
            WorkerResponse::Error(message) => {
                self.cancel_requested = false;
                self.active_job = None;
                if let AppState::Computing { job, .. } = &self.state {
                    if *job == WorkerJobKind::DescriptorCalculation {
                        self.log_descriptor_error_and_continue(&message);
                        return;
                    }
                }
                self.set_error(message);
            }
        }
    }
}

/// Runs this executable entrypoint and orchestrates its top-level workflow.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut terminal = setup_terminal()?;
    let (worker_tx, worker_rx, worker_cancel_flag, worker_handle) = spawn_worker();

    let mut app = App::new(worker_tx, worker_rx, worker_cancel_flag);
    let run_result = run_app(&mut terminal, &mut app);

    let _ = app.worker_tx.send(WorkerRequest::Shutdown);
    let _ = worker_handle.join();

    restore_terminal(&mut terminal)?;
    run_result?;
    Ok(())
}

/// Implements terminal.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend)
}

/// Implements terminal.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()
}

/// Runs app.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut App) -> io::Result<()> {
    loop {
        app.drain_worker_messages();
        app.tick();

        terminal.draw(|frame| render(frame, app))?;

        if !event::poll(FRAME_INTERVAL)? {
            continue;
        }

        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        if app.handle_key(key) {
            return Ok(());
        }
    }
}

/// Implements the behavior encoded by this function.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render(frame: &mut Frame, app: &App) {
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(10),
        Constraint::Length(8),
    ])
    .areas(frame.area());

    render_header(frame, header, app);

    match app.active_tab {
        0 => render_import_tab(frame, body, app),
        1 => render_descriptor_tab(frame, body, app),
        2 => render_model_selection_tab(frame, body, app),
        3 => render_hyperparameter_tab(frame, body, app),
        4 => render_visuals_tab(frame, body, app),
        5 => render_diagnostics_tab(frame, body, app),
        _ => {}
    }

    render_footer(frame, footer, app);

    match &app.state {
        AppState::Computing { message, job } => {
            render_progress_popup(frame, app, message, *job, SPINNER_FRAMES[app.spinner_frame]);
        }
        AppState::ErrorPopup(message) => {
            render_error_popup(frame, message);
        }
        AppState::TargetSelectionOpen => {
            render_target_selection_popup(frame, app);
        }
        AppState::FileBrowserOpen(path) => {
            render_file_browser_popup(frame, app, path);
        }
        AppState::Idle => {}
    }
}

/// Implements header.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let [title_area, tabs_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(3)]).areas(area);

    let title = Paragraph::new("FerrumChem TUI — Non-blocking QSAR Workbench").style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    frame.render_widget(title, title_area);

    let tab_labels: Vec<Line> = TAB_TITLES.iter().map(|title| Line::from(*title)).collect();
    let tabs = Tabs::new(tab_labels)
        .select(app.active_tab)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Workflow Tabs"),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .divider(" | ");
    frame.render_widget(tabs, tabs_area);
}

/// Implements footer.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let tab_help = match app.active_tab {
        0 => "Import: f=open file browser (.csv/.smi/.sdf)",
        1 => {
            "Descriptor: Space toggle, a/n all/none, r fast-ML preset, Enter calculate, Esc cancel active compute, v/c filters, e export matrix"
        }
        2 => {
            "Model Select: m cycle mode (AutoQsar/ART2-A/Manual), a/A manual model, t targets, z scaler, x split, b export model, u import model"
        }
        3 => "Parameters: ,/. page, 1-6 edit, Enter save, Esc cancel",
        4 => {
            "Analyze: Enter train (Auto/Manual), g ART2-A cluster, y toggle CSV extras, i toggle importance panel, [/] or ,/. importance target, p predict file, s export predictions, K export clusters, b export model, u import model, o color, Esc cancel active compute"
        }
        _ => "",
    };

    let footer = Paragraph::new(format!(
        "Status: {}\nKeys: q=quit, Left/Right=switch tabs, Esc=close popup/stop editing. {}",
        app.status_message, tab_help
    ))
    .wrap(Wrap { trim: true })
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Status / Help"),
    );

    frame.render_widget(footer, area);
}

/// Implements import tab.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_import_tab(frame: &mut Frame, area: Rect, app: &App) {
    let [top, middle, bottom] = Layout::vertical([
        Constraint::Length(5),
        Constraint::Length(7),
        Constraint::Min(8),
    ])
    .areas(area);

    let source_text = app
        .loaded_from_path
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(no file loaded)".to_string());

    let top_widget = Paragraph::new(format!(
        "Input source: {}\nPress 'f' to choose .csv/.smi/.sdf file from browser.",
        source_text
    ))
    .wrap(Wrap { trim: true })
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Dataset Import"),
    );
    frame.render_widget(top_widget, top);

    let property_count = app
        .loaded_properties
        .first()
        .map(|row| row.len())
        .unwrap_or(0);
    let summary = Paragraph::new(format!(
        "Loaded molecules: {}\nFailed rows skipped: {}\nSMILES rows: {}\nProperty pairs (first row): {}",
        app.loaded_molecules.len(),
        app.last_dataset_failed_rows,
        app.loaded_smiles.len(),
        property_count
    ))
    .wrap(Wrap { trim: true })
    .block(Block::default().borders(Borders::ALL).title("Dataset Summary"));
    frame.render_widget(summary, middle);

    let target_preview = if app.target_columns.is_empty() {
        "No target columns detected yet.".to_string()
    } else {
        app.target_columns
            .iter()
            .take(20)
            .enumerate()
            .map(|(i, target)| format!("{:>2}. {}", i + 1, target.display_label()))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let preview = Paragraph::new(target_preview)
        .wrap(Wrap { trim: true })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Detected Target Columns (CSV + descriptors)"),
        );
    frame.render_widget(preview, bottom);
}

/// Implements descriptor tab.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_descriptor_tab(frame: &mut Frame, area: Rect, app: &App) {
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(38), Constraint::Percentage(62)]).areas(area);

    let selected_count = app.selected_descriptors().len();
    let selected_value_count = app.selected_descriptor_value_count();
    let total_value_count = app
        .available_descriptors
        .iter()
        .map(|(descriptor, _)| descriptor.feature_len())
        .sum::<usize>();
    let fast_ml_descriptor_count = BatchDescriptor::fast_ml_preset().len();
    let fast_ml_value_count = BatchDescriptor::fast_ml_preset()
        .iter()
        .map(|descriptor| descriptor.feature_len())
        .sum::<usize>();
    let source_text = app
        .loaded_from_path
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(no file loaded)".to_string());
    let matrix_text = app
        .feature_matrix
        .as_ref()
        .map(|m| format!("{} x {}", m.rows, m.cols))
        .unwrap_or_else(|| "(none)".to_string());

    let left_text = format!(
        "Input Dataset\n\nSource file: {}\nLoaded molecules: {}\nFeature matrix: {}\n\nDescriptor selection:\nEnabled {}/{} descriptors\nOutput width: {}/{} values per molecule\nFast ML preset: {} descriptors, {} values/mol (press 'r')\n\nActions:\n- Press 'f' to open file browser\n- Press Enter to calculate selected descriptors\n- Press Esc while computing to cancel active compute\n- Press 'v' variance-threshold filter\n- Press 'c' correlation filter\n- Press 'e' export current feature matrix CSV",
        source_text,
        app.loaded_molecules.len(),
        matrix_text,
        selected_count,
        app.available_descriptors.len(),
        selected_value_count,
        total_value_count,
        fast_ml_descriptor_count,
        fast_ml_value_count,
    );

    let left_widget = Paragraph::new(left_text).wrap(Wrap { trim: true }).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Dataset & Controls"),
    );
    frame.render_widget(left_widget, left);

    let [list_area, preview_area] =
        Layout::vertical([Constraint::Percentage(58), Constraint::Percentage(42)]).areas(right);

    let descriptor_items: Vec<ListItem> = app
        .available_descriptors
        .iter()
        .map(|(descriptor, enabled)| {
            let marker = if *enabled { "X" } else { " " };
            ListItem::new(format!(
                "[{}] {} ({} values)",
                marker,
                descriptor.name(),
                descriptor.feature_len()
            ))
        })
        .collect();

    let mut list_state = ListState::default();
    list_state.select(app.descriptor_list_state.selected());

    let list = List::new(descriptor_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Descriptor Toggle List"),
        )
        .highlight_symbol(">> ")
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_stateful_widget(list, list_area, &mut list_state);

    if let Some(matrix) = &app.feature_matrix {
        let preview_cols = matrix.cols.min(PREVIEW_COLS);
        let preview_rows = matrix.rows.min(PREVIEW_ROWS);

        let mut header_cells = vec!["Row".to_string()];
        for name in matrix.feature_names.iter().take(preview_cols) {
            header_cells.push(truncate_label(name, 16));
        }

        let header = Row::new(header_cells).style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

        let mut rows = Vec::with_capacity(preview_rows);
        for row_idx in 0..preview_rows {
            let mut cells = vec![format!("#{}", row_idx + 1)];
            for value in matrix.row(row_idx).iter().take(preview_cols) {
                cells.push(format!("{:.3}", value));
            }
            rows.push(Row::new(cells));
        }

        let mut constraints = vec![Constraint::Length(6)];
        constraints.extend((0..preview_cols).map(|_| Constraint::Min(10)));

        let table = Table::new(rows, constraints).header(header).block(
            Block::default().borders(Borders::ALL).title(format!(
                "Output Preview (showing {}x{} of {}x{})",
                preview_rows, preview_cols, matrix.rows, matrix.cols
            )),
        );

        frame.render_widget(table, preview_area);
    } else {
        let placeholder = Paragraph::new(
            "No descriptor output yet.\n\nAfter loading molecules and pressing Enter, a sample output table appears here.",
        )
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("Output Preview"));
        frame.render_widget(placeholder, preview_area);
    }
}

/// Implements model selection tab.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_model_selection_tab(frame: &mut Frame, area: Rect, app: &App) {
    let [card_auto, card_art2a, card_manual] = Layout::horizontal([
        Constraint::Percentage(33),
        Constraint::Percentage(34),
        Constraint::Percentage(33),
    ])
    .areas(area);
    let card_areas = [card_auto, card_art2a, card_manual];

    let cards_data = [
        (
            MlMode::AutoQsar,
            "AutoQsar",
            "Select best supervised model automatically.",
        ),
        (
            MlMode::Art2a,
            "ART2-A",
            "Unsupervised clustering and diversity analysis.",
        ),
        (
            MlMode::Manual,
            "Manual",
            "Choose specific algorithm (RF, k-NN, PLS, Elastic Net).",
        ),
    ];

    for (idx, (mode, title, body)) in cards_data.into_iter().enumerate() {
        let active = app.selected_ml_mode == mode;
        let border = if active { Color::Yellow } else { Color::Cyan };
        let content = if mode == MlMode::Manual {
            format!(
                "{}\n\nCurrent manual model:\n{}",
                body,
                app.manual_algo_selection.label()
            )
        } else {
            body.to_string()
        };
        let widget = Paragraph::new(content)
            .wrap(Wrap { trim: true })
            .style(if active {
                Style::default().add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(border))
                    .title(if active {
                        format!("{} [selected]", title)
                    } else {
                        title.to_string()
                    }),
            );
        frame.render_widget(widget, card_areas[idx]);
    }

    let selected_targets = app.selected_targets_in_order();
    let target_summary = if selected_targets.is_empty() {
        "(none)".to_string()
    } else {
        truncate_label(
            &selected_targets
                .iter()
                .map(|target| target.display_label())
                .collect::<Vec<_>>()
                .join(", "),
            64,
        )
    };
    let overlay = centered_rect(96, 26, area);
    let info = Paragraph::new(format!(
        "Selected mode: {} (m to cycle)\nTargets: {} (t to choose)\nScaler: {} (z)\nSplit: {} (x)\nManual model: {} (a/A)\nART2-A CSV extras: {} (y)",
        app.selected_ml_mode.label(),
        target_summary,
        app.training_config.scaler.label(),
        app.training_config.split_strategy.label(),
        app.manual_algo_selection.label(),
        if app.include_csv_numeric_in_clustering {
            "on"
        } else {
            "off"
        }
    ))
    .wrap(Wrap { trim: true })
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Selection Controls"),
    );
    frame.render_widget(info, overlay);
}

/// Implements hyperparameter tab.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_hyperparameter_tab(frame: &mut Frame, area: Rect, app: &App) {
    let [top, bottom] = Layout::vertical([Constraint::Length(6), Constraint::Min(8)]).areas(area);

    let mode_text = match app.selected_ml_mode {
        MlMode::AutoQsar => {
            "AutoQsar mode: model-specific knobs are hidden; common controls still editable."
        }
        MlMode::Art2a => {
            "ART2-A mode: edit vigilance/learning-rate/max-iter; common controls still editable."
        }
        MlMode::Manual => "Manual mode: edit selected model hyperparameters below.",
    };
    let top_widget = Paragraph::new(mode_text).wrap(Wrap { trim: true }).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Parameter Mode"),
    );
    frame.render_widget(top_widget, top);

    let mut lines = Vec::new();
    lines.push(format!(
        "Page {}/{} (','/'.' to cycle)",
        app.ml_param_page
            .min(app.ml_param_page_count().saturating_sub(1))
            + 1,
        app.ml_param_page_count()
    ));
    lines.push(String::new());
    for entry in app.ml_param_entries() {
        lines.push(format!(
            "{}: {} = {}",
            entry.hotkey, entry.label, entry.value
        ));
    }
    if let Some(param) = app.editing_ml_param {
        lines.push(String::new());
        lines.push(format!(
            "Editing {}: {}",
            ml_param_label(param),
            app.ml_param_input.value()
        ));
    }

    let bottom_widget = Paragraph::new(lines.join("\n"))
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("Parameters"));
    frame.render_widget(bottom_widget, bottom);

    if app.active_tab == 3 && app.editing_ml_param.is_some() && matches!(app.state, AppState::Idle)
    {
        let cursor_x = bottom
            .x
            .saturating_add(2)
            .saturating_add(app.ml_param_input.visual_cursor() as u16);
        let cursor_y = bottom.y.saturating_add(bottom.height.saturating_sub(2));
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

/// Implements predict visual tab.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_visuals_tab(frame: &mut Frame, area: Rect, app: &App) {
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(area);

    render_pca_panel(frame, left, app);

    let [profile_area, cluster_area] = Layout::vertical([
        Constraint::Percentage(50),
        Constraint::Percentage(50),
    ])
    .areas(right);
    render_descriptor_profile_table(frame, profile_area, app);
    render_art2a_cluster_summary(frame, cluster_area, app);
}

/// Implements render diagnostics tab.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
fn render_diagnostics_tab(frame: &mut Frame, area: Rect, app: &App) {
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(area);

    render_regression_diagnostics_panel(frame, left, app);

    let [prediction_area, importance_area] = Layout::vertical([
        Constraint::Percentage(40),
        Constraint::Percentage(60),
    ])
    .areas(right);

    render_prediction_summary_panel(frame, prediction_area, app);
    if app.show_feature_importance {
        render_feature_importance_panel(frame, importance_area, app);
    } else {
        let block = Block::default()
            .title("Feature Importance")
            .borders(Borders::ALL);
        let placeholder = Paragraph::new("\n\nPress 'I' to calculate and show Feature Importances (requires Random Forest model).")
            .alignment(Alignment::Center)
            .block(block);
        frame.render_widget(placeholder, importance_area);
    }
}

/// Implements pca panel.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_pca_panel(frame: &mut Frame, area: Rect, app: &App) {
    if let Some(projection) = &app.projection {
        let [canvas_area, info_area] =
            Layout::horizontal([Constraint::Percentage(72), Constraint::Percentage(28)])
                .areas(area);

        let points_all = projection.points.clone();
        let maybe_target_overlay = bucket_points_by_values(
            &projection.points,
            app.target_overlay_values.as_deref().unwrap_or(&[]),
        );
        let cluster_buckets = bucket_points_by_clusters(&projection.points, &app.cluster_labels, 6);
        let ad_buckets = bucket_points_by_binary_flags(&projection.points, &app.ad_in_domain_flags);
        let color_mode = match app.color_mode {
            VisualizationColorMode::Uniform => VisualizationColorMode::Uniform,
            VisualizationColorMode::TargetOverlay => {
                if maybe_target_overlay.is_some() {
                    VisualizationColorMode::TargetOverlay
                } else {
                    VisualizationColorMode::Uniform
                }
            }
            VisualizationColorMode::ClusterIds => {
                if cluster_buckets.is_some() {
                    VisualizationColorMode::ClusterIds
                } else {
                    VisualizationColorMode::Uniform
                }
            }
            VisualizationColorMode::ApplicabilityDomain => {
                if ad_buckets.is_some() {
                    VisualizationColorMode::ApplicabilityDomain
                } else {
                    VisualizationColorMode::Uniform
                }
            }
        };
        let target_overlay_for_canvas = maybe_target_overlay.clone();
        let cluster_buckets_for_canvas = cluster_buckets.clone();
        let ad_buckets_for_canvas = ad_buckets.clone();

        let cluster_centroids = compute_cluster_centroids(&projection.points, &app.cluster_labels);

        let canvas = Canvas::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(
                        "PCA Scatter (PC1: {:.1}%, PC2: {:.1}%)",
                        projection.explained_variance[0] * 100.0,
                        projection.explained_variance[1] * 100.0
                    )),
            )
            .x_bounds(projection.x_bounds)
            .y_bounds(projection.y_bounds)
            .paint(move |ctx| match color_mode {
                VisualizationColorMode::TargetOverlay => {
                    if let Some((low, mid, high)) = target_overlay_for_canvas.clone() {
                        if !low.is_empty() {
                            ctx.draw(&Points {
                                coords: &low,
                                color: Color::Blue,
                            });
                        }
                        if !mid.is_empty() {
                            ctx.draw(&Points {
                                coords: &mid,
                                color: Color::Yellow,
                            });
                        }
                        if !high.is_empty() {
                            ctx.draw(&Points {
                                coords: &high,
                                color: Color::Red,
                            });
                        }
                    }
                }
                VisualizationColorMode::ClusterIds => {
                    if let Some(buckets) = cluster_buckets_for_canvas.clone() {
                        let palette = [
                            Color::Blue,
                            Color::Cyan,
                            Color::Green,
                            Color::Yellow,
                            Color::Magenta,
                            Color::Red,
                        ];
                        for (idx, bucket) in buckets.iter().enumerate() {
                            if bucket.is_empty() {
                                continue;
                            }
                            ctx.draw(&Points {
                                coords: bucket,
                                color: palette[idx % palette.len()],
                            });
                        }
                    }
                }
                VisualizationColorMode::ApplicabilityDomain => {
                    if let Some((in_domain, out_domain)) = ad_buckets_for_canvas.clone() {
                        if !in_domain.is_empty() {
                            ctx.draw(&Points {
                                coords: &in_domain,
                                color: Color::Green,
                            });
                        }
                        if !out_domain.is_empty() {
                            ctx.draw(&Points {
                                coords: &out_domain,
                                color: Color::Red,
                            });
                        }
                    }
                }
                VisualizationColorMode::Uniform => {
                    ctx.draw(&Points {
                        coords: &points_all,
                        color: Color::Cyan,
                    });
                }
            });
        frame.render_widget(canvas, canvas_area);

        // Overlay cluster centroids (cross marks) regardless of color mode when available.
        if !cluster_centroids.is_empty() {
            let centroid_canvas = Canvas::default()
                .x_bounds(projection.x_bounds)
                .y_bounds(projection.y_bounds)
                .paint(move |ctx| {
                    for &(x, y) in &cluster_centroids {
                        let dx = (projection.x_bounds[1] - projection.x_bounds[0]).abs() * 0.01;
                        let dy = (projection.y_bounds[1] - projection.y_bounds[0]).abs() * 0.01;
                        ctx.draw(&CanvasLine {
                            x1: x - dx,
                            y1: y - dy,
                            x2: x + dx,
                            y2: y + dy,
                            color: Color::White,
                        });
                        ctx.draw(&CanvasLine {
                            x1: x - dx,
                            y1: y + dy,
                            x2: x + dx,
                            y2: y - dy,
                            color: Color::White,
                        });
                    }
                });
            frame.render_widget(centroid_canvas, canvas_area);
        }

        let (ad_in_count, ad_out_count) = ad_buckets
            .as_ref()
            .map(|(in_domain, out_domain)| (in_domain.len(), out_domain.len()))
            .unwrap_or((0, 0));

        let legend = match color_mode {
            VisualizationColorMode::TargetOverlay => "\n\n🎨 Legend (Target):\n🔵 Low (Bottom 33%)\n🟡 Mid (Middle 33%)\n🔴 High (Top 33%)",
            VisualizationColorMode::ClusterIds => "\n\n🎨 Legend (Clusters):\n🔵 Clst 1  🟡 Clst 4\n🟢 Clst 2  🟣 Clst 5\n🔴 Clst 3  🟤 Clst 6+",
            VisualizationColorMode::ApplicabilityDomain => "\n\n🎨 Legend (AD):\n🟢 In Domain\n🔴 Out of Domain (Outliers)",
            VisualizationColorMode::Uniform => "\n\n🎨 Legend:\n🔵 All Points",
        };

        let info = Paragraph::new(format!(
            "📊 Projection Summary\n\nPoints: {}\nPC1 var: {:.1}%\nPC2 var: {:.1}%\nTotal Expl: {:.1}%\n\nColor mode: {} (toggle: 'o')\nClusters: {}\nTarget overlay: {}\nAD in/out: {}/{}{}",
            projection.points.len(),
            projection.explained_variance[0] * 100.0,
            projection.explained_variance[1] * 100.0,
            (projection.explained_variance[0] + projection.explained_variance[1]) * 100.0,
            color_mode.label(),
            app.cluster_sizes.len(),
            if maybe_target_overlay.is_some() {
                "Available"
            } else if app.target_overlay_error.is_some() {
                "Invalid target"
            } else if app.selected_target_columns.is_empty() {
                "No targets"
            } else {
                "Disabled"
            },
            ad_in_count,
            ad_out_count,
            legend
        ))
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("PCA Info"));
        frame.render_widget(info, info_area);
    } else {
        let placeholder = Paragraph::new(
            "No PCA projection available yet.\n\nCalculate descriptors in tab 1 to generate a projection.",
        )
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("PCA Scatter"));
        frame.render_widget(placeholder, area);
    }
}

/// Implements regression diagnostics panel.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_regression_diagnostics_panel(frame: &mut Frame, area: Rect, app: &App) {
    let primary_target_label = app.trained_primary_target.as_deref().unwrap_or("n/a");
    if app.train_actuals.is_empty() || app.train_actuals.len() != app.training_predictions.len() {
        let placeholder = Paragraph::new(
            "No regression diagnostics available.\n\nTrain a model in Tab 2 to view Predicted vs Actual and residual diagnostics.",
        )
        .wrap(Wrap { trim: true })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(
                    "Regression Diagnostics (primary: {})",
                    truncate_label(primary_target_label, 24)
                )),
        );
        frame.render_widget(placeholder, area);
        return;
    }

    let [plot_area, details_area] =
        Layout::vertical([Constraint::Percentage(66), Constraint::Percentage(34)]).areas(area);

    let min_actual = app
        .train_actuals
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_actual = app
        .train_actuals
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_pred = app
        .training_predictions
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_pred = app
        .training_predictions
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_axis = min_actual.min(min_pred);
    let max_axis = max_actual.max(max_pred);
    let pad = (max_axis - min_axis).max(1e-9) * 0.05;
    let bounds = [min_axis - pad, max_axis + pad];
    let points = app
        .train_actuals
        .iter()
        .copied()
        .zip(app.training_predictions.iter().copied())
        .collect::<Vec<_>>();

    let canvas = Canvas::default()
        .block(Block::default().borders(Borders::ALL).title(format!(
            "Predicted vs Actual ({})",
            truncate_label(primary_target_label, 20)
        )))
        .x_bounds(bounds)
        .y_bounds(bounds)
        .paint(move |ctx| {
            ctx.draw(&CanvasLine {
                x1: bounds[0],
                y1: bounds[0],
                x2: bounds[1],
                y2: bounds[1],
                color: Color::DarkGray,
            });
            ctx.draw(&Points {
                coords: &points,
                color: Color::Cyan,
            });
        });
    frame.render_widget(canvas, plot_area);

    let residuals = app
        .train_actuals
        .iter()
        .zip(app.training_predictions.iter())
        .map(|(a, p)| p - a)
        .collect::<Vec<_>>();
    let mut abs_residuals = app
        .train_actuals
        .iter()
        .zip(app.training_predictions.iter())
        .map(|(a, p)| (p - a).abs())
        .collect::<Vec<_>>();
    abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p95_abs = abs_residuals
        .get(((abs_residuals.len().saturating_sub(1)) as f64 * 0.95).round() as usize)
        .copied()
        .unwrap_or(0.0);
    let bias = if residuals.is_empty() {
        0.0
    } else {
        residuals.iter().sum::<f64>() / residuals.len() as f64
    };
    let residual_std = if residuals.is_empty() {
        0.0
    } else {
        let variance = residuals
            .iter()
            .map(|r| {
                let d = *r - bias;
                d * d
            })
            .sum::<f64>()
            / residuals.len() as f64;
        variance.sqrt()
    };
    let outliers_2sigma = if residual_std <= 1e-12 {
        0usize
    } else {
        residuals
            .iter()
            .filter(|r| (**r - bias).abs() > 2.0 * residual_std)
            .count()
    };
    let outliers_3sigma = if residual_std <= 1e-12 {
        0usize
    } else {
        residuals
            .iter()
            .filter(|r| (**r - bias).abs() > 3.0 * residual_std)
            .count()
    };

    let [summary_area, bins_area] =
        Layout::horizontal([Constraint::Percentage(54), Constraint::Percentage(46)])
            .areas(details_area);

    let train_r2 = app.trained_pipeline.as_ref().map_or(0.0, |p| p.train_r2);
    let holdout_r2 = app.last_holdout_r2.unwrap_or(0.0);
    
    let interpretation = if train_r2 > 0.8 && holdout_r2 > 0.7 {
        "EXCELLENT: Strong correlation and generalizes well. Safe for predictions."
    } else if train_r2 > 0.8 && holdout_r2 <= 0.4 {
        "OVERFITTING: High train R2 but poor holdout generalization."
    } else if train_r2 < 0.5 {
        "UNDERFITTING: Fails to capture data patterns. Try more complex models."
    } else {
        "MODERATE: Captures some variance, leaves room for improvement."
    };

    let stats = Paragraph::new(format!(
        "🎯 Primary target: {}\n\n📊 Metrics:\nTrain n: {} | Train R2: {:.4} | RMSE: {:.4} | MAE: {:.4}\nTrain Pearson r: {:.4}\nHoldout R2: {:.4} | RMSE: {:.4} | MAE: {:.4}\n\n📉 Residuals:\nBias: {:.4} | σ: {:.4} | |Residual| p95: {:.4}\nOutliers (>2σ / >3σ): {} / {}\n\n💡 Interpretation:\n{}",
        primary_target_label,
        app.train_actuals.len(),
        train_r2,
        app.last_train_rmse.unwrap_or(0.0),
        app.last_train_mae.unwrap_or(0.0),
        app.last_train_pearson.unwrap_or(0.0),
        holdout_r2,
        app.last_holdout_rmse.unwrap_or(0.0),
        app.last_holdout_mae.unwrap_or(0.0),
        bias,
        residual_std,
        p95_abs,
        outliers_2sigma,
        outliers_3sigma,
        interpretation
    ))
    .wrap(Wrap { trim: true })
    .block(Block::default().borders(Borders::ALL).title("Residual Summary"));
    frame.render_widget(stats, summary_area);

    let bins = residual_distribution_bins(&residuals, bias, residual_std);
    let n = residuals.len().max(1) as f64;
    let rows = bins
        .iter()
        .map(|(label, count)| {
            let pct = (*count as f64 / n) * 100.0;
            let width = if *count == 0 {
                0
            } else {
                (((*count as f64 / n) * 12.0).round().max(1.0)) as usize
            };
            Row::new(vec![
                (*label).to_string(),
                count.to_string(),
                format!("{:.1}%", pct),
                "#".repeat(width),
            ])
        })
        .collect::<Vec<_>>();
    let header = Row::new(vec!["Bin", "N", "%", "Bar"]).style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let table = Table::new(
        rows,
        [
            Constraint::Percentage(43),
            Constraint::Length(5),
            Constraint::Length(7),
            Constraint::Percentage(25),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Residual Distribution (z-score bins)"),
    );
    frame.render_widget(table, bins_area);
}

/// Implements feature importance panel.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_feature_importance_panel(frame: &mut Frame, area: Rect, app: &App) {
    let importance_summary = app.current_importance_target_summary();
    let title = importance_summary
        .map(|summary| {
            format!(
                "Top Feature Importance: {} ({}) [ / ]",
                truncate_label(&summary.target_display_label, 18),
                truncate_label(&summary.pipeline_name, 12)
            )
        })
        .unwrap_or_else(|| "Top Feature Importance [ / ]".to_string());
    if app.top_feature_importances.is_empty() {
        let placeholder_text = if let Some(summary) = importance_summary {
            format!(
                "No feature-importance data for {} ({}).\n\nThis model may not expose importances (e.g. k-NN).\nUse '[', ']', ',' or '.' to switch trained targets.",
                summary.target_display_label, summary.pipeline_name
            )
        } else {
            "No feature-importance data.\n\nAvailable after training RF, PLS, or Elastic Net models.".to_string()
        };
        let placeholder = Paragraph::new(placeholder_text)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title(title.clone()));
        frame.render_widget(placeholder, area);
        return;
    }

    let max_score = app
        .top_feature_importances
        .iter()
        .map(|(_, score)| *score)
        .fold(0.0_f64, f64::max)
        .max(1e-12);

    let rows = app
        .top_feature_importances
        .iter()
        .take(10)
        .enumerate()
        .map(|(idx, (name, score))| {
            let bars = (((*score / max_score) * 14.0).round() as usize).max(1);
            let bar = "#".repeat(bars);
            Row::new(vec![
                format!("#{}", idx + 1),
                truncate_label(name, 35),
                format!("{:.4}", score),
                bar,
            ])
        })
        .collect::<Vec<_>>();

    let header = Row::new(vec!["Rank", "Feature", "Score", "Bar"]).style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let table = Table::new(
        rows,
        [
            Constraint::Length(6),
            Constraint::Percentage(55),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(table, area);
}

/// Implements descriptor profile table.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_descriptor_profile_table(frame: &mut Frame, area: Rect, app: &App) {
    if app.descriptor_profiles.is_empty() {
        let placeholder = Paragraph::new(
            "No descriptor profile yet.\n\nCalculate descriptors in tab 1 to populate mean/min/max/stddev statistics.",
        )
        .wrap(Wrap { trim: true })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Descriptor Profiling"),
        );
        frame.render_widget(placeholder, area);
        return;
    }

    let visible_rows = area.height.saturating_sub(4) as usize;
    let max_offset = app
        .descriptor_profiles
        .len()
        .saturating_sub(visible_rows.max(1));
    let offset = app.profile_scroll.min(max_offset);

    let rows: Vec<Row> = app
        .descriptor_profiles
        .iter()
        .skip(offset)
        .take(visible_rows.max(1))
        .map(|row| {
            let style = if row.zero_variance {
                Style::default().fg(Color::Red)
            } else {
                Style::default()
            };
            Row::new(vec![
                truncate_label(&row.name, 18),
                format!("{:.4}", row.mean),
                format!("{:.4}", row.min),
                format!("{:.4}", row.max),
                format!("{:.4}", row.stddev),
                if row.zero_variance { "yes" } else { "no" }.to_string(),
            ])
            .style(style)
        })
        .collect();

    let header = Row::new(vec![
        "Descriptor",
        "Mean",
        "Min",
        "Max",
        "StdDev",
        "ZeroVar",
    ])
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );

    let title = format!(
        "Descriptor Profiling (rows {}-{} of {}, Up/Down scroll)",
        offset + 1,
        (offset + rows.len()).min(app.descriptor_profiles.len()),
        app.descriptor_profiles.len()
    );

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(34),
            Constraint::Percentage(13),
            Constraint::Percentage(13),
            Constraint::Percentage(13),
            Constraint::Percentage(13),
            Constraint::Percentage(14),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(title));

    frame.render_widget(table, area);
}

/// Implements progress popup.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_progress_popup(
    frame: &mut Frame,
    app: &App,
    message: &str,
    job: WorkerJobKind,
    spinner: char,
) {
    let area = centered_rect(60, 28, frame.area());
    frame.render_widget(Clear, area);
    let [text_area, gauge_area] =
        Layout::vertical([Constraint::Length(5), Constraint::Length(3)]).areas(area);

    let mut progress_hint = format!("{} {}", spinner, "Working...");
    if !job.is_cancelable() {
        progress_hint.push_str("\nThis step will finish automatically (cancellation disabled).");
    }
    if job.is_cancelable() {
        if app.cancel_requested {
            progress_hint = "Cancellation requested... finishing current row.".to_string();
        } else {
            progress_hint.push_str(&format!("\nPress Esc to cancel {}.", job.label()));
        }
    }
    let text = Paragraph::new(format!("{}\n\n{}", message, progress_hint))
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("Computing"));
    frame.render_widget(text, text_area);

    let percent = if app.progress_percent < 0.0 {
        ((app.spinner_frame as f32 + 1.0) / SPINNER_FRAMES.len() as f32).clamp(0.0, 1.0)
    } else {
        app.progress_percent.clamp(0.0, 1.0)
    };
    let label = if app.progress_percent < 0.0 {
        "indeterminate".to_string()
    } else {
        format!("{:.0}%", percent * 100.0)
    };
    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Progress"))
        .gauge_style(Style::default().fg(Color::Cyan).bg(Color::Black))
        .ratio(percent as f64)
        .label(label);
    frame.render_widget(gauge, gauge_area);
}

/// Implements error popup.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_error_popup(frame: &mut Frame, message: &str) {
    let area = centered_rect(66, 30, frame.area());
    frame.render_widget(Clear, area);
    let popup = Paragraph::new(format!(
        "{}\n\nPress Enter or Esc to close this message.",
        message
    ))
    .wrap(Wrap { trim: true })
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Error")
            .border_style(Style::default().fg(Color::Red)),
    );
    frame.render_widget(popup, area);
}

/// Implements file browser popup.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_file_browser_popup(frame: &mut Frame, app: &App, current_path: &Path) {
    let area = centered_rect(74, 70, frame.area());
    frame.render_widget(Clear, area);

    let [header, list_area, footer] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(8),
        Constraint::Length(3),
    ])
    .areas(area);

    let mode = match app.browser_purpose {
        BrowserPurpose::Dataset => "dataset",
        BrowserPurpose::Prediction => "prediction",
        BrowserPurpose::ModelImport => "model bundle",
    };

    let header_widget = Paragraph::new(format!(
        "Select {} file {}\nCurrent directory: {}",
        mode,
        match app.browser_purpose {
            BrowserPurpose::ModelImport => "(.fcbundle/.bin)",
            _ => "(.csv/.smi/.sdf)",
        },
        current_path.display()
    ))
    .wrap(Wrap { trim: true })
    .block(Block::default().borders(Borders::ALL).title("File Browser"));
    frame.render_widget(header_widget, header);

    let items: Vec<ListItem> = app
        .browser_entries
        .iter()
        .map(|path| {
            let mut label = path
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| path.display().to_string());
            if path.is_dir() {
                label.push('/');
            }
            ListItem::new(label)
        })
        .collect();

    let mut list_state = ListState::default();
    list_state.select(app.browser_list_state.selected());

    let list = if items.is_empty() {
        List::new(vec![ListItem::new("(no matching files in this directory)")])
    } else {
        List::new(items)
    }
    .block(Block::default().borders(Borders::ALL).title("Entries"))
    .highlight_symbol(">> ")
    .highlight_style(
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
    );

    frame.render_stateful_widget(list, list_area, &mut list_state);

    let footer_widget =
        Paragraph::new("Up/Down: navigate  Enter: open/select  Backspace: parent  Esc: close")
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title("Browser Keys"));
    frame.render_widget(footer_widget, footer);
}

/// Implements target selection popup.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_target_selection_popup(frame: &mut Frame, app: &App) {
    let area = centered_rect(62, 62, frame.area());
    frame.render_widget(Clear, area);

    let [header, list_area, footer] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(8),
        Constraint::Length(3),
    ])
    .areas(area);

    let selected_count = app.selected_target_columns.len();
    let header_widget = Paragraph::new(format!(
        "Select supervised target column(s)\nSelected: {} of {}",
        selected_count,
        app.target_columns.len()
    ))
    .wrap(Wrap { trim: true })
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Target Selection"),
    );
    frame.render_widget(header_widget, header);

    let items: Vec<ListItem> = app
        .target_columns
        .iter()
        .map(|target| {
            let checked = if app.selected_target_columns.contains(target.key.as_str()) {
                "[x]"
            } else {
                "[ ]"
            };
            ListItem::new(format!("{} {}", checked, target.display_label()))
        })
        .collect();
    let mut list_state = ListState::default();
    list_state.select(app.target_column_list_state.selected());

    let list = if items.is_empty() {
        List::new(vec![ListItem::new("(no target columns found)")])
    } else {
        List::new(items)
    }
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Target columns"),
    )
    .highlight_symbol(">> ")
    .highlight_style(
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
    );
    frame.render_stateful_widget(list, list_area, &mut list_state);

    let footer_widget =
        Paragraph::new("Up/Down: move  Space: toggle  a: all  n: none  Enter/Esc: close")
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Selection Keys"),
            );
    frame.render_widget(footer_widget, footer);
}

/// Implements art2a cluster summary.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_art2a_cluster_summary(frame: &mut Frame, area: Rect, app: &App) {
    if app.cluster_summaries.is_empty() {
        let placeholder = Paragraph::new(
            "No ART2-A diagnostics yet.\n\nPress 'g' in Tab 5 (Analyze) after descriptor calculation to compute clusters.",
        )
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("ART2-A Cluster Summary"));
        frame.render_widget(placeholder, area);
        return;
    }

    let rows = app
        .cluster_summaries
        .iter()
        .take(6)
        .map(|summary| {
            Row::new(vec![
                format!("C{}", summary.cluster_id),
                summary.size.to_string(),
                format!("{:.1}%", summary.fraction * 100.0),
                format!("{:.3}", summary.mean_similarity),
                format!("{:.3}", summary.min_similarity),
                format!("{:.3}", summary.max_similarity),
            ])
        })
        .collect::<Vec<_>>();
    let header = Row::new(vec!["Cluster", "N", "%", "MeanSim", "MinSim", "MaxSim"]).style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Length(5),
            Constraint::Length(7),
            Constraint::Length(9),
            Constraint::Length(9),
            Constraint::Length(9),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(format!(
        "ART2-A Cluster Summary ({} clusters)",
        app.cluster_summaries.len()
    )));
    frame.render_widget(table, area);
}

/// Implements prediction summary panel.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn render_prediction_summary_panel(frame: &mut Frame, area: Rect, app: &App) {
    if app.prediction_rows.is_empty() {
        let placeholder =
            Paragraph::new("No predictions loaded.\nUse 'p' to run prediction and 's' to export.")
                .wrap(Wrap { trim: true })
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Prediction Summary"),
                );
        frame.render_widget(placeholder, area);
        return;
    }

    let source_text = app
        .prediction_source
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(none)".to_string());
    let output_names = if app.prediction_target_names.is_empty() {
        "(none)".to_string()
    } else {
        truncate_label(&app.prediction_target_names.join(", "), 56)
    };
    let preview = app
        .prediction_rows
        .iter()
        .take(2)
        .enumerate()
        .map(|(idx, row)| {
            let first = row.predictions.first().copied().unwrap_or(0.0);
            format!(
                "#{} {} -> {:.4}",
                idx + 1,
                truncate_label(&row.smiles, 22),
                first
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let widget = Paragraph::new(format!(
        "Source: {}\nRows: {}\nOutputs: {}\nPreview:\n{}",
        source_text,
        app.prediction_rows.len(),
        output_names,
        if preview.is_empty() {
            "(empty)".to_string()
        } else {
            preview
        }
    ))
    .wrap(Wrap { trim: true })
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Prediction Summary"),
    );
    frame.render_widget(widget, area);
}

/// Implements rect.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let [top, middle, _bottom] = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .areas(area);

    let [_left, center, _right] = Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .areas(middle);

    let _ = top;
    center
}

/// Implements descriptor error log.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn append_descriptor_error_log(message: &str) -> Result<(), String> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(DESCRIPTOR_ERROR_LOG_PATH)
        .map_err(|error| format!("open {} failed: {}", DESCRIPTOR_ERROR_LOG_PATH, error))?;

    let timestamp_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system time error: {}", error))?
        .as_secs();
    writeln!(
        file,
        "[{}] descriptor-calculation-error: {}",
        timestamp_secs, message
    )
    .map_err(|error| format!("write {} failed: {}", DESCRIPTOR_ERROR_LOG_PATH, error))
}

/// Derives a UI label from output target key (e.g. `alogp__csv` -> `[CSV] alogp`).
fn display_label_from_output_target_name(output_target_name: &str) -> String {
    let Some((name, source_suffix)) = output_target_name.rsplit_once("__") else {
        return output_target_name.to_string();
    };
    match source_suffix.to_ascii_lowercase().as_str() {
        "csv" => format!("[CSV] {}", name),
        "desc" => format!("[DESC] {}", name),
        _ => output_target_name.to_string(),
    }
}

/// Builds cluster export rows from latest ART2-A predictions or in-memory clustering state.
fn build_cluster_export_rows(
    prediction_rows: &[PredictionOutputRow],
    prediction_target_names: &[String],
    cluster_labels: &[usize],
    loaded_smiles: &[String],
) -> Result<Vec<ClusterExportRow>, String> {
    if !prediction_rows.is_empty() {
        let cluster_idx = prediction_target_names
            .iter()
            .position(|name| name == "ART2A_ClusterId");
        let similarity_idx = prediction_target_names
            .iter()
            .position(|name| name == "ART2A_Similarity");
        let in_domain_idx = prediction_target_names
            .iter()
            .position(|name| name == "ART2A_InDomain");

        if let Some(cluster_idx) = cluster_idx {
            let mut rows = Vec::with_capacity(prediction_rows.len());
            for row in prediction_rows {
                let cluster_raw = row.predictions.get(cluster_idx).copied().ok_or_else(|| {
                    "ART2A_ClusterId column missing in prediction row.".to_string()
                })?;
                if !cluster_raw.is_finite() || cluster_raw < 0.0 {
                    return Err(
                        "ART2A_ClusterId contains non-finite or negative value.".to_string()
                    );
                }
                let similarity = similarity_idx.and_then(|idx| row.predictions.get(idx).copied());
                let in_domain = in_domain_idx
                    .and_then(|idx| row.predictions.get(idx).copied())
                    .map(|value| value >= 0.5);
                rows.push(ClusterExportRow {
                    smiles: row.smiles.clone(),
                    cluster_id: cluster_raw.round() as usize,
                    similarity,
                    in_domain,
                });
            }
            return Ok(rows);
        }
    }

    if !cluster_labels.is_empty() {
        if loaded_smiles.len() < cluster_labels.len() {
            return Err(format!(
                "Cannot export ART2-A clusters: {} SMILES entries for {} cluster labels.",
                loaded_smiles.len(),
                cluster_labels.len()
            ));
        }
        let rows = cluster_labels
            .iter()
            .enumerate()
            .map(|(idx, &cluster_id)| ClusterExportRow {
                smiles: loaded_smiles[idx].clone(),
                cluster_id,
                similarity: None,
                in_domain: None,
            })
            .collect::<Vec<_>>();
        return Ok(rows);
    }

    Err(
        "No ART2-A cluster outputs available. Run prediction with ART2-A output or cluster with 'g' first."
            .to_string(),
    )
}

/// Resolves a UI target-display label from output target key (e.g. `alogp__csv`).
fn resolve_target_display_label(
    output_target_name: &str,
    target_columns: &[TargetColumn],
) -> String {
    target_columns
        .iter()
        .find(|column| column.output_name() == output_target_name)
        .map(TargetColumn::display_label)
        .unwrap_or_else(|| display_label_from_output_target_name(output_target_name))
}

/// Implements worker.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn spawn_worker() -> (
    Sender<WorkerRequest>,
    Receiver<WorkerResponse>,
    Arc<AtomicBool>,
    thread::JoinHandle<()>,
) {
    let (request_tx, request_rx) = mpsc::channel::<WorkerRequest>();
    let (response_tx, response_rx) = mpsc::channel::<WorkerResponse>();
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let app_cancel_flag = Arc::clone(&cancel_flag);

    let handle = thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || worker_loop(request_rx, response_tx, cancel_flag))
        .unwrap();

    (request_tx, response_rx, app_cancel_flag, handle)
}

/// Implements loop.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn worker_loop(
    request_rx: Receiver<WorkerRequest>,
    response_tx: Sender<WorkerResponse>,
    cancel_flag: Arc<AtomicBool>,
) {
    while let Ok(request) = request_rx.recv() {
        if let WorkerRequest::Shutdown = &request {
            return;
        }
        let request_label = match &request {
            WorkerRequest::LoadDataset(_) => "LoadDataset",
            WorkerRequest::CalculateDescriptors { .. } => "CalculateDescriptors",
            WorkerRequest::TrainModel { .. } => "TrainModel",
            WorkerRequest::RunArt2aClustering { .. } => "RunArt2aClustering",
            WorkerRequest::PredictFile { .. } => "PredictFile",
            WorkerRequest::ExportPredictions { .. } => "ExportPredictions",
            WorkerRequest::ExportClusters { .. } => "ExportClusters",
            WorkerRequest::ApplyVarianceFilter { .. } => "ApplyVarianceFilter",
            WorkerRequest::ApplyCorrelationFilter { .. } => "ApplyCorrelationFilter",
            WorkerRequest::ExportFeatureMatrix { .. } => "ExportFeatureMatrix",
            WorkerRequest::ExportModelBundle { .. } => "ExportModelBundle",
            WorkerRequest::ImportModelBundle { .. } => "ImportModelBundle",
            WorkerRequest::Shutdown => unreachable!("Shutdown handled before request execution"),
        };

        let _ = response_tx.send(WorkerResponse::Progress(0.0));

        let cancel_flag = Arc::clone(&cancel_flag);
        let guarded = panic::catch_unwind(AssertUnwindSafe(|| match request {
            WorkerRequest::LoadDataset(path) => handle_load_dataset(path),
            WorkerRequest::CalculateDescriptors {
                molecules,
                descriptors,
            } => handle_calculate_descriptors(
                molecules,
                descriptors,
                Some(&response_tx),
                &cancel_flag,
            ),
            WorkerRequest::TrainModel {
                matrix,
                target_series,
                smiles,
                config,
                target_columns,
            } => handle_train_model(
                matrix,
                target_series,
                smiles,
                config,
                target_columns,
                Some(&response_tx),
                &cancel_flag,
            ),
            WorkerRequest::RunArt2aClustering {
                matrix,
                config,
                scaler,
            } => handle_art2a_clustering(matrix, config, scaler, Some(&response_tx), &cancel_flag),
            WorkerRequest::PredictFile {
                path,
                pipeline_bundle_bytes,
                art2a_model_bytes,
                descriptors,
                selected_feature_indices,
                expected_feature_names,
            } => handle_predict_file(
                path,
                pipeline_bundle_bytes,
                art2a_model_bytes,
                descriptors,
                selected_feature_indices,
                expected_feature_names,
                &cancel_flag,
            ),
            WorkerRequest::ExportPredictions {
                path,
                target_names,
                rows,
            } => handle_export_predictions(path, target_names, rows),
            WorkerRequest::ExportClusters { path, rows } => handle_export_clusters(path, rows),
            WorkerRequest::ApplyVarianceFilter { matrix, threshold } => {
                handle_apply_variance_filter(matrix, threshold)
            }
            WorkerRequest::ApplyCorrelationFilter { matrix, threshold } => {
                handle_apply_correlation_filter(matrix, threshold)
            }
            WorkerRequest::ExportFeatureMatrix {
                path,
                smiles,
                matrix,
            } => handle_export_feature_matrix(path, smiles, matrix),
            WorkerRequest::ExportModelBundle { path, bundle } => {
                handle_export_model_bundle(path, bundle)
            }
            WorkerRequest::ImportModelBundle { path } => handle_import_model_bundle(path),
            WorkerRequest::Shutdown => unreachable!("Shutdown handled before request execution"),
        }));

        let result = match guarded {
            Ok(result) => result,
            Err(panic_payload) => {
                let panic_message = if let Some(message) = panic_payload.downcast_ref::<&str>() {
                    (*message).to_string()
                } else if let Some(message) = panic_payload.downcast_ref::<String>() {
                    message.clone()
                } else {
                    "non-string panic payload".to_string()
                };
                Err(format!(
                    "Worker panic during {}: {}",
                    request_label, panic_message
                ))
            }
        };

        let response = match result {
            Ok(response) => response,
            Err(error) => WorkerResponse::Error(error),
        };

        let _ = response_tx.send(WorkerResponse::Progress(1.0));

        if response_tx.send(response).is_err() {
            return;
        }
    }
}

/// Implements load dataset.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn handle_load_dataset(path: PathBuf) -> Result<WorkerResponse, String> {
    let extension = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    let (molecules, smiles, properties, failed_rows) = match extension.as_str() {
        "csv" => load_dataset_from_csv(&path)?,
        "smi" | "smiles" => load_dataset_from_smi(&path)?,
        "sdf" => load_dataset_from_sdf(&path)?,
        _ => {
            return Err(format!(
                "Unsupported file extension for '{}'. Use .csv, .smi, or .sdf",
                path.display()
            ));
        }
    };

    if molecules.is_empty() {
        return Err(format!(
            "No valid molecules parsed from '{}'.",
            path.display()
        ));
    }

    Ok(WorkerResponse::DatasetLoaded {
        path,
        molecules,
        smiles,
        properties,
        failed_rows,
    })
}

/// Implements calculate descriptors.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn handle_calculate_descriptors(
    mut molecules: Vec<Molecule>,
    descriptors: Vec<BatchDescriptor>,
    progress_tx: Option<&Sender<WorkerResponse>>,
    cancel_flag: &Arc<AtomicBool>,
) -> Result<WorkerResponse, String> {
    if cancel_flag.load(AtomicOrdering::Relaxed) {
        return Ok(WorkerResponse::JobCanceled {
            job: WorkerJobKind::DescriptorCalculation,
        });
    }

    if molecules.is_empty() {
        return Err("Cannot calculate descriptors: no molecules provided.".to_string());
    }
    if descriptors.is_empty() {
        return Err("Cannot calculate descriptors: descriptor set is empty.".to_string());
    }

    if let Some(tx) = progress_tx {
        let _ = tx.send(WorkerResponse::Progress(0.20));
    }
    let feature_names = descriptor_feature_names(&descriptors);
    let flat = match calculate_batch_flat_static_cancellable(&mut molecules, &descriptors, &|| {
        cancel_flag.load(AtomicOrdering::Relaxed)
    }) {
        Ok(flat) => flat,
        Err(()) => {
            return Ok(WorkerResponse::JobCanceled {
                job: WorkerJobKind::DescriptorCalculation,
            });
        }
    };
    if let Some(tx) = progress_tx {
        let _ = tx.send(WorkerResponse::Progress(0.75));
    }

    if flat.cols != feature_names.len() {
        return Err(format!(
            "Descriptor output mismatch: got {} columns, expected {} names.",
            flat.cols,
            feature_names.len()
        ));
    }

    let payload = FeatureMatrixPayload {
        rows: flat.rows,
        cols: flat.cols,
        feature_names,
        data: flat.data.into_iter().map(|v| v as f64).collect(),
        descriptors,
        selected_feature_indices: (0..flat.cols).collect(),
    };

    let profiles = build_profiles(&payload);
    let projection = build_projection(&payload);
    if let Some(tx) = progress_tx {
        let _ = tx.send(WorkerResponse::Progress(0.95));
    }

    Ok(WorkerResponse::DescriptorsCalculated {
        matrix: payload,
        profiles,
        projection,
    })
}

/// Implements train model.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn handle_train_model(
    matrix: FeatureMatrixPayload,
    target_series: Vec<(String, Vec<f64>)>,
    smiles: Vec<String>,
    config: MlTrainingConfig,
    target_columns: Vec<TargetColumn>,
    progress_tx: Option<&Sender<WorkerResponse>>,
    cancel_flag: &Arc<AtomicBool>,
) -> Result<WorkerResponse, String> {
    if cancel_flag.load(AtomicOrdering::Relaxed) {
        return Ok(WorkerResponse::JobCanceled {
            job: WorkerJobKind::ModelTraining,
        });
    }
    if matrix.rows == 0 || matrix.cols == 0 {
        return Err("Cannot train model: feature matrix is empty.".to_string());
    }
    if target_series.is_empty() {
        return Err("Cannot train model: no target columns selected.".to_string());
    }
    for (target_name, targets) in &target_series {
        if targets.len() != matrix.rows {
            return Err(format!(
                "Cannot train model for target '{}': targets={} rows={}",
                target_name,
                targets.len(),
                matrix.rows
            ));
        }
    }

    let mut target_summaries = Vec::with_capacity(target_series.len());
    let mut pipeline_bundle = Vec::with_capacity(target_series.len());

    let mut primary_target = String::new();
    let mut primary_target_display_label = String::new();
    let mut primary_pipeline: Option<QsarPipeline> = None;
    let mut primary_train_actuals = Vec::new();
    let mut primary_train_predictions = Vec::new();
    let mut primary_preview_predictions = Vec::new();
    let mut primary_ad_in_domain_flags = Vec::new();
    let mut primary_top_feature_importances = Vec::new();
    let mut primary_rmse = 0.0;
    let mut primary_mae = 0.0;
    let mut primary_pearson = 0.0;
    let mut primary_holdout_r2 = 0.0;
    let mut primary_holdout_rmse = 0.0;
    let mut primary_holdout_mae = 0.0;
    let mut primary_holdout_pearson = 0.0;
    let mut primary_split_label = String::new();
    let mut primary_split_train_size = 0usize;
    let mut primary_split_test_size = 0usize;

    let base_features = Array2::from_shape_vec((matrix.rows, matrix.cols), matrix.data.clone())
        .map_err(|error| format!("Failed to shape feature matrix for training: {}", error))?;
    let total_targets = target_series.len().max(1);
    for (target_idx, (target_name, targets)) in target_series.into_iter().enumerate() {
        let target_display_label = resolve_target_display_label(&target_name, &target_columns);
        let target_start = target_idx as f32 / total_targets as f32;
        let target_span = 1.0 / total_targets as f32;
        let report_target_progress = |phase: f32| {
            if let Some(tx) = progress_tx {
                let local = phase.clamp(0.0, 1.0);
                let global = (0.05 + 0.90 * (target_start + target_span * local)).min(0.98);
                let _ = tx.send(WorkerResponse::Progress(global));
            }
        };
        report_target_progress(0.0);
        if cancel_flag.load(AtomicOrdering::Relaxed) {
            return Ok(WorkerResponse::JobCanceled {
                job: WorkerJobKind::ModelTraining,
            });
        }

        let dataset = QsarDataset::new(
            base_features.clone(),
            Array1::from(targets),
            matrix.feature_names.clone(),
        );
        report_target_progress(0.10);

        if cancel_flag.load(AtomicOrdering::Relaxed) {
            return Ok(WorkerResponse::JobCanceled {
                job: WorkerJobKind::ModelTraining,
            });
        }
        let split = split_dataset(&dataset, &smiles, &config)?;
        if split.train.n_samples() < 2 || split.test.n_samples() < 1 {
            return Err("Split produced insufficient train/test rows for diagnostics.".to_string());
        }
        report_target_progress(0.22);

        if cancel_flag.load(AtomicOrdering::Relaxed) {
            return Ok(WorkerResponse::JobCanceled {
                job: WorkerJobKind::ModelTraining,
            });
        }
        let mut report_pipeline_progress = |phase_fraction: f32, _label: &str| {
            // Reserve [0.22, 0.62] of target-local progress for model training internals.
            report_target_progress(0.22 + 0.40 * phase_fraction.clamp(0.0, 1.0));
        };
        let pipeline =
            train_pipeline_with_progress(&dataset, &config, &mut report_pipeline_progress)?;
        report_target_progress(0.62);

        if cancel_flag.load(AtomicOrdering::Relaxed) {
            return Ok(WorkerResponse::JobCanceled {
                job: WorkerJobKind::ModelTraining,
            });
        }
        let train_predictions = pipeline.predict(&dataset.features).to_vec();
        report_target_progress(0.72);
        let train_actuals = dataset.targets.to_vec();
        let ad_in_domain_flags = pipeline
            .ad
            .as_ref()
            .map(|ad| ad.training_in_domain_flags())
            .filter(|flags| flags.len() == dataset.features.nrows())
            .unwrap_or_else(|| {
                (0..dataset.features.nrows())
                    .map(|row_idx| {
                        let sample = dataset.features.row(row_idx).to_vec();
                        pipeline.is_in_domain(&sample)
                    })
                    .collect::<Vec<_>>()
            });
        report_target_progress(0.82);
        let top_feature_importances =
            extract_top_feature_importances(&pipeline, &matrix.feature_names, 10);
        let preview_predictions = train_predictions
            .iter()
            .take(PREVIEW_ROWS)
            .copied()
            .collect::<Vec<_>>();
        let rmse = metrics::rmse(&train_actuals, &train_predictions);
        let mae = metrics::mae(&train_actuals, &train_predictions);
        let pearson = pearson_correlation(&train_actuals, &train_predictions);
        let holdout_predictions = pipeline.predict(&split.test.features).to_vec();
        let holdout_targets = split.test.targets.to_vec();
        let holdout_r2 = metrics::r_squared(&holdout_targets, &holdout_predictions);
        let holdout_rmse = metrics::rmse(&holdout_targets, &holdout_predictions);
        let holdout_mae = metrics::mae(&holdout_targets, &holdout_predictions);
        let holdout_pearson = pearson_correlation(&holdout_targets, &holdout_predictions);
        report_target_progress(0.94);
        let split_label = config.split_strategy.label().to_string();
        let split_train_size = split.train.n_samples();
        let split_test_size = split.test.n_samples();

        pipeline_bundle.push((target_name.clone(), pipeline.clone()));
        target_summaries.push(TargetTrainingSummary {
            target: target_name.clone(),
            target_display_label: target_display_label.clone(),
            pipeline_name: pipeline.name.clone(),
            cv_q2: pipeline.cv_q2,
            train_r2: pipeline.train_r2,
            rmse,
            mae,
            pearson,
            holdout_r2,
            holdout_rmse,
            holdout_mae,
            holdout_pearson,
            split_label: split_label.clone(),
            split_train_size,
            split_test_size,
            top_feature_importances: top_feature_importances.clone(),
        });

        if target_idx == 0 {
            primary_target = target_name;
            primary_target_display_label = target_display_label;
            primary_pipeline = Some(pipeline);
            primary_train_actuals = train_actuals;
            primary_train_predictions = train_predictions;
            primary_preview_predictions = preview_predictions;
            primary_ad_in_domain_flags = ad_in_domain_flags;
            primary_top_feature_importances = top_feature_importances;
            primary_rmse = rmse;
            primary_mae = mae;
            primary_pearson = pearson;
            primary_holdout_r2 = holdout_r2;
            primary_holdout_rmse = holdout_rmse;
            primary_holdout_mae = holdout_mae;
            primary_holdout_pearson = holdout_pearson;
            primary_split_label = split_label;
            primary_split_train_size = split_train_size;
            primary_split_test_size = split_test_size;
        }
        report_target_progress(1.0);
    }

    if cancel_flag.load(AtomicOrdering::Relaxed) {
        return Ok(WorkerResponse::JobCanceled {
            job: WorkerJobKind::ModelTraining,
        });
    }

    let primary_pipeline =
        primary_pipeline.ok_or_else(|| "No trained pipeline produced.".to_string())?;
    if let Some(tx) = progress_tx {
        let _ = tx.send(WorkerResponse::Progress(0.98));
    }
    let pipeline_bundle_bytes = bincode::serialize(&pipeline_bundle)
        .map_err(|error| format!("Failed to serialize trained pipeline bundle: {}", error))?;

    Ok(WorkerResponse::ModelTrained {
        primary_target,
        primary_target_display_label,
        pipeline_name: primary_pipeline.name.clone(),
        cv_q2: primary_pipeline.cv_q2,
        train_r2: primary_pipeline.train_r2,
        rmse: primary_rmse,
        mae: primary_mae,
        pearson: primary_pearson,
        train_actuals: primary_train_actuals,
        ad_in_domain_flags: primary_ad_in_domain_flags,
        top_feature_importances: primary_top_feature_importances,
        holdout_r2: primary_holdout_r2,
        holdout_rmse: primary_holdout_rmse,
        holdout_mae: primary_holdout_mae,
        holdout_pearson: primary_holdout_pearson,
        split_label: primary_split_label,
        split_train_size: primary_split_train_size,
        split_test_size: primary_split_test_size,
        target_summaries,
        pipeline_bundle_bytes,
        descriptors: matrix.descriptors.clone(),
        selected_feature_indices: matrix.selected_feature_indices.clone(),
        expected_feature_names: matrix.feature_names.clone(),
        preview_predictions: primary_preview_predictions,
        train_predictions: primary_train_predictions,
    })
}

/// Implements art2a clustering.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn handle_art2a_clustering(
    matrix: FeatureMatrixPayload,
    config: Art2aConfig,
    scaler: MlScalerChoice,
    progress_tx: Option<&Sender<WorkerResponse>>,
    cancel_flag: &Arc<AtomicBool>,
) -> Result<WorkerResponse, String> {
    if cancel_flag.load(AtomicOrdering::Relaxed) {
        return Ok(WorkerResponse::JobCanceled {
            job: WorkerJobKind::Art2aClustering,
        });
    }
    if matrix.rows < 2 || matrix.cols == 0 {
        return Err("ART2-A requires at least 2 rows and 1 feature.".to_string());
    }
    if !(0.0..1.0).contains(&config.vigilance) {
        return Err("ART2-A vigilance must be in (0,1).".to_string());
    }
    if !(0.0..=1.0).contains(&config.learning_rate) || config.learning_rate <= 0.0 {
        return Err("ART2-A learning rate must be in (0,1].".to_string());
    }
    if config.max_iter == 0 {
        return Err("ART2-A max_iter must be >= 1.".to_string());
    }

    let sanitized_matrix_data = matrix
        .data
        .iter()
        .map(|v| if v.is_finite() { *v } else { 0.0 })
        .collect::<Vec<_>>();

    let data = Array2::from_shape_vec((matrix.rows, matrix.cols), sanitized_matrix_data)
        .map_err(|error| format!("Failed to shape matrix for ART2-A: {}", error))?;
    if let Some(tx) = progress_tx {
        let _ = tx.send(WorkerResponse::Progress(0.08));
    }
    let (fitted_scaler, scaled) = match scaler {
        MlScalerChoice::Standard => {
            let scaler = StandardScaler::fit(&data).unwrap();
            let scaled = scaler.transform(&data);
            (ScalerType::Standard(scaler), scaled)
        }
        MlScalerChoice::MinMax => {
            let scaler = MinMaxScaler::fit(&data).unwrap();
            let scaled = scaler.transform(&data);
            (ScalerType::MinMax(scaler), scaled)
        }
    };

    let vectors: Vec<Vec<f64>> = scaled
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .map(|value| if value.is_finite() { *value } else { 0.0 })
                .collect::<Vec<_>>()
        })
        .collect();
    if let Some(tx) = progress_tx {
        let _ = tx.send(WorkerResponse::Progress(0.12));
    }
    let mut model = Art2a::new(config.vigilance, config.learning_rate, config.max_iter);
    let completed = model.fit_with_progress(&vectors, |progress| {
        if let Some(tx) = progress_tx {
            let global = (0.12 + 0.76 * progress.clamp(0.0, 1.0)).min(0.90);
            let _ = tx.send(WorkerResponse::Progress(global));
        }
        cancel_flag.load(AtomicOrdering::Relaxed)
    });
    if !completed {
        return Ok(WorkerResponse::JobCanceled {
            job: WorkerJobKind::Art2aClustering,
        });
    }
    if cancel_flag.load(AtomicOrdering::Relaxed) {
        return Ok(WorkerResponse::JobCanceled {
            job: WorkerJobKind::Art2aClustering,
        });
    }
    if model.labels.len() != matrix.rows {
        return Err(format!(
            "ART2-A label length mismatch: {} labels for {} rows.",
            model.labels.len(),
            matrix.rows
        ));
    }

    let labels = model.labels.clone();
    let cluster_sizes = model.cluster_sizes.clone();
    let mut sim_buckets: Vec<Vec<f64>> = vec![Vec::new(); model.n_clusters()];
    for (row_idx, row) in vectors.iter().enumerate() {
        if cancel_flag.load(AtomicOrdering::Relaxed) {
            return Ok(WorkerResponse::JobCanceled {
                job: WorkerJobKind::Art2aClustering,
            });
        }
        let (pred_cluster, similarity) = model.predict_one(row);
        let assigned_cluster = labels[row_idx];
        debug_assert_eq!(pred_cluster, assigned_cluster);
        if let Some(bucket) = sim_buckets.get_mut(assigned_cluster) {
            bucket.push(similarity);
        }
    }
    if let Some(tx) = progress_tx {
        let _ = tx.send(WorkerResponse::Progress(0.95));
    }
    let total = labels.len().max(1) as f64;
    let mut summaries = Vec::with_capacity(cluster_sizes.len());
    for (cluster_id, &size) in cluster_sizes.iter().enumerate() {
        let sims = sim_buckets
            .get(cluster_id)
            .map_or(&[][..], |v| v.as_slice());
        let (mean_similarity, min_similarity, max_similarity) = if sims.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let mean = sims.iter().sum::<f64>() / sims.len() as f64;
            let min = sims.iter().copied().fold(f64::INFINITY, f64::min);
            let max = sims.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (mean, min, max)
        };
        summaries.push(Art2aClusterSummary {
            cluster_id,
            size,
            fraction: size as f64 / total,
            mean_similarity,
            min_similarity,
            max_similarity,
        });
    }

    let art2a_model_bytes = bincode::serialize(&(model, fitted_scaler, matrix.cols))
        .map_err(|error| format!("Failed to serialize ART2-A model bundle: {}", error))?;
    let projection = build_projection(&matrix);
    if let Some(tx) = progress_tx {
        let _ = tx.send(WorkerResponse::Progress(0.98));
    }
    Ok(WorkerResponse::ClusteringCompleted {
        labels,
        cluster_sizes,
        summaries,
        art2a_model_bytes,
        projection,
        descriptors: matrix.descriptors.clone(),
        selected_feature_indices: matrix.selected_feature_indices.clone(),
        expected_feature_names: matrix.feature_names.clone(),
    })
}

/// Implements predict file.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn handle_predict_file(
    path: PathBuf,
    pipeline_bundle_bytes: Option<Vec<u8>>,
    art2a_model_bytes: Option<Vec<u8>>,
    descriptors: Vec<BatchDescriptor>,
    selected_feature_indices: Vec<usize>,
    expected_feature_names: Vec<String>,
    cancel_flag: &Arc<AtomicBool>,
) -> Result<WorkerResponse, String> {
    if cancel_flag.load(AtomicOrdering::Relaxed) {
        return Ok(WorkerResponse::JobCanceled {
            job: WorkerJobKind::Prediction,
        });
    }
    if pipeline_bundle_bytes.is_none() && art2a_model_bytes.is_none() {
        return Err(
            "Prediction failed: no supervised model or ART2-A model available. Train one first."
                .to_string(),
        );
    }
    if descriptors.is_empty() {
        return Err("Prediction failed: descriptor selection is empty.".to_string());
    }

    let extension = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    let (mut molecules, smiles, properties, failed_rows) = match extension.as_str() {
        "csv" => load_dataset_from_csv(&path)?,
        "smi" | "smiles" => load_dataset_from_smi(&path)?,
        "sdf" => load_dataset_from_sdf(&path)?,
        _ => {
            return Err(format!(
                "Unsupported prediction input '{}'. Use .csv, .smi, or .sdf",
                path.display()
            ));
        }
    };

    if molecules.is_empty() {
        return Err(format!(
            "Prediction input '{}' produced no valid molecules.",
            path.display()
        ));
    }

    let flat = match calculate_batch_flat_static_cancellable(&mut molecules, &descriptors, &|| {
        cancel_flag.load(AtomicOrdering::Relaxed)
    }) {
        Ok(flat) => flat,
        Err(()) => {
            return Ok(WorkerResponse::JobCanceled {
                job: WorkerJobKind::Prediction,
            });
        }
    };
    let source_feature_names = descriptor_feature_names(&descriptors);
    if flat.cols != source_feature_names.len() {
        return Err(format!(
            "Prediction source feature mismatch: got {} columns from descriptor set.",
            flat.cols
        ));
    }
    let selected_flat = build_prediction_flat_by_expected_names(
        &flat,
        &source_feature_names,
        &properties,
        &expected_feature_names,
    )?;
    if selected_flat.cols == 0 {
        return Err("Prediction failed: expected feature layout is empty.".to_string());
    }

    let features = Array2::from_shape_vec(
        (selected_flat.rows, selected_flat.cols),
        selected_flat.data.into_iter().map(|v| v as f64).collect(),
    )
    .map_err(|error| format!("Failed to shape prediction feature matrix: {}", error))?;
    let projection_payload = FeatureMatrixPayload {
        rows: features.nrows(),
        cols: features.ncols(),
        feature_names: expected_feature_names.clone(),
        data: features.iter().copied().collect(),
        descriptors: descriptors.clone(),
        selected_feature_indices: selected_feature_indices.clone(),
    };

    let maybe_pipeline_bundle = pipeline_bundle_bytes
        .map(|bytes| {
            bincode::deserialize::<Vec<(String, QsarPipeline)>>(&bytes).map_err(|error| {
                format!(
                    "Failed to deserialize pipeline bundle for prediction: {}",
                    error
                )
            })
        })
        .transpose()?;
    if let Some(pipeline_bundle) = &maybe_pipeline_bundle {
        if pipeline_bundle.is_empty() {
            return Err("Prediction failed: trained pipeline bundle is empty.".to_string());
        }
        let feature_layout_mismatch = pipeline_bundle
            .iter()
            .any(|(_, pipeline)| pipeline.feature_names != expected_feature_names);
        if feature_layout_mismatch {
            return Err(
                "Prediction failed: descriptor set does not match the trained pipeline feature layout."
                    .to_string(),
            );
        }
        let model_feature_count = pipeline_bundle[0].1.feature_names.len();
        if selected_flat.cols != model_feature_count {
            return Err(format!(
                "Prediction feature mismatch: got {} columns, model expects {}.",
                selected_flat.cols, model_feature_count
            ));
        }
    }

    let mut target_predictions = Vec::new();
    let mut target_names = Vec::new();
    if let Some(pipeline_bundle) = maybe_pipeline_bundle {
        target_predictions.reserve(pipeline_bundle.len());
        target_names.reserve(pipeline_bundle.len());
        for (target_name, pipeline) in pipeline_bundle {
            target_names.push(target_name);
            target_predictions.push(pipeline.predict(&features).to_vec());
        }
    }

    let maybe_art2a_bundle = art2a_model_bytes
        .map(|bytes| {
            bincode::deserialize::<(Art2a, ScalerType, usize)>(&bytes).map_err(|error| {
                format!(
                    "Failed to deserialize ART2-A model bundle for prediction: {}",
                    error
                )
            })
        })
        .transpose()?;

    if let Some((model, scaler, expected_feature_count)) = &maybe_art2a_bundle {
        if model.prototypes.is_empty() {
            return Err(
                "Prediction failed: ART2-A model has no learned prototypes. Re-run clustering."
                    .to_string(),
            );
        }
        if *expected_feature_count != selected_flat.cols {
            return Err(format!(
                "Prediction failed: ART2-A feature mismatch (bundle expects {}, got {}).",
                expected_feature_count, selected_flat.cols
            ));
        }
        if model.prototypes[0].len() != selected_flat.cols {
            return Err(format!(
                "Prediction failed: ART2-A feature mismatch (model expects {}, got {}).",
                model.prototypes[0].len(),
                selected_flat.cols
            ));
        }
        let art2a_features = match scaler {
            ScalerType::Standard(s) => s.transform(&features),
            ScalerType::MinMax(s) => s.transform(&features),
            ScalerType::None => features.clone(),
        };
        let mut cluster_ids = Vec::with_capacity(features.nrows());
        let mut similarities = Vec::with_capacity(features.nrows());
        let mut in_domains = Vec::with_capacity(features.nrows());
        for row_idx in 0..art2a_features.nrows() {
            let sample = art2a_features.row(row_idx).to_vec();
            let (cluster_id, similarity) = model.predict_one(&sample);
            cluster_ids.push(cluster_id as f64);
            similarities.push(similarity);
            in_domains.push(if model.is_in_domain(&sample) {
                1.0
            } else {
                0.0
            });
        }
        target_names.push("ART2A_ClusterId".to_string());
        target_names.push("ART2A_Similarity".to_string());
        target_names.push("ART2A_InDomain".to_string());
        target_predictions.push(cluster_ids);
        target_predictions.push(similarities);
        target_predictions.push(in_domains);
    }
    if target_predictions.is_empty() {
        return Err("Prediction failed: no outputs were generated.".to_string());
    }

    let mut rows = Vec::with_capacity(smiles.len());
    for (row_idx, smiles_value) in smiles.into_iter().enumerate() {
        if cancel_flag.load(AtomicOrdering::Relaxed) {
            return Ok(WorkerResponse::JobCanceled {
                job: WorkerJobKind::Prediction,
            });
        }
        let predictions = target_predictions
            .iter()
            .map(|column| column[row_idx])
            .collect::<Vec<_>>();
        rows.push(PredictionOutputRow {
            smiles: smiles_value,
            predictions,
        });
    }

    Ok(WorkerResponse::PredictionsReady {
        path,
        target_names,
        rows,
        skipped_rows: failed_rows,
        projection: build_projection(&projection_payload),
    })
}

/// Implements pipeline bundle.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn decode_pipeline_bundle(bytes: &[u8], primary_target: &str) -> Result<QsarPipeline, String> {
    let bundle: Vec<(String, QsarPipeline)> = bincode::deserialize(bytes)
        .map_err(|error| format!("Failed to deserialize pipeline bundle: {}", error))?;
    if bundle.is_empty() {
        return Err("Pipeline bundle is empty.".to_string());
    }
    if let Some((_, pipeline)) = bundle
        .iter()
        .find(|(target, _)| target.eq_ignore_ascii_case(primary_target))
    {
        return Ok(pipeline.clone());
    }
    Ok(bundle[0].1.clone())
}



/// Trains pipeline with optional progress callback.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn train_pipeline_with_progress(
    data: &QsarDataset,
    config: &MlTrainingConfig,
    on_progress: &mut dyn FnMut(f32, &str),
) -> Result<QsarPipeline, String> {
    if data.n_samples() < 3 {
        return Err("Need at least 3 samples for training.".to_string());
    }
    let cv_folds = config.cv_folds.min(data.n_samples()).max(2);

    if config.algorithm == MlAlgorithm::AutoQsar {
        if config.scaler != MlScalerChoice::Standard {
            return Err(
                "AutoQsar currently requires StandardScaler. Switch scaler to Standard."
                    .to_string(),
            );
        }
        let mut label_buf = String::new();
        let mut bridge = |progress: AutoQsarProgress| {
            let stage_text = match progress.stage {
                AutoQsarProgressStage::Scaling => "AutoQsar: scaling",
                AutoQsarProgressStage::CandidateStart { label } => {
                    label_buf.clear();
                    label_buf.push_str("AutoQsar: ");
                    label_buf.push_str(label);
                    label_buf.push_str(" CV start");
                    &label_buf
                }
                AutoQsarProgressStage::CandidateDone { label, q2, best_q2 } => {
                    label_buf.clear();
                    label_buf.push_str("AutoQsar: ");
                    label_buf.push_str(label);
                    label_buf.push_str(" done (q2=");
                    label_buf.push_str(&format!("{q2:.4}"));
                    label_buf.push_str(", best=");
                    label_buf.push_str(&format!("{best_q2:.4}"));
                    label_buf.push(')');
                    &label_buf
                }
                AutoQsarProgressStage::AdFit => "AutoQsar: fitting AD",
                AutoQsarProgressStage::Complete => "AutoQsar: complete",
            };
            on_progress(progress.fraction(), stage_text);
        };
        return Ok(AutoQsar::regression_with_progress(
            data,
            cv_folds,
            config.seed,
            &mut bridge,
        ));
    }

    let (scaler, scaled) = match config.scaler {
        MlScalerChoice::Standard => {
            let scaler = StandardScaler::fit(&data.features).unwrap();
            let scaled = scaler.transform(&data.features);
            (ScalerType::Standard(scaler), scaled)
        }
        MlScalerChoice::MinMax => {
            let scaler = MinMaxScaler::fit(&data.features).unwrap();
            let scaled = scaler.transform(&data.features);
            (ScalerType::MinMax(scaler), scaled)
        }
    };

    match config.algorithm {
        MlAlgorithm::AutoQsar => unreachable!(),
        MlAlgorithm::RandomForest => train_rf_pipeline(data, scaled, scaler, config, cv_folds),
        MlAlgorithm::Knn => train_knn_pipeline(data, scaled, scaler, config, cv_folds),
        MlAlgorithm::Pls => train_pls_pipeline(data, scaled, scaler, config, cv_folds),
        MlAlgorithm::ElasticNet => {
            train_elastic_net_pipeline(data, scaled, scaler, config, cv_folds)
        }
    }
}

/// Implements dataset.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn split_dataset(
    data: &QsarDataset,
    smiles: &[String],
    config: &MlTrainingConfig,
) -> Result<TrainTestSplit, String> {
    if smiles.len() != data.n_samples() {
        return Err(format!(
            "Cannot split dataset: {} smiles for {} samples.",
            smiles.len(),
            data.n_samples()
        ));
    }
    let test_fraction = config.test_fraction.clamp(0.01, 0.99);
    let split = match config.split_strategy {
        SplitStrategy::Random => random_split(data, test_fraction, config.seed),
        SplitStrategy::Stratified => stratified_split(data, test_fraction, config.seed),
        SplitStrategy::Scaffold => {
            let mut mols = Vec::with_capacity(smiles.len());
            for smi in smiles {
                let mol = parse_smiles(smi)
                    .map_err(|e| format!("Scaffold split failed to parse '{}': {}", smi, e))?;
                mols.push(mol);
            }
            let keys = scaffold_keys_for_molecules(&mols);
            scaffold_split(data, &keys, test_fraction, config.seed)
        }
    };
    Ok(split)
}

/// Implements keys for molecules.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn scaffold_keys_for_molecules(molecules: &[Molecule]) -> Vec<String> {
    molecules
        .iter()
        .map(|mol| {
            let clone = mol.clone();
            let rings = find_symmetrized_sssr_bonds(&clone);
            if rings.is_empty() {
                return format!("acyclic:{}:{}", mol.num_atoms(), mol.num_bonds());
            }
            let mut ring_sizes: Vec<usize> = rings.iter().map(|r| r.len()).collect();
            ring_sizes.sort_unstable();
            let mut ring_atoms = BTreeSet::new();
            for ring in rings {
                for &bond_idx in &ring {
                    let bond = &clone.bonds[bond_idx as usize];
                    ring_atoms.insert(bond.source as usize);
                    ring_atoms.insert(bond.target as usize);
                }
            }
            let aromatic_atoms = ring_atoms
                .iter()
                .filter(|&&idx| clone.atoms[idx].is_aromatic())
                .count();
            format!(
                "rings:{}|atoms:{}|arom:{}|ring_atoms:{}",
                ring_sizes
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join("-"),
                clone.num_atoms(),
                aromatic_atoms,
                ring_atoms.len()
            )
        })
        .collect()
}

/// Trains rf pipeline.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn train_rf_pipeline(
    data: &QsarDataset,
    scaled: Array2<f64>,
    scaler: ScalerType,
    config: &MlTrainingConfig,
    cv_folds: usize,
) -> Result<QsarPipeline, String> {
    let n_trees = config.rf_n_estimators.max(1);
    let max_depth = config.rf_max_depth.max(1);
    let min_samples_split = config.rf_min_samples_split.max(2);
    let max_features_fraction = config.rf_max_features_fraction.clamp(0.01, 1.0);

    let q2 = cv_regression_q2_configurable(
        &scaled,
        &data.targets,
        cv_folds,
        config.seed,
        |train_x, train_y| {
            let model = RandomForestRegressor::fit(train_x.view(), train_y.view(),
                n_trees,
                max_depth,
                min_samples_split,
                max_features_fraction,
                config.seed,
            );
            ModelType::RandomForestRegressor(model)
        },
    );

    let model = RandomForestRegressor::fit(scaled.view(), data.targets.view(),
        n_trees,
        max_depth,
        min_samples_split,
        max_features_fraction,
        config.seed,
    );
    let train_predictions = model.predict(scaled.view());
    let train_targets = data.targets.to_vec();
    let train_preds = train_predictions.to_vec();
    let train_r2 = metrics::r_squared(&train_targets, &train_preds);
    let ad = KnnDistanceAD::fit(&scaled, 5, 2.0);

    Ok(QsarPipeline {
        name: format!("RF-{n_trees}"),
        task: TaskType::Regression,
        scaler,
        pca: None,
        model: ModelType::RandomForestRegressor(model),
        ad: Some(ad),
        feature_names: data.feature_names.clone(),
        train_r2,
        cv_q2: q2,
    })
}

/// Trains knn pipeline.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn train_knn_pipeline(
    data: &QsarDataset,
    scaled: Array2<f64>,
    scaler: ScalerType,
    config: &MlTrainingConfig,
    cv_folds: usize,
) -> Result<QsarPipeline, String> {
    let k = config
        .knn_k
        .max(1)
        .min(data.n_samples().saturating_sub(1).max(1));

    let q2 = cv_regression_q2_configurable(
        &scaled,
        &data.targets,
        cv_folds,
        config.seed,
        |train_x, train_y| {
            let train_k = k.min(train_x.nrows().max(1));
            let model = KnnRegressor::fit(train_x.view(), train_y.view(),
                train_k,
                DistanceMetric::Euclidean,
            );
            ModelType::KnnRegressor(model)
        },
    );

    let model = KnnRegressor::fit(scaled.view(), data.targets.view(),
        k.min(scaled.nrows().max(1)),
        DistanceMetric::Euclidean,
    );
    let train_predictions = model.predict(scaled.view());
    let train_targets = data.targets.to_vec();
    let train_preds = train_predictions.to_vec();
    let train_r2 = metrics::r_squared(&train_targets, &train_preds);
    let ad = KnnDistanceAD::fit(&scaled, 5, 2.0);

    Ok(QsarPipeline {
        name: format!("kNN-{k}"),
        task: TaskType::Regression,
        scaler,
        pca: None,
        model: ModelType::KnnRegressor(model),
        ad: Some(ad),
        feature_names: data.feature_names.clone(),
        train_r2,
        cv_q2: q2,
    })
}

/// Trains pls pipeline.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn train_pls_pipeline(
    data: &QsarDataset,
    scaled: Array2<f64>,
    scaler: ScalerType,
    config: &MlTrainingConfig,
    cv_folds: usize,
) -> Result<QsarPipeline, String> {
    let max_allowed = data.n_features().min(data.n_samples()).max(1);
    let components = config.pls_components.clamp(1, max_allowed);

    let q2 = cv_regression_q2_configurable(
        &scaled,
        &data.targets,
        cv_folds,
        config.seed,
        |train_x, train_y| {
            let nc = components.min(train_x.ncols().min(train_x.nrows()).max(1));
            let model = PlsRegression::fit(train_x.view(), train_y.view(), nc);
            ModelType::PlsRegression(model)
        },
    );

    let model = PlsRegression::fit(scaled.view(), data.targets.view(), components);
    let train_predictions = model.predict(scaled.view());
    let train_targets = data.targets.to_vec();
    let train_preds = train_predictions.to_vec();
    let train_r2 = metrics::r_squared(&train_targets, &train_preds);
    let ad = KnnDistanceAD::fit(&scaled, 5, 2.0);

    Ok(QsarPipeline {
        name: format!("PLS-{components}"),
        task: TaskType::Regression,
        scaler,
        pca: None,
        model: ModelType::PlsRegression(model),
        ad: Some(ad),
        feature_names: data.feature_names.clone(),
        train_r2,
        cv_q2: q2,
    })
}

/// Trains elastic net pipeline.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn train_elastic_net_pipeline(
    data: &QsarDataset,
    scaled: Array2<f64>,
    scaler: ScalerType,
    config: &MlTrainingConfig,
    cv_folds: usize,
) -> Result<QsarPipeline, String> {
    let alpha = config.en_alpha.max(1e-12);
    let l1_ratio = config.en_l1_ratio.clamp(0.0, 1.0);
    let max_iter = config.en_max_iter.max(1);
    let tol = config.en_tol.max(1e-12);

    let q2 = cv_regression_q2_configurable(
        &scaled,
        &data.targets,
        cv_folds,
        config.seed,
        |train_x, train_y| {
            let model = ElasticNet::fit(train_x.view(), train_y.view(), alpha, l1_ratio, max_iter, tol);
            ModelType::ElasticNet(model)
        },
    );

    let model = ElasticNet::fit(scaled.view(), data.targets.view(), alpha, l1_ratio, max_iter, tol);
    let train_predictions = model.predict(scaled.view());
    let train_targets = data.targets.to_vec();
    let train_preds = train_predictions.to_vec();
    let train_r2 = metrics::r_squared(&train_targets, &train_preds);
    let ad = KnnDistanceAD::fit(&scaled, 5, 2.0);

    Ok(QsarPipeline {
        name: format!("EN-a{alpha:.3}-l1{l1_ratio:.2}"),
        task: TaskType::Regression,
        scaler,
        pca: None,
        model: ModelType::ElasticNet(model),
        ad: Some(ad),
        feature_names: data.feature_names.clone(),
        train_r2,
        cv_q2: q2,
    })
}

/// Implements regression q2 configurable.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn cv_regression_q2_configurable<F>(
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
    let folds = n_folds.min(n).max(2);
    let splits = cv::k_fold(n, folds, seed);
    let mut all_preds = vec![0.0f64; n];

    for (train_idx, test_idx) in &splits.folds {
        let train_x = select_rows(features, train_idx);
        let train_y = select_elements(targets, train_idx);
        let test_x = select_rows(features, test_idx);

        let model = build_model(&train_x, &train_y);
        let preds = match &model {
            ModelType::KnnRegressor(m) => m.predict(test_x.view()),
            ModelType::RandomForestRegressor(m) => m.predict(test_x.view()),
            ModelType::PlsRegression(m) => m.predict(test_x.view()),
            ModelType::ElasticNet(m) => m.predict(test_x.view()),
            _ => Array1::zeros(test_idx.len()),
        };

        for (i, &idx) in test_idx.iter().enumerate() {
            all_preds[idx] = preds[i];
        }
    }

    let target_values = targets.to_vec();
    metrics::q_squared(&target_values, &all_preds)
}

/// Implements correlation.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn pearson_correlation(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return 0.0;
    }
    let n = actual.len() as f64;
    let mean_a = actual.iter().sum::<f64>() / n;
    let mean_p = predicted.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_p = 0.0;
    for (&a, &p) in actual.iter().zip(predicted.iter()) {
        let da = a - mean_a;
        let dp = p - mean_p;
        num += da * dp;
        den_a += da * da;
        den_p += dp * dp;
    }
    let denom = (den_a * den_p).sqrt();
    if denom <= 1e-15 { 0.0 } else { num / denom }
}

/// Implements export predictions.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn handle_export_predictions(
    path: PathBuf,
    target_names: Vec<String>,
    rows: Vec<PredictionOutputRow>,
) -> Result<WorkerResponse, String> {
    if rows.is_empty() {
        return Err("No predictions to export.".to_string());
    }
    if target_names.is_empty() {
        return Err("No prediction target columns available for export.".to_string());
    }
    if rows
        .iter()
        .any(|row| row.predictions.len() != target_names.len())
    {
        return Err("Prediction export failed: row/target column mismatch.".to_string());
    }

    let mut output = String::from("SMILES");
    for target_name in &target_names {
        output.push(',');
        output.push_str(target_name);
    }
    output.push('\n');

    for row in &rows {
        output.push_str(&row.smiles);
        for prediction in &row.predictions {
            output.push(',');
            output.push_str(&format!("{:.8}", prediction));
        }
        output.push('\n');
    }

    fs::write(&path, output).map_err(|error| {
        format!(
            "Failed to write predictions to '{}': {}",
            path.display(),
            error
        )
    })?;

    Ok(WorkerResponse::PredictionsExported {
        path,
        count: rows.len(),
    })
}

/// Exports ART2-A cluster assignments with SMILES for downstream inspection.
fn handle_export_clusters(
    path: PathBuf,
    rows: Vec<ClusterExportRow>,
) -> Result<WorkerResponse, String> {
    if rows.is_empty() {
        return Err("No ART2-A cluster rows available for export.".to_string());
    }

    let include_similarity = rows.iter().any(|row| row.similarity.is_some());
    let include_in_domain = rows.iter().any(|row| row.in_domain.is_some());

    let mut output = String::from("SMILES,ART2A_ClusterId");
    if include_similarity {
        output.push_str(",ART2A_Similarity");
    }
    if include_in_domain {
        output.push_str(",ART2A_InDomain");
    }
    output.push('\n');

    for row in &rows {
        output.push_str(&row.smiles);
        output.push(',');
        output.push_str(&row.cluster_id.to_string());
        if include_similarity {
            output.push(',');
            if let Some(value) = row.similarity {
                output.push_str(&format!("{:.8}", value));
            }
        }
        if include_in_domain {
            output.push(',');
            if let Some(value) = row.in_domain {
                output.push_str(if value { "1" } else { "0" });
            }
        }
        output.push('\n');
    }

    fs::write(&path, output).map_err(|error| {
        format!(
            "Failed to write ART2-A clusters to '{}': {}",
            path.display(),
            error
        )
    })?;

    Ok(WorkerResponse::ClustersExported {
        path,
        count: rows.len(),
    })
}

/// Exports a persisted model bundle artifact.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn handle_export_model_bundle(
    path: PathBuf,
    bundle: PersistedModelBundle,
) -> Result<WorkerResponse, String> {
    if bundle.pipeline_bundle_bytes.is_none() && bundle.art2a_model_bytes.is_none() {
        return Err("Cannot export model bundle: no model payload present.".to_string());
    }
    if bundle.descriptors.is_empty() {
        return Err("Cannot export model bundle: descriptor set is empty.".to_string());
    }
    if bundle
        .descriptors
        .iter()
        .any(|name| descriptor_from_name(name).is_none())
    {
        return Err(
            "Cannot export model bundle: descriptor set contains unknown names.".to_string(),
        );
    }
    if bundle.expected_feature_names.is_empty() {
        return Err("Cannot export model bundle: expected feature names are empty.".to_string());
    }
    if bundle.selected_feature_indices.is_empty() {
        return Err("Cannot export model bundle: selected feature indices are empty.".to_string());
    }
    if bundle
        .selected_feature_indices
        .iter()
        .any(|&idx| idx >= bundle.expected_feature_names.len())
    {
        return Err(format!(
            "Cannot export model bundle: selected feature indices exceed feature-name width (indices={}, width={}).",
            bundle.selected_feature_indices.len(),
            bundle.expected_feature_names.len()
        ));
    }

    let bytes = bincode::serialize(&bundle)
        .map_err(|error| format!("Failed to serialize model bundle: {}", error))?;
    fs::write(&path, bytes).map_err(|error| {
        format!(
            "Failed to write model bundle to '{}': {}",
            path.display(),
            error
        )
    })?;
    Ok(WorkerResponse::ModelBundleExported { path })
}

/// Imports a persisted model bundle artifact.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn handle_import_model_bundle(path: PathBuf) -> Result<WorkerResponse, String> {
    let bytes = fs::read(&path).map_err(|error| {
        format!(
            "Failed to read model bundle from '{}': {}",
            path.display(),
            error
        )
    })?;
    let bundle: PersistedModelBundle = bincode::deserialize(&bytes)
        .map_err(|error| format!("Failed to deserialize model bundle: {}", error))?;

    if bundle.schema_version != MODEL_BUNDLE_SCHEMA_VERSION {
        return Err(format!(
            "Unsupported model bundle schema version {} (expected {}).",
            bundle.schema_version, MODEL_BUNDLE_SCHEMA_VERSION
        ));
    }
    if bundle.pipeline_bundle_bytes.is_none() && bundle.art2a_model_bytes.is_none() {
        return Err("Imported model bundle contains no model payload.".to_string());
    }
    if bundle.descriptors.is_empty() {
        return Err("Imported model bundle descriptor set is empty.".to_string());
    }
    let descriptors = bundle
        .descriptors
        .iter()
        .map(|name| {
            descriptor_from_name(name).ok_or_else(|| {
                format!(
                    "Imported model bundle references unknown descriptor '{}'.",
                    name
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if bundle.expected_feature_names.is_empty() {
        return Err("Imported model bundle expected feature names are empty.".to_string());
    }
    if bundle.selected_feature_indices.is_empty() {
        return Err("Imported model bundle selected feature indices are empty.".to_string());
    }
    let expected_names = descriptor_feature_names(&descriptors);
    for &selected_idx in &bundle.selected_feature_indices {
        let bundle_name = bundle
            .expected_feature_names
            .get(selected_idx)
            .ok_or_else(|| {
                format!(
                    "Imported model bundle missing expected feature name at selected index {}.",
                    selected_idx
                )
            })?;
        if let Some(csv_name) = bundle_name.strip_prefix("CSV::") {
            if csv_name.is_empty() {
                return Err(format!(
                    "Imported model bundle has invalid CSV feature marker at selected index {}.",
                    selected_idx
                ));
            }
            continue;
        }
        let expected_name = expected_names.get(selected_idx).ok_or_else(|| {
            format!(
                "Imported model bundle descriptor layout missing selected index {}.",
                selected_idx
            )
        })?;
        if expected_name != bundle_name {
            return Err(format!(
                "Imported model bundle descriptor layout mismatch at selected index {} ('{}' vs '{}').",
                selected_idx, expected_name, bundle_name
            ));
        }
    }

    Ok(WorkerResponse::ModelBundleImported { path, bundle })
}

/// Implements apply variance filter.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn handle_apply_variance_filter(
    matrix: FeatureMatrixPayload,
    threshold: f64,
) -> Result<WorkerResponse, String> {
    if matrix.rows == 0 || matrix.cols == 0 {
        return Err("Cannot apply variance filter: feature matrix is empty.".to_string());
    }
    let mut filtered = matrix;
    let data = Array2::from_shape_vec((filtered.rows, filtered.cols), filtered.data.clone())
        .map_err(|error| format!("Failed to shape matrix for variance filter: {}", error))?;
    let keep = variance_threshold(&data, threshold);
    if keep.is_empty() {
        return Err(format!(
            "Variance filter removed all features at threshold {:.2e}.",
            threshold
        ));
    }
    if keep.len() == filtered.cols {
        let profiles = build_profiles(&filtered);
        let projection = build_projection(&filtered);
        return Ok(WorkerResponse::FeatureFilterApplied {
            matrix: filtered,
            profiles,
            projection,
            removed_features: 0,
            filter_label: format!("Variance threshold ({:.2e})", threshold),
        });
    }
    filtered = select_feature_columns(filtered, &keep)?;
    let removed_features = data.ncols() - keep.len();
    let profiles = build_profiles(&filtered);
    let projection = build_projection(&filtered);
    Ok(WorkerResponse::FeatureFilterApplied {
        matrix: filtered,
        profiles,
        projection,
        removed_features,
        filter_label: format!("Variance threshold ({:.2e})", threshold),
    })
}

/// Implements apply correlation filter.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn handle_apply_correlation_filter(
    matrix: FeatureMatrixPayload,
    threshold: f64,
) -> Result<WorkerResponse, String> {
    if matrix.rows == 0 || matrix.cols == 0 {
        return Err("Cannot apply correlation filter: feature matrix is empty.".to_string());
    }
    if matrix.cols < 2 {
        return Err("Correlation filter needs at least 2 features.".to_string());
    }
    let mut filtered = matrix;
    let data = Array2::from_shape_vec((filtered.rows, filtered.cols), filtered.data.clone())
        .map_err(|error| format!("Failed to shape matrix for correlation filter: {}", error))?;
    let keep = correlation_filter(&data, threshold);
    if keep.is_empty() {
        return Err(format!(
            "Correlation filter removed all features at threshold {:.2}.",
            threshold
        ));
    }
    if keep.len() == filtered.cols {
        let profiles = build_profiles(&filtered);
        let projection = build_projection(&filtered);
        return Ok(WorkerResponse::FeatureFilterApplied {
            matrix: filtered,
            profiles,
            projection,
            removed_features: 0,
            filter_label: format!("Correlation filter (|r| > {:.2})", threshold),
        });
    }
    filtered = select_feature_columns(filtered, &keep)?;
    let removed_features = data.ncols() - keep.len();
    let profiles = build_profiles(&filtered);
    let projection = build_projection(&filtered);
    Ok(WorkerResponse::FeatureFilterApplied {
        matrix: filtered,
        profiles,
        projection,
        removed_features,
        filter_label: format!("Correlation filter (|r| > {:.2})", threshold),
    })
}

/// Implements export feature matrix.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn handle_export_feature_matrix(
    path: PathBuf,
    smiles: Vec<String>,
    matrix: FeatureMatrixPayload,
) -> Result<WorkerResponse, String> {
    if matrix.rows == 0 || matrix.cols == 0 {
        return Err("Cannot export feature matrix: matrix is empty.".to_string());
    }
    if smiles.len() < matrix.rows {
        return Err(format!(
            "Cannot export feature matrix: {} SMILES rows for {} matrix rows.",
            smiles.len(),
            matrix.rows
        ));
    }
    let flat = FlatDescriptorMatrix {
        rows: matrix.rows,
        cols: matrix.cols,
        data: matrix.data.iter().map(|v| *v as f32).collect(),
    };
    let mut output = Vec::new();
    write_csv_flat(
        &mut output,
        &smiles[..matrix.rows],
        &matrix.feature_names,
        &flat,
    )
    .map_err(|error| format!("Failed to render feature CSV: {}", error))?;
    fs::write(&path, output).map_err(|error| {
        format!(
            "Failed to write feature matrix to '{}': {}",
            path.display(),
            error
        )
    })?;
    Ok(WorkerResponse::FeatureMatrixExported {
        path,
        rows: matrix.rows,
        cols: matrix.cols,
    })
}

/// Implements feature columns.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn select_feature_columns(
    matrix: FeatureMatrixPayload,
    keep: &[usize],
) -> Result<FeatureMatrixPayload, String> {
    if keep.is_empty() {
        return Err("Feature selection produced an empty keep-set.".to_string());
    }
    for &idx in keep {
        if idx >= matrix.cols {
            return Err(format!(
                "Feature selection index {} out of bounds for {} columns.",
                idx, matrix.cols
            ));
        }
    }
    let mut new_data = Vec::with_capacity(matrix.rows * keep.len());
    for row in 0..matrix.rows {
        let base = row * matrix.cols;
        for &col in keep {
            new_data.push(matrix.data[base + col]);
        }
    }
    let mut new_names = Vec::with_capacity(keep.len());
    let mut new_selected = Vec::with_capacity(keep.len());
    for &col in keep {
        new_names.push(
            matrix
                .feature_names
                .get(col)
                .cloned()
                .unwrap_or_else(|| format!("feature_{}", col)),
        );
        new_selected.push(
            matrix
                .selected_feature_indices
                .get(col)
                .copied()
                .unwrap_or(col),
        );
    }
    Ok(FeatureMatrixPayload {
        rows: matrix.rows,
        cols: keep.len(),
        feature_names: new_names,
        data: new_data,
        descriptors: matrix.descriptors,
        selected_feature_indices: new_selected,
    })
}

/// Loads dataset from csv.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn load_dataset_from_csv(
    path: &Path,
) -> Result<
    (
        Vec<Molecule>,
        Vec<String>,
        Vec<Vec<(Arc<str>, Box<str>)>>,
        usize,
    ),
    String,
> {
    let records = CsvReader::from_path(path)
        .map_err(|error| format!("Failed to open CSV '{}': {}", path.display(), error))?
        .with_properties(true)
        .parse()
        .map_err(|error| format!("Failed to parse CSV '{}': {}", path.display(), error))?;

    let mut molecules = Vec::new();
    let mut smiles_vec = Vec::new();
    let mut properties_vec = Vec::new();
    let mut failed_rows = 0usize;

    for record in records {
        if let Some(molecule) = record.molecule {
            molecules.push(molecule);
            smiles_vec.push(record.smiles);
            properties_vec.push(record.properties);
        } else {
            failed_rows += 1;
        }
    }

    Ok((molecules, smiles_vec, properties_vec, failed_rows))
}

/// Loads dataset from smi.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn load_dataset_from_smi(
    path: &Path,
) -> Result<
    (
        Vec<Molecule>,
        Vec<String>,
        Vec<Vec<(Arc<str>, Box<str>)>>,
        usize,
    ),
    String,
> {
    let content = fs::read_to_string(path)
        .map_err(|error| format!("Failed to read SMI '{}': {}", path.display(), error))?;

    let mut molecules = Vec::new();
    let mut smiles_vec = Vec::new();
    let mut properties_vec = Vec::new();
    let mut failed_rows = 0usize;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let smiles = trimmed
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_string();
        if smiles.is_empty() {
            continue;
        }

        match parse_smiles(&smiles) {
            Ok(molecule) => {
                molecules.push(molecule);
                smiles_vec.push(smiles);
                properties_vec.push(Vec::new());
            }
            Err(_) => {
                failed_rows += 1;
            }
        }
    }

    Ok((molecules, smiles_vec, properties_vec, failed_rows))
}

/// Implements load dataset from sdf.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
fn load_dataset_from_sdf(
    path: &Path,
) -> Result<
    (
        Vec<Molecule>,
        Vec<String>,
        Vec<Vec<(Arc<str>, Box<str>)>>,
        usize,
    ),
    String,
> {
    match ferrum_core::io::sdf::parse_sdf(path) {
        Ok(molecules) => {
            let num = molecules.len();
            let smiles_vec = vec![String::from("SDF_Molecule"); num];
            let properties_vec = vec![Vec::new(); num];
            Ok((molecules, smiles_vec, properties_vec, 0))
        }
        Err(e) => Err(format!("Failed to parse SDF '{}': {}", path.display(), e)),
    }
}

/// Implements feature names.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn descriptor_feature_names(descriptors: &[BatchDescriptor]) -> Vec<String> {
    let mut names = Vec::new();
    for descriptor in descriptors {
        let feature_len = descriptor.feature_len();
        if feature_len == 1 {
            names.push(descriptor.name().to_string());
        } else {
            for idx in 0..feature_len {
                names.push(format!("{}[{}]", descriptor.name(), idx));
            }
        }
    }
    names
}

/// Implements build prediction flat by expected names.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// System-layer note: component execution and pipeline orchestration.
fn build_prediction_flat_by_expected_names(
    source: &FlatDescriptorMatrix,
    source_feature_names: &[String],
    properties: &[Vec<(Arc<str>, Box<str>)>],
    expected_feature_names: &[String],
) -> Result<FlatDescriptorMatrix, String> {
    if expected_feature_names.is_empty() {
        return Ok(FlatDescriptorMatrix {
            rows: source.rows,
            cols: 0,
            data: Vec::new(),
        });
    }

    let mut descriptor_index_by_name = HashMap::new();
    for (idx, name) in source_feature_names.iter().enumerate() {
        descriptor_index_by_name.insert(name.as_str(), idx);
    }

    let needs_csv_features = expected_feature_names
        .iter()
        .any(|name| name.starts_with("CSV::"));
    if needs_csv_features && properties.len() != source.rows {
        return Err(format!(
            "Prediction failed: CSV feature alignment mismatch ({} property rows for {} molecules).",
            properties.len(),
            source.rows
        ));
    }

    let mut data = Vec::with_capacity(source.rows * expected_feature_names.len());
    for row_idx in 0..source.rows {
        let source_offset = row_idx * source.cols;
        let row_properties = properties.get(row_idx);
        for feature_name in expected_feature_names {
            if let Some(csv_name) = feature_name.strip_prefix("CSV::") {
                let row_properties = row_properties.ok_or_else(|| {
                    format!(
                        "Prediction failed: missing CSV properties for row #{} while resolving '{}'.",
                        row_idx + 1,
                        feature_name
                    )
                })?;
                let maybe_raw = row_properties
                    .iter()
                    .find(|(name, _)| name.as_ref() == csv_name)
                    .or_else(|| {
                        row_properties
                            .iter()
                            .find(|(name, _)| name.as_ref().eq_ignore_ascii_case(csv_name))
                    })
                    .map(|(_, value)| value.trim());
                let raw = maybe_raw.ok_or_else(|| {
                    format!(
                        "Prediction failed: CSV feature '{}' missing at row #{}.",
                        csv_name,
                        row_idx + 1
                    )
                })?;
                let parsed = raw.parse::<f32>().map_err(|_| {
                    format!(
                        "Prediction failed: CSV feature '{}' has non-numeric value '{}' at row #{}.",
                        csv_name,
                        raw,
                        row_idx + 1
                    )
                })?;
                data.push(parsed);
                continue;
            }

            let col_idx = descriptor_index_by_name
                .get(feature_name.as_str())
                .copied()
                .ok_or_else(|| {
                    format!(
                        "Prediction failed: expected descriptor feature '{}' is unavailable in the selected descriptor set.",
                        feature_name
                    )
                })?;
            data.push(source.data[source_offset + col_idx]);
        }
    }

    Ok(FlatDescriptorMatrix {
        rows: source.rows,
        cols: expected_feature_names.len(),
        data,
    })
}

/// Resolves a descriptor by its canonical batch name.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn descriptor_from_name(name: &str) -> Option<BatchDescriptor> {
    BatchDescriptor::all()
        .iter()
        .copied()
        .find(|descriptor| descriptor.name() == name)
}

/// Implements targets.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn extract_targets(
    properties: &[Vec<(Arc<str>, Box<str>)>],
    target_column: &str,
) -> Result<Vec<f64>, String> {
    let mut targets = Vec::with_capacity(properties.len());

    for (row_idx, row_props) in properties.iter().enumerate() {
        let maybe_value = row_props
            .iter()
            .find(|(name, _)| name.as_ref() == target_column)
            .or_else(|| {
                row_props
                    .iter()
                    .find(|(name, _)| name.as_ref().eq_ignore_ascii_case(target_column))
            })
            .map(|(_, value)| value.trim().to_string());

        let Some(raw_value) = maybe_value else {
            return Err(format!(
                "Target column '{}' not found in loaded CSV properties (first missing row #{})",
                target_column,
                row_idx + 1
            ));
        };

        let parsed = raw_value.parse::<f64>().map_err(|_| {
            format!(
                "Target column '{}' contains non-numeric value '{}' at row #{}",
                target_column,
                raw_value,
                row_idx + 1
            )
        })?;

        targets.push(parsed);
    }

    Ok(targets)
}

/// Builds profiles.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn build_profiles(matrix: &FeatureMatrixPayload) -> Vec<DescriptorProfileRow> {
    if matrix.rows == 0 || matrix.cols == 0 {
        return Vec::new();
    }

    let n = matrix.rows as f64;
    let mut output = Vec::with_capacity(matrix.cols);

    for col in 0..matrix.cols {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        let mut sum = 0.0;

        for row in 0..matrix.rows {
            let value = matrix.data[row * matrix.cols + col];
            min_v = min_v.min(value);
            max_v = max_v.max(value);
            sum += value;
        }

        let mean = sum / n;
        let mut var_sum = 0.0;
        for row in 0..matrix.rows {
            let value = matrix.data[row * matrix.cols + col];
            var_sum += (value - mean) * (value - mean);
        }

        let variance = var_sum / n;
        output.push(DescriptorProfileRow {
            name: matrix
                .feature_names
                .get(col)
                .cloned()
                .unwrap_or_else(|| format!("feature_{}", col)),
            mean,
            min: min_v,
            max: max_v,
            stddev: variance.sqrt(),
            zero_variance: variance < 1e-12,
        });
    }

    output
}

/// Builds projection.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn build_projection(matrix: &FeatureMatrixPayload) -> Option<ProjectionData> {
    if matrix.rows < 3 || matrix.cols < 2 {
        return None;
    }

    let sanitized_matrix_data = matrix
        .data
        .iter()
        .map(|v| if v.is_finite() { *v } else { 0.0 })
        .collect::<Vec<_>>();
    let data = Array2::from_shape_vec((matrix.rows, matrix.cols), sanitized_matrix_data).ok()?;
    let scaler = StandardScaler::fit(&data).unwrap();
    let mut scaled = scaler.transform(&data);
    scaled.iter_mut().for_each(|value| {
        if !value.is_finite() {
            *value = 0.0;
        }
    });
    let (pca, mut projected) = Pca::fit_transform(&scaled, 2).unwrap();
    projected.iter_mut().for_each(|value| {
        if !value.is_finite() {
            *value = 0.0;
        }
    });

    let pc1 = projected.column(0);
    let pc2 = projected.column(1);

    let x_min = pc1.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = pc1.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_min = pc2.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = pc2.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (x_max - x_min).max(0.01) * 0.05;
    let y_pad = (y_max - y_min).max(0.01) * 0.05;

    Some(ProjectionData {
        points: pc1.iter().zip(pc2.iter()).map(|(&x, &y)| (x, y)).collect(),
        x_bounds: [x_min - x_pad, x_max + x_pad],
        y_bounds: [y_min - y_pad, y_max + y_pad],
        explained_variance: [
            pca.explained_variance_ratio
                .get(0)
                .copied()
                .filter(|value| value.is_finite())
                .unwrap_or(0.0),
            pca.explained_variance_ratio
                .get(1)
                .copied()
                .filter(|value| value.is_finite())
                .unwrap_or(0.0),
        ],
    })
}

/// Implements points by values.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn bucket_points_by_values(
    points: &[(f64, f64)],
    values: &[f64],
) -> Option<(Vec<(f64, f64)>, Vec<(f64, f64)>, Vec<(f64, f64)>)> {
    if points.is_empty() || points.len() != values.len() {
        return None;
    }

    let min_v = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !min_v.is_finite() || !max_v.is_finite() {
        return None;
    }

    let span = (max_v - min_v).max(1e-12);
    let low_cut = min_v + span / 3.0;
    let high_cut = min_v + (2.0 * span / 3.0);

    let mut low = Vec::new();
    let mut mid = Vec::new();
    let mut high = Vec::new();

    for (point, value) in points.iter().zip(values.iter().copied()) {
        if value <= low_cut {
            low.push(*point);
        } else if value <= high_cut {
            mid.push(*point);
        } else {
            high.push(*point);
        }
    }

    Some((low, mid, high))
}

/// Implements param label.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn ml_param_label(param: MlParamKey) -> &'static str {
    match param {
        MlParamKey::CvFolds => "CV folds",
        MlParamKey::TestFraction => "Test fraction",
        MlParamKey::Seed => "Seed",
        MlParamKey::Art2aVigilance => "ART2-A rho",
        MlParamKey::Art2aLearningRate => "ART2-A eta",
        MlParamKey::Art2aMaxIter => "ART2-A max iter",
        MlParamKey::KnnK => "k-NN k",
        MlParamKey::RfTrees => "RF trees",
        MlParamKey::RfMaxDepth => "RF max depth",
        MlParamKey::RfMinSamplesSplit => "RF min split",
        MlParamKey::RfMaxFeaturesFraction => "RF max feat frac",
        MlParamKey::PlsComponents => "PLS components",
        MlParamKey::EnAlpha => "EN alpha",
        MlParamKey::EnL1Ratio => "EN l1_ratio",
        MlParamKey::EnMaxIter => "EN max iter",
        MlParamKey::EnTol => "EN tol",
    }
}

/// Implements rows.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
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
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn select_elements(data: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        out.push(data[idx]);
    }
    Array1::from_vec(out)
}

/// Implements points by clusters.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn bucket_points_by_clusters(
    points: &[(f64, f64)],
    labels: &[usize],
    max_groups: usize,
) -> Option<Vec<Vec<(f64, f64)>>> {
    if points.is_empty() || labels.len() != points.len() || max_groups == 0 {
        return None;
    }
    let max_label = labels.iter().copied().max()?;
    let mut buckets = vec![Vec::new(); max_label + 1];
    for (point, &label) in points.iter().zip(labels.iter()) {
        buckets[label].push(*point);
    }
    // Keep largest groups first, merge remainder into final bucket.
    let mut indexed: Vec<(usize, Vec<(f64, f64)>)> = buckets
        .into_iter()
        .enumerate()
        .filter(|(_, b)| !b.is_empty())
        .collect();
    indexed.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    if indexed.is_empty() {
        return None;
    }
    if indexed.len() <= max_groups {
        return Some(indexed.into_iter().map(|(_, b)| b).collect());
    }
    let mut kept: Vec<Vec<(f64, f64)>> = indexed
        .iter()
        .take(max_groups.saturating_sub(1))
        .map(|(_, b)| b.clone())
        .collect();
    let mut merged = Vec::new();
    for (_, bucket) in indexed.into_iter().skip(max_groups.saturating_sub(1)) {
        merged.extend(bucket);
    }
    kept.push(merged);
    Some(kept)
}

/// Implements cluster centroids.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn compute_cluster_centroids(points: &[(f64, f64)], labels: &[usize]) -> Vec<(f64, f64)> {
    if points.is_empty() || labels.len() != points.len() {
        return Vec::new();
    }
    let mut sums: HashMap<usize, (f64, f64, usize)> = HashMap::new();
    for ((x, y), &label) in points.iter().zip(labels.iter()) {
        let entry = sums.entry(label).or_insert((0.0, 0.0, 0));
        entry.0 += *x;
        entry.1 += *y;
        entry.2 += 1;
    }
    let mut items = sums.into_iter().collect::<Vec<_>>();
    items.sort_by_key(|(label, _)| *label);
    items
        .into_iter()
        .filter_map(|(_, (sum_x, sum_y, count))| {
            if count == 0 {
                None
            } else {
                Some((sum_x / count as f64, sum_y / count as f64))
            }
        })
        .collect()
}

/// Implements points by binary flags.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn bucket_points_by_binary_flags(
    points: &[(f64, f64)],
    flags: &[bool],
) -> Option<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
    if points.is_empty() || flags.len() != points.len() {
        return None;
    }
    let mut in_domain = Vec::new();
    let mut out_domain = Vec::new();
    for (point, in_flag) in points.iter().zip(flags.iter().copied()) {
        if in_flag {
            in_domain.push(*point);
        } else {
            out_domain.push(*point);
        }
    }
    Some((in_domain, out_domain))
}

/// Computes cluster-size histogram from per-row cluster labels.
fn cluster_size_histogram(labels: &[usize]) -> Vec<usize> {
    if labels.is_empty() {
        return Vec::new();
    }
    let max_label = labels.iter().copied().max().unwrap_or(0);
    let mut sizes = vec![0usize; max_label + 1];
    for &label in labels {
        sizes[label] += 1;
    }
    sizes
}

/// Rebuilds ART2-A cluster summary rows from prediction outputs.
fn cluster_summaries_from_predictions(
    rows: &[PredictionOutputRow],
    cluster_idx: usize,
    similarity_idx: Option<usize>,
) -> Vec<Art2aClusterSummary> {
    if rows.is_empty() {
        return Vec::new();
    }
    let mut bucket: HashMap<usize, Vec<f64>> = HashMap::new();
    for row in rows {
        let Some(cluster_value) = row.predictions.get(cluster_idx).copied() else {
            continue;
        };
        if !cluster_value.is_finite() || cluster_value < 0.0 {
            continue;
        }
        let cluster_id = cluster_value.round() as usize;
        let similarity = similarity_idx
            .and_then(|idx| row.predictions.get(idx).copied())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);
        bucket.entry(cluster_id).or_default().push(similarity);
    }
    if bucket.is_empty() {
        return Vec::new();
    }

    let total = bucket.values().map(Vec::len).sum::<usize>().max(1) as f64;
    let mut cluster_ids = bucket.keys().copied().collect::<Vec<_>>();
    cluster_ids.sort_unstable();
    cluster_ids
        .into_iter()
        .map(|cluster_id| {
            let sims = bucket.remove(&cluster_id).unwrap_or_default();
            let size = sims.len();
            let (mean_similarity, min_similarity, max_similarity) = if sims.is_empty() {
                (0.0, 0.0, 0.0)
            } else {
                let mean = sims.iter().sum::<f64>() / sims.len() as f64;
                let min = sims.iter().copied().fold(f64::INFINITY, f64::min);
                let max = sims.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (mean, min, max)
            };
            Art2aClusterSummary {
                cluster_id,
                size,
                fraction: size as f64 / total,
                mean_similarity,
                min_similarity,
                max_similarity,
            }
        })
        .collect()
}

/// Implements top feature importances.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn extract_top_feature_importances(
    pipeline: &QsarPipeline,
    feature_names: &[String],
    top_n: usize,
) -> Vec<(String, f64)> {
    if top_n == 0 || feature_names.is_empty() {
        return Vec::new();
    }

    let raw_scores: Option<Vec<f64>> = match &pipeline.model {
        ModelType::RandomForestRegressor(model) => Some(model.feature_importances.clone()),
        ModelType::PlsRegression(model) => {
            Some(model.coefficients.iter().map(|v| v.abs()).collect())
        }
        ModelType::ElasticNet(model) => Some(model.coefficients.iter().map(|v| v.abs()).collect()),
        _ => None,
    };

    let Some(mut scores) = raw_scores else {
        return Vec::new();
    };

    if scores.len() != feature_names.len() {
        let limit = scores.len().min(feature_names.len());
        scores.truncate(limit);
    }

    let mut indexed = scores
        .into_iter()
        .enumerate()
        .map(|(idx, score)| {
            (
                feature_names
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("feature_{}", idx)),
                score,
            )
        })
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    indexed.into_iter().take(top_n).collect()
}

/// Implements distribution bins.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn residual_distribution_bins(
    residuals: &[f64],
    mean: f64,
    stddev: f64,
) -> [(&'static str, usize); 5] {
    if residuals.is_empty() || stddev <= 1e-12 {
        return [
            ("<= -2σ", 0),
            ("(-2σ, -1σ]", 0),
            ("(-1σ, +1σ]", residuals.len()),
            ("(+1σ, +2σ]", 0),
            ("> +2σ", 0),
        ];
    }

    let mut bins = [0usize; 5];
    for residual in residuals {
        let z = (*residual - mean) / stddev;
        let idx = if z <= -2.0 {
            0
        } else if z <= -1.0 {
            1
        } else if z <= 1.0 {
            2
        } else if z <= 2.0 {
            3
        } else {
            4
        };
        bins[idx] += 1;
    }

    [
        ("<= -2σ", bins[0]),
        ("(-2σ, -1σ]", bins[1]),
        ("(-1σ, +1σ]", bins[2]),
        ("(+1σ, +2σ]", bins[3]),
        ("> +2σ", bins[4]),
    ]
}

/// Implements label.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn truncate_label(input: &str, max_len: usize) -> String {
    if input.chars().count() <= max_len {
        return input.to_string();
    }

    let mut out = input
        .chars()
        .take(max_len.saturating_sub(1))
        .collect::<String>();
    out.push('~');
    out
}

/// Reads browser entries.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn read_browser_entries(dir: &Path, purpose: BrowserPurpose) -> io::Result<Vec<PathBuf>> {
    let mut entries = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .file_name()
            .map(|name| name.to_string_lossy().starts_with('.'))
            .unwrap_or(false)
        {
            continue;
        }

        if path.is_dir() || is_supported_browser_file(&path, purpose) {
            entries.push(path);
        }
    }

    entries.sort_by(|a, b| match (a.is_dir(), b.is_dir()) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => a
            .file_name()
            .and_then(|name| name.to_str())
            .cmp(&b.file_name().and_then(|name| name.to_str())),
    });

    Ok(entries)
}

/// Checks whether supported data file.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn is_supported_data_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("csv") | Some("smi") | Some("smiles")
    )
}

/// Checks whether supported model bundle file.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn is_supported_model_bundle_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("fcbundle") | Some("bin")
    )
}

/// Checks whether supported browser file.
///
/// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn is_supported_browser_file(path: &Path, purpose: BrowserPurpose) -> bool {
    match purpose {
        BrowserPurpose::Dataset | BrowserPurpose::Prediction => is_supported_data_file(path),
        BrowserPurpose::ModelImport => is_supported_model_bundle_file(path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    /// Implements canceled flag.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn canceled_flag() -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        flag.store(true, Ordering::Relaxed);
        flag
    }

    /// Implements empty matrix payload.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn empty_matrix_payload() -> FeatureMatrixPayload {
        FeatureMatrixPayload {
            rows: 0,
            cols: 0,
            feature_names: Vec::new(),
            data: Vec::new(),
            descriptors: Vec::new(),
            selected_feature_indices: Vec::new(),
        }
    }

    #[test]
    /// Implements test supported browser file filters by purpose.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_supported_browser_file_filters_by_purpose() {
        assert!(is_supported_browser_file(
            Path::new("data.csv"),
            BrowserPurpose::Dataset
        ));
        assert!(is_supported_browser_file(
            Path::new("data.smi"),
            BrowserPurpose::Prediction
        ));
        assert!(!is_supported_browser_file(
            Path::new("model.fcbundle"),
            BrowserPurpose::Dataset
        ));

        assert!(is_supported_browser_file(
            Path::new("model.fcbundle"),
            BrowserPurpose::ModelImport
        ));
        assert!(is_supported_browser_file(
            Path::new("model.bin"),
            BrowserPurpose::ModelImport
        ));
        assert!(!is_supported_browser_file(
            Path::new("data.csv"),
            BrowserPurpose::ModelImport
        ));
    }

    /// Implements make test app.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn make_test_app() -> App {
        let (worker_tx, _worker_rx_unused) = mpsc::channel::<WorkerRequest>();
        let (_response_tx_unused, worker_rx) = mpsc::channel::<WorkerResponse>();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        App::new(worker_tx, worker_rx, cancel_flag)
    }

    #[test]
    /// Implements test descriptor handler returns job canceled when flag set.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_descriptor_handler_returns_job_canceled_when_flag_set() {
        let flag = canceled_flag();
        let response = handle_calculate_descriptors(Vec::new(), Vec::new(), None, &flag)
            .expect("canceled path should return Ok(JobCanceled)");
        assert!(matches!(
            response,
            WorkerResponse::JobCanceled {
                job: WorkerJobKind::DescriptorCalculation
            }
        ));
    }

    #[test]
    /// Implements test training handler returns job canceled when flag set.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_training_handler_returns_job_canceled_when_flag_set() {
        let flag = canceled_flag();
        let response = handle_train_model(
            empty_matrix_payload(),
            Vec::new(),
            Vec::new(),
            MlTrainingConfig::default(),
            Vec::new(),
            None,
            &flag,
        )
        .expect("canceled path should return Ok(JobCanceled)");
        assert!(matches!(
            response,
            WorkerResponse::JobCanceled {
                job: WorkerJobKind::ModelTraining
            }
        ));
    }

    #[test]
    /// Implements test art2a handler returns job canceled when flag set.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_art2a_handler_returns_job_canceled_when_flag_set() {
        let flag = canceled_flag();
        let response = handle_art2a_clustering(
            empty_matrix_payload(),
            Art2aConfig {
                vigilance: DEFAULT_ART2A_VIGILANCE,
                learning_rate: DEFAULT_ART2A_LEARNING_RATE,
                max_iter: DEFAULT_ART2A_MAX_ITER,
            },
            MlScalerChoice::Standard,
            None,
            &flag,
        )
        .expect("canceled path should return Ok(JobCanceled)");
        assert!(matches!(
            response,
            WorkerResponse::JobCanceled {
                job: WorkerJobKind::Art2aClustering
            }
        ));
    }

    #[test]
    /// Implements test prediction handler returns job canceled when flag set.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_prediction_handler_returns_job_canceled_when_flag_set() {
        let flag = canceled_flag();
        let response = handle_predict_file(
            PathBuf::from("/tmp/does_not_matter.csv"),
            None,
            None,
            vec![BatchDescriptor::MolecularWeight],
            Vec::new(),
            Vec::new(),
            &flag,
        )
        .expect("canceled path should return Ok(JobCanceled)");
        assert!(matches!(
            response,
            WorkerResponse::JobCanceled {
                job: WorkerJobKind::Prediction
            }
        ));
    }

    #[test]
    /// Implements test fast ml preset only enables preset descriptors.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_fast_ml_preset_only_enables_preset_descriptors() {
        let (worker_tx, _worker_rx_unused) = mpsc::channel::<WorkerRequest>();
        let (_response_tx_unused, worker_rx) = mpsc::channel::<WorkerResponse>();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let mut app = App::new(worker_tx, worker_rx, cancel_flag);

        app.set_all_descriptors(true);
        app.apply_fast_ml_descriptor_preset();

        let selected = app.selected_descriptors();
        let preset = BatchDescriptor::fast_ml_preset();
        assert_eq!(
            selected.len(),
            preset.len(),
            "Preset selection size should match BatchDescriptor::fast_ml_preset"
        );
        for descriptor in preset {
            assert!(
                selected.contains(descriptor),
                "Preset descriptor should be enabled: {:?}",
                descriptor
            );
        }
    }

    #[test]
    /// Implements test selected descriptor value count matches enabled sum.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_selected_descriptor_value_count_matches_enabled_sum() {
        let mut app = make_test_app();

        app.set_all_descriptors(false);
        if let Some((_, enabled)) = app.available_descriptors.get_mut(0) {
            *enabled = true;
        }
        if let Some((_, enabled)) = app.available_descriptors.get_mut(1) {
            *enabled = true;
        }

        let expected = app.available_descriptors[0].0.feature_len()
            + app.available_descriptors[1].0.feature_len();
        assert_eq!(app.selected_descriptor_value_count(), expected);
    }

    #[test]
    /// Implements test rebuild target columns includes csv and descriptor sources.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_rebuild_target_columns_includes_csv_and_descriptor_sources() {
        let mut app = make_test_app();
        app.loaded_properties = vec![
            vec![(Arc::<str>::from("alogp"), "1.2".to_string().into())],
            vec![(Arc::<str>::from("alogp"), "2.4".to_string().into())],
        ];
        app.feature_matrix = Some(FeatureMatrix {
            rows: 2,
            cols: 2,
            feature_names: vec!["alogp".to_string(), "MW".to_string()],
            data: vec![1.2, 100.0, 2.4, 120.0],
            descriptors: vec![BatchDescriptor::LogP],
            selected_feature_indices: vec![0, 1],
        });

        app.rebuild_target_columns();

        let labels = app
            .target_columns
            .iter()
            .map(TargetColumn::display_label)
            .collect::<Vec<_>>();
        assert!(
            labels.contains(&"[CSV] alogp".to_string()),
            "CSV target should be present"
        );
        assert!(
            labels.contains(&"[DESC] alogp".to_string()),
            "Descriptor target should be present even if name overlaps CSV"
        );
        assert!(
            labels.contains(&"[DESC] MW".to_string()),
            "Descriptor-only target should be present"
        );
    }

    #[test]
    /// Implements test descriptor target overlay uses matrix values.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_descriptor_target_overlay_uses_matrix_values() {
        let mut app = make_test_app();
        app.loaded_properties = vec![
            vec![(Arc::<str>::from("alogp"), "1.0".to_string().into())],
            vec![(Arc::<str>::from("alogp"), "2.0".to_string().into())],
        ];
        app.feature_matrix = Some(FeatureMatrix {
            rows: 2,
            cols: 1,
            feature_names: vec!["MW".to_string()],
            data: vec![101.5, 202.5],
            descriptors: vec![BatchDescriptor::MolecularWeight],
            selected_feature_indices: vec![0],
        });
        app.rebuild_target_columns();
        app.selected_target_columns.clear();
        let descriptor_key = app
            .target_columns
            .iter()
            .find(|target| target.source == TargetColumnSource::Descriptor && target.name == "MW")
            .expect("descriptor target should exist")
            .key
            .clone();
        app.selected_target_columns.insert(descriptor_key);

        app.refresh_target_overlay_cache();

        assert_eq!(app.target_overlay_error, None);
        assert_eq!(
            app.target_overlay_values,
            Some(vec![101.5, 202.5]),
            "Descriptor target overlay should come from feature matrix column values"
        );
    }

    #[test]
    /// Implements test descriptor target resolves drop index and matrix drop.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_descriptor_target_resolves_drop_index_and_matrix_drop() {
        let mut app = make_test_app();
        app.loaded_properties = vec![vec![], vec![]];
        app.feature_matrix = Some(FeatureMatrix {
            rows: 2,
            cols: 2,
            feature_names: vec!["A".to_string(), "B".to_string()],
            data: vec![1.0, 10.0, 2.0, 20.0],
            descriptors: vec![BatchDescriptor::MolecularWeight, BatchDescriptor::LogP],
            selected_feature_indices: vec![0, 1],
        });
        app.rebuild_target_columns();
        let descriptor_target = app
            .target_columns
            .iter()
            .find(|target| target.source == TargetColumnSource::Descriptor && target.name == "A")
            .expect("descriptor target A should exist")
            .clone();

        let (values, drop_idx) = app
            .resolve_training_target_values(&descriptor_target)
            .expect("descriptor target should resolve");
        assert_eq!(values, vec![1.0, 2.0]);
        assert_eq!(drop_idx, Some(0));

        let payload = app
            .feature_matrix
            .as_ref()
            .expect("feature matrix should be present")
            .to_payload();
        let reduced = select_feature_columns(payload, &[1]).expect("column selection should work");
        assert_eq!(reduced.cols, 1);
        assert_eq!(reduced.feature_names, vec!["B".to_string()]);
        assert_eq!(reduced.data, vec![10.0, 20.0]);
    }

    #[test]
    /// Implements test csv target resolves drop index when descriptor name matches.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_csv_target_resolves_drop_index_when_descriptor_name_matches() {
        let mut app = make_test_app();
        app.loaded_properties = vec![
            vec![(Arc::<str>::from("alogp"), "1.2".to_string().into())],
            vec![(Arc::<str>::from("alogp"), "2.4".to_string().into())],
        ];
        app.feature_matrix = Some(FeatureMatrix {
            rows: 2,
            cols: 2,
            feature_names: vec!["alogp".to_string(), "MW".to_string()],
            data: vec![1.2, 100.0, 2.4, 120.0],
            descriptors: vec![BatchDescriptor::LogP, BatchDescriptor::MolecularWeight],
            selected_feature_indices: vec![0, 1],
        });
        app.rebuild_target_columns();
        let csv_target = app
            .target_columns
            .iter()
            .find(|target| target.source == TargetColumnSource::Csv && target.name == "alogp")
            .expect("csv target alogp should exist")
            .clone();

        let (values, drop_idx) = app
            .resolve_training_target_values(&csv_target)
            .expect("csv target should resolve");
        assert_eq!(values, vec![1.2, 2.4]);
        assert_eq!(
            drop_idx,
            Some(0),
            "matching descriptor feature should be removed to avoid target leakage"
        );
    }

    #[test]
    /// Implements test build projection sanitizes non finite values.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_build_projection_sanitizes_non_finite_values() {
        let matrix = FeatureMatrixPayload {
            rows: 4,
            cols: 2,
            feature_names: vec!["f0".to_string(), "f1".to_string()],
            data: vec![1.0, 2.0, f64::NAN, 4.0, f64::INFINITY, 6.0, 7.0, 8.0],
            descriptors: vec![BatchDescriptor::MolecularWeight, BatchDescriptor::AtomCount],
            selected_feature_indices: vec![0, 1],
        };

        let projection = build_projection(&matrix).expect("projection should be built");
        assert_eq!(projection.points.len(), matrix.rows);
        assert!(
            projection
                .points
                .iter()
                .all(|(x, y)| x.is_finite() && y.is_finite())
        );
        assert!(projection.explained_variance.iter().all(|v| v.is_finite()));
    }

    #[test]
    /// Implements test art2a clustering sanitizes non finite matrix values.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_art2a_clustering_sanitizes_non_finite_matrix_values() {
        let matrix = FeatureMatrixPayload {
            rows: 4,
            cols: 3,
            feature_names: vec!["f0".to_string(), "f1".to_string(), "f2".to_string()],
            data: vec![
                1.0,
                0.0,
                0.1,
                f64::NAN,
                0.2,
                0.0,
                f64::INFINITY,
                0.1,
                0.0,
                0.8,
                0.1,
                f64::NEG_INFINITY,
            ],
            descriptors: vec![
                BatchDescriptor::MolecularWeight,
                BatchDescriptor::AtomCount,
                BatchDescriptor::BondCount,
            ],
            selected_feature_indices: vec![0, 1, 2],
        };
        let response = handle_art2a_clustering(
            matrix,
            Art2aConfig {
                vigilance: 0.8,
                learning_rate: 0.3,
                max_iter: 20,
            },
            MlScalerChoice::Standard,
            None,
            &Arc::new(AtomicBool::new(false)),
        )
        .expect("clustering should succeed");

        match response {
            WorkerResponse::ClusteringCompleted {
                labels,
                cluster_sizes,
                summaries,
                projection,
                ..
            } => {
                assert_eq!(labels.len(), 4);
                assert!(!cluster_sizes.is_empty());
                assert_eq!(cluster_sizes.iter().sum::<usize>(), 4);
                assert_eq!(summaries.len(), cluster_sizes.len());
                assert!(
                    projection.is_some(),
                    "clustering should return PCA projection"
                );
            }
            other => panic!("Expected ClusteringCompleted, got {:?}", other),
        }
    }

    #[test]
    /// Implements test clustering payload appends csv numeric columns row aligned.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_clustering_payload_appends_csv_numeric_columns_row_aligned() {
        let mut app = make_test_app();
        app.feature_matrix = Some(FeatureMatrix {
            rows: 2,
            cols: 2,
            feature_names: vec!["D0".to_string(), "D1".to_string()],
            data: vec![1.0, 2.0, 3.0, 4.0],
            descriptors: vec![BatchDescriptor::MolecularWeight, BatchDescriptor::LogP],
            selected_feature_indices: vec![0, 1],
        });
        app.loaded_properties = vec![
            vec![
                (Arc::<str>::from("target_a"), "10.0".to_string().into()),
                (Arc::<str>::from("target_b"), "20.0".to_string().into()),
            ],
            vec![
                (Arc::<str>::from("target_a"), "11.0".to_string().into()),
                (Arc::<str>::from("target_b"), "21.0".to_string().into()),
            ],
        ];
        app.include_csv_numeric_in_clustering = true;

        let (payload, added) = app
            .clustering_payload_with_optional_csv_targets()
            .expect("payload should build");
        assert_eq!(added, 2);
        assert_eq!(payload.rows, 2);
        assert_eq!(payload.cols, 4);
        assert_eq!(
            payload.data,
            vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 11.0, 21.0]
        );
        assert_eq!(
            payload.feature_names,
            vec![
                "D0".to_string(),
                "D1".to_string(),
                "CSV::target_a".to_string(),
                "CSV::target_b".to_string()
            ]
        );
    }

    #[test]
    /// Implements test compute cluster centroids returns expected means.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_compute_cluster_centroids_returns_expected_means() {
        let points = vec![(0.0, 0.0), (2.0, 2.0), (10.0, 0.0), (12.0, 2.0)];
        let labels = vec![0usize, 0usize, 1usize, 1usize];
        let centroids = compute_cluster_centroids(&points, &labels);
        assert_eq!(centroids.len(), 2);
        assert_eq!(centroids[0], (1.0, 1.0));
        assert_eq!(centroids[1], (11.0, 1.0));
    }

    #[test]
    /// Implements test cluster size histogram counts each cluster.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_cluster_size_histogram_counts_each_cluster() {
        let labels = vec![2usize, 0usize, 2usize, 1usize, 1usize, 1usize];
        let sizes = cluster_size_histogram(&labels);
        assert_eq!(sizes, vec![1usize, 3usize, 2usize]);
    }

    #[test]
    /// Implements test cluster summaries from predictions reconstructs stats.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_cluster_summaries_from_predictions_reconstructs_stats() {
        let rows = vec![
            PredictionOutputRow {
                smiles: "A".to_string(),
                predictions: vec![1.0, 0.8],
            },
            PredictionOutputRow {
                smiles: "B".to_string(),
                predictions: vec![0.0, 0.5],
            },
            PredictionOutputRow {
                smiles: "C".to_string(),
                predictions: vec![1.0, 0.6],
            },
        ];
        let summaries = cluster_summaries_from_predictions(&rows, 0, Some(1));
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].cluster_id, 0);
        assert_eq!(summaries[0].size, 1);
        assert!((summaries[0].mean_similarity - 0.5).abs() < 1e-12);
        assert_eq!(summaries[1].cluster_id, 1);
        assert_eq!(summaries[1].size, 2);
        assert!((summaries[1].mean_similarity - 0.7).abs() < 1e-12);
    }

    #[test]
    /// Implements test display label from output target name maps known suffixes.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_display_label_from_output_target_name_maps_known_suffixes() {
        assert_eq!(
            display_label_from_output_target_name("alogp__csv"),
            "[CSV] alogp"
        );
        assert_eq!(
            display_label_from_output_target_name("MW__desc"),
            "[DESC] MW"
        );
        assert_eq!(
            display_label_from_output_target_name("raw_target"),
            "raw_target"
        );
    }

    #[test]
    /// Implements test importance target cycle updates active importance table.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_importance_target_cycle_updates_active_importance_table() {
        let mut app = make_test_app();
        let mk_summary =
            |target: &str, label: &str, pipeline: &str, score: f64| TargetTrainingSummary {
                target: target.to_string(),
                target_display_label: label.to_string(),
                pipeline_name: pipeline.to_string(),
                cv_q2: 0.0,
                train_r2: 0.0,
                rmse: 0.0,
                mae: 0.0,
                pearson: 0.0,
                holdout_r2: 0.0,
                holdout_rmse: 0.0,
                holdout_mae: 0.0,
                holdout_pearson: 0.0,
                split_label: "Random".to_string(),
                split_train_size: 0,
                split_test_size: 0,
                top_feature_importances: vec![("feat".to_string(), score)],
            };
        app.trained_target_summaries = vec![
            mk_summary("alogp__csv", "[CSV] alogp", "RandomForest", 0.8),
            mk_summary("MW__desc", "[DESC] MW", "ElasticNet", 0.3),
        ];
        app.importance_target_index = 0;
        app.sync_importance_view_from_index();
        assert_eq!(app.top_feature_importances, vec![("feat".to_string(), 0.8)]);

        app.cycle_importance_target(1);
        assert_eq!(app.importance_target_index, 1);
        assert_eq!(app.top_feature_importances, vec![("feat".to_string(), 0.3)]);
        assert!(app.status_message.contains("[DESC] MW"));

        app.cycle_importance_target(1);
        assert_eq!(app.importance_target_index, 0);
        assert_eq!(app.top_feature_importances, vec![("feat".to_string(), 0.8)]);
    }

    #[test]
    /// Implements test resolve active model selection falls back to supervised when layouts differ.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_resolve_active_model_selection_falls_back_to_supervised_when_layouts_differ() {
        let mut app = make_test_app();
        app.pipeline_bundle_bytes = Some(vec![1, 2, 3]);
        app.art2a_model_bytes = Some(vec![4, 5, 6]);
        app.pipeline_feature_layout = Some(ModelFeatureLayout::from_parts(
            vec![BatchDescriptor::MolecularWeight],
            vec![0],
            vec!["MW".to_string()],
        ));
        app.art2a_feature_layout = Some(ModelFeatureLayout::from_parts(
            vec![BatchDescriptor::LogP],
            vec![0],
            vec!["LogP".to_string()],
        ));

        let selection = app
            .resolve_active_model_selection()
            .expect("selection should resolve");
        assert!(selection.pipeline_bundle_bytes.is_some());
        assert!(selection.art2a_model_bytes.is_none());
        assert!(selection.warning.is_some());
        assert_eq!(
            selection.layout.expected_feature_names,
            vec!["MW".to_string()]
        );
    }

    #[test]
    /// Implements test build prediction flat by expected names handles csv features.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_build_prediction_flat_by_expected_names_handles_csv_features() {
        let source = FlatDescriptorMatrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let source_feature_names = vec!["D0".to_string(), "D1".to_string()];
        let properties = vec![
            vec![(Arc::<str>::from("target"), "10.0".to_string().into())],
            vec![(Arc::<str>::from("target"), "11.0".to_string().into())],
        ];
        let expected_feature_names = vec![
            "D1".to_string(),
            "CSV::target".to_string(),
            "D0".to_string(),
        ];

        let merged = build_prediction_flat_by_expected_names(
            &source,
            &source_feature_names,
            &properties,
            &expected_feature_names,
        )
        .expect("merged features should be built");

        assert_eq!(merged.rows, 2);
        assert_eq!(merged.cols, 3);
        assert_eq!(merged.data, vec![2.0, 10.0, 1.0, 4.0, 11.0, 3.0]);
    }

    #[test]
    /// Implements test import model bundle allows csv augmented feature layout.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_import_model_bundle_allows_csv_augmented_feature_layout() {
        let bundle = PersistedModelBundle {
            schema_version: MODEL_BUNDLE_SCHEMA_VERSION,
            pipeline_bundle_bytes: None,
            art2a_model_bytes: Some(vec![1, 2, 3]),
            descriptors: vec![BatchDescriptor::MolecularWeight.name().to_string()],
            selected_feature_indices: vec![0, 1],
            expected_feature_names: vec!["MolecularWeight".to_string(), "CSV::target".to_string()],
        };
        let bytes = bincode::serialize(&bundle).expect("bundle should serialize");
        let path = std::env::temp_dir().join(format!(
            "ferrum_tui_bundle_{}_{}.fcbundle",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock should be after epoch")
                .as_nanos()
        ));
        std::fs::write(&path, bytes).expect("bundle should be written");

        let response =
            handle_import_model_bundle(path.clone()).expect("CSV-augmented bundle should import");
        assert!(matches!(
            response,
            WorkerResponse::ModelBundleImported { .. }
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    /// Implements test build cluster export rows prefers prediction art2a outputs.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_build_cluster_export_rows_prefers_prediction_art2a_outputs() {
        let prediction_rows = vec![
            PredictionOutputRow {
                smiles: "CCO".to_string(),
                predictions: vec![2.0, 0.82, 1.0],
            },
            PredictionOutputRow {
                smiles: "CCN".to_string(),
                predictions: vec![1.0, 0.41, 0.0],
            },
        ];
        let target_names = vec![
            "ART2A_ClusterId".to_string(),
            "ART2A_Similarity".to_string(),
            "ART2A_InDomain".to_string(),
        ];

        let rows = build_cluster_export_rows(&prediction_rows, &target_names, &[], &[])
            .expect("cluster rows should build");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].cluster_id, 2);
        assert_eq!(rows[0].similarity, Some(0.82));
        assert_eq!(rows[0].in_domain, Some(true));
        assert_eq!(rows[1].cluster_id, 1);
        assert_eq!(rows[1].in_domain, Some(false));
    }

    #[test]
    /// Implements test build cluster export rows falls back to loaded labels and smiles.
    ///
    /// Function context: `crates/ferrum_cli/src/bin/tui.rs`.
    /// System-layer note: component execution and pipeline orchestration.
    fn test_build_cluster_export_rows_falls_back_to_loaded_labels_and_smiles() {
        let labels = vec![0usize, 2usize];
        let smiles = vec!["C".to_string(), "CC".to_string()];
        let rows = build_cluster_export_rows(&[], &[], &labels, &smiles)
            .expect("fallback rows should build");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].smiles, "C");
        assert_eq!(rows[0].cluster_id, 0);
        assert_eq!(rows[0].similarity, None);
        assert_eq!(rows[0].in_domain, None);
        assert_eq!(rows[1].smiles, "CC");
        assert_eq!(rows[1].cluster_id, 2);
    }
}
