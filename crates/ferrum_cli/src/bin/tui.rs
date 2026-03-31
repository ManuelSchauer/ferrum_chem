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
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
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

use ferrum_core::desc::batch::{
    BatchDescriptor, FlatDescriptorMatrix, calculate_batch_flat_static,
    calculate_batch_flat_static_cancellable, write_csv_flat,
};
use ferrum_core::algo::rings::find_symmetrized_sssr_bonds;
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
use ferrum_core::ml::pipeline::{ModelType, QsarPipeline, ScalerType, TaskType};
use ferrum_core::ml::pls::PlsRegression;
use ferrum_core::ml::random_forest::RandomForestRegressor;
use ferrum_core::ml::scaling::{MinMaxScaler, StandardScaler};
use ferrum_core::ml::splitting::{TrainTestSplit, random_split, scaffold_split, stratified_split};
use ferrum_core::mol::molecule::Molecule;

const FRAME_INTERVAL: Duration = Duration::from_millis(16);
const PREVIEW_ROWS: usize = 8;
const PREVIEW_COLS: usize = 5;
const DEFAULT_EXPORT_PATH: &str = "/tmp/ferrum_tui_predictions.csv";
const DEFAULT_FEATURE_EXPORT_PATH: &str = "/tmp/ferrum_tui_features.csv";
const DEFAULT_VARIANCE_THRESHOLD: f64 = 1e-12;
const DEFAULT_CORRELATION_THRESHOLD: f64 = 0.95;
const DEFAULT_TEST_FRACTION: f64 = 0.20;
const DEFAULT_ART2A_VIGILANCE: f64 = 0.80;
const DEFAULT_ART2A_LEARNING_RATE: f64 = 0.30;
const DEFAULT_ART2A_MAX_ITER: usize = 50;
const DESCRIPTOR_COMPUTING_MESSAGE: &str = "Calculating descriptors...";
const DESCRIPTOR_ERROR_LOG_PATH: &str = "/tmp/ferrum_tui_descriptor_errors.log";
const SPINNER_FRAMES: [char; 4] = ['|', '/', '-', '\\'];

const TAB_TITLES: [&str; 5] = [
    "1: Import",
    "2: Descriptor",
    "3: Model Select",
    "4: Parameters",
    "5: Analyze",
];

/// Top-level UI state.
#[derive(Debug, Clone)]
enum AppState {
    Idle,
    Computing(String),
    ErrorPopup(String),
    FileBrowserOpen(PathBuf),
    TargetSelectionOpen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BrowserPurpose {
    Dataset,
    Prediction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlAlgorithm {
    AutoQsar,
    RandomForest,
    Knn,
    Pls,
    ElasticNet,
}

impl MlAlgorithm {
    fn all() -> [Self; 5] {
        [
            Self::AutoQsar,
            Self::RandomForest,
            Self::Knn,
            Self::Pls,
            Self::ElasticNet,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            Self::AutoQsar => "AutoQsar",
            Self::RandomForest => "Random Forest",
            Self::Knn => "k-NN",
            Self::Pls => "PLS",
            Self::ElasticNet => "Elastic Net",
        }
    }

    fn cycle(self, delta: isize) -> Self {
        let all = Self::all();
        let idx = all
            .iter()
            .position(|candidate| *candidate == self)
            .unwrap_or(0);
        let next = (idx as isize + delta).rem_euclid(all.len() as isize) as usize;
        all[next]
    }

    fn manual_all() -> [Self; 4] {
        [Self::RandomForest, Self::Knn, Self::Pls, Self::ElasticNet]
    }

    fn is_manual(self) -> bool {
        self != Self::AutoQsar
    }

    fn manual_cycle(self, delta: isize) -> Self {
        let all = Self::manual_all();
        let idx = all.iter().position(|candidate| *candidate == self).unwrap_or(0);
        let next = (idx as isize + delta).rem_euclid(all.len() as isize) as usize;
        all[next]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlMode {
    AutoQsar,
    Art2a,
    Manual,
}

impl MlMode {
    fn toggle(self) -> Self {
        match self {
            Self::AutoQsar => Self::Art2a,
            Self::Art2a => Self::Manual,
            Self::Manual => Self::AutoQsar,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::AutoQsar => "AutoQsar",
            Self::Art2a => "ART2-A",
            Self::Manual => "Manual",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlScalerChoice {
    Standard,
    MinMax,
}

impl MlScalerChoice {
    fn label(self) -> &'static str {
        match self {
            Self::Standard => "StandardScaler",
            Self::MinMax => "MinMaxScaler",
        }
    }

    fn toggle(self) -> Self {
        match self {
            Self::Standard => Self::MinMax,
            Self::MinMax => Self::Standard,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SplitStrategy {
    Random,
    Stratified,
    Scaffold,
}

impl SplitStrategy {
    fn cycle(self) -> Self {
        match self {
            Self::Random => Self::Stratified,
            Self::Stratified => Self::Scaffold,
            Self::Scaffold => Self::Random,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Random => "Random",
            Self::Stratified => "Stratified",
            Self::Scaffold => "Scaffold",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisualizationColorMode {
    Uniform,
    TargetOverlay,
    ClusterIds,
    ApplicabilityDomain,
}

impl VisualizationColorMode {
    fn cycle(self) -> Self {
        match self {
            Self::Uniform => Self::TargetOverlay,
            Self::TargetOverlay => Self::ClusterIds,
            Self::ClusterIds => Self::ApplicabilityDomain,
            Self::ApplicabilityDomain => Self::Uniform,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::TargetOverlay => "target overlay",
            Self::ClusterIds => "cluster IDs",
            Self::ApplicabilityDomain => "applicability domain",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl FeatureMatrix {
    fn row(&self, idx: usize) -> &[f64] {
        debug_assert!(idx < self.rows);
        let start = idx * self.cols;
        &self.data[start..start + self.cols]
    }

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
struct Art2aConfig {
    vigilance: f64,
    learning_rate: f64,
    max_iter: usize,
}

#[derive(Debug, Clone)]
struct FeatureMatrixPayload {
    rows: usize,
    cols: usize,
    feature_names: Vec<String>,
    data: Vec<f64>,
    descriptors: Vec<BatchDescriptor>,
    selected_feature_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct TargetTrainingSummary {
    target: String,
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
}

#[derive(Debug, Clone)]
struct PredictionOutputRow {
    smiles: String,
    predictions: Vec<f64>,
}

#[derive(Debug, Clone)]
struct Art2aClusterSummary {
    cluster_id: usize,
    size: usize,
    fraction: f64,
    mean_similarity: f64,
    min_similarity: f64,
    max_similarity: f64,
}

#[derive(Debug)]
enum WorkerRequest {
    LoadDataset(PathBuf),
    CancelDescriptors,
    CalculateDescriptors {
        molecules: Vec<Molecule>,
        descriptors: Vec<BatchDescriptor>,
    },
    TrainModel {
        matrix: FeatureMatrixPayload,
        target_series: Vec<(String, Vec<f64>)>,
        smiles: Vec<String>,
        config: MlTrainingConfig,
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
    Shutdown,
}

#[derive(Debug)]
enum WorkerResponse {
    Progress(f32),
    DatasetLoaded {
        path: PathBuf,
        molecules: Vec<Molecule>,
        smiles: Vec<String>,
        properties: Vec<Vec<(String, String)>>,
        failed_rows: usize,
    },
    DescriptorsCalculated {
        matrix: FeatureMatrixPayload,
        profiles: Vec<DescriptorProfileRow>,
        projection: Option<ProjectionData>,
    },
    ModelTrained {
        primary_target: String,
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
        preview_predictions: Vec<f64>,
        train_predictions: Vec<f64>,
    },
    ClusteringCompleted {
        labels: Vec<usize>,
        cluster_sizes: Vec<usize>,
        summaries: Vec<Art2aClusterSummary>,
        art2a_model_bytes: Vec<u8>,
    },
    PredictionsReady {
        path: PathBuf,
        target_names: Vec<String>,
        rows: Vec<PredictionOutputRow>,
        skipped_rows: usize,
    },
    PredictionsExported {
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
    DescriptorsCanceled,
    FeatureMatrixExported {
        path: PathBuf,
        rows: usize,
        cols: usize,
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
    color_mode: VisualizationColorMode,

    // Additional state needed for complete workflows
    loaded_smiles: Vec<String>,
    loaded_properties: Vec<Vec<(String, String)>>,
    last_dataset_failed_rows: usize,
    loaded_from_path: Option<PathBuf>,
    target_columns: Vec<String>,
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

    prediction_rows: Vec<PredictionOutputRow>,
    prediction_target_names: Vec<String>,
    prediction_source: Option<PathBuf>,
    prediction_export_path: PathBuf,
    feature_export_path: PathBuf,

    pipeline_bundle_bytes: Option<Vec<u8>>,

    status_message: String,
    progress_percent: f32,
    profile_scroll: usize,
    spinner_frame: usize,

    browser_purpose: BrowserPurpose,
    browser_entries: Vec<PathBuf>,
    browser_list_state: ListState,

    worker_tx: Sender<WorkerRequest>,
    worker_rx: Receiver<WorkerResponse>,
    descriptor_cancel_requested: bool,
}

impl App {
    fn new(worker_tx: Sender<WorkerRequest>, worker_rx: Receiver<WorkerResponse>) -> Self {
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

            prediction_rows: Vec::new(),
            prediction_target_names: Vec::new(),
            prediction_source: None,
            prediction_export_path: PathBuf::from(DEFAULT_EXPORT_PATH),
            feature_export_path: PathBuf::from(DEFAULT_FEATURE_EXPORT_PATH),

            pipeline_bundle_bytes: None,

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
            descriptor_cancel_requested: false,
        }
    }

    fn tick(&mut self) {
        if matches!(self.state, AppState::Computing(_)) {
            self.spinner_frame = (self.spinner_frame + 1) % SPINNER_FRAMES.len();
        }
    }

    fn selected_descriptors(&self) -> Vec<BatchDescriptor> {
        self.available_descriptors
            .iter()
            .filter_map(|(descriptor, enabled)| (*enabled).then_some(*descriptor))
            .collect()
    }

    fn primary_selected_target(&self) -> Option<&str> {
        self.target_columns
            .iter()
            .find(|name| self.selected_target_columns.contains(name.as_str()))
            .map(|name| name.as_str())
    }

    fn selected_targets_in_order(&self) -> Vec<String> {
        self.target_columns
            .iter()
            .filter(|name| self.selected_target_columns.contains(name.as_str()))
            .cloned()
            .collect()
    }

    fn refresh_target_overlay_cache(&mut self) {
        let Some(target_column) = self.primary_selected_target().map(str::to_string) else {
            self.target_overlay_values = None;
            self.target_overlay_error = None;
            return;
        };
        match extract_targets(&self.loaded_properties, &target_column) {
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

    fn ml_mode_summary(&self) -> String {
        let selected = self.selected_targets_in_order();
        if selected.is_empty() {
            return "Mode: Unsupervised (select target(s) to enable supervised training)".to_string();
        }
        let primary = &selected[0];
        if self.feature_matrix.is_none() {
            if self.target_overlay_values.is_some() {
                return format!(
                    "Mode: Targets selected [{}], compute descriptors to enable supervised training",
                    truncate_label(&selected.join(", "), 48)
                );
            }
            if let Some(error) = &self.target_overlay_error {
                return format!("Mode: Primary target invalid ({})", truncate_label(error, 52));
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
            return format!("Mode: Primary target invalid ({})", truncate_label(error, 52));
        }
        format!("Mode: Target pending validation ('{}')", primary)
    }

    fn rebuild_target_columns(&mut self) {
        let mut stats: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for row in &self.loaded_properties {
            for (name, value) in row {
                let entry = stats.entry(name.clone()).or_insert((0usize, 0usize));
                entry.0 += 1;
                if value.trim().parse::<f64>().is_ok() {
                    entry.1 += 1;
                }
            }
        }

        let row_count = self.loaded_properties.len();
        let mut numeric_candidates = stats
            .into_iter()
            .filter_map(|(name, (present_count, numeric_count))| {
                (present_count == row_count && row_count > 0 && numeric_count == row_count)
                    .then_some(name)
            })
            .collect::<Vec<_>>();
        numeric_candidates.sort();
        self.target_columns = numeric_candidates;

        self.selected_target_columns
            .retain(|name| self.target_columns.iter().any(|candidate| candidate == name));
        if self.selected_target_columns.is_empty() {
            for name in &self.target_columns {
                self.selected_target_columns.insert(name.clone());
            }
        }

        let selected_idx = self
            .target_columns
            .iter()
            .position(|name| self.selected_target_columns.contains(name.as_str()))
            .or_else(|| (!self.target_columns.is_empty()).then_some(0));
        self.target_column_list_state.select(selected_idx);
        self.refresh_target_overlay_cache();
    }

    fn open_target_selection_popup(&mut self) {
        if self.target_columns.is_empty() {
            self.status_message =
                "No numeric target columns found in loaded CSV properties.".to_string();
            return;
        }
        let selected_idx = self
            .target_columns
            .iter()
            .position(|name| self.selected_target_columns.contains(name.as_str()))
            .or_else(|| Some(0));
        self.target_column_list_state.select(selected_idx);
        self.state = AppState::TargetSelectionOpen;
    }

    fn toggle_selected_target_column(&mut self) {
        let Some(selected) = self.target_column_list_state.selected() else {
            return;
        };
        let Some(name) = self.target_columns.get(selected) else {
            return;
        };
        if self.selected_target_columns.contains(name.as_str()) {
            self.selected_target_columns.remove(name.as_str());
        } else {
            self.selected_target_columns.insert(name.clone());
        }
        self.refresh_target_overlay_cache();
    }

    fn set_all_target_columns(&mut self, enabled: bool) {
        self.selected_target_columns.clear();
        if enabled {
            for name in &self.target_columns {
                self.selected_target_columns.insert(name.clone());
            }
        }
        self.refresh_target_overlay_cache();
    }

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
                self.target_column_list_state.select(Some((current + 1) % len));
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

    fn fallback_stratified_when_target_unavailable(&mut self) -> bool {
        if self.training_config.split_strategy == SplitStrategy::Stratified
            && !self.has_target_for_stratified_split()
        {
            self.training_config.split_strategy = SplitStrategy::Random;
            return true;
        }
        false
    }

    fn active_algorithm(&self) -> MlAlgorithm {
        match self.selected_ml_mode {
            MlMode::AutoQsar => MlAlgorithm::AutoQsar,
            MlMode::Art2a => MlAlgorithm::AutoQsar,
            MlMode::Manual => self.manual_algo_selection,
        }
    }

    fn ml_param_page_count(&self) -> usize {
        match self.selected_ml_mode {
            MlMode::AutoQsar | MlMode::Art2a => 1,
            MlMode::Manual => match self.active_algorithm() {
                MlAlgorithm::RandomForest | MlAlgorithm::ElasticNet => 2,
                _ => 1,
            },
        }
    }

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

    fn set_error(&mut self, message: String) {
        self.progress_percent = 0.0;
        self.state = AppState::ErrorPopup(message);
    }

    fn begin_computing(&mut self, message: String) {
        self.progress_percent = -1.0;
        self.state = AppState::Computing(message);
    }

    fn send_worker(&mut self, request: WorkerRequest) {
        if let Err(error) = self.worker_tx.send(request) {
            self.set_error(format!("Worker channel error: {}", error));
        }
    }

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
        self.state = AppState::Idle;
    }

    fn next_tab(&mut self) {
        self.active_tab = (self.active_tab + 1) % TAB_TITLES.len();
    }

    fn prev_tab(&mut self) {
        if self.active_tab == 0 {
            self.active_tab = TAB_TITLES.len() - 1;
        } else {
            self.active_tab -= 1;
        }
    }

    fn update_descriptor_selection(&mut self, delta: isize) {
        let len = self.available_descriptors.len();
        if len == 0 {
            return;
        }
        let current = self.descriptor_list_state.selected().unwrap_or(0) as isize;
        let next = (current + delta).rem_euclid(len as isize) as usize;
        self.descriptor_list_state.select(Some(next));
    }

    fn toggle_selected_descriptor(&mut self) {
        let Some(index) = self.descriptor_list_state.selected() else {
            return;
        };
        if let Some((_, enabled)) = self.available_descriptors.get_mut(index) {
            *enabled = !*enabled;
        }
    }

    fn set_all_descriptors(&mut self, enabled: bool) {
        for (_, flag) in &mut self.available_descriptors {
            *flag = enabled;
        }
    }

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
        };

        match read_browser_entries(&start_dir) {
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

    fn navigate_browser_to(&mut self, dir: PathBuf) {
        match read_browser_entries(&dir) {
            Ok(entries) => {
                self.browser_entries = entries;
                let selected = (!self.browser_entries.is_empty()).then_some(0);
                self.browser_list_state.select(selected);
                self.state = AppState::FileBrowserOpen(dir);
            }
            Err(error) => self.set_error(format!("Failed to read '{}': {}", dir.display(), error)),
        }
    }

    fn request_dataset_load(&mut self, path: PathBuf) {
        self.begin_computing(format!("Loading molecules from '{}'", path.display()));
        self.send_worker(WorkerRequest::LoadDataset(path));
    }

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

        self.descriptor_cancel_requested = false;
        self.begin_computing(DESCRIPTOR_COMPUTING_MESSAGE.to_string());
        self.send_worker(WorkerRequest::CalculateDescriptors {
            molecules: self.loaded_molecules.clone(),
            descriptors,
        });
    }

    fn request_descriptor_cancellation(&mut self) {
        if self.descriptor_cancel_requested {
            self.status_message = "Descriptor cancellation already requested; waiting for worker to stop.".to_string();
            return;
        }
        self.descriptor_cancel_requested = true;
        self.status_message = "Cancelling descriptor calculation...".to_string();
        self.send_worker(WorkerRequest::CancelDescriptors);
    }

    fn request_variance_filter(&mut self) {
        let payload = {
            let Some(matrix) = &self.feature_matrix else {
                self.set_error("No feature matrix available. Calculate descriptors first.".to_string());
                return;
            };
            if matrix.cols == 0 {
                self.set_error("Feature matrix has zero columns.".to_string());
                return;
            }
            matrix.to_payload()
        };
        self.begin_computing(format!(
            "Applying variance threshold ({:.2e})...",
            DEFAULT_VARIANCE_THRESHOLD
        ));
        self.send_worker(WorkerRequest::ApplyVarianceFilter {
            matrix: payload,
            threshold: DEFAULT_VARIANCE_THRESHOLD,
        });
    }

    fn request_correlation_filter(&mut self) {
        let payload = {
            let Some(matrix) = &self.feature_matrix else {
                self.set_error("No feature matrix available. Calculate descriptors first.".to_string());
                return;
            };
            if matrix.cols <= 1 {
                self.set_error("Correlation filter needs at least 2 features.".to_string());
                return;
            }
            matrix.to_payload()
        };
        self.begin_computing(format!(
            "Applying correlation filter (|r| > {:.2})...",
            DEFAULT_CORRELATION_THRESHOLD
        ));
        self.send_worker(WorkerRequest::ApplyCorrelationFilter {
            matrix: payload,
            threshold: DEFAULT_CORRELATION_THRESHOLD,
        });
    }

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
        self.begin_computing(format!(
            "Exporting feature matrix to '{}'...",
            path.display()
        ));
        self.send_worker(WorkerRequest::ExportFeatureMatrix {
            path,
            smiles: self.loaded_smiles.clone(),
            matrix: payload,
        });
    }

    fn request_training(&mut self) {
        if self.feature_matrix.is_none() {
            self.set_error("Calculate descriptors first in Tab 2 (Descriptor).".to_string());
            return;
        }
        self.refresh_target_overlay_cache();
        let Some(matrix_rows) = self.feature_matrix.as_ref().map(|m| m.rows) else {
            self.set_error("Calculate descriptors first in Tab 2 (Descriptor).".to_string());
            return;
        };

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
        for target_name in selected_targets {
            let values = match extract_targets(&self.loaded_properties, &target_name) {
                Ok(values) => values,
                Err(error) => {
                    self.set_error(format!("Target '{}' validation failed: {}", target_name, error));
                    return;
                }
            };
            if values.len() != matrix_rows {
                self.set_error(format!(
                    "Target '{}' length mismatch: {} targets for {} feature rows.",
                    target_name,
                    values.len(),
                    matrix_rows
                ));
                return;
            }
            target_series.push((target_name, values));
        }
        if self.loaded_smiles.len() != matrix_rows {
            self.set_error(format!(
                "SMILES row mismatch: {} smiles for {} feature rows.",
                self.loaded_smiles.len(),
                matrix_rows
            ));
            return;
        }
        let matrix_payload = self
            .feature_matrix
            .as_ref()
            .map(FeatureMatrix::to_payload)
            .unwrap_or_else(|| unreachable!("matrix presence validated above"));

        self.training_config.algorithm = self.active_algorithm();
        self.begin_computing(format!(
            "Training {} on {} target(s) with {}...",
            self.training_config.algorithm.label(),
            target_series.len(),
            self.training_config.scaler.label()
        ));
        self.send_worker(WorkerRequest::TrainModel {
            matrix: matrix_payload,
            target_series,
            smiles: self.loaded_smiles.clone(),
            config: self.training_config.clone(),
        });
    }

    fn request_art2a_clustering(&mut self) {
        let matrix_payload = {
            let Some(matrix) = &self.feature_matrix else {
                self.set_error("No feature matrix available. Calculate descriptors first.".to_string());
                return;
            };
            if matrix.rows < 2 || matrix.cols == 0 {
                self.set_error("ART2-A clustering needs at least 2 rows and 1 feature.".to_string());
                return;
            }
            matrix.to_payload()
        };
        self.begin_computing(format!(
            "Running ART2-A clustering (rho={:.2}, eta={:.2}, scaler={})...",
            self.art2a_config.vigilance,
            self.art2a_config.learning_rate,
            self.training_config.scaler.label()
        ));
        self.send_worker(WorkerRequest::RunArt2aClustering {
            matrix: matrix_payload,
            config: self.art2a_config.clone(),
            scaler: self.training_config.scaler,
        });
    }

    fn request_prediction(&mut self, path: PathBuf) {
        if self.pipeline_bundle_bytes.is_none() && self.art2a_model_bytes.is_none() {
            self.set_error(
                "No supervised model or ART2-A model available. Train model (Enter) or run ART2-A (g) first.".to_string(),
            );
            return;
        }
        let (descriptors, selected_feature_indices, expected_feature_names) = {
            let Some(matrix) = &self.feature_matrix else {
                self.set_error(
                    "No descriptor matrix available. Calculate descriptors before prediction."
                        .to_string(),
                );
                return;
            };

            if matrix.descriptors.is_empty() {
                self.set_error("No descriptor selection cached for prediction.".to_string());
                return;
            }
            (
                matrix.descriptors.clone(),
                matrix.selected_feature_indices.clone(),
                matrix.feature_names.clone(),
            )
        };

        self.begin_computing(format!("Running prediction for '{}'...", path.display()));
        self.send_worker(WorkerRequest::PredictFile {
            path,
            pipeline_bundle_bytes: self.pipeline_bundle_bytes.clone(),
            art2a_model_bytes: self.art2a_model_bytes.clone(),
            descriptors,
            selected_feature_indices,
            expected_feature_names,
        });
    }

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

        self.begin_computing(format!(
            "Exporting predictions to '{}'...",
            export_path.display()
        ));
        self.send_worker(WorkerRequest::ExportPredictions {
            path: export_path,
            target_names: self.prediction_target_names.clone(),
            rows: self.prediction_rows.clone(),
        });
    }

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
                }
            }
            _ => {}
        }
    }

    fn handle_import_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('f') => self.open_file_browser(BrowserPurpose::Dataset),
            _ => {}
        }
    }

    fn handle_descriptor_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.update_descriptor_selection(-1),
            KeyCode::Down | KeyCode::Char('j') => self.update_descriptor_selection(1),
            KeyCode::Char(' ') => self.toggle_selected_descriptor(),
            KeyCode::Char('a') => self.set_all_descriptors(true),
            KeyCode::Char('n') => self.set_all_descriptors(false),
            KeyCode::Char('f') => self.open_file_browser(BrowserPurpose::Dataset),
            KeyCode::Char('v') => self.request_variance_filter(),
            KeyCode::Char('c') => self.request_correlation_filter(),
            KeyCode::Char('e') => self.request_export_feature_matrix(),
            KeyCode::Enter => self.request_descriptor_calculation(),
            _ => {}
        }
    }

    fn handle_model_select_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('t') => {
                self.open_target_selection_popup();
            }
            KeyCode::Char('m') => {
                self.selected_ml_mode = self.selected_ml_mode.toggle();
                self.ml_param_page = 0;
                self.status_message =
                    format!("ML mode set to {}.", self.selected_ml_mode.label());
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
            _ => {}
        }
    }

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
                self.status_message =
                    format!("ART2-A vigilance set to {:.2}.", self.art2a_config.vigilance);
            }
            KeyCode::Char(']') => {
                self.art2a_config.vigilance = (self.art2a_config.vigilance + 0.05).min(0.99);
                self.status_message =
                    format!("ART2-A vigilance set to {:.2}.", self.art2a_config.vigilance);
            }
            KeyCode::Char(';') => {
                self.art2a_config.learning_rate = (self.art2a_config.learning_rate - 0.05).max(0.05);
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
            _ => {}
        }
    }

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
            KeyCode::Char('p') => {
                self.open_file_browser(BrowserPurpose::Prediction);
            }
            KeyCode::Char('s') => {
                self.request_export_predictions();
            }
            KeyCode::Char('o') => {
                self.color_mode = self.color_mode.cycle();
                self.status_message = format!("Visualization color mode: {}.", self.color_mode.label());
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
            AppState::Computing(_) => {
                if matches!(key.code, KeyCode::Esc)
                    && !self.descriptor_cancel_requested
                    && matches!(&self.state, AppState::Computing(active) if active == DESCRIPTOR_COMPUTING_MESSAGE)
                {
                    self.request_descriptor_cancellation();
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
                        self.status_message = "Stratified split requires valid target; split reverted to Random.".to_string();
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

    fn drain_worker_messages(&mut self) {
        loop {
            match self.worker_rx.try_recv() {
                Ok(response) => self.apply_worker_response(response),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.set_error("Worker thread disconnected unexpectedly.".to_string());
                    break;
                }
            }
        }
    }

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
                self.last_dataset_failed_rows = failed_rows;
                self.loaded_molecules = molecules;
                self.loaded_smiles = smiles;
                self.loaded_properties = properties;
                self.loaded_from_path = Some(path.clone());
                self.rebuild_target_columns();

                // Reset downstream state because source data changed.
                self.feature_matrix = None;
                self.descriptor_profiles.clear();
                self.projection = None;
                self.train_actuals.clear();
                self.training_predictions.clear();
                self.ad_in_domain_flags.clear();
                self.top_feature_importances.clear();
                self.trained_target_summaries.clear();
                self.trained_pipeline = None;
                self.pipeline_bundle_bytes = None;
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
                self.descriptor_cancel_requested = false;
                self.progress_percent = 0.0;
                let rows = matrix.rows;
                let cols = matrix.cols;
                self.feature_matrix = Some(FeatureMatrix::from_payload(matrix));
                self.descriptor_profiles = profiles;
                self.projection = projection;
                self.refresh_target_overlay_cache();
                let split_fallback = self.fallback_stratified_when_target_unavailable();
                self.profile_scroll = 0;

                // New feature set invalidates existing trained model and predictions.
                self.train_actuals.clear();
                self.training_predictions.clear();
                self.ad_in_domain_flags.clear();
                self.top_feature_importances.clear();
                self.trained_target_summaries.clear();
                self.trained_pipeline = None;
                self.pipeline_bundle_bytes = None;
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
                target_summaries,
                pipeline_bundle_bytes,
                preview_predictions,
                train_predictions,
            } => match decode_pipeline_bundle(&pipeline_bundle_bytes, &primary_target) {
                Ok(pipeline) => {
                    self.progress_percent = 0.0;
                    self.trained_pipeline = Some(pipeline);
                    self.trained_target_summaries = target_summaries;
                    self.pipeline_bundle_bytes = Some(pipeline_bundle_bytes);
                    self.train_actuals = train_actuals;
                    self.training_predictions = train_predictions;
                    self.ad_in_domain_flags = ad_in_domain_flags;
                    self.top_feature_importances = top_feature_importances;
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

                    let preview = preview_predictions
                        .iter()
                        .map(|v| format!("{:.3}", v))
                        .collect::<Vec<_>>()
                        .join(", ");

                    self.status_message = format!(
                        "Training complete for {} target(s). Primary '{}': {} (Q2={:.4}, train R2={:.4}, holdout R2={:.4}) [{} split {}:{}]. Preview: [{}]",
                        self.trained_target_summaries.len(), primary_target, pipeline_name, cv_q2, train_r2, holdout_r2, split_label, split_train_size, split_test_size, preview
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
            } => {
                self.progress_percent = 0.0;
                self.prediction_target_names = target_names;
                self.prediction_rows = rows;
                self.prediction_source = Some(path.clone());
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
                self.status_message =
                    format!("Exported {} predictions to '{}'.", count, path.display());
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
                let rows = matrix.rows;
                let cols = matrix.cols;
                self.feature_matrix = Some(FeatureMatrix::from_payload(matrix));
                self.descriptor_profiles = profiles;
                self.projection = projection;
                self.refresh_target_overlay_cache();
                let split_fallback = self.fallback_stratified_when_target_unavailable();
                self.profile_scroll = 0;
                self.train_actuals.clear();
                self.training_predictions.clear();
                self.ad_in_domain_flags.clear();
                self.top_feature_importances.clear();
                self.trained_target_summaries.clear();
                self.trained_pipeline = None;
                self.pipeline_bundle_bytes = None;
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
            } => {
                self.progress_percent = 0.0;
                self.cluster_labels = labels;
                self.cluster_sizes = cluster_sizes.clone();
                self.cluster_summaries = summaries;
                self.art2a_model_bytes = Some(art2a_model_bytes);
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
                self.status_message = format!(
                    "Exported feature matrix ({}x{}) to '{}'.",
                    rows,
                    cols,
                    path.display()
                );
                self.state = AppState::Idle;
            }
            WorkerResponse::DescriptorsCanceled => {
                self.descriptor_cancel_requested = false;
                self.progress_percent = 0.0;
                self.status_message = "Descriptor calculation canceled.".to_string();
                self.state = AppState::Idle;
            }
            WorkerResponse::Error(message) => {
                self.descriptor_cancel_requested = false;
                if let AppState::Computing(active) = &self.state {
                    if active == DESCRIPTOR_COMPUTING_MESSAGE {
                        self.log_descriptor_error_and_continue(&message);
                        return;
                    }
                }
                self.set_error(message);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut terminal = setup_terminal()?;
    let (worker_tx, worker_rx, worker_handle) = spawn_worker();

    let mut app = App::new(worker_tx, worker_rx);
    let run_result = run_app(&mut terminal, &mut app);

    let _ = app.worker_tx.send(WorkerRequest::Shutdown);
    let _ = worker_handle.join();

    restore_terminal(&mut terminal)?;
    run_result?;
    Ok(())
}

fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()
}

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

fn render(frame: &mut Frame, app: &App) {
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(10),
        Constraint::Length(4),
    ])
    .areas(frame.area());

    render_header(frame, header, app);

    match app.active_tab {
        0 => render_import_tab(frame, body, app),
        1 => render_descriptor_tab(frame, body, app),
        2 => render_model_selection_tab(frame, body, app),
        3 => render_hyperparameter_tab(frame, body, app),
        4 => render_predict_visual_tab(frame, body, app),
        _ => {}
    }

    render_footer(frame, footer, app);

    match &app.state {
        AppState::Computing(message) => {
            render_progress_popup(frame, app, message, SPINNER_FRAMES[app.spinner_frame]);
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

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let tab_help = match app.active_tab {
        0 => "Import: f=open file browser (.csv/.smi)",
        1 => "Descriptor: Space toggle, a/n all/none, Enter calculate, Esc cancel active calc, v/c filters, e export matrix",
        2 => "Model Select: m cycle mode (AutoQsar/ART2-A/Manual), a/A manual model, t targets, z scaler, x split",
        3 => "Parameters: ,/. page, 1-6 edit, Enter save, Esc cancel",
        4 => "Analyze: Enter train (Auto/Manual), g ART2-A cluster, p predict file, s export predictions, o color",
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
        "Input source: {}\nPress 'f' to choose .csv/.smi file from browser.",
        source_text
    ))
    .wrap(Wrap { trim: true })
    .block(Block::default().borders(Borders::ALL).title("Dataset Import"));
    frame.render_widget(top_widget, top);

    let property_count = app.loaded_properties.first().map(|row| row.len()).unwrap_or(0);
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
        "No numeric target columns detected.".to_string()
    } else {
        app.target_columns
            .iter()
            .take(20)
            .enumerate()
            .map(|(i, name)| format!("{:>2}. {}", i + 1, name))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let preview = Paragraph::new(target_preview)
        .wrap(Wrap { trim: true })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Detected Numeric Property Columns"),
        );
    frame.render_widget(preview, bottom);
}

fn render_descriptor_tab(frame: &mut Frame, area: Rect, app: &App) {
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(38), Constraint::Percentage(62)]).areas(area);

    let selected_count = app.selected_descriptors().len();
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
        "Input Dataset\n\nSource file: {}\nLoaded molecules: {}\nFeature matrix: {}\n\nDescriptor selection:\nEnabled {}/{} descriptors\n\nActions:\n- Press 'f' to open file browser\n- Press Enter to calculate selected descriptors\n- Press Esc while computing to cancel descriptor calculation\n- Press 'v' variance-threshold filter\n- Press 'c' correlation filter\n- Press 'e' export current feature matrix CSV",
        source_text,
        app.loaded_molecules.len(),
        matrix_text,
        selected_count,
        app.available_descriptors.len(),
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
            ListItem::new(format!("[{}] {}", marker, descriptor.name()))
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
            format!("{}\n\nCurrent manual model:\n{}", body, app.manual_algo_selection.label())
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
        truncate_label(&selected_targets.join(", "), 64)
    };
    let overlay = centered_rect(96, 26, area);
    let info = Paragraph::new(format!(
        "Selected mode: {} (m to cycle)\nTargets: {} (t to choose)\nScaler: {} (z)\nSplit: {} (x)\nManual model: {} (a/A)",
        app.selected_ml_mode.label(),
        target_summary,
        app.training_config.scaler.label(),
        app.training_config.split_strategy.label(),
        app.manual_algo_selection.label()
    ))
    .wrap(Wrap { trim: true })
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Selection Controls"),
    );
    frame.render_widget(info, overlay);
}

fn render_hyperparameter_tab(frame: &mut Frame, area: Rect, app: &App) {
    let [top, bottom] =
        Layout::vertical([Constraint::Length(6), Constraint::Min(8)]).areas(area);

    let mode_text = match app.selected_ml_mode {
        MlMode::AutoQsar => {
            "AutoQsar mode: model-specific knobs are hidden; common controls still editable."
        }
        MlMode::Art2a => {
            "ART2-A mode: edit vigilance/learning-rate/max-iter; common controls still editable."
        }
        MlMode::Manual => "Manual mode: edit selected model hyperparameters below.",
    };
    let top_widget = Paragraph::new(mode_text)
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("Parameter Mode"));
    frame.render_widget(top_widget, top);

    let mut lines = Vec::new();
    lines.push(format!(
        "Page {}/{} (','/'.' to cycle)",
        app.ml_param_page.min(app.ml_param_page_count().saturating_sub(1)) + 1,
        app.ml_param_page_count()
    ));
    lines.push(String::new());
    for entry in app.ml_param_entries() {
        lines.push(format!("{}: {} = {}", entry.hotkey, entry.label, entry.value));
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

    if app.active_tab == 3 && app.editing_ml_param.is_some() && matches!(app.state, AppState::Idle) {
        let cursor_x = bottom
            .x
            .saturating_add(2)
            .saturating_add(app.ml_param_input.visual_cursor() as u16);
        let cursor_y = bottom.y.saturating_add(bottom.height.saturating_sub(2));
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

fn render_predict_visual_tab(frame: &mut Frame, area: Rect, app: &App) {
    let [top, bottom] =
        Layout::vertical([Constraint::Percentage(58), Constraint::Percentage(42)]).areas(area);

    render_pca_panel(frame, top, app);

    let [regression_area, right_area] =
        Layout::horizontal([Constraint::Percentage(52), Constraint::Percentage(48)]).areas(bottom);
    render_regression_diagnostics_panel(frame, regression_area, app);

    let [importance_area, profile_area, prediction_area, cluster_area] = Layout::vertical([
        Constraint::Percentage(24),
        Constraint::Percentage(36),
        Constraint::Percentage(20),
        Constraint::Percentage(20),
    ])
    .areas(right_area);
    render_feature_importance_panel(frame, importance_area, app);
    render_descriptor_profile_table(frame, profile_area, app);
    render_prediction_summary_panel(frame, prediction_area, app);
    render_art2a_cluster_summary(frame, cluster_area, app);
}

fn render_pca_panel(frame: &mut Frame, area: Rect, app: &App) {
    if let Some(projection) = &app.projection {
        let [canvas_area, info_area] =
            Layout::horizontal([Constraint::Percentage(72), Constraint::Percentage(28)]).areas(area);

        let points_all = projection.points.clone();
        let maybe_target_overlay = bucket_points_by_values(
            &projection.points,
            app.target_overlay_values.as_deref().unwrap_or(&[]),
        );
        let cluster_buckets =
            bucket_points_by_clusters(&projection.points, &app.cluster_labels, 6);
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

        let canvas = Canvas::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("PCA Scatter (PC1 vs PC2)"),
            )
            .x_bounds(projection.x_bounds)
            .y_bounds(projection.y_bounds)
            .paint(move |ctx| {
                match color_mode {
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
                }
            });
        frame.render_widget(canvas, canvas_area);

        let (ad_in_count, ad_out_count) = ad_buckets
            .as_ref()
            .map(|(in_domain, out_domain)| (in_domain.len(), out_domain.len()))
            .unwrap_or((0, 0));

        let info = Paragraph::new(format!(
            "Projection Summary\n\nPoints: {}\nPC1 variance: {:.1}%\nPC2 variance: {:.1}%\nTotal: {:.1}%\n\nX range: [{:.2}, {:.2}]\nY range: [{:.2}, {:.2}]\n\nColor mode: {}\nClusters: {}\nTarget overlay: {}\nAD in/out: {}/{}",
            projection.points.len(),
            projection.explained_variance[0] * 100.0,
            projection.explained_variance[1] * 100.0,
            (projection.explained_variance[0] + projection.explained_variance[1]) * 100.0,
            projection.x_bounds[0],
            projection.x_bounds[1],
            projection.y_bounds[0],
            projection.y_bounds[1],
            color_mode.label(),
            app.cluster_sizes.len(),
            if maybe_target_overlay.is_some() {
                "available"
            } else if app.target_overlay_error.is_some() {
                "invalid target"
            } else if app.selected_target_columns.is_empty() {
                "disabled (no target)"
            } else {
                "unavailable"
            },
            ad_in_count,
            ad_out_count,
        ))
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("PCA Details"));
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

fn render_regression_diagnostics_panel(frame: &mut Frame, area: Rect, app: &App) {
    if app.train_actuals.is_empty() || app.train_actuals.len() != app.training_predictions.len() {
        let placeholder = Paragraph::new(
            "No regression diagnostics available.\n\nTrain a model in Tab 2 to view Predicted vs Actual and residual diagnostics.",
        )
        .wrap(Wrap { trim: true })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Regression Diagnostics"),
        );
        frame.render_widget(placeholder, area);
        return;
    }

    let [plot_area, details_area] =
        Layout::vertical([Constraint::Percentage(66), Constraint::Percentage(34)]).areas(area);

    let min_actual = app.train_actuals.iter().copied().fold(f64::INFINITY, f64::min);
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
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Predicted vs Actual"),
        )
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
        Layout::horizontal([Constraint::Percentage(54), Constraint::Percentage(46)]).areas(details_area);

    let stats = Paragraph::new(format!(
        "Train n: {}\nTrain R2: {:.4} | RMSE: {:.4} | MAE: {:.4}\nTrain Pearson r: {:.4}\nResidual bias: {:.4} | Residual σ: {:.4}\n|Residual| p95: {:.4}\nOutliers (>2σ / >3σ): {} / {}\nHoldout R2: {:.4} | RMSE: {:.4} | MAE: {:.4}",
        app.train_actuals.len(),
        app.trained_pipeline.as_ref().map_or(0.0, |p| p.train_r2),
        app.last_train_rmse.unwrap_or(0.0),
        app.last_train_mae.unwrap_or(0.0),
        app.last_train_pearson.unwrap_or(0.0),
        bias,
        residual_std,
        p95_abs,
        outliers_2sigma,
        outliers_3sigma,
        app.last_holdout_r2.unwrap_or(0.0),
        app.last_holdout_rmse.unwrap_or(0.0),
        app.last_holdout_mae.unwrap_or(0.0),
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

fn render_feature_importance_panel(frame: &mut Frame, area: Rect, app: &App) {
    if app.top_feature_importances.is_empty() {
        let placeholder = Paragraph::new(
            "No feature-importance data.\n\nAvailable after training RF, PLS, or Elastic Net models.",
        )
        .wrap(Wrap { trim: true })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Top Feature Importance"),
        );
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
                truncate_label(name, 16),
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
            Constraint::Percentage(40),
            Constraint::Percentage(24),
            Constraint::Percentage(30),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title("Top Feature Importance"));
    frame.render_widget(table, area);
}

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

fn render_progress_popup(frame: &mut Frame, app: &App, message: &str, spinner: char) {
    let area = centered_rect(60, 28, frame.area());
    frame.render_widget(Clear, area);
    let [text_area, gauge_area] =
        Layout::vertical([Constraint::Length(5), Constraint::Length(3)]).areas(area);

    let mut progress_hint = format!("{} {}", spinner, "Working...");
    if message == DESCRIPTOR_COMPUTING_MESSAGE {
        if app.descriptor_cancel_requested {
            progress_hint = "Cancellation requested... finishing current row.".to_string();
        } else {
            progress_hint.push_str("\nPress Esc to cancel descriptor calculation.");
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
    };

    let header_widget = Paragraph::new(format!(
        "Select {} file (.csv/.smi)\nCurrent directory: {}",
        mode,
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
    .block(Block::default().borders(Borders::ALL).title("Target Selection"));
    frame.render_widget(header_widget, header);

    let items: Vec<ListItem> = app
        .target_columns
        .iter()
        .map(|name| {
            let checked = if app.selected_target_columns.contains(name.as_str()) {
                "[x]"
            } else {
                "[ ]"
            };
            ListItem::new(format!("{} {}", checked, name))
        })
        .collect();
    let mut list_state = ListState::default();
    list_state.select(app.target_column_list_state.selected());

    let list = if items.is_empty() {
        List::new(vec![ListItem::new("(no numeric target columns found)")])
    } else {
        List::new(items)
    }
    .block(Block::default().borders(Borders::ALL).title("Numeric columns"))
    .highlight_symbol(">> ")
    .highlight_style(
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
    );
    frame.render_stateful_widget(list, list_area, &mut list_state);

    let footer_widget = Paragraph::new(
        "Up/Down: move  Space: toggle  a: all  n: none  Enter/Esc: close",
    )
    .wrap(Wrap { trim: true })
    .block(Block::default().borders(Borders::ALL).title("Selection Keys"));
    frame.render_widget(footer_widget, footer);
}

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

fn render_prediction_summary_panel(frame: &mut Frame, area: Rect, app: &App) {
    if app.prediction_rows.is_empty() {
        let placeholder = Paragraph::new(
            "No predictions loaded.\nUse 'p' to run prediction and 's' to export.",
        )
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("Prediction Summary"));
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
            format!("#{} {} -> {:.4}", idx + 1, truncate_label(&row.smiles, 22), first)
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
    .block(Block::default().borders(Borders::ALL).title("Prediction Summary"));
    frame.render_widget(widget, area);
}

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

fn spawn_worker() -> (
    Sender<WorkerRequest>,
    Receiver<WorkerResponse>,
    thread::JoinHandle<()>,
) {
    let (request_tx, request_rx) = mpsc::channel::<WorkerRequest>();
    let (response_tx, response_rx) = mpsc::channel::<WorkerResponse>();
    let descriptor_cancel_flag = Arc::new(AtomicBool::new(false));

    let handle = thread::spawn(move || worker_loop(request_rx, response_tx, descriptor_cancel_flag));

    (request_tx, response_rx, handle)
}

fn worker_loop(
    request_rx: Receiver<WorkerRequest>,
    response_tx: Sender<WorkerResponse>,
    descriptor_cancel_flag: Arc<AtomicBool>,
) {
    while let Ok(request) = request_rx.recv() {
        if let WorkerRequest::Shutdown = &request {
            return;
        }
        let request_label = match &request {
            WorkerRequest::LoadDataset(_) => "LoadDataset",
            WorkerRequest::CancelDescriptors => "CancelDescriptors",
            WorkerRequest::CalculateDescriptors { .. } => "CalculateDescriptors",
            WorkerRequest::TrainModel { .. } => "TrainModel",
            WorkerRequest::RunArt2aClustering { .. } => "RunArt2aClustering",
            WorkerRequest::PredictFile { .. } => "PredictFile",
            WorkerRequest::ExportPredictions { .. } => "ExportPredictions",
            WorkerRequest::ApplyVarianceFilter { .. } => "ApplyVarianceFilter",
            WorkerRequest::ApplyCorrelationFilter { .. } => "ApplyCorrelationFilter",
            WorkerRequest::ExportFeatureMatrix { .. } => "ExportFeatureMatrix",
            WorkerRequest::Shutdown => unreachable!("Shutdown handled before request execution"),
        };

        if matches!(request, WorkerRequest::CancelDescriptors) {
            descriptor_cancel_flag.store(true, AtomicOrdering::Relaxed);
            continue;
        }

        let _ = response_tx.send(WorkerResponse::Progress(0.0));

        descriptor_cancel_flag.store(false, AtomicOrdering::Relaxed);
        let cancel_flag = Arc::clone(&descriptor_cancel_flag);
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
            } => handle_train_model(matrix, target_series, smiles, config, Some(&response_tx)),
            WorkerRequest::RunArt2aClustering {
                matrix,
                config,
                scaler,
            } => {
                handle_art2a_clustering(matrix, config, scaler)
            }
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
            ),
            WorkerRequest::ExportPredictions {
                path,
                target_names,
                rows,
            } => {
                handle_export_predictions(path, target_names, rows)
            }
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
            WorkerRequest::CancelDescriptors => unreachable!("handled before request execution"),
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

fn handle_load_dataset(path: PathBuf) -> Result<WorkerResponse, String> {
    let extension = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    let (molecules, smiles, properties, failed_rows) = match extension.as_str() {
        "csv" => load_dataset_from_csv(&path)?,
        "smi" | "smiles" => load_dataset_from_smi(&path)?,
        _ => {
            return Err(format!(
                "Unsupported file extension for '{}'. Use .csv or .smi",
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

fn handle_calculate_descriptors(
    mut molecules: Vec<Molecule>,
    descriptors: Vec<BatchDescriptor>,
    progress_tx: Option<&Sender<WorkerResponse>>,
    cancel_flag: &Arc<AtomicBool>,
) -> Result<WorkerResponse, String> {
    if cancel_flag.load(AtomicOrdering::Relaxed) {
        return Ok(WorkerResponse::DescriptorsCanceled);
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
    let flat = match calculate_batch_flat_static_cancellable(
        &mut molecules,
        &descriptors,
        &|| cancel_flag.load(AtomicOrdering::Relaxed),
    ) {
        Ok(flat) => flat,
        Err(()) => return Ok(WorkerResponse::DescriptorsCanceled),
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

fn handle_train_model(
    matrix: FeatureMatrixPayload,
    target_series: Vec<(String, Vec<f64>)>,
    smiles: Vec<String>,
    config: MlTrainingConfig,
    progress_tx: Option<&Sender<WorkerResponse>>,
) -> Result<WorkerResponse, String> {
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

    let total_targets = target_series.len().max(1);
    for (target_idx, (target_name, targets)) in target_series.into_iter().enumerate() {
        if let Some(tx) = progress_tx {
            let start = (target_idx as f32 / total_targets as f32).clamp(0.0, 1.0);
            let _ = tx.send(WorkerResponse::Progress((0.10 + 0.80 * start).min(0.95)));
        }
        let features = Array2::from_shape_vec((matrix.rows, matrix.cols), matrix.data.clone())
            .map_err(|error| format!("Failed to shape feature matrix for training: {}", error))?;
        let dataset = QsarDataset::new(
            features,
            Array1::from(targets),
            matrix.feature_names.clone(),
        );
        let split = split_dataset(&dataset, &smiles, &config)?;
        if split.train.n_samples() < 2 || split.test.n_samples() < 1 {
            return Err("Split produced insufficient train/test rows for diagnostics.".to_string());
        }

        let pipeline = train_pipeline(&dataset, &config)?;
        let train_predictions = pipeline.predict(&dataset.features).to_vec();
        let train_actuals = dataset.targets.to_vec();
        let ad_in_domain_flags = (0..dataset.features.nrows())
            .map(|row_idx| {
                let sample = dataset.features.row(row_idx).to_vec();
                pipeline.is_in_domain(&sample)
            })
            .collect::<Vec<_>>();
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
        let split_label = config.split_strategy.label().to_string();
        let split_train_size = split.train.n_samples();
        let split_test_size = split.test.n_samples();

        pipeline_bundle.push((target_name.clone(), pipeline.clone()));
        target_summaries.push(TargetTrainingSummary {
            target: target_name.clone(),
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
        });

        if target_idx == 0 {
            primary_target = target_name;
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

        if let Some(tx) = progress_tx {
            let done = ((target_idx + 1) as f32 / total_targets as f32).clamp(0.0, 1.0);
            let _ = tx.send(WorkerResponse::Progress((0.10 + 0.80 * done).min(0.96)));
        }
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
        preview_predictions: primary_preview_predictions,
        train_predictions: primary_train_predictions,
    })
}

fn handle_art2a_clustering(
    matrix: FeatureMatrixPayload,
    config: Art2aConfig,
    scaler: MlScalerChoice,
) -> Result<WorkerResponse, String> {
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

    let data = Array2::from_shape_vec((matrix.rows, matrix.cols), matrix.data.clone())
        .map_err(|error| format!("Failed to shape matrix for ART2-A: {}", error))?;
    let (fitted_scaler, scaled) = match scaler {
        MlScalerChoice::Standard => {
            let scaler = StandardScaler::fit(&data);
            let scaled = scaler.transform(&data);
            (ScalerType::Standard(scaler), scaled)
        }
        MlScalerChoice::MinMax => {
            let scaler = MinMaxScaler::fit(&data);
            let scaled = scaler.transform(&data);
            (ScalerType::MinMax(scaler), scaled)
        }
    };

    let vectors: Vec<Vec<f64>> = scaled.rows().into_iter().map(|r| r.to_vec()).collect();
    let mut model = Art2a::new(config.vigilance, config.learning_rate, config.max_iter);
    model.fit(&vectors);
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
        let (pred_cluster, similarity) = model.predict_one(row);
        let assigned_cluster = labels[row_idx];
        debug_assert_eq!(pred_cluster, assigned_cluster);
        if let Some(bucket) = sim_buckets.get_mut(assigned_cluster) {
            bucket.push(similarity);
        }
    }
    let total = labels.len().max(1) as f64;
    let mut summaries = Vec::with_capacity(cluster_sizes.len());
    for (cluster_id, &size) in cluster_sizes.iter().enumerate() {
        let sims = sim_buckets.get(cluster_id).map_or(&[][..], |v| v.as_slice());
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
    Ok(WorkerResponse::ClusteringCompleted {
        labels,
        cluster_sizes,
        summaries,
        art2a_model_bytes,
    })
}

fn handle_predict_file(
    path: PathBuf,
    pipeline_bundle_bytes: Option<Vec<u8>>,
    art2a_model_bytes: Option<Vec<u8>>,
    descriptors: Vec<BatchDescriptor>,
    selected_feature_indices: Vec<usize>,
    expected_feature_names: Vec<String>,
) -> Result<WorkerResponse, String> {
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

    let (mut molecules, smiles, _properties, failed_rows) = match extension.as_str() {
        "csv" => load_dataset_from_csv(&path)?,
        "smi" | "smiles" => load_dataset_from_smi(&path)?,
        _ => {
            return Err(format!(
                "Unsupported prediction input '{}'. Use .csv or .smi",
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

    let flat = calculate_batch_flat_static(&mut molecules, &descriptors);
    if flat.cols != descriptor_feature_names(&descriptors).len() {
        return Err(format!(
            "Prediction source feature mismatch: got {} columns from descriptor set.",
            flat.cols
        ));
    }
    let selected_flat = apply_feature_subset_to_flat(&flat, &selected_feature_indices).ok_or_else(
        || {
            format!(
                "Prediction feature subset invalid: {} selected columns for {} source columns.",
                selected_feature_indices.len(),
                flat.cols
            )
        },
    )?;

    let features = Array2::from_shape_vec(
        (selected_flat.rows, selected_flat.cols),
        selected_flat.data.into_iter().map(|v| v as f64).collect(),
    )
    .map_err(|error| format!("Failed to shape prediction feature matrix: {}", error))?;

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
                selected_flat.cols,
                model_feature_count
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
                expected_feature_count,
                selected_flat.cols
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
            in_domains.push(if model.is_in_domain(&sample) { 1.0 } else { 0.0 });
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
    })
}

fn decode_pipeline_bundle(
    bytes: &[u8],
    primary_target: &str,
) -> Result<QsarPipeline, String> {
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

fn train_pipeline(data: &QsarDataset, config: &MlTrainingConfig) -> Result<QsarPipeline, String> {
    if data.n_samples() < 3 {
        return Err("Need at least 3 samples for training.".to_string());
    }
    let cv_folds = config.cv_folds.min(data.n_samples()).max(2);

    if config.algorithm == MlAlgorithm::AutoQsar {
        if config.scaler != MlScalerChoice::Standard {
            return Err(
                "AutoQsar currently requires StandardScaler. Switch scaler to Standard.".to_string(),
            );
        }
        return Ok(ferrum_core::ml::pipeline::AutoQsar::regression(
            data,
            cv_folds,
            config.seed,
        ));
    }

    let (scaler, scaled) = match config.scaler {
        MlScalerChoice::Standard => {
            let scaler = StandardScaler::fit(&data.features);
            let scaled = scaler.transform(&data.features);
            (ScalerType::Standard(scaler), scaled)
        }
        MlScalerChoice::MinMax => {
            let scaler = MinMaxScaler::fit(&data.features);
            let scaled = scaler.transform(&data.features);
            (ScalerType::MinMax(scaler), scaled)
        }
    };

    match config.algorithm {
        MlAlgorithm::AutoQsar => unreachable!(),
        MlAlgorithm::RandomForest => train_rf_pipeline(data, scaled, scaler, config, cv_folds),
        MlAlgorithm::Knn => train_knn_pipeline(data, scaled, scaler, config, cv_folds),
        MlAlgorithm::Pls => train_pls_pipeline(data, scaled, scaler, config, cv_folds),
        MlAlgorithm::ElasticNet => train_elastic_net_pipeline(data, scaled, scaler, config, cv_folds),
    }
}

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
                    let bond = &clone.bonds[bond_idx];
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
            let model = RandomForestRegressor::fit(
                train_x,
                train_y,
                n_trees,
                max_depth,
                min_samples_split,
                max_features_fraction,
                config.seed,
            );
            ModelType::RandomForestRegressor(model)
        },
    );

    let model = RandomForestRegressor::fit(
        &scaled,
        &data.targets,
        n_trees,
        max_depth,
        min_samples_split,
        max_features_fraction,
        config.seed,
    );
    let train_predictions = model.predict(&scaled);
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

fn train_knn_pipeline(
    data: &QsarDataset,
    scaled: Array2<f64>,
    scaler: ScalerType,
    config: &MlTrainingConfig,
    cv_folds: usize,
) -> Result<QsarPipeline, String> {
    let k = config.knn_k.max(1).min(data.n_samples().saturating_sub(1).max(1));

    let q2 = cv_regression_q2_configurable(
        &scaled,
        &data.targets,
        cv_folds,
        config.seed,
        |train_x, train_y| {
            let train_k = k.min(train_x.nrows().max(1));
            let model = KnnRegressor::fit(
                train_x.clone(),
                train_y.clone(),
                train_k,
                DistanceMetric::Euclidean,
            );
            ModelType::KnnRegressor(model)
        },
    );

    let model = KnnRegressor::fit(
        scaled.clone(),
        data.targets.clone(),
        k.min(scaled.nrows().max(1)),
        DistanceMetric::Euclidean,
    );
    let train_predictions = model.predict(&scaled);
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
            let model = PlsRegression::fit(train_x, train_y, nc);
            ModelType::PlsRegression(model)
        },
    );

    let model = PlsRegression::fit(&scaled, &data.targets, components);
    let train_predictions = model.predict(&scaled);
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
            let model = ElasticNet::fit(train_x, train_y, alpha, l1_ratio, max_iter, tol);
            ModelType::ElasticNet(model)
        },
    );

    let model = ElasticNet::fit(&scaled, &data.targets, alpha, l1_ratio, max_iter, tol);
    let train_predictions = model.predict(&scaled);
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

    let target_values = targets.to_vec();
    metrics::q_squared(&target_values, &all_preds)
}

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

fn apply_feature_subset_to_flat(
    matrix: &FlatDescriptorMatrix,
    selected_feature_indices: &[usize],
) -> Option<FlatDescriptorMatrix> {
    if matrix.cols == 0 {
        return None;
    }
    if selected_feature_indices.is_empty() {
        return None;
    }
    if selected_feature_indices.iter().any(|&idx| idx >= matrix.cols) {
        return None;
    }
    let mut data = Vec::with_capacity(matrix.rows * selected_feature_indices.len());
    for row in 0..matrix.rows {
        let start = row * matrix.cols;
        for &col in selected_feature_indices {
            data.push(matrix.data[start + col]);
        }
    }
    Some(FlatDescriptorMatrix {
        rows: matrix.rows,
        cols: selected_feature_indices.len(),
        data,
    })
}

fn load_dataset_from_csv(
    path: &Path,
) -> Result<
    (
        Vec<Molecule>,
        Vec<String>,
        Vec<Vec<(String, String)>>,
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

fn load_dataset_from_smi(
    path: &Path,
) -> Result<
    (
        Vec<Molecule>,
        Vec<String>,
        Vec<Vec<(String, String)>>,
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

fn extract_targets(
    properties: &[Vec<(String, String)>],
    target_column: &str,
) -> Result<Vec<f64>, String> {
    let mut targets = Vec::with_capacity(properties.len());

    for (row_idx, row_props) in properties.iter().enumerate() {
        let maybe_value = row_props
            .iter()
            .find(|(name, _)| name == target_column)
            .or_else(|| {
                row_props
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(target_column))
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

fn build_projection(matrix: &FeatureMatrixPayload) -> Option<ProjectionData> {
    if matrix.rows < 3 || matrix.cols < 2 {
        return None;
    }

    let data = Array2::from_shape_vec((matrix.rows, matrix.cols), matrix.data.clone()).ok()?;
    let scaler = StandardScaler::fit(&data);
    let scaled = scaler.transform(&data);
    let (pca, projected) = Pca::fit_transform(&scaled, 2);

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
            *pca.explained_variance_ratio.get(0).unwrap_or(&0.0),
            *pca.explained_variance_ratio.get(1).unwrap_or(&0.0),
        ],
    })
}

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
        ModelType::PlsRegression(model) => Some(model.coefficients.iter().map(|v| v.abs()).collect()),
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

fn read_browser_entries(dir: &Path) -> io::Result<Vec<PathBuf>> {
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

        if path.is_dir() || is_supported_data_file(&path) {
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

fn is_supported_data_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("csv") | Some("smi") | Some("smiles")
    )
}
